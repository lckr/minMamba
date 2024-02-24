"""
Full definition of a Mamba SSM Language Model, all of it in this single file.
In spirit of Karparthys mingpt with focus on clairity and
without many of the performance-boosting tricks (such as custom GPU kernels)
of the official implementation.

References:
1) the official Mamba implementation released by the authors:
https://github.com/state-spaces/mamba
2) Andrej Kaparthys mingpt
https://github.com/karpathy/minGPT
3) Ryan Bradys nanoMamba implementation
https://github.com/rjb7731/nanoMamba 
"""
import math
import typing
from dataclasses import dataclass

import einops
import torch


@dataclass
class Args:
    d_model: int
    n_layer: int
    vocab_size: int
    bias: bool = False
    conv_bias: bool = True
    d_conv: int = 4
    d_state: int = 16  # Dim of B and C
    dt_rank: int | typing.Literal["auto"] = "auto"  # Rank of 𝚫
    expand: int = 2
    pad_vocab_size_multiple: int = 8

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class MambaBlock(torch.nn.Module):
    """
              ┌──┐            ┌──────┐   ┌────┐   ┌───┐
            ┌─┘  ├─ x ───────►│conv1d├──►│siLU├──►│ssm├───► * ─────► y
         ┌──┘    │            └──────┘   └────┘   └───┘     ▲
    x───►│in_proj│                       ┌────┐             │
         └──┐    ├─ z ──────────────────►│siLU├─────────────┘
            └─┐  │                       └────┘
              └──┘
    """

    def __init__(self, args: Args) -> None:
        super().__init__()

        self.args = args

        self.in_proj = torch.nn.Linear(
            in_features=args.d_model, out_features=args.d_inner * 2, bias=args.bias
        )

        self.conv1d = torch.nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.activation = "silu"
        self.act = torch.nn.SiLU()

        # SSM projection matrices s_B(X), s_C(x), s_𝚫(x)
        self.x_proj = torch.nn.Linear(
            in_features=args.d_inner,
            out_features=args.dt_rank
            + args.d_state * 2,  # delta.shape + B.shape + C.shape
            bias=False,
        )

        # Broadcast 𝚫_rank to d_inner
        self.dt_proj = torch.nn.Linear(
            in_features=args.dt_rank, out_features=args.d_inner, bias=True
        )

        # Global A param
        # S4D real initialization
        A = einops.repeat(torch.arange(1, args.d_state + 1), "n -> d n", d=args.d_inner)
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = torch.nn.Parameter(torch.ones(args.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = torch.nn.Linear(
            in_features=args.d_inner, out_features=args.d_model, bias=args.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (B,L,D) tensor

        Returns:
            torch.Tensor: Shape (B,L,D) tensor
        """
        (B, L, D) = x.shape

        # Input projection
        x_and_z = self.in_proj(x)  # Projection D -> d_inner:  (B,L,2*d_inner)
        (x, z) = x_and_z.split(
            split_size=[self.args.d_inner, self.args.d_inner], dim=-1
        )  # (B,L,d_inner), (B,L,d_inner)

        # Convolution
        x = einops.rearrange(x, "B L d_inner -> B d_inner L")
        x = self.conv1d(x)[:, :, :L]
        x = einops.rearrange(x, "B d_inner L -> B L d_inner")

        # Non-linearity
        x = self.act(x)

        # SSM
        y = self.ssm(x)

        y = y * self.act(z)

        return self.out_proj(y)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        delta, A, B, C, D = self.compute_state_space_params(x)
        A_bar, B_bar = MambaBlock.discretize(delta, A, B)
        y = MambaBlock.selective_scan(x, A_bar, B_bar, C, D)
        return y

    @staticmethod
    def discretize(
        delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        A_bar = torch.exp(
            einops.einsum(delta, A, "B L D, D N -> B D L N")
        )  # Zero-order hold discretization
        B_bar = einops.einsum(delta, B, "B L D, B L N -> B D L N")

        return A_bar, B_bar

    @staticmethod
    def selective_scan(
        x: torch.Tensor,  # (B,L,D)
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
        D: typing.Optional[torch.Tensor] = None,
    ):
        """_summary_
            h(t + 1) = Ah(t) + Bx(t)
            y(t)     = Ch(t) + Dx(t)

        Reference Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86

        Args:
            x (torch.Tensor): (B,L,D) tensor
            A_bar (torch.Tensor): (B,D,L,N) tensor
            B_bar (torch.Tensor): (B,D,L,N) tensor
            C (torch.Tensor): (B,L,D) tensor
            D (torch.Tensor): (D,) tensor - skip connection parameter

        Returns:
            torch.Tensor: Shape (B,L,D) tensor
        """
        batch, seq_len, dim, d_state = (
            x.shape[0],
            x.shape[1],
            A_bar.shape[1],
            A_bar.shape[3],
        )  # B, L, D, N

        B_bar_x = einops.einsum(B_bar, x, "B D L N, B L D -> B D L N")

        # Init hidden states
        h = A_bar.new_zeros((batch, dim, d_state))

        # Linear recurrence
        ys = []
        for i in range(seq_len):
            h = A_bar[:, :, i] * h + B_bar_x[:, :, i]
            y = einops.einsum(h, C[:, i, :], "B D N, B N -> B D")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B,L,D)

        # Skip connections
        if D is not None:
            y = y + x * D

        return y

    def compute_state_space_params(
        self, x: torch.Tensor
    ) -> typing.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:

        # prepare input-independent ssm params
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # Compute input-dependent ssm params
        x_dbl = self.x_proj(x)  # (B,L, delta.shape + B.shape + C.shape)
        delta, B, C = x_dbl.split(
            split_size=[self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1
        )

        delta = self.dt_proj(delta)  # (B,L,D)
        delta = torch.nn.functional.softplus(delta)

        return delta, A, B, C, D


class ResidualMambaBlock(torch.nn.Module):

    def __init__(self, args: Args) -> None:
        super().__init__()
        self.mixer = MambaBlock(args=args)
        self.norm = RMSNorm(normalized_shape=args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class Backbone(torch.nn.Module):
    def __init__(
        self,
        args: Args,
    ):
        self.args = args
        super().__init__()
        self.embedding = torch.nn.Embedding(args.vocab_size, args.d_model)
        self.layers = torch.nn.Sequential(
            *[ResidualMambaBlock(args) for _ in range(args.n_layer)]
        )
        self.norm_f = RMSNorm(normalized_shape=args.d_model)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.layers(x)
        x = self.norm_f(x)
        return x


class MambaLMModel(torch.nn.Module):

    def __init__(self, args: Args, device: typing.Optional[str] = None) -> None:
        self.args = args

        super().__init__()
        self.backbone = Backbone(args=args)
        self.lm_head = torch.nn.Linear(
            in_features=args.d_model, out_features=args.vocab_size, bias=False
        )

        self.tie_weights()

    def tie_weights(self):
        # See "Weight Tying" paper (https://arxiv.org/abs/1608.05859)
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids, position_ids=None):
        x = self.backbone(input_ids)
        logits = self.lm_head(x)
        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str) -> "MambaLMModel":
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        import json

        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        # Load config
        cfg = json.load(
            open(
                cached_file(
                    pretrained_model_name,
                    CONFIG_NAME,
                    _raise_exceptions_for_missing_entries=False,
                ),
                "r",
            )
        )

        args = Args(
            d_model=cfg["d_model"], n_layer=cfg["n_layer"], vocab_size=cfg["vocab_size"]
        )
        model = MambaLMModel(args)

        state_dict = torch.load(
            cached_file(
                pretrained_model_name,
                WEIGHTS_NAME,
                _raise_exceptions_for_missing_entries=False,
            ),
            weights_only=True,
        )

        assert set(model.state_dict().keys()) ^ set(state_dict.keys()) == set(), set(
            model.state_dict().keys()
        ) ^ set(state_dict.keys())

        model.load_state_dict(state_dict=state_dict)

        return model
    
    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):

            # forward the model to get the logits for the index in the sequence
            logits = self(idx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output
