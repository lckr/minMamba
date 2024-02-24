
# minMamba



A simple PyTorch re-implementation of [Mamba](https://github.com/state-spaces/mamba) in a single file. minMamba tries to be small, clean, interpretable and educational.

### Library Installation

If you want to `import minmamba` into your project:

```
git clone https://github.com/lckr/minMamba.git
cd minMamba
pip install -e .
```

### Usage

Here's how you'd load a pretrained Mamba model from Huggingface Hub:

```python
import minmamba.model
pretrained_model = minmamba.model.MambaLMModel.from_pretrained("state-spaces/mamba-130m")
```

And here's how you'd run inference with it:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") # tokenizer used by "state-spaces/mamba-130m"

input_seq = tokenizer("A fish is a ", return_tensors="pt")["input_ids"]
gen_seq = pretrained_model.generate(input_seq, 100)
print(tokenizer.decode(gen_seq[0]))
```


### References

Code:
- [state-spaces/mamba](https://github.com/state-spaces/mamba
) the official Mamba implementation released by the authors
- [karpathy/minGPT](https://github.com/karpathy/minGPT) Andrej Kaparthys mingpt
- [rjb7731/nanoMamba](https://github.com/rjb7731/nanoMamba) Ryan Bradys nanoMamba implementation

Paper:
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> Paper: https://arxiv.org/abs/2312.00752

### License

MIT
