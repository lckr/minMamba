from setuptools import setup

setup(
    name="minMamba",
    version="0.0.1",
    author="lckr",
    packages=["minmamba"],
    description="A PyTorch re-implementation of Mamba",
    license="MIT",
    install_requires=[
        "torch",
    ],
)
