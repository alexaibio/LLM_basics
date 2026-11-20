# A set of working examples to explain various LLM architectures

## Environment (one shared env for all projects)
Create a conda env with PyTorch specific for your platform, then activate it.
uv adds the rest on top (see `pyproject.toml`).
```bash
source activate p311_torch
cd LLM_basics                     
uv venv --python "$CONDA_PREFIX/bin/python" --system-site-packages  # venv can also see conda's packages
uv sync
```

Activate `source .venv/bin/activate` if you use it in terminal

## Attention All You Need
Implementation of famous 2017 articles, a Transformer Encoder-Decoder architechure for machine translation.

Based on Umar Jamil video: https://www.youtube.com/watch?v=ISNdQcPhsts
and github: https://github.com/hkproj/pytorch-transformer

Run:
```bash
cd AttentionAllYouNeed
uv run python train.py
```


## GPT
Implementation of GPT decoder for text generation.

Based on Andrej Karpathy video: https://www.youtube.com/watch?v=kCc8FmEb1nY

