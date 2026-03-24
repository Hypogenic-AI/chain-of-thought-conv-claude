# Datasets for CoT Convergence Research

This directory contains datasets used to study whether Chain-of-Thought (CoT) prompting
causes LLM outputs to converge compared to non-CoT prompting.

## Dataset Summary

| Dataset        | Split       | Examples | Task Type              | Used For                        |
|----------------|-------------|----------|------------------------|---------------------------------|
| GSM8K          | train       | 7,473    | Grade-school math      | Easy arithmetic reasoning       |
| GSM8K          | test        | 1,319    | Grade-school math      | Primary eval split              |
| CommonsenseQA  | train       | 9,741    | Commonsense reasoning  | Non-math reasoning baseline     |
| CommonsenseQA  | validation  | 1,221    | Commonsense reasoning  | Eval (test labels are hidden)   |
| CommonsenseQA  | test        | 1,140    | Commonsense reasoning  | Held-out (no gold labels)       |
| ARC-Challenge  | train       | 1,119    | Science QA (hard)      | Hard multiple-choice reasoning  |
| ARC-Challenge  | validation  | 299      | Science QA (hard)      | Dev split                       |
| ARC-Challenge  | test        | 1,172    | Science QA (hard)      | Primary eval split              |

## Directory Structure

```
datasets/
├── gsm8k/
│   ├── train.jsonl          # 7,473 examples (git-ignored)
│   ├── test.jsonl           # 1,319 examples (git-ignored)
│   └── sample_5.json        # First 5 test examples (tracked)
├── commonsenseqa/
│   ├── train.jsonl          # 9,741 examples (git-ignored)
│   ├── validation.jsonl     # 1,221 examples (git-ignored)
│   ├── test.jsonl           # 1,140 examples (git-ignored)
│   └── sample_5.json        # First 5 validation examples (tracked)
├── arc_challenge/
│   ├── train.jsonl          # 1,119 examples (git-ignored)
│   ├── validation.jsonl     # 299 examples (git-ignored)
│   ├── test.jsonl           # 1,172 examples (git-ignored)
│   └── sample_5.json        # First 5 test examples (tracked)
├── .gitignore               # Excludes *.jsonl files
└── README.md                # This file
```

The `.jsonl` files are excluded from git (see `.gitignore`). Only the `sample_5.json`
files are tracked in version control.

## Re-downloading the Data

Activate the project virtualenv and run the snippet below. Requires the `datasets`
library (`uv pip install datasets`).

```python
from datasets import load_dataset
import json, os

datasets_to_download = [
    ("gsm8k",          "main",          "gsm8k"),
    ("commonsense_qa", None,            "commonsenseqa"),
    ("ai2_arc",        "ARC-Challenge", "arc_challenge"),
]

BASE = "/workspaces/chain-of-thought-conv-claude/datasets"

for hf_name, config, local_dir in datasets_to_download:
    path = os.path.join(BASE, local_dir)
    os.makedirs(path, exist_ok=True)
    ds = load_dataset(hf_name, config) if config else load_dataset(hf_name)
    for split_name, split_data in ds.items():
        out = os.path.join(path, f"{split_name}.jsonl")
        with open(out, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
    print(f"{local_dir}: {dict({k: len(v) for k, v in ds.items()})}")
```

## Dataset Details

### GSM8K
- **Source:** Cobbe et al. (2021), HuggingFace `gsm8k` / config `main`
- **Format:** `{"question": "...", "answer": "...<final answer>"}`. The `answer` field
  contains a step-by-step solution followed by `#### <number>`.
- **Relevance:** Canonical CoT benchmark; answers require multi-step arithmetic.
  Ideal for measuring whether CoT causes models to produce similar reasoning chains.

### CommonsenseQA
- **Source:** Talmor et al. (2019), HuggingFace `commonsense_qa`
- **Format:** Multiple-choice with 5 options (A–E). Gold labels are available for
  train and validation; test labels are withheld by the benchmark.
- **Relevance:** Tests commonsense reasoning rather than math. Useful as a
  non-numerical contrast to GSM8K.

### ARC-Challenge
- **Source:** Clark et al. (2018), HuggingFace `ai2_arc` / config `ARC-Challenge`
- **Format:** Multiple-choice science questions (4 options). The Challenge partition
  contains questions that simple retrieval/co-occurrence methods fail on.
- **Relevance:** Hard reasoning questions that require scientific knowledge; tests
  whether CoT convergence effects generalise beyond math.
