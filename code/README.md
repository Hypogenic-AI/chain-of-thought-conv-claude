# Cloned Repositories

## 1. introspective-self-consistency
- **URL**: https://github.com/JacobPfau/introspective-self-consistency
- **Paper**: "Self-Consistency of Large Language Models under Ambiguity" (2023)
- **Purpose**: Benchmark and evaluation code for measuring cross-context self-consistency of LLMs on ambiguous integer sequences. Includes dataset of 197 sequences and evaluation scripts.
- **Location**: code/introspective-self-consistency/
- **Key files**: main.py, src/, data/, conf/ (Hydra configs)
- **Notes**: Uses GPT API calls; requires OpenAI API key. Can be adapted to test CoT vs non-CoT consistency.

## 2. fillerTokens
- **URL**: https://github.com/JacobPfau/fillerTokens
- **Paper**: "Let's Think Dot by Dot: Hidden Computation in Transformer Language Models" (2024)
- **Purpose**: Training and evaluation code for studying whether meaningless filler tokens can substitute for chain-of-thought reasoning. Demonstrates that computational benefit of CoT can come from token budget, not semantic content.
- **Location**: code/fillerTokens/
- **Key files**: scripts/, src/, data/
- **Notes**: Trains small LLaMA models from scratch on synthetic tasks. Useful for understanding what CoT actually provides computationally.

## 3. pararel
- **URL**: https://github.com/yanaiela/pararel
- **Paper**: "Measuring and Improving Consistency in Pretrained Language Models" (2021)
- **Purpose**: PARAREL benchmark - 328 paraphrase patterns for 38 factual relations. Measures consistency of factual knowledge retrieval across paraphrases in masked LMs.
- **Location**: code/pararel/
- **Key files**: pararel/ (Python package), data/, notebooks/
- **Notes**: Pre-CoT baseline for consistency measurement. Provides the paradigm of measuring consistency via paraphrased queries that can be extended to CoT settings.
