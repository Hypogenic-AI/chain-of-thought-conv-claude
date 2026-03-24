# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project investigating whether Chain of Thought (CoT) prompting causes models to converge more in their representations, outputs, and reasoning paths.

## Papers

Total papers downloaded: 25

| # | Title | Year | Citations | File |
|---|-------|------|-----------|------|
| 1 | Chain of Thought Prompting Elicits Reasoning in LLMs | 2022 | 16,531 | papers/2201_11903_*.pdf |
| 2 | Self-Consistency Improves CoT Reasoning | 2022 | 6,155 | papers/2203_11171_*.pdf |
| 3 | Large Language Models are Zero-Shot Reasoners | 2022 | 6,636 | papers/2205_11916_*.pdf |
| 4 | STaR: Bootstrapping Reasoning With Reasoning | 2022 | 780 | papers/2203_14465_*.pdf |
| 5 | Measuring and Improving Consistency in PLMs | 2021 | 455 | papers/2102_01017_*.pdf |
| 6 | Answer Convergence as Early Stopping Signal | 2025 | 27 | papers/2506_02536_*.pdf |
| 7 | Mapping the Minds of LLMs (Graph Analysis) | 2025 | 11 | papers/2505_13890_*.pdf |
| 8 | Two Failures of Self-Consistency | 2023 | 53 | papers/2305_14279_*.pdf |
| 9 | Robust Answers, Fragile Logic (Decoupling) | 2025 | 5 | papers/2505_17406_*.pdf |
| 10 | Let's Think Dot by Dot (Filler Tokens) | 2024 | 152 | papers/2404_15758_*.pdf |
| 11 | Representational Capacity with CoT | 2024 | 26 | papers/2406_14197_*.pdf |
| 12 | Feature Extraction and Steering for CoT | 2025 | 13 | papers/2505_15634_*.pdf |
| 13 | Self-Consistency under Ambiguity | 2023 | 19 | papers/2310_13439_*.pdf |
| 14 | Why Can LLMs Generate Correct CoTs? | 2023 | 21 | papers/2310_13571_*.pdf |
| 15 | Quiet-STaR: Think Before Speaking | 2024 | 247 | papers/2403_09629_*.pdf |
| 16 | Think Deep, Not Just Long | 2026 | 1 | papers/2602_13517_*.pdf |
| 17 | Molecular Structure of Thought | 2026 | 0 | papers/2601_06002_*.pdf |
| 18 | Compressed Chain of Thought | 2024 | 133 | papers/2412_13171_*.pdf |
| 19 | Why think step-by-step? | 2023 | 142 | papers/2304_03843_*.pdf |
| 20 | Formal Comparison: CoT vs Latent Thought | 2025 | 3 | papers/2509_25239_*.pdf |
| 21 | Training LLMs to Reason in Latent Space | 2024 | 397 | papers/2412_06769_*.pdf |
| 22 | Contrastive Chain-of-Thought Prompting | 2023 | 50 | papers/2311_09277_*.pdf |
| 23 | CoT is Not True Reasoning | 2025 | 3 | papers/2506_02878_*.pdf |
| 24 | Hardness of Faithful CoT | 2024 | 23 | papers/2406_10625_*.pdf |
| 25 | Propositional Interpretability in AI | 2025 | 16 | papers/2501_15740_*.pdf |

See papers/README.md for detailed descriptions of each paper.

## Datasets

Total datasets downloaded: 3

| Name | Source | Eval Size | Task | Location |
|------|--------|-----------|------|----------|
| GSM8K | HuggingFace (openai/gsm8k) | 1,319 test | Math reasoning | datasets/gsm8k/ |
| CommonsenseQA | HuggingFace (tau/commonsense_qa) | 1,221 val | Commonsense QA | datasets/commonsenseqa/ |
| ARC-Challenge | HuggingFace (allenai/ai2_arc) | 1,172 test | Science QA | datasets/arc_challenge/ |

See datasets/README.md for download instructions and format details.

## Code Repositories

Total repositories cloned: 3

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| introspective-self-consistency | github.com/JacobPfau/introspective-self-consistency | Self-consistency measurement | code/introspective-self-consistency/ |
| fillerTokens | github.com/JacobPfau/fillerTokens | Filler token experiments | code/fillerTokens/ |
| pararel | github.com/yanaiela/pararel | Paraphrase consistency benchmark | code/pararel/ |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with diligent mode for "chain of thought reasoning convergence language models" (123 results)
2. Curated 27 most relevant papers based on relevance scores and direct applicability to convergence question
3. Downloaded 25 papers successfully from arXiv (2 without arXiv IDs were skipped)
4. Deep-read 12 papers with full chunk-by-chunk analysis; abstract-reviewed remaining 13
5. Datasets selected based on frequency of use in reviewed papers and coverage of reasoning types

### Selection Criteria
- Papers: Prioritized those directly studying CoT mechanisms, consistency, convergence, or representational analysis. Included foundational papers (Wei 2022, Wang 2022) and recent analytical work.
- Datasets: Selected the three most commonly used benchmarks covering math, commonsense, and science reasoning — sufficient for diverse convergence testing.
- Code: Cloned repos with reusable evaluation frameworks for consistency measurement.

### Challenges Encountered
- Semantic Scholar API rate limiting required multiple retry cycles for arXiv ID lookups
- 2 papers lacked arXiv IDs (one about LLM thought divergence/convergence for image generation — less relevant)
- cot-lms repo (Merrill & Sabharwal) not found on GitHub despite being referenced in paper

### Gaps and Workarounds
- No existing codebase directly implements "CoT convergence measurement" — this will need to be built
- Most papers study output-level convergence; representation-level convergence tools need to be developed
- No cross-model convergence studies exist in the literature — this is a key novel contribution

## Recommendations for Experiment Design

### 1. Primary Datasets
- **GSM8K test** (1,319 examples): Universal CoT benchmark, math reasoning
- **CommonsenseQA val** (1,221 examples): Non-mathematical reasoning
- **ARC-Challenge test** (1,172 examples): Science domain for cross-domain testing

### 2. Baseline Methods
- Direct prompting (no CoT) — primary control
- Zero-shot CoT ("Let's think step by step")
- Few-shot CoT (manual exemplars from Wei et al.)
- Self-consistency (Wang et al.) — majority vote over samples

### 3. Evaluation Metrics
- **Answer agreement rate**: % of samples producing same final answer
- **Output embedding similarity**: Cosine similarity between sentence embeddings of full responses
- **Output entropy**: Shannon entropy of answer distribution
- **Reasoning path similarity**: BLEU/ROUGE between reasoning chains
- **Accuracy**: Standard task performance for context

### 4. Code to Adapt/Reuse
- **introspective-self-consistency**: Framework for measuring cross-context consistency; adapt for CoT vs non-CoT comparison
- **pararel**: Paraphrase-based consistency testing paradigm; extend to reasoning tasks
- **fillerTokens**: Control condition (filler vs CoT) for isolating semantic vs computational effects

### 5. Experimental Design
- Compare within-model convergence (same model, multiple samples, CoT vs non-CoT)
- Compare cross-model convergence (different models, same prompts, CoT vs non-CoT)
- Use temperature > 0 for sampling (T=0.7 following Wang et al.)
- Generate 20-40 samples per prompt per condition
- Test at least 2-3 different model families (e.g., GPT, Claude, LLaMA)
- Analyze convergence as function of task difficulty
