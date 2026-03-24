# Does Chain of Thought Cause Models to Converge More?

An empirical study investigating whether Chain of Thought (CoT) prompting causes LLM outputs to become more similar — both within a single model across samples, and across different models.

## Key Findings

- **CoT increases convergence on math tasks**: All three GPT-4.1 family models showed significantly higher answer agreement with CoT on GSM8K (Cohen's d up to +2.39, p < 0.001)
- **CoT decreases convergence on MC tasks**: On CommonsenseQA and ARC-Challenge, CoT reduced answer agreement compared to direct prompting (Cohen's d up to -0.91, p < 0.001)
- **Cross-model convergence is dramatically affected**: Models that agree only 15% of the time without CoT on math jump to 94% agreement with CoT
- **Convergence tracks accuracy**: CoT converges outputs toward correct answers on math, but introduces "overthinking" divergence on easier MC tasks
- **Smaller models benefit most**: The convergence-inducing effect of CoT is strongest for the smallest model (nano)

## Project Structure

```
├── REPORT.md              # Full research report with results and analysis
├── planning.md            # Research plan and experimental design
├── literature_review.md   # Comprehensive literature review
├── resources.md           # Catalog of datasets, papers, and code
├── src/
│   ├── experiment.py      # Main experiment script (API calls)
│   └── analysis.py        # Analysis and visualization
├── results/
│   ├── all_responses.json # 2,700 raw API responses
│   ├── metrics_summary.json
│   └── plots/             # 5 visualization files
├── datasets/              # GSM8K, CommonsenseQA, ARC-Challenge
├── papers/                # 25 downloaded research papers
└── code/                  # 3 cloned baseline repositories
```

## Reproduce

```bash
source .venv/bin/activate
export OPENAI_API_KEY=<your-key>
python src/experiment.py   # ~45 min, 2,700 API calls
python src/analysis.py     # Generates metrics and plots
```

## Method

- 3 models: GPT-4.1, GPT-4.1-mini, GPT-4.1-nano
- 3 datasets: GSM8K (math), CommonsenseQA (commonsense), ARC-Challenge (science)
- 2 conditions: Direct prompting vs. zero-shot CoT
- 5 samples per question at temperature=0.7
- Metrics: Answer Agreement Rate, Cross-Model Agreement, Output Entropy, Accuracy
- Statistical tests: Wilcoxon signed-rank (paired by question), Cohen's d effect sizes

See [REPORT.md](REPORT.md) for full details.
