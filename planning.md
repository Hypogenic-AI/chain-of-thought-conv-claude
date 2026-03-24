# Research Plan: Does Chain of Thought Cause Models to Converge More?

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly perceived as converging in their outputs and "persona" — different models giving similar-sounding answers. Chain of Thought (CoT) prompting is now ubiquitous, but its effect on output diversity/convergence across and within models is unstudied. Understanding whether CoT homogenizes model outputs has implications for model selection, ensemble design, and the diversity of AI-generated content.

### Gap in Existing Work
The literature review reveals:
- **Output-level convergence** (answer agreement) is well-studied within single models via self-consistency (Wang et al. 2022)
- **Reasoning-path convergence** has been examined structurally (Xiong et al. 2025) and shown to decouple from answer convergence (MATCHA, 2025)
- **Cross-model convergence** — whether different models produce more similar outputs under CoT — is **completely unexplored**
- No study compares convergence *with vs. without CoT* as the independent variable across model families

### Our Novel Contribution
We conduct the first empirical study of whether CoT increases or decreases **cross-model convergence** (similarity of outputs across different LLMs) and **within-model convergence** (consistency of outputs across multiple samples from the same model). We test this across three reasoning domains (math, commonsense, science) using state-of-the-art models.

### Experiment Justification
- **Experiment 1 (Within-model convergence)**: Measures whether CoT makes a single model's outputs more self-consistent. Needed to establish baseline convergence effect before cross-model comparison.
- **Experiment 2 (Cross-model convergence)**: Measures whether CoT makes different models' outputs more similar to each other. This is the core novel question.
- **Experiment 3 (Domain modulation)**: Tests whether convergence effects vary by task type (math vs. commonsense vs. science). Needed to understand generalizability.

## Research Question
Does Chain of Thought prompting cause large language models to produce more convergent outputs (both within a single model and across different models) compared to direct prompting?

## Hypothesis Decomposition

### H1: Within-model answer convergence
CoT increases the agreement rate of final answers across multiple samples from the same model (compared to direct prompting).

### H2: Cross-model answer convergence
CoT increases the agreement rate of final answers across different models on the same questions (compared to direct prompting).

### H3: Cross-model response similarity
CoT increases the semantic similarity of full responses across different models (compared to direct prompting).

### H4: Domain modulation
The convergence effect of CoT is stronger for well-structured tasks (math) than for open-ended tasks (commonsense).

## Proposed Methodology

### Approach
We prompt multiple LLMs on questions from three benchmarks under two conditions: **direct prompting** (no CoT) and **zero-shot CoT** ("Let's think step by step"). For each question × model × condition, we generate multiple samples (N=10) at temperature=0.7. We then measure convergence using answer agreement rates, output entropy, and embedding-based similarity.

### Models (via API)
1. **GPT-4.1** (OpenAI API) — `gpt-4.1`
2. **Claude Sonnet 4.5** (OpenRouter) — `anthropic/claude-sonnet-4-5`
3. **Gemini 2.5 Flash** (OpenRouter) — `google/gemini-2.5-flash-preview`

### Datasets (sampled subsets)
- **GSM8K**: 50 questions from test set (math reasoning)
- **CommonsenseQA**: 50 questions from validation set (commonsense)
- **ARC-Challenge**: 50 questions from test set (science)

### Experimental Steps
1. Sample 50 questions from each dataset (stratified by difficulty where possible)
2. For each question, construct two prompts: direct and CoT
3. For each question × prompt × model, generate 10 responses at T=0.7
4. Extract final answers from all responses
5. Compute within-model and cross-model convergence metrics
6. Statistical comparison of convergence under CoT vs. direct

### Baselines
- **Direct prompting**: Primary control (no CoT instruction)
- **Random baseline**: Expected agreement rate under uniform random answering

### Evaluation Metrics
1. **Answer Agreement Rate (AAR)**: Fraction of sample pairs with matching final answers (within-model)
2. **Cross-Model Agreement Rate (CMAR)**: Fraction of model pairs agreeing on final answer per question
3. **Output Entropy**: Shannon entropy of the answer distribution per question
4. **Embedding Cosine Similarity**: Average pairwise cosine similarity of response embeddings (using text-embedding-3-small)
5. **Accuracy**: Standard task accuracy for context

### Statistical Analysis Plan
- Paired t-tests (or Wilcoxon signed-rank) comparing CoT vs. direct on each metric, paired by question
- Effect sizes (Cohen's d)
- 95% confidence intervals
- Bonferroni correction for multiple comparisons across datasets
- Significance level: α = 0.05

## Expected Outcomes
- **H1 supported**: CoT increases within-model answer agreement (consistent with self-consistency literature)
- **H2 direction unclear**: CoT may increase cross-model agreement (models converge on correct answers) or decrease it (models use different reasoning strategies)
- **H3 direction unclear**: Full response similarity may decrease even if answer similarity increases (decoupling effect)
- **H4 supported**: Math tasks should show stronger convergence than commonsense

## Timeline and Milestones
1. Environment setup & code: 20 min
2. Run experiments (API calls): 60-90 min
3. Analysis & visualization: 30 min
4. Documentation: 20 min

## Potential Challenges
- API rate limits → use exponential backoff, cache responses
- Cost management → limit to 50 questions × 3 datasets × 3 models × 2 conditions × 10 samples = 9,000 API calls
- Answer extraction → need robust parsing for different response formats
- OpenRouter availability → fallback to additional OpenAI models if needed

## Success Criteria
- Complete data collection for at least 2 models × 2 datasets × 2 conditions
- Statistical tests with clear results (significant or null)
- Visualizations showing convergence patterns
- Documented findings in REPORT.md
