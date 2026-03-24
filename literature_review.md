# Literature Review: Does Chain of Thought Cause Models to Converge More?

## Research Area Overview

Chain-of-thought (CoT) prompting has become the dominant paradigm for eliciting reasoning from large language models. Since Wei et al. (2022) demonstrated that prompting models to "think step by step" dramatically improves reasoning performance, a rich body of work has explored CoT's mechanisms, limitations, and theoretical foundations. This review examines a specific question: **does CoT cause models to produce more convergent outputs** — in terms of final answers, reasoning paths, and internal representations — compared to direct prompting?

The question of convergence spans multiple levels:
1. **Answer-level convergence**: Do models produce more consistent final answers with CoT?
2. **Reasoning-path convergence**: Do different reasoning chains converge to similar intermediate steps?
3. **Representation-level convergence**: Do internal model activations become more similar under CoT?
4. **Cross-model convergence**: Do different models produce more similar outputs when using CoT?

## Key Papers

### Paper 1: Chain of Thought Prompting Elicits Reasoning in Large Language Models
- **Authors**: Wei, Wang, Schuurmans, Bosma, Xia, Chi, Le, Zhou
- **Year**: 2022 (NeurIPS)
- **arXiv**: 2201.11903
- **Key Contribution**: Foundational CoT paper. Demonstrates that adding intermediate reasoning steps to few-shot prompts dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks.
- **Methodology**: Few-shot prompting with manually-written CoT exemplars; greedy decoding; tested on GPT-3 and PaLM at various scales.
- **Datasets Used**: GSM8K, SVAMP, ASDiv, AQuA, MAWPS (arithmetic); CSQA, StrategyQA, Date/Sports Understanding (commonsense); Last Letter Concatenation, Coin Flip (symbolic)
- **Results**: CoT only helps at ~100B+ parameters. Ablations show semantic content of the chain matters (not just extra tokens). Performance is robust across different annotators and exemplar orderings.
- **Relevance to Convergence**: CoT narrows the output distribution toward correct answers at large scale, robust to prompt variation. However, the paper does not study representational convergence or whether different models converge on similar chains. The ablation ruling out "dots only" suggests convergence is in semantic reasoning structure, not just token count.
- **Code Available**: No

### Paper 2: Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **Authors**: Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou
- **Year**: 2022 (ICLR 2023)
- **arXiv**: 2203.11171
- **Key Contribution**: Introduces self-consistency — sampling multiple CoT paths and taking majority vote over final answers. Demonstrates that correct reasoning paths converge on the same answer while incorrect paths scatter.
- **Methodology**: Sample m=40 diverse reasoning paths (temperature=0.7), aggregate final answers by majority vote. No additional training.
- **Datasets Used**: GSM8K, SVAMP, ASDiv, AQuA, MultiArith, AddSub (arithmetic); CSQA, StrategyQA, ARC (commonsense); Last Letter, Coin Flip (symbolic); ANLI, e-SNLI, RTE, BoolQ, HotpotQA (NLP)
- **Results**: +17.9% on GSM8K for PaLM-540B over greedy CoT. Diversity of paths (not quantity) drives gains. Consistency percentage correlates strongly with accuracy (useful as uncertainty estimate). Crucially, normalized token probabilities are nearly uniform across correct and incorrect paths — the model cannot distinguish them by confidence.
- **Relevance to Convergence**: **Central paper.** Establishes that CoT creates output-level convergence (final answer plurality) without confidence-level convergence (model assigns similar probabilities to all paths). Correct answers act as attractors across diverse reasoning trajectories. The mechanism exploits answer convergence while the reasoning paths themselves remain diverse.
- **Code Available**: No (but straightforward to implement)

### Paper 3: Answer Convergence as a Signal for Early Stopping in Reasoning
- **Authors**: Liu, Wang
- **Year**: 2025
- **arXiv**: 2506.02536
- **Key Contribution**: Studies within-trace answer convergence — shows models' predicted answers stabilize well before the full CoT completes. Proposes Answer Convergence Ratio (ACR) metric and early stopping methods.
- **Methodology**: Incrementally truncate CoT at sentence boundaries, force answer generation after each truncation. Measure when answer stabilizes. Test three early stopping strategies.
- **Datasets Used**: NaturalQuestions, GSM8K, MATH-500, GPQA-Diamond, AIME 2024
- **Results**: Models converge to final answer after ~60% of reasoning steps on average. Larger models converge earlier. On easy tasks (NQ), extended reasoning can "overthink" and diverge from correct answer. ACR is task-difficulty-modulated.
- **Relevance to Convergence**: Directly establishes that answer convergence within a single generation is the norm, provides the ACR metric as a replicable convergence measure, and shows convergence is modulated by model scale and task difficulty. The ~40% redundant reasoning steps suggest substantial over-computation.
- **Code Available**: Yes — HuggingFace Spaces: reasoning_earlystop

### Paper 4: Mapping the Minds of LLMs: A Graph-Based Analysis of Reasoning LLM
- **Authors**: Xiong et al.
- **Year**: 2025
- **arXiv**: 2505.13890
- **Key Contribution**: Converts CoT traces into directed reasoning graphs and defines structural metrics including Convergence Ratio (gamma_C) — the proportion of nodes with in-degree > 1, measuring how reasoning threads are synthesized.
- **Methodology**: Three-stage pipeline: segment CoT into units, cluster into reasoning steps, detect semantic edges between steps. Define four graph metrics: exploration density, branching ratio, convergence ratio, linearity.
- **Datasets Used**: GPQA-Diamond
- **Results**: Convergence ratio gamma_C strongly predicts accuracy (r=0.68). Zero-shot prompting produces richer, more convergent graphs than few-shot. Few-shot prompting collapses reasoning structure toward linear chains. Larger models produce more convergent graphs.
- **Relevance to Convergence**: **Provides the most structurally precise definition of convergence** — measuring integration of multiple reasoning threads. Shows convergence is malleable by prompting strategy (zero-shot > few-shot) and that it causally associates with correctness. Critical methodological contribution.
- **Code Available**: Promised but not confirmed

### Paper 5: Two Failures of Self-Consistency in the Multi-Step Reasoning of LLMs
- **Authors**: Chen, Zhong, Hao, Li, Liu, Huang, Ma, Gu, Li
- **Year**: 2023 (TMLR 2024)
- **arXiv**: 2305.14279
- **Key Contribution**: Formalizes hypothetical consistency (can a model predict its own output?) and compositional consistency (does substituting intermediate answers preserve final answers?). Shows both are poor in GPT-3/4-era models.
- **Methodology**: Hypothetical: 5-way multiple choice on model's own completions. Compositional: substitute model's sub-expression answers into parent expressions, check if final answer changes.
- **Datasets Used**: Wikipedia, DailyDialog (hypothetical); Synthetic arithmetic, GeoQuery (compositional)
- **Results**: Hypothetical consistency near random chance for all but largest models. Compositional consistency <50% on arithmetic. 90% of failures are type-2: correct sub-answer but wrong final when substituted back.
- **Relevance to Convergence**: Shows CoT steps do not converge in a logically integrated way — models fail to maintain consistency when their own intermediate outputs are reintroduced. Establishes the baseline inconsistency that CoT must overcome.
- **Code Available**: No

### Paper 6: Robust Answers, Fragile Logic: Probing the Decoupling Hypothesis
- **Authors**: Various
- **Year**: 2025
- **arXiv**: 2505.17406
- **Key Contribution**: Introduces MATCHA framework showing that LLMs maintain correct answers under perturbation while generating wildly inconsistent reasoning chains — the "Decoupling Hypothesis."
- **Datasets Used**: GSM8K, commonsense and logical reasoning benchmarks
- **Results**: Answer-level convergence coexists with reasoning-level divergence. Adversarial perturbations reveal reasoning is often post-hoc rationalization.
- **Relevance to Convergence**: Directly demonstrates that answer convergence and reasoning convergence are independent. CoT may converge answers without converging reasoning processes.
- **Code Available**: Not confirmed

### Paper 7: Let's Think Dot by Dot: Hidden Computation in Transformer Language Models
- **Authors**: Pfau, Merrill, Bowman
- **Year**: 2024
- **arXiv**: 2404.15758
- **Key Contribution**: Shows transformers can use meaningless filler tokens (dots) to solve hard tasks, demonstrating CoT's benefit can come from computational capacity (token budget) rather than semantic reasoning.
- **Datasets Used**: Custom synthetic: 3SUM, 2SUM-Transform
- **Results**: 34M-parameter LLaMA achieves 100% on 3SUM with filler tokens. Filler benefits scale with task complexity.
- **Relevance to Convergence**: Dissociates semantic CoT from computational benefit. Convergence observed with CoT may be partly a token-budget artifact rather than evidence of coherent reasoning.
- **Code Available**: Yes — https://github.com/JacobPfau/fillerTokens

### Paper 8: On the Representational Capacity of Neural Language Models with Chain-of-Thought
- **Authors**: Merrill, Sabharwal
- **Year**: 2024
- **arXiv**: 2406.14197
- **Key Contribution**: Proves formally that CoT strictly expands the representational capacity of both RNN and transformer LMs. CoT-augmented LMs are equivalent to probabilistic Turing machines.
- **Results**: CoT enables models to represent non-deterministic distributions over outputs, theoretically allowing more diverse (not more convergent) generation.
- **Relevance to Convergence**: Provides formal basis for why CoT could increase output diversity rather than convergence — the model gains access to a richer class of distributions. The convergence question is thus: does CoT's expanded capacity get used to converge or diverge?
- **Code Available**: Attempted at https://github.com/rycolab/cot-lms (not found)

### Paper 9: Self-Consistency of Large Language Models under Ambiguity
- **Authors**: Pfau, Merrill, Bowman
- **Year**: 2023
- **arXiv**: 2310.13439
- **Key Contribution**: Measures cross-context self-consistency on ambiguous integer sequences. Finds consistency improves with model capability but models are poorly calibrated about their own consistency.
- **Datasets Used**: Custom: 197 integer sequences (140 unambiguous, 57 ambiguous)
- **Results**: Consistency 67-82% across GPT-3 to GPT-4. Even "converged" models internally assign significant probability mass to inconsistent alternatives.
- **Relevance to Convergence**: Output-level convergence increases with scale, but internal probability distributions remain divergent. Surface convergence may mask internal uncertainty.
- **Code Available**: Yes — https://github.com/JacobPfau/introspective-self-consistency

### Paper 10: Measuring and Improving Consistency in Pretrained Language Models
- **Authors**: Elazar, Kassner, Goldberg, Schütze
- **Year**: 2021
- **arXiv**: 2102.01017
- **Key Contribution**: PARAREL benchmark — 328 paraphrase patterns for measuring factual consistency in masked LMs. Establishes pre-CoT baseline: models are inconsistent across paraphrases.
- **Datasets Used**: PARAREL (328 patterns, 38 relations, ~27K KB tuples)
- **Results**: Best model (BERT-large) only 61.1% consistent. Strong correlation between accuracy and consistency. Wikipedia-dominant training may help consistency.
- **Relevance to Convergence**: Pre-CoT baseline establishing that LMs are inherently inconsistent. Provides the floor against which CoT's potential convergence-inducing effect should be measured.
- **Code Available**: Yes — https://github.com/yanaiela/pararel

### Additional Papers (Abstract-level review)

- **CoT is Not True Reasoning** (2506.02878, 2025): Theoretical argument that CoT constrains sampling toward training-distribution patterns, not genuine reasoning. Predicts CoT causes convergence to familiar formats, not logical correctness.
- **Feature Extraction and Steering for Enhanced CoT** (2505.15634, 2025): Uses SAEs to extract and steer CoT features. Shows reasoning depth is tunable via activation steering, with non-monotonic convergence effects.
- **Compressed Chain of Thought** (2412.13171, 2024): Dense token representations can substitute for verbose CoT. Relevant to whether convergence depends on reasoning verbosity.
- **Why think step-by-step?** (2304.03843, 2023): Bayesian analysis showing CoT works because it enables local factorization of complex distributions. Theoretical basis for when CoT should improve convergence.
- **On the Hardness of Faithful CoT** (2406.10625, 2024): Shows faithful CoT (where chain causally determines answer) is computationally harder than unfaithful CoT. Relevant to decoupling of answer and reasoning convergence.
- **Propositional Interpretability** (2501.15740, 2025): Framework for interpreting model propositions during reasoning. Could be used to measure internal convergence.

## Common Methodologies

1. **Sampling and aggregation** (Wang et al., 2022): Sample multiple CoT paths, measure answer convergence via majority voting. Standard for studying output-level convergence.
2. **Truncation analysis** (Liu & Wang, 2025): Progressively truncate CoT and measure when answer stabilizes. Provides within-trace convergence dynamics.
3. **Graph-based structural analysis** (Xiong et al., 2025): Convert CoT to directed graphs, measure structural convergence via graph metrics.
4. **Consistency testing** (Chen et al., 2023; Elazar et al., 2021): Paraphrase or compositionally transform queries, measure answer stability.
5. **Perturbation probing** (MATCHA, 2025): Apply adversarial perturbations, measure whether answer and reasoning converge/diverge independently.

## Standard Baselines

- **Direct prompting** (no CoT): The primary control condition
- **Greedy CoT decoding**: Single CoT path via temperature=0
- **Self-consistency (majority vote)**: Multiple sampled CoT paths with answer aggregation
- **Filler tokens**: Meaningless tokens replacing CoT content (computational control)

## Evaluation Metrics

- **Accuracy**: Task performance (% correct answers)
- **Self-consistency rate**: Agreement across multiple sampled outputs
- **Answer Convergence Ratio (ACR)**: Proportion of chain needed before answer stabilizes
- **Convergence Ratio (gamma_C)**: Graph-based metric of reasoning thread synthesis
- **Compositional consistency**: Answer stability under intermediate-result substitution
- **Cosine similarity**: For internal representation convergence (not yet studied for CoT)
- **Output entropy**: Diversity of generated outputs (proposed metric)

## Datasets in the Literature

| Dataset | Used In | Task Type | Size |
|---------|---------|-----------|------|
| GSM8K | Wei 2022, Wang 2022, Liu 2025, MATCHA | Math reasoning | 1,319 test |
| MATH-500 | Liu 2025 | Hard math | 500 test |
| GPQA-Diamond | Liu 2025, Xiong 2025 | Graduate QA | ~200 |
| CommonsenseQA | Wei 2022, Wang 2022 | Commonsense | 1,221 val |
| StrategyQA | Wei 2022, Wang 2022 | Multi-hop | ~2,300 |
| ARC-Challenge | Wang 2022 | Science QA | 1,172 test |
| NaturalQuestions | Liu 2025 | Open QA | 3,610 test |
| AIME 2024 | Liu 2025 | Competition math | 30 |

## Gaps and Opportunities

1. **No study directly compares representation-level convergence with vs. without CoT.** All existing work measures output-level or structural convergence; internal activation similarity under CoT remains unstudied.
2. **Cross-model convergence is unexplored.** Do different models produce more similar outputs when using CoT? This is the "persona convergence" aspect of the hypothesis.
3. **The relationship between answer convergence and reasoning convergence needs quantification.** MATCHA shows they decouple, but the degree and conditions of decoupling are unknown.
4. **Task difficulty modulates convergence** (shown by ACR analysis) but a systematic study across diverse domains is missing.
5. **No study compares CoT convergence across model families** (e.g., GPT vs. Claude vs. LLaMA) to assess whether CoT homogenizes outputs across architectures.

## Recommendations for Our Experiment

### Recommended Datasets
- **GSM8K test** (1,319 examples): Primary benchmark, universally used
- **CommonsenseQA validation** (1,221 examples): Tests non-mathematical reasoning
- **ARC-Challenge test** (1,172 examples): Science reasoning, different domain

### Recommended Baselines
- Direct prompting (no CoT) as primary control
- Zero-shot CoT ("Let's think step by step")
- Few-shot CoT with manual exemplars
- Self-consistency (multiple samples + majority vote)

### Recommended Metrics
- **Output similarity** (BLEU, ROUGE, or embedding cosine similarity between runs)
- **Answer agreement rate** across multiple samples
- **Output entropy** (diversity of generated answers)
- **Representation cosine similarity** (if internal activations are accessible)
- **Standard accuracy** for performance context

### Methodological Considerations
- Use temperature > 0 for sampling to measure true output diversity
- Compare multiple runs per prompt to distinguish stochastic variation from systematic convergence
- Test both within-model convergence (same model, different runs) and cross-model convergence (different models, same prompts)
- Include task difficulty as a variable (easy math vs. hard math vs. commonsense)
- The "convergence" question has multiple valid operationalizations — clearly define which level is being measured
