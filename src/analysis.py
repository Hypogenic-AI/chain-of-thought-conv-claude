"""
Analysis: Measure convergence metrics from experiment results.

Metrics computed:
1. Answer Agreement Rate (AAR) — within-model consistency
2. Cross-Model Agreement Rate (CMAR) — cross-model convergence
3. Output Entropy — diversity of answer distributions
4. Accuracy — for context
5. Embedding Cosine Similarity — semantic similarity of responses
"""

import json
import re
import os
import sys
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path("/workspaces/chain-of-thought-conv-claude")
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Answer Extraction ──────────────────────────────────────────────────────

def extract_answer(response, answer_type):
    """Extract the final answer from a model response."""
    if response is None:
        return None

    text = response.strip()

    # Try "Answer: X" pattern first
    match = re.search(r"[Aa]nswer:\s*([^\n]+)", text)
    if match:
        ans = match.group(1).strip().rstrip(".")
        if answer_type == "numeric":
            # Extract number
            nums = re.findall(r"-?[\d,]+\.?\d*", ans)
            if nums:
                return nums[-1].replace(",", "")
        else:
            # Extract letter
            letters = re.findall(r"[A-E]", ans.upper())
            if letters:
                return letters[0]

    if answer_type == "numeric":
        # Look for #### pattern (GSM8K style)
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "")
        # Last number in text
        nums = re.findall(r"-?[\d,]+\.?\d*", text)
        if nums:
            return nums[-1].replace(",", "")
    else:
        # For MC, look for standalone letter at end or in parentheses
        # Check last line
        last_line = text.strip().split("\n")[-1]
        letters = re.findall(r"\b([A-E])\b", last_line.upper())
        if letters:
            return letters[-1]
        # Check for (X) pattern
        parens = re.findall(r"\(([A-E])\)", text.upper())
        if parens:
            return parens[-1]

    return None


# ── Metrics ────────────────────────────────────────────────────────────────

def answer_agreement_rate(answers):
    """Compute pairwise agreement rate among a list of answers."""
    valid = [a for a in answers if a is not None]
    if len(valid) < 2:
        return np.nan
    pairs = list(combinations(range(len(valid)), 2))
    agreements = sum(1 for i, j in pairs if valid[i] == valid[j])
    return agreements / len(pairs)


def output_entropy(answers):
    """Compute Shannon entropy of the answer distribution."""
    valid = [a for a in answers if a is not None]
    if len(valid) == 0:
        return np.nan
    counts = Counter(valid)
    probs = np.array(list(counts.values())) / len(valid)
    return -np.sum(probs * np.log2(probs + 1e-12))


def accuracy(answers, gold):
    """Compute accuracy (fraction matching gold answer)."""
    valid = [a for a in answers if a is not None]
    if len(valid) == 0:
        return np.nan
    return sum(1 for a in valid if str(a).strip() == str(gold).strip()) / len(valid)


# ── Main Analysis ──────────────────────────────────────────────────────────

def load_results():
    """Load experiment results."""
    with open(RESULTS_DIR / "all_responses.json") as f:
        return json.load(f)


def organize_results(results):
    """Organize results into nested dict: model -> condition -> question_id -> [responses]."""
    organized = {}
    for r in results:
        model = r["model"]
        condition = r["condition"]
        qid = r["question_id"]
        if model not in organized:
            organized[model] = {}
        if condition not in organized[model]:
            organized[model][condition] = {}
        if qid not in organized[model][condition]:
            organized[model][condition][qid] = []
        organized[model][condition][qid].append(r)
    return organized


def compute_metrics(results):
    """Compute all convergence metrics."""
    organized = organize_results(results)
    models = sorted(organized.keys())
    conditions = ["direct", "cot"]

    # Build question metadata
    q_meta = {}
    for r in results:
        q_meta[r["question_id"]] = {
            "dataset": r["dataset"],
            "gold_answer": r["gold_answer"],
            "answer_type": r["answer_type"],
        }

    all_qids = sorted(q_meta.keys())

    # ── Within-model metrics ──
    within_model = []
    for model in models:
        for condition in conditions:
            if condition not in organized.get(model, {}):
                continue
            for qid in all_qids:
                if qid not in organized[model][condition]:
                    continue
                responses = organized[model][condition][qid]
                answers = [extract_answer(r["response"], q_meta[qid]["answer_type"])
                           for r in responses]
                aar = answer_agreement_rate(answers)
                ent = output_entropy(answers)
                acc = accuracy(answers, q_meta[qid]["gold_answer"])

                within_model.append({
                    "model": model,
                    "condition": condition,
                    "question_id": qid,
                    "dataset": q_meta[qid]["dataset"],
                    "aar": aar,
                    "entropy": ent,
                    "accuracy": acc,
                    "n_valid": sum(1 for a in answers if a is not None),
                    "n_unique": len(set(a for a in answers if a is not None)),
                })

    # ── Cross-model metrics ──
    cross_model = []
    model_pairs = list(combinations(models, 2))
    for condition in conditions:
        for qid in all_qids:
            for m1, m2 in model_pairs:
                if (condition not in organized.get(m1, {}) or
                    condition not in organized.get(m2, {}) or
                    qid not in organized[m1][condition] or
                    qid not in organized[m2][condition]):
                    continue

                answers1 = [extract_answer(r["response"], q_meta[qid]["answer_type"])
                            for r in organized[m1][condition][qid]]
                answers2 = [extract_answer(r["response"], q_meta[qid]["answer_type"])
                            for r in organized[m2][condition][qid]]

                # Modal answer for each model
                valid1 = [a for a in answers1 if a is not None]
                valid2 = [a for a in answers2 if a is not None]
                if not valid1 or not valid2:
                    continue

                modal1 = Counter(valid1).most_common(1)[0][0]
                modal2 = Counter(valid2).most_common(1)[0][0]
                agrees = 1 if modal1 == modal2 else 0

                # Pairwise agreement across all samples
                pair_agreements = 0
                pair_total = 0
                for a1 in valid1:
                    for a2 in valid2:
                        pair_total += 1
                        if a1 == a2:
                            pair_agreements += 1
                pairwise_rate = pair_agreements / pair_total if pair_total > 0 else np.nan

                cross_model.append({
                    "model_pair": f"{m1} vs {m2}",
                    "condition": condition,
                    "question_id": qid,
                    "dataset": q_meta[qid]["dataset"],
                    "modal_agreement": agrees,
                    "pairwise_agreement": pairwise_rate,
                })

    return within_model, cross_model


def aggregate_and_test(within_model, cross_model):
    """Aggregate metrics and run statistical tests."""
    results_summary = {}

    # ── Within-model aggregation ──
    models = sorted(set(r["model"] for r in within_model))
    datasets = sorted(set(r["dataset"] for r in within_model))

    print("\n" + "=" * 70)
    print("WITHIN-MODEL CONVERGENCE (Answer Agreement Rate)")
    print("=" * 70)

    for model in models:
        for dataset in datasets:
            direct_vals = [r["aar"] for r in within_model
                           if r["model"] == model and r["dataset"] == dataset
                           and r["condition"] == "direct" and not np.isnan(r["aar"])]
            cot_vals = [r["aar"] for r in within_model
                        if r["model"] == model and r["dataset"] == dataset
                        and r["condition"] == "cot" and not np.isnan(r["aar"])]

            if direct_vals and cot_vals:
                # Pair by question
                direct_by_q = {r["question_id"]: r["aar"] for r in within_model
                               if r["model"] == model and r["dataset"] == dataset
                               and r["condition"] == "direct" and not np.isnan(r["aar"])}
                cot_by_q = {r["question_id"]: r["aar"] for r in within_model
                            if r["model"] == model and r["dataset"] == dataset
                            and r["condition"] == "cot" and not np.isnan(r["aar"])}
                common_qs = sorted(set(direct_by_q.keys()) & set(cot_by_q.keys()))
                if len(common_qs) >= 5:
                    d_paired = [direct_by_q[q] for q in common_qs]
                    c_paired = [cot_by_q[q] for q in common_qs]
                    stat, pval = stats.wilcoxon(d_paired, c_paired, alternative="two-sided")
                    diff = np.mean(c_paired) - np.mean(d_paired)
                    # Cohen's d
                    diffs = np.array(c_paired) - np.array(d_paired)
                    cohens_d = np.mean(diffs) / (np.std(diffs) + 1e-12)
                else:
                    pval = np.nan
                    diff = np.nan
                    cohens_d = np.nan

                print(f"\n  {model} | {dataset}:")
                print(f"    Direct AAR: {np.mean(direct_vals):.3f} ± {np.std(direct_vals):.3f}")
                print(f"    CoT AAR:    {np.mean(cot_vals):.3f} ± {np.std(cot_vals):.3f}")
                print(f"    Diff (CoT - Direct): {diff:+.3f}, Cohen's d: {cohens_d:.3f}, p={pval:.4f}")

                key = f"within_{model}_{dataset}"
                results_summary[key] = {
                    "model": model,
                    "dataset": dataset,
                    "direct_aar_mean": float(np.mean(direct_vals)),
                    "direct_aar_std": float(np.std(direct_vals)),
                    "cot_aar_mean": float(np.mean(cot_vals)),
                    "cot_aar_std": float(np.std(cot_vals)),
                    "diff": float(diff),
                    "cohens_d": float(cohens_d),
                    "p_value": float(pval) if not np.isnan(pval) else None,
                    "n_questions": len(common_qs) if len(common_qs) >= 5 else 0,
                }

    # ── Accuracy ──
    print("\n" + "=" * 70)
    print("ACCURACY")
    print("=" * 70)

    for model in models:
        for dataset in datasets:
            for condition in ["direct", "cot"]:
                acc_vals = [r["accuracy"] for r in within_model
                            if r["model"] == model and r["dataset"] == dataset
                            and r["condition"] == condition and not np.isnan(r["accuracy"])]
                if acc_vals:
                    print(f"  {model} | {dataset} | {condition}: {np.mean(acc_vals):.3f}")
                    results_summary[f"acc_{model}_{dataset}_{condition}"] = float(np.mean(acc_vals))

    # ── Cross-model aggregation ──
    print("\n" + "=" * 70)
    print("CROSS-MODEL CONVERGENCE (Pairwise Agreement Rate)")
    print("=" * 70)

    model_pairs = sorted(set(r["model_pair"] for r in cross_model))
    for pair in model_pairs:
        for dataset in datasets:
            direct_vals = [r["pairwise_agreement"] for r in cross_model
                           if r["model_pair"] == pair and r["dataset"] == dataset
                           and r["condition"] == "direct" and not np.isnan(r["pairwise_agreement"])]
            cot_vals = [r["pairwise_agreement"] for r in cross_model
                        if r["model_pair"] == pair and r["dataset"] == dataset
                        and r["condition"] == "cot" and not np.isnan(r["pairwise_agreement"])]

            if direct_vals and cot_vals:
                direct_by_q = {r["question_id"]: r["pairwise_agreement"] for r in cross_model
                               if r["model_pair"] == pair and r["dataset"] == dataset
                               and r["condition"] == "direct" and not np.isnan(r["pairwise_agreement"])}
                cot_by_q = {r["question_id"]: r["pairwise_agreement"] for r in cross_model
                            if r["model_pair"] == pair and r["dataset"] == dataset
                            and r["condition"] == "cot" and not np.isnan(r["pairwise_agreement"])}
                common_qs = sorted(set(direct_by_q.keys()) & set(cot_by_q.keys()))

                if len(common_qs) >= 5:
                    d_paired = [direct_by_q[q] for q in common_qs]
                    c_paired = [cot_by_q[q] for q in common_qs]
                    stat, pval = stats.wilcoxon(d_paired, c_paired, alternative="two-sided")
                    diff = np.mean(c_paired) - np.mean(d_paired)
                    diffs = np.array(c_paired) - np.array(d_paired)
                    cohens_d = np.mean(diffs) / (np.std(diffs) + 1e-12)
                else:
                    pval = np.nan
                    diff = np.nan
                    cohens_d = np.nan

                print(f"\n  {pair} | {dataset}:")
                print(f"    Direct: {np.mean(direct_vals):.3f} ± {np.std(direct_vals):.3f}")
                print(f"    CoT:    {np.mean(cot_vals):.3f} ± {np.std(cot_vals):.3f}")
                print(f"    Diff (CoT - Direct): {diff:+.3f}, Cohen's d: {cohens_d:.3f}, p={pval:.4f}")

                key = f"cross_{pair}_{dataset}"
                results_summary[key] = {
                    "model_pair": pair,
                    "dataset": dataset,
                    "direct_mean": float(np.mean(direct_vals)),
                    "direct_std": float(np.std(direct_vals)),
                    "cot_mean": float(np.mean(cot_vals)),
                    "cot_std": float(np.std(cot_vals)),
                    "diff": float(diff),
                    "cohens_d": float(cohens_d),
                    "p_value": float(pval) if not np.isnan(pval) else None,
                    "n_questions": len(common_qs) if len(common_qs) >= 5 else 0,
                }

    # ── Entropy ──
    print("\n" + "=" * 70)
    print("OUTPUT ENTROPY (lower = more convergent)")
    print("=" * 70)

    for model in models:
        for dataset in datasets:
            for condition in ["direct", "cot"]:
                ent_vals = [r["entropy"] for r in within_model
                            if r["model"] == model and r["dataset"] == dataset
                            and r["condition"] == condition and not np.isnan(r["entropy"])]
                if ent_vals:
                    print(f"  {model} | {dataset} | {condition}: {np.mean(ent_vals):.3f} ± {np.std(ent_vals):.3f}")

    return results_summary


# ── Visualization ──────────────────────────────────────────────────────────

def create_visualizations(within_model, cross_model):
    """Create comprehensive visualizations."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    models = sorted(set(r["model"] for r in within_model))
    datasets = sorted(set(r["dataset"] for r in within_model))
    conditions = ["direct", "cot"]

    # ── Plot 1: Within-model AAR by condition and dataset ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for di, dataset in enumerate(datasets):
        ax = axes[di]
        data_for_plot = []
        for model in models:
            for condition in conditions:
                vals = [r["aar"] for r in within_model
                        if r["model"] == model and r["dataset"] == dataset
                        and r["condition"] == condition and not np.isnan(r["aar"])]
                for v in vals:
                    data_for_plot.append({"Model": model, "Condition": condition.upper(), "AAR": v})

        if data_for_plot:
            import pandas as pd
            df = pd.DataFrame(data_for_plot)
            sns.boxplot(data=df, x="Model", y="AAR", hue="Condition", ax=ax,
                        palette={"DIRECT": "#4c72b0", "COT": "#dd8452"})
            ax.set_title(f"{dataset.upper()}", fontsize=13)
            ax.set_xlabel("")
            ax.set_ylabel("Answer Agreement Rate" if di == 0 else "")
            ax.tick_params(axis='x', rotation=15)
            if di > 0:
                ax.legend().remove()

    fig.suptitle("Within-Model Answer Convergence: Direct vs CoT", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "within_model_aar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: within_model_aar.png")

    # ── Plot 2: Cross-model agreement by condition and dataset ──
    model_pairs = sorted(set(r["model_pair"] for r in cross_model))
    if cross_model:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for di, dataset in enumerate(datasets):
            ax = axes[di]
            data_for_plot = []
            for pair in model_pairs:
                for condition in conditions:
                    vals = [r["pairwise_agreement"] for r in cross_model
                            if r["model_pair"] == pair and r["dataset"] == dataset
                            and r["condition"] == condition
                            and not np.isnan(r["pairwise_agreement"])]
                    for v in vals:
                        short_pair = pair.replace("gpt-4.1", "4.1").replace("-mini", "m").replace("-nano", "n")
                        data_for_plot.append({
                            "Model Pair": short_pair,
                            "Condition": condition.upper(),
                            "Agreement": v,
                        })

            if data_for_plot:
                import pandas as pd
                df = pd.DataFrame(data_for_plot)
                sns.boxplot(data=df, x="Model Pair", y="Agreement", hue="Condition", ax=ax,
                            palette={"DIRECT": "#4c72b0", "COT": "#dd8452"})
                ax.set_title(f"{dataset.upper()}", fontsize=13)
                ax.set_xlabel("")
                ax.set_ylabel("Cross-Model Pairwise Agreement" if di == 0 else "")
                ax.tick_params(axis='x', rotation=15)
                if di > 0:
                    ax.legend().remove()

        fig.suptitle("Cross-Model Answer Convergence: Direct vs CoT", fontsize=15, y=1.02)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "cross_model_agreement.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: cross_model_agreement.png")

    # ── Plot 3: Accuracy comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for di, dataset in enumerate(datasets):
        ax = axes[di]
        data_for_plot = []
        for model in models:
            for condition in conditions:
                vals = [r["accuracy"] for r in within_model
                        if r["model"] == model and r["dataset"] == dataset
                        and r["condition"] == condition and not np.isnan(r["accuracy"])]
                for v in vals:
                    data_for_plot.append({"Model": model, "Condition": condition.upper(), "Accuracy": v})

        if data_for_plot:
            import pandas as pd
            df = pd.DataFrame(data_for_plot)
            sns.barplot(data=df, x="Model", y="Accuracy", hue="Condition", ax=ax,
                        palette={"DIRECT": "#4c72b0", "COT": "#dd8452"}, errorbar="sd")
            ax.set_title(f"{dataset.upper()}", fontsize=13)
            ax.set_xlabel("")
            ax.set_ylabel("Accuracy" if di == 0 else "")
            ax.tick_params(axis='x', rotation=15)
            if di > 0:
                ax.legend().remove()

    fig.suptitle("Accuracy: Direct vs CoT", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: accuracy_comparison.png")

    # ── Plot 4: Entropy comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for di, dataset in enumerate(datasets):
        ax = axes[di]
        data_for_plot = []
        for model in models:
            for condition in conditions:
                vals = [r["entropy"] for r in within_model
                        if r["model"] == model and r["dataset"] == dataset
                        and r["condition"] == condition and not np.isnan(r["entropy"])]
                for v in vals:
                    data_for_plot.append({"Model": model, "Condition": condition.upper(), "Entropy": v})

        if data_for_plot:
            import pandas as pd
            df = pd.DataFrame(data_for_plot)
            sns.boxplot(data=df, x="Model", y="Entropy", hue="Condition", ax=ax,
                        palette={"DIRECT": "#4c72b0", "COT": "#dd8452"})
            ax.set_title(f"{dataset.upper()}", fontsize=13)
            ax.set_xlabel("")
            ax.set_ylabel("Output Entropy (bits)" if di == 0 else "")
            ax.tick_params(axis='x', rotation=15)
            if di > 0:
                ax.legend().remove()

    fig.suptitle("Output Entropy: Direct vs CoT (lower = more convergent)", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "entropy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: entropy_comparison.png")

    # ── Plot 5: Summary heatmap ──
    summary_data = {}
    for model in models:
        for dataset in datasets:
            for condition in conditions:
                vals = [r["aar"] for r in within_model
                        if r["model"] == model and r["dataset"] == dataset
                        and r["condition"] == condition and not np.isnan(r["aar"])]
                if vals:
                    key = f"{model}\n{condition}"
                    if key not in summary_data:
                        summary_data[key] = {}
                    summary_data[key][dataset] = np.mean(vals)

    if summary_data:
        import pandas as pd
        df = pd.DataFrame(summary_data).T
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    vmin=0, vmax=1, linewidths=0.5)
        ax.set_title("Within-Model Answer Agreement Rate Summary", fontsize=13)
        ax.set_ylabel("")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "summary_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: summary_heatmap.png")


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    results = load_results()
    print(f"  Loaded {len(results)} responses")

    print("\nComputing metrics...")
    within_model, cross_model = compute_metrics(results)
    print(f"  Within-model records: {len(within_model)}")
    print(f"  Cross-model records: {len(cross_model)}")

    print("\nRunning statistical tests...")
    summary = aggregate_and_test(within_model, cross_model)

    # Save summary
    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics summary to {RESULTS_DIR / 'metrics_summary.json'}")

    print("\nCreating visualizations...")
    create_visualizations(within_model, cross_model)

    # Save raw metrics
    with open(RESULTS_DIR / "within_model_metrics.json", "w") as f:
        json.dump(within_model, f, indent=2)
    with open(RESULTS_DIR / "cross_model_metrics.json", "w") as f:
        json.dump(cross_model, f, indent=2)

    print("\nAnalysis complete!")
    return summary


if __name__ == "__main__":
    main()
