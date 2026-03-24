"""
Experiment: Does Chain of Thought Cause Models to Converge More?

This script runs the core experiment:
1. Loads questions from GSM8K, CommonsenseQA, and ARC-Challenge
2. Prompts multiple LLMs under direct and CoT conditions
3. Generates multiple samples per question per condition
4. Saves all responses for analysis
"""

import json
import os
import random
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_QUESTIONS_PER_DATASET = 30
NUM_SAMPLES = 5
TEMPERATURE = 0.7
MAX_TOKENS_DIRECT = 150
MAX_TOKENS_COT = 600

BASE_DIR = Path("/workspaces/chain-of-thought-conv-claude")
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Models configuration
MODELS = {
    "gpt-4.1": {
        "client_type": "openai",
        "model_id": "gpt-4.1",
    },
    "gpt-4.1-mini": {
        "client_type": "openai",
        "model_id": "gpt-4.1-mini",
    },
    "gpt-4.1-nano": {
        "client_type": "openai",
        "model_id": "gpt-4.1-nano",
    },
}

# ── Data Loading ───────────────────────────────────────────────────────────

def load_gsm8k(n=NUM_QUESTIONS_PER_DATASET):
    """Load GSM8K test questions."""
    with open(DATASETS_DIR / "gsm8k" / "test.jsonl") as f:
        data = [json.loads(line) for line in f]
    random.seed(SEED)
    sampled = random.sample(data, min(n, len(data)))
    questions = []
    for item in sampled:
        # Extract numeric answer after ####
        answer_text = item["answer"]
        final_answer = answer_text.split("####")[-1].strip()
        questions.append({
            "id": hashlib.md5(item["question"].encode()).hexdigest()[:12],
            "dataset": "gsm8k",
            "question": item["question"],
            "gold_answer": final_answer,
            "answer_type": "numeric",
        })
    return questions


def load_commonsenseqa(n=NUM_QUESTIONS_PER_DATASET):
    """Load CommonsenseQA validation questions."""
    with open(DATASETS_DIR / "commonsenseqa" / "validation.jsonl") as f:
        data = [json.loads(line) for line in f]
    random.seed(SEED + 1)
    sampled = random.sample(data, min(n, len(data)))
    questions = []
    for item in sampled:
        choices_str = " ".join(
            f"({l}) {t}" for l, t in zip(item["choices"]["label"], item["choices"]["text"])
        )
        questions.append({
            "id": item["id"][:12],
            "dataset": "commonsenseqa",
            "question": f"{item['question']}\n{choices_str}",
            "gold_answer": item["answerKey"],
            "answer_type": "multiple_choice",
        })
    return questions


def load_arc_challenge(n=NUM_QUESTIONS_PER_DATASET):
    """Load ARC-Challenge test questions."""
    with open(DATASETS_DIR / "arc_challenge" / "test.jsonl") as f:
        data = [json.loads(line) for line in f]
    random.seed(SEED + 2)
    sampled = random.sample(data, min(n, len(data)))
    questions = []
    for item in sampled:
        choices_str = " ".join(
            f"({l}) {t}" for l, t in zip(item["choices"]["label"], item["choices"]["text"])
        )
        questions.append({
            "id": item["id"][:12],
            "dataset": "arc_challenge",
            "question": f"{item['question']}\n{choices_str}",
            "gold_answer": item["answerKey"],
            "answer_type": "multiple_choice",
        })
    return questions


# ── Prompt Construction ────────────────────────────────────────────────────

def make_direct_prompt(question, answer_type):
    """Create a direct (no CoT) prompt."""
    if answer_type == "numeric":
        return (
            f"{question}\n\n"
            "Give your final numerical answer only, with no explanation. "
            "Format: Answer: <number>"
        )
    else:
        return (
            f"{question}\n\n"
            "Give your answer as a single letter only, with no explanation. "
            "Format: Answer: <letter>"
        )


def make_cot_prompt(question, answer_type):
    """Create a Chain of Thought prompt."""
    if answer_type == "numeric":
        return (
            f"{question}\n\n"
            "Let's think step by step. Show your reasoning, then give your "
            "final numerical answer. Format your final answer as: Answer: <number>"
        )
    else:
        return (
            f"{question}\n\n"
            "Let's think step by step. Show your reasoning, then give your "
            "final answer as a single letter. Format your final answer as: Answer: <letter>"
        )


# ── API Calling ────────────────────────────────────────────────────────────

def get_client(model_config):
    """Get the appropriate API client."""
    if model_config["client_type"] == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif model_config["client_type"] == "openrouter":
        return OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
    raise ValueError(f"Unknown client type: {model_config['client_type']}")


def cache_key(model_name, question_id, condition, sample_idx):
    """Generate a cache key for a specific API call."""
    return f"{model_name}__{question_id}__{condition}__{sample_idx}"


def call_model(client, model_id, prompt, max_tokens, temperature=TEMPERATURE):
    """Call an LLM API with retry logic."""
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                seed=None,  # We want diversity across samples
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {e}, waiting {wait}s")
            time.sleep(wait)
    return None


def run_single_query(model_name, model_config, question, condition, sample_idx):
    """Run a single query and cache the result."""
    ckey = cache_key(model_name, question["id"], condition, sample_idx)
    cache_file = CACHE_DIR / f"{ckey}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    if condition == "direct":
        prompt = make_direct_prompt(question["question"], question["answer_type"])
        max_tokens = MAX_TOKENS_DIRECT
    else:
        prompt = make_cot_prompt(question["question"], question["answer_type"])
        max_tokens = MAX_TOKENS_COT

    client = get_client(model_config)
    response_text = call_model(client, model_config["model_id"], prompt, max_tokens)

    result = {
        "model": model_name,
        "question_id": question["id"],
        "dataset": question["dataset"],
        "condition": condition,
        "sample_idx": sample_idx,
        "prompt": prompt,
        "response": response_text,
        "gold_answer": question["gold_answer"],
        "answer_type": question["answer_type"],
    }

    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


# ── Main Experiment Runner ─────────────────────────────────────────────────

def run_experiment():
    """Run the full experiment."""
    print("Loading datasets...")
    all_questions = []
    all_questions.extend(load_gsm8k())
    all_questions.extend(load_commonsenseqa())
    all_questions.extend(load_arc_challenge())
    print(f"  Loaded {len(all_questions)} questions total")

    conditions = ["direct", "cot"]
    all_results = []
    total_calls = len(MODELS) * len(all_questions) * len(conditions) * NUM_SAMPLES
    print(f"Total API calls planned: {total_calls}")

    completed = 0
    for model_name, model_config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        client = get_client(model_config)

        for condition in conditions:
            print(f"\n  Condition: {condition}")
            for qi, question in enumerate(all_questions):
                for sample_idx in range(NUM_SAMPLES):
                    result = run_single_query(
                        model_name, model_config, question, condition, sample_idx
                    )
                    all_results.append(result)
                    completed += 1

                if (qi + 1) % 10 == 0:
                    print(f"    Questions completed: {qi+1}/{len(all_questions)} "
                          f"(total calls: {completed}/{total_calls})")

    # Save all results
    output_file = RESULTS_DIR / "all_responses.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {output_file}")

    return all_results


if __name__ == "__main__":
    run_experiment()
