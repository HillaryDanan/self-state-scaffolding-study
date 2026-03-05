#!/usr/bin/env python3
"""
Self-State Scaffolding Study — Main Runner
============================================
Generates novel operators, runs all conditions across all providers,
and saves raw trial data for analysis.

Usage:
    python3 run_study.py                    # Run everything
    python3 run_study.py --providers anthropic  # Single provider
    python3 run_study.py --conditions bare memory  # Subset of conditions
    python3 run_study.py --n-operators 5    # Quick test run

Reference:
    Danan, H. (2025). Abstraction-Intelligence: Exploring abstraction
    as a primitive of intelligent systems.
    https://github.com/HillaryDanan/abstraction-intelligence
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

from config import (
    MODELS, CONDITIONS, N_OPERATORS, RANDOM_SEED,
    DATA_DIR, ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
)
from operators.generator import generate_full_stimulus_set, save_stimulus_set
from conditions import bare, memory, stakes, full_mcu


CONDITION_MODULES = {
    "bare": bare,
    "memory": memory,
    "stakes": stakes,
    "full_mcu": full_mcu,
}


def get_provider(provider_key: str):
    """Initialize the API provider."""
    model_cfg = MODELS[provider_key]

    if model_cfg["provider"] == "anthropic":
        from providers.anthropic_api import AnthropicProvider
        return AnthropicProvider(model_cfg["model_id"], model_cfg["display_name"])
    elif model_cfg["provider"] == "openai":
        from providers.openai_api import OpenAIProvider
        return OpenAIProvider(model_cfg["model_id"], model_cfg["display_name"])
    elif model_cfg["provider"] == "google":
        from providers.google_api import GoogleProvider
        return GoogleProvider(model_cfg["model_id"], model_cfg["display_name"])
    else:
        raise ValueError(f"Unknown provider: {model_cfg['provider']}")


def check_api_keys(providers: List[str]) -> bool:
    """Verify required API keys are set."""
    key_map = {
        "anthropic": ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        "openai": ("OPENAI_API_KEY", OPENAI_API_KEY),
        "google": ("GOOGLE_API_KEY", GOOGLE_API_KEY),
    }
    all_ok = True
    for p in providers:
        name, value = key_map[p]
        if not value:
            print(f"  ✗ {name} not set!")
            all_ok = False
        else:
            print(f"  ✓ {name} is set")
    return all_ok


def run_condition(
    provider,
    condition_module,
    condition_name: str,
    problems: List[Dict],
) -> List[Dict]:
    """
    Run all problems for a single condition, maintaining trial history
    for memory/stakes/full_mcu conditions.
    """
    trial_history = []
    results = []

    for i, problem in enumerate(problems):
        print(f"      Problem {i+1}/{len(problems)} [{problem['difficulty']}]", end="", flush=True)

        # Run trial
        response = condition_module.run_trial(provider, problem, trial_history)

        # Score it
        correct = False
        if response.get("answer") is not None and problem.get("ground_truth") is not None:
            try:
                # Allow small floating point tolerance
                correct = abs(float(response["answer"]) - float(problem["ground_truth"])) < 0.01
            except (ValueError, TypeError):
                correct = False

        # Build trial record
        trial = {
            "problem_index": i,
            "operator_name": problem["operator_name"],
            "difficulty": problem["difficulty"],
            "question": problem["question"],
            "ground_truth": problem["ground_truth"],
            "answer": response.get("answer"),
            "confidence": response.get("confidence"),
            "correct": correct,
            "raw_response": response.get("raw_response", ""),
            "self_reflection": response.get("self_reflection", ""),
            "self_model_update": response.get("self_model_update", ""),
            "parse_error": response.get("parse_error", False),
            "condition": condition_name,
        }

        results.append(trial)

        # Update history for sequential conditions
        trial_history.append(trial)

        mark = "✓" if correct else "✗"
        conf_str = f"{response.get('confidence', 0):.0%}" if response.get("confidence") is not None else "?"
        print(f" → {mark} (conf={conf_str})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Self-State Scaffolding Study")
    parser.add_argument(
        "--providers", nargs="+", default=list(MODELS.keys()),
        help="Which providers to test (default: all)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=CONDITIONS,
        help="Which conditions to run (default: all)",
    )
    parser.add_argument(
        "--n-operators", type=int, default=N_OPERATORS,
        help=f"Number of novel operators (default: {N_OPERATORS})",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("SELF-STATE SCAFFOLDING STUDY")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Providers: {args.providers}")
    print(f"Conditions: {args.conditions}")
    print(f"Operators: {args.n_operators}")
    print(f"Seed: {args.seed}")
    print()

    # Check API keys
    print("Checking API keys...")
    if not check_api_keys(args.providers):
        print("\nPlease set missing API keys and retry.")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export GOOGLE_API_KEY='your-key'")
        return

    # Generate stimuli
    print(f"\nGenerating {args.n_operators} novel operators...")
    stimuli = generate_full_stimulus_set(seed=args.seed, n_operators=args.n_operators)
    os.makedirs(DATA_DIR, exist_ok=True)
    stimuli_path = os.path.join(DATA_DIR, f"stimuli_{timestamp}.json")
    save_stimulus_set(stimuli, stimuli_path)

    problems = stimuli["problems"]
    n_per_condition = len(problems)
    total_calls = n_per_condition * len(args.conditions) * len(args.providers)
    print(f"  {len(problems)} problems × {len(args.conditions)} conditions × {len(args.providers)} providers = {total_calls} API calls")

    # Estimate cost
    est_cost = total_calls * 0.003  # Rough estimate: ~$0.003 per call for mid-tier
    print(f"  Estimated cost: ~${est_cost:.2f}")
    print()

    # Run study
    all_results = {}
    start_time = time.time()

    for provider_key in args.providers:
        print(f"{'='*50}")
        print(f"PROVIDER: {MODELS[provider_key]['display_name']}")
        print(f"{'='*50}")

        try:
            provider = get_provider(provider_key)
        except Exception as e:
            print(f"  Failed to initialize {provider_key}: {e}")
            continue

        provider_results = {}

        for condition_name in args.conditions:
            print(f"\n  --- Condition: {condition_name} ---")
            condition_module = CONDITION_MODULES[condition_name]

            results = run_condition(
                provider, condition_module, condition_name, problems
            )

            provider_results[condition_name] = results

            # Quick summary
            valid = [r for r in results if not r.get("parse_error")]
            correct = sum(1 for r in valid if r["correct"])
            print(f"  Summary: {correct}/{len(valid)} correct ({correct/max(len(valid),1):.0%})")

        all_results[MODELS[provider_key]["display_name"]] = provider_results

        # Save incrementally
        save_path = os.path.join(DATA_DIR, f"raw_results_{provider_key}_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(provider_results, f, indent=2, default=str)
        print(f"\n  Saved to {save_path}")

    # Save combined results
    combined_path = os.path.join(DATA_DIR, f"all_results_{timestamp}.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"STUDY COMPLETE — {elapsed/60:.1f} minutes")
    print(f"Results saved to: {combined_path}")
    print(f"Run analysis:  python3 analyze_results.py --input {combined_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
