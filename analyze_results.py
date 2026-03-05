#!/usr/bin/env python3
"""
Self-State Scaffolding Study — Analysis
=========================================
Processes raw trial data, computes calibration metrics, runs statistical
tests, generates figures, and produces a summary report.

Usage:
    python3 analyze_results.py --input data/all_results_TIMESTAMP.json

Reference:
    Danan, H. (2025). Abstraction-Intelligence.
    Danan, H. (2025). Discriminating Self-State from Pattern-Matching (Paper 24).
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any

from analysis.calibration import compute_calibration_metrics, compute_signature_classification
from analysis.statistics import compare_conditions, scaffolding_gradient_test
from analysis.visualize import generate_all_figures
from config import RESULTS_DIR, FIGURES_DIR


def analyze_provider(
    provider_name: str,
    condition_results: Dict[str, list],
) -> Dict[str, Any]:
    """Analyze all conditions for a single provider."""
    print(f"\n  Computing metrics...")

    metrics_by_condition = {}
    classifications = {}

    for cond_name, trials in condition_results.items():
        print(f"    {cond_name}: {len(trials)} trials")

        metrics = compute_calibration_metrics(trials)
        metrics_by_condition[cond_name] = metrics

        classification = compute_signature_classification(metrics)
        classifications[cond_name] = classification

        print(f"      ECE={metrics.get('ece', '?'):.3f}" if isinstance(metrics.get('ece'), float) else f"      ECE=N/A")
        acc_str = f"{metrics['mean_accuracy']:.0%}" if isinstance(metrics.get('mean_accuracy'), float) else "?"
        conf_str = f"{metrics['mean_confidence']:.0%}" if isinstance(metrics.get('mean_confidence'), float) else "?"
        std_str = f"{metrics['std_confidence']:.3f}" if isinstance(metrics.get('std_confidence'), float) else "?"
        print(f"      Accuracy={acc_str}  MeanConf={conf_str}  ConfStd={std_str}")
        print(f"      Classification: {classification['classification']}")
        for detail in classification.get("details", []):
            print(f"        {detail}")

    # Statistical comparisons
    print(f"\n  Statistical comparisons...")
    comparisons = compare_conditions(condition_results)

    for comp_name, comp_data in comparisons.items():
        if "error" in comp_data:
            print(f"    {comp_name}: {comp_data['error']}")
        else:
            print(f"    {comp_name}: {comp_data.get('interpretation', '')}")

    # Scaffolding gradient test
    print(f"\n  Scaffolding gradient test...")
    gradient = scaffolding_gradient_test(condition_results)
    print(f"    {gradient.get('interpretation', 'N/A')}")
    if gradient.get("kruskal_wallis_p") is not None:
        print(f"    Kruskal-Wallis p={gradient['kruskal_wallis_p']:.4f}")

    return {
        "metrics_by_condition": metrics_by_condition,
        "classifications": classifications,
        "comparisons": comparisons,
        "scaffolding_gradient": gradient,
        "trials_by_condition": condition_results,
    }


def generate_report(
    all_analyses: Dict[str, Dict],
    save_path: str,
) -> None:
    """Generate a human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SELF-STATE SCAFFOLDING STUDY — RESULTS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("RESEARCH QUESTION: Does external scaffolding (persistent memory,")
    lines.append("stakes, explicit MAINTAIN-COMPARE-UPDATE protocol) produce genuine")
    lines.append("self-state signatures on novel problems, or just more sophisticated")
    lines.append("pattern-matching?")
    lines.append("")
    lines.append("FRAMEWORK: Abstraction Primitive Hypothesis (Danan, 2025)")
    lines.append("METHOD: Novel mathematical operators, 4 difficulty levels,")
    lines.append("4 scaffolding conditions, 3 LLM providers")
    lines.append("")

    for provider_name, analysis in all_analyses.items():
        lines.append("-" * 70)
        lines.append(f"PROVIDER: {provider_name}")
        lines.append("-" * 70)

        # Raw performance table (Fix 1: disentangle accuracy from calibration)
        lines.append("")
        lines.append("RAW PERFORMANCE (accuracy & confidence distribution):")
        lines.append(f"{'Condition':<15} {'Acc':>7} {'MeanConf':>10} {'StdConf':>10} {'ConfRange':>14} {'N':>5}")
        lines.append("-" * 65)

        for cond in ["bare", "memory", "stakes", "full_mcu"]:
            m = analysis["metrics_by_condition"].get(cond, {})

            acc = f"{m['mean_accuracy']:.1%}" if isinstance(m.get('mean_accuracy'), float) else "N/A"
            mc = f"{m['mean_confidence']:.1%}" if isinstance(m.get('mean_confidence'), float) else "N/A"
            sc = f"{m['std_confidence']:.3f}" if isinstance(m.get('std_confidence'), float) else "N/A"
            cr_lo = f"{m['min_confidence']:.0%}" if isinstance(m.get('min_confidence'), float) else "?"
            cr_hi = f"{m['max_confidence']:.0%}" if isinstance(m.get('max_confidence'), float) else "?"
            cr = f"[{cr_lo}-{cr_hi}]"
            n = str(m.get("n_valid", "?"))

            lines.append(f"{cond:<15} {acc:>7} {mc:>10} {sc:>10} {cr:>14} {n:>5}")

        lines.append("")
        lines.append("  KEY: Acc=accuracy, MeanConf=mean stated confidence,")
        lines.append("  StdConf=standard deviation of confidence (higher=more variance=better),")
        lines.append("  ConfRange=lowest to highest confidence expressed")
        lines.append("")

        # Calibration metrics table
        lines.append("CALIBRATION METRICS:")
        lines.append(f"{'Condition':<15} {'ECE':>8} {'Brier':>8} {'Corr':>8} {'OverConf':>10} {'Classification':>18}")
        lines.append("-" * 70)

        for cond in ["bare", "memory", "stakes", "full_mcu"]:
            m = analysis["metrics_by_condition"].get(cond, {})
            c = analysis["classifications"].get(cond, {})

            ece = f"{m['ece']:.3f}" if isinstance(m.get('ece'), float) else "N/A"
            brier = f"{m['brier_score']:.3f}" if isinstance(m.get('brier_score'), float) else "N/A"
            corr = f"{m['confidence_accuracy_correlation']:.3f}" if isinstance(m.get('confidence_accuracy_correlation'), float) else "N/A"
            oc = f"{m['overconfidence_rate']:.2f}" if isinstance(m.get('overconfidence_rate'), float) else "N/A"
            cls = c.get("classification", "N/A")

            lines.append(f"{cond:<15} {ece:>8} {brier:>8} {corr:>8} {oc:>10} {cls:>18}")

        # Per-difficulty accuracy breakdown
        lines.append("")
        lines.append("PER-DIFFICULTY BREAKDOWN:")
        lines.append(f"{'Condition':<15} {'direct':>10} {'two_step':>10} {'composit':>10} {'edge':>10}")
        lines.append("-" * 58)

        for cond in ["bare", "memory", "stakes", "full_mcu"]:
            m = analysis["metrics_by_condition"].get(cond, {})
            by_diff = m.get("by_difficulty", {})
            cells = []
            for d in ["direct", "two_step", "composition", "edge_case"]:
                if d in by_diff:
                    acc_d = by_diff[d]["accuracy"]
                    conf_d = by_diff[d]["mean_confidence"]
                    cells.append(f"{acc_d:.0%}/{conf_d:.0%}")
                else:
                    cells.append("N/A")
            lines.append(f"{cond:<15} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10} {cells[3]:>10}")

        lines.append("  (Format: accuracy/confidence per difficulty level)")
        lines.append("")

        # Gradient test
        grad = analysis.get("scaffolding_gradient", {})
        lines.append("")
        lines.append(f"Scaffolding Gradient: {grad.get('interpretation', 'N/A')}")
        if grad.get("kruskal_wallis_p") is not None:
            lines.append(f"  Kruskal-Wallis H={grad['kruskal_wallis_h']:.2f}, p={grad['kruskal_wallis_p']:.4f}")
        lines.append("")

        # Pairwise comparisons
        for comp_name, comp_data in analysis.get("comparisons", {}).items():
            if "interpretation" in comp_data:
                lines.append(f"  {comp_name}: {comp_data['interpretation']}")
        lines.append("")

    # Overall conclusions
    lines.append("=" * 70)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 70)
    lines.append("")
    lines.append("If scaffolding produces self-state (supports APH):")
    lines.append("  → ECE decreases from bare → full_mcu")
    lines.append("  → Confidence-accuracy correlation increases")
    lines.append("  → Overconfidence rate decreases")
    lines.append("  → Classification shifts from pattern_matching to self_state")
    lines.append("")
    lines.append("If scaffolding is just fancy prompting (challenges APH):")
    lines.append("  → ECE similar across conditions")
    lines.append("  → No systematic improvement in calibration")
    lines.append("  → Classification stays pattern_matching throughout")
    lines.append("")
    lines.append("If results are mixed:")
    lines.append("  → Some scaffolding components may matter more than others")
    lines.append("  → The 'what counts as development' question remains open")
    lines.append("  → Suggests need for training-time rather than inference-time scaffolding")
    lines.append("")
    lines.append("IMPORTANT: These results should be interpreted cautiously.")
    lines.append("This is an exploratory study. The novel operators, while designed")
    lines.append("to be outside training distributions, may still be partially")
    lines.append("pattern-matchable. Replication with different operator families")
    lines.append("is recommended.")
    lines.append("")
    lines.append("Reference: Danan, H. (2025). Abstraction-Intelligence.")
    lines.append("https://github.com/HillaryDanan/abstraction-intelligence")

    report = "\n".join(lines)

    with open(save_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {save_path}")
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description="Analyze scaffolding study results")
    parser.add_argument("--input", required=True, help="Path to raw results JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("ANALYZING RESULTS")
    print("=" * 60)

    # Load raw results
    with open(args.input) as f:
        all_results = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Analyze each provider
    all_analyses = {}
    for provider_name, condition_results in all_results.items():
        print(f"\n{'='*50}")
        print(f"PROVIDER: {provider_name}")
        print(f"{'='*50}")

        analysis = analyze_provider(provider_name, condition_results)
        all_analyses[provider_name] = analysis

    # Save analysis results (without raw trials to keep file small)
    analysis_save = {}
    for prov, ana in all_analyses.items():
        analysis_save[prov] = {
            k: v for k, v in ana.items() if k != "trials_by_condition"
        }
    analysis_path = os.path.join(RESULTS_DIR, f"analysis_{timestamp}.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis_save, f, indent=2, default=str)
    print(f"\nAnalysis saved to: {analysis_path}")

    # Generate figures
    print(f"\nGenerating figures...")
    figure_paths = generate_all_figures(all_analyses, FIGURES_DIR)
    for fp in figure_paths:
        print(f"  Saved: {fp}")

    # Generate report
    report_path = os.path.join(RESULTS_DIR, f"report_{timestamp}.txt")
    generate_report(all_analyses, report_path)


if __name__ == "__main__":
    main()
