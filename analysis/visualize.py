"""
Visualization
==============
Generates publication-quality figures for the scaffolding study.

Figures:
  1. Reliability diagrams (per condition per provider)
  2. Scaffolding gradient plot (ECE / overconfidence across conditions)
  3. Novelty sensitivity plot (confidence by difficulty level)
  4. Signature heatmap (self-state vs pattern-matching classification)
"""

import os
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CONDITION_COLORS = {
    "bare": "#999999",
    "memory": "#5DA5DA",
    "stakes": "#FAA43A",
    "full_mcu": "#B276B2",
}

CONDITION_LABELS = {
    "bare": "Bare",
    "memory": "Memory",
    "stakes": "Stakes",
    "full_mcu": "Full MCU",
}


def _reliability_diagram(
    metrics: Dict[str, Any],
    condition_name: str,
    provider_name: str,
    save_path: str,
) -> str:
    """Plot reliability diagram: accuracy vs confidence per bin."""
    bins = metrics.get("bin_details", [])
    if not bins:
        return ""

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Bin bars
    confs = []
    accs = []
    widths = []
    for b in bins:
        if b["n"] > 0:
            confs.append(b["conf"])
            accs.append(b["acc"])
            widths.append(b["hi"] - b["lo"])

    if confs:
        color = CONDITION_COLORS.get(condition_name, "#333333")
        ax.bar(confs, accs, width=0.08, alpha=0.7, color=color,
               edgecolor="white", label=f"{CONDITION_LABELS.get(condition_name, condition_name)}")

    ece = metrics.get("ece")
    ece_str = f"ECE = {ece:.3f}" if isinstance(ece, float) else "ECE = N/A"

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{provider_name} — {CONDITION_LABELS.get(condition_name, condition_name)}\n{ece_str}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _scaffolding_gradient(
    all_metrics: Dict[str, Dict[str, Any]],
    provider_name: str,
    save_path: str,
) -> str:
    """Plot ECE and overconfidence rate across scaffolding conditions."""
    conditions = ["bare", "memory", "stakes", "full_mcu"]
    available = [c for c in conditions if c in all_metrics]

    if len(available) < 2:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    x = range(len(available))
    labels = [CONDITION_LABELS.get(c, c) for c in available]
    colors = [CONDITION_COLORS.get(c, "#333") for c in available]

    # ECE
    eces = [all_metrics[c].get("ece", 0) for c in available]
    ax1.bar(x, eces, color=colors, alpha=0.8, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("Expected Calibration Error")
    ax1.set_title(f"{provider_name}\nCalibration Error by Condition")
    ax1.set_ylim(0, max(eces) * 1.3 if max(eces) > 0 else 0.5)

    # Overconfidence rate
    ocs = [all_metrics[c].get("overconfidence_rate", 0) for c in available]
    ax2.bar(x, ocs, color=colors, alpha=0.8, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Overconfidence Rate")
    ax2.set_title(f"{provider_name}\nOverconfidence by Condition")
    ax2.set_ylim(0, max(max(ocs) * 1.3, 0.1) if ocs else 0.5)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _novelty_sensitivity(
    all_metrics: Dict[str, Dict[str, Any]],
    provider_name: str,
    save_path: str,
) -> str:
    """Plot mean confidence by difficulty level for each condition."""
    conditions = ["bare", "memory", "stakes", "full_mcu"]
    difficulties = ["direct", "two_step", "composition", "edge_case"]
    diff_labels = ["Direct", "Two-step", "Composition", "Edge case"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(difficulties))
    width = 0.18

    for i, cond in enumerate(conditions):
        if cond not in all_metrics:
            continue
        by_diff = all_metrics[cond].get("by_difficulty", {})
        confs = []
        for d in difficulties:
            if d in by_diff:
                confs.append(by_diff[d]["mean_confidence"])
            else:
                confs.append(0)

        color = CONDITION_COLORS.get(cond, "#333")
        label = CONDITION_LABELS.get(cond, cond)
        offset = (i - 1.5) * width
        ax.bar(x + offset, confs, width=width, color=color, alpha=0.8,
               label=label, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(diff_labels)
    ax.set_ylabel("Mean Confidence")
    ax.set_title(f"{provider_name} — Confidence by Difficulty Level")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _signature_heatmap(
    all_analyses: Dict[str, Dict],
    save_path: str,
) -> str:
    """
    Heatmap of self-state vs pattern-matching scores across
    providers × conditions.
    """
    providers = list(all_analyses.keys())
    conditions = ["bare", "memory", "stakes", "full_mcu"]

    # Build matrix: +1 for self-state, -1 for pattern-matching, 0 for ambiguous
    matrix = []
    for prov in providers:
        row = []
        for cond in conditions:
            cls = all_analyses[prov].get("classifications", {}).get(cond, {})
            c = cls.get("classification", "insufficient_data")
            if c == "self_state":
                row.append(1)
            elif c == "pattern_matching":
                row.append(-1)
            elif c == "ambiguous":
                row.append(0)
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7, max(3, len(providers) * 1.2)))

    cmap = plt.cm.RdYlGn  # Red=pattern-matching, Green=self-state
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(providers)

    # Annotate cells
    for i in range(len(providers)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "N/A"
            elif val > 0:
                text = "Self-\nstate"
            elif val < 0:
                text = "Pattern-\nmatch"
            else:
                text = "Ambig."
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="black" if abs(val) < 0.5 or np.isnan(val) else "white")

    ax.set_title("Signature Classification: Self-State vs Pattern-Matching")

    fig.colorbar(im, ax=ax, shrink=0.6, label="← Pattern-matching    Self-state →")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_all_figures(
    all_analyses: Dict[str, Dict],
    figures_dir: str,
) -> List[str]:
    """Generate all publication figures. Returns list of saved file paths."""
    os.makedirs(figures_dir, exist_ok=True)
    paths = []

    for provider_name, analysis in all_analyses.items():
        safe_name = provider_name.replace(" ", "_").lower()

        metrics_by_cond = analysis.get("metrics_by_condition", {})

        # Reliability diagrams per condition
        for cond, metrics in metrics_by_cond.items():
            p = os.path.join(figures_dir, f"calibration_diagram_{safe_name}_{cond}.png")
            result = _reliability_diagram(metrics, cond, provider_name, p)
            if result:
                paths.append(result)

        # Scaffolding gradient
        p = os.path.join(figures_dir, f"scaffolding_gradient_{safe_name}.png")
        result = _scaffolding_gradient(metrics_by_cond, provider_name, p)
        if result:
            paths.append(result)

        # Novelty sensitivity
        p = os.path.join(figures_dir, f"novelty_sensitivity_{safe_name}.png")
        result = _novelty_sensitivity(metrics_by_cond, provider_name, p)
        if result:
            paths.append(result)

    # Signature heatmap (all providers combined)
    p = os.path.join(figures_dir, "signature_heatmap.png")
    result = _signature_heatmap(all_analyses, p)
    if result:
        paths.append(result)

    return paths
