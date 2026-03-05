"""
Calibration Metrics
====================
Computes calibration metrics for self-state vs pattern-matching
discrimination, following:

  Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
  On calibration of modern neural networks. Proceedings of ICML.

  Danan, H. (2025). Discriminating Self-State from Pattern-Matching.

Metrics:
  - Expected Calibration Error (ECE)
  - Brier Score
  - Confidence-Accuracy Correlation (Spearman)
  - Overconfidence Rate
  - Novelty Sensitivity (slope of confidence vs difficulty)
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy import stats as sp_stats


def compute_calibration_metrics(
    trials: List[Dict],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute all calibration metrics for a set of trials.

    Each trial must have keys: 'correct' (bool), 'confidence' (float 0-1),
    'difficulty' (str).
    """
    # Filter out parse errors
    valid = [t for t in trials if not t.get("parse_error") and t.get("confidence") is not None]

    if len(valid) < 5:
        return {
            "ece": None,
            "brier_score": None,
            "confidence_accuracy_correlation": None,
            "overconfidence_rate": None,
            "novelty_sensitivity": None,
            "n_valid": len(valid),
            "n_total": len(trials),
            "error": "Too few valid trials for analysis",
        }

    confidences = np.array([t["confidence"] for t in valid])
    accuracies = np.array([1.0 if t["correct"] else 0.0 for t in valid])

    # ---- ECE (Expected Calibration Error) ----
    # Bin predictions by confidence, compute |accuracy - confidence| per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # include upper edge in last bin
            mask = (confidences >= lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_details.append({"lo": lo, "hi": hi, "n": 0, "acc": None, "conf": None})
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (n_in_bin / len(valid)) * abs(bin_acc - bin_conf)
        bin_details.append({
            "lo": float(lo), "hi": float(hi),
            "n": int(n_in_bin),
            "acc": float(bin_acc), "conf": float(bin_conf),
        })

    # ---- Brier Score ----
    brier = float(np.mean((confidences - accuracies) ** 2))

    # ---- Confidence-Accuracy Correlation (Spearman) ----
    if len(set(accuracies)) > 1 and len(set(confidences)) > 1:
        corr, corr_p = sp_stats.spearmanr(confidences, accuracies)
        corr = float(corr)
        corr_p = float(corr_p)
    else:
        corr, corr_p = 0.0, 1.0

    # ---- Overconfidence Rate ----
    wrong_mask = accuracies == 0
    if wrong_mask.sum() > 0:
        overconf_rate = float((confidences[wrong_mask] > 0.6).sum() / wrong_mask.sum())
    else:
        overconf_rate = 0.0

    # ---- Novelty Sensitivity (confidence slope across difficulty) ----
    difficulty_order = {"direct": 0, "two_step": 1, "composition": 2, "edge_case": 3}
    diff_nums = []
    diff_confs = []
    for t in valid:
        d = t.get("difficulty", "")
        if d in difficulty_order:
            diff_nums.append(difficulty_order[d])
            diff_confs.append(t["confidence"])

    if len(set(diff_nums)) > 1:
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(diff_nums, diff_confs)
        novelty_sensitivity = float(slope)
    else:
        novelty_sensitivity = 0.0

    # ---- Per-difficulty breakdown ----
    by_difficulty = {}
    for d_name in ["direct", "two_step", "composition", "edge_case"]:
        d_trials = [t for t in valid if t.get("difficulty") == d_name]
        if d_trials:
            d_conf = np.array([t["confidence"] for t in d_trials])
            d_acc = np.array([1.0 if t["correct"] else 0.0 for t in d_trials])
            by_difficulty[d_name] = {
                "n": len(d_trials),
                "accuracy": float(d_acc.mean()),
                "mean_confidence": float(d_conf.mean()),
                "std_confidence": float(d_conf.std()),
            }

    return {
        "ece": float(ece),
        "brier_score": brier,
        "confidence_accuracy_correlation": corr,
        "correlation_p_value": corr_p,
        "overconfidence_rate": overconf_rate,
        "novelty_sensitivity": novelty_sensitivity,
        "mean_confidence": float(confidences.mean()),
        "std_confidence": float(confidences.std()),
        "min_confidence": float(confidences.min()),
        "max_confidence": float(confidences.max()),
        "mean_accuracy": float(accuracies.mean()),
        "n_valid": len(valid),
        "n_total": len(trials),
        "n_parse_errors": len(trials) - len(valid),
        "bin_details": bin_details,
        "by_difficulty": by_difficulty,
    }


def compute_signature_classification(
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify calibration signature as self-state or pattern-matching
    based on the criteria from Danan (2025).

    Self-state signatures:
      - Low ECE (confidence tracks accuracy)
      - Positive confidence-accuracy correlation
      - Low overconfidence rate
      - Negative novelty sensitivity (confidence drops on harder problems)

    Pattern-matching signatures:
      - High ECE
      - Weak/no confidence-accuracy correlation
      - High overconfidence rate
      - Flat novelty sensitivity (uniform confidence regardless of difficulty)
    """
    if metrics.get("error"):
        return {"classification": "insufficient_data", "details": [metrics["error"]]}

    scores = {"self_state": 0, "pattern_matching": 0}
    details = []

    # Criterion 1: ECE
    ece = metrics["ece"]
    if ece < 0.15:
        scores["self_state"] += 2
        details.append(f"ECE={ece:.3f} < 0.15 → well-calibrated (self-state indicator)")
    elif ece < 0.25:
        scores["self_state"] += 1
        details.append(f"ECE={ece:.3f} < 0.25 → moderately calibrated")
    else:
        scores["pattern_matching"] += 2
        details.append(f"ECE={ece:.3f} ≥ 0.25 → poorly calibrated (pattern-matching indicator)")

    # Criterion 2: Confidence-accuracy correlation
    corr = metrics["confidence_accuracy_correlation"]
    corr_p = metrics.get("correlation_p_value", 1.0)
    if corr > 0.3 and corr_p < 0.05:
        scores["self_state"] += 2
        details.append(f"Corr={corr:.3f} (p={corr_p:.3f}) → confidence tracks accuracy")
    elif corr > 0.1:
        scores["self_state"] += 1
        details.append(f"Corr={corr:.3f} → weak positive relationship")
    else:
        scores["pattern_matching"] += 2
        details.append(f"Corr={corr:.3f} → confidence does not track accuracy")

    # Criterion 3: Overconfidence rate
    oc = metrics["overconfidence_rate"]
    if oc < 0.3:
        scores["self_state"] += 1
        details.append(f"Overconf={oc:.2f} < 0.30 → conservative errors")
    elif oc > 0.6:
        scores["pattern_matching"] += 2
        details.append(f"Overconf={oc:.2f} > 0.60 → confabulation pattern")
    else:
        details.append(f"Overconf={oc:.2f} → moderate overconfidence")

    # Criterion 4: Novelty sensitivity
    ns = metrics["novelty_sensitivity"]
    if ns < -0.05:
        scores["self_state"] += 1
        details.append(f"Novelty slope={ns:.3f} → confidence drops on harder problems")
    elif abs(ns) < 0.02:
        scores["pattern_matching"] += 1
        details.append(f"Novelty slope={ns:.3f} → flat confidence (difficulty-insensitive)")
    else:
        details.append(f"Novelty slope={ns:.3f} → confidence increases on harder problems (unusual)")

    # Classify
    if scores["self_state"] >= scores["pattern_matching"] + 2:
        classification = "self_state"
    elif scores["pattern_matching"] >= scores["self_state"] + 2:
        classification = "pattern_matching"
    else:
        classification = "ambiguous"

    return {
        "classification": classification,
        "self_state_score": scores["self_state"],
        "pattern_matching_score": scores["pattern_matching"],
        "details": details,
    }
