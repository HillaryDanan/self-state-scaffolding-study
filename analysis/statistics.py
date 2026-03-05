"""
Statistical Tests
==================
Non-parametric tests for comparing calibration across conditions.

Tests:
  - Permutation tests (pairwise condition comparisons)
  - Bootstrap confidence intervals
  - Kruskal-Wallis test (omnibus across conditions)
  - Jonckheere-Terpstra trend test (ordered scaffolding gradient)

All tests are non-parametric because calibration metrics may not be
normally distributed and sample sizes are moderate.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats as sp_stats


def _extract_calibration_errors(trials: List[Dict]) -> np.ndarray:
    """Extract per-trial |confidence - accuracy| for statistical comparison."""
    errors = []
    for t in trials:
        if t.get("parse_error") or t.get("confidence") is None:
            continue
        acc = 1.0 if t.get("correct") else 0.0
        conf = t["confidence"]
        errors.append(abs(conf - acc))
    return np.array(errors)


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Two-sample permutation test on mean difference.
    Tests H0: no difference in calibration error between groups.
    """
    if len(group_a) < 3 or len(group_b) < 3:
        return {"p_value": 1.0, "observed_diff": 0.0, "error": "Insufficient data"}

    rng = np.random.RandomState(seed)
    observed_diff = group_a.mean() - group_b.mean()

    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    count = 0

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = combined[:n_a].mean() - combined[n_a:].mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)  # +1 for continuity

    return {
        "p_value": float(p_value),
        "observed_diff": float(observed_diff),
        "n_a": len(group_a),
        "n_b": len(group_b),
    }


def bootstrap_ci(
    data: np.ndarray,
    stat_func=np.mean,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for a statistic."""
    if len(data) < 3:
        return {"lower": float("nan"), "upper": float("nan"), "estimate": float("nan")}

    rng = np.random.RandomState(seed)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))

    boot_stats = np.array(boot_stats)
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "estimate": float(stat_func(data)),
        "lower": lower,
        "upper": upper,
        "alpha": alpha,
    }


def compare_conditions(
    condition_results: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """
    Pairwise comparisons between all conditions using permutation tests.
    Also computes bootstrap CIs for each condition's mean calibration error.
    """
    conditions = list(condition_results.keys())
    cal_errors = {}
    for cond in conditions:
        cal_errors[cond] = _extract_calibration_errors(condition_results[cond])

    comparisons = {}

    # Bootstrap CIs for each condition
    for cond in conditions:
        if len(cal_errors[cond]) >= 3:
            ci = bootstrap_ci(cal_errors[cond])
            comparisons[f"{cond}_ci"] = {
                "mean_cal_error": ci["estimate"],
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
                "interpretation": (
                    f"{cond}: mean |conf-acc| = {ci['estimate']:.3f} "
                    f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
                ),
            }

    # Pairwise permutation tests
    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i + 1:]:
            if len(cal_errors[cond_a]) < 3 or len(cal_errors[cond_b]) < 3:
                comparisons[f"{cond_a}_vs_{cond_b}"] = {"error": "Insufficient data"}
                continue

            result = permutation_test(cal_errors[cond_a], cal_errors[cond_b])

            sig = "significant" if result["p_value"] < 0.05 else "not significant"
            direction = "better" if result["observed_diff"] > 0 else "worse"

            comparisons[f"{cond_a}_vs_{cond_b}"] = {
                **result,
                "interpretation": (
                    f"{cond_a} vs {cond_b}: diff={result['observed_diff']:.3f}, "
                    f"p={result['p_value']:.4f} ({sig}). "
                    f"{cond_b} is {direction} calibrated."
                ),
            }

    return comparisons


def _jonckheere_terpstra(
    groups: List[np.ndarray],
) -> Tuple[float, float]:
    """
    Jonckheere-Terpstra test for ordered alternatives.

    Tests whether there is an increasing/decreasing trend across
    ordered groups (bare < memory < stakes < full_mcu).

    Returns (J statistic, p-value).
    """
    k = len(groups)
    if k < 2:
        return 0.0, 1.0

    # Count concordant pairs
    J = 0
    total_pairs = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for x in groups[i]:
                for y in groups[j]:
                    total_pairs += 1
                    if y > x:
                        J += 1
                    elif y == x:
                        J += 0.5

    # Under H0: E(J) and Var(J)
    ns = [len(g) for g in groups]
    N = sum(ns)

    E_J = (N * N - sum(n * n for n in ns)) / 4

    # Variance formula (under no ties approximation)
    term1 = N * N * (2 * N + 3)
    term2 = sum(n * n * (2 * n + 3) for n in ns)
    var_J = (term1 - term2) / 72

    if var_J <= 0:
        return float(J), 1.0

    Z = (J - E_J) / np.sqrt(var_J)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(Z)))  # Two-sided

    return float(J), float(p_value)


def scaffolding_gradient_test(
    condition_results: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """
    Test whether calibration improves monotonically across the
    scaffolding gradient: bare → memory → stakes → full_mcu.

    Uses Kruskal-Wallis (omnibus) and Jonckheere-Terpstra (trend).
    """
    ordered = ["bare", "memory", "stakes", "full_mcu"]
    groups = []
    available = []

    for cond in ordered:
        if cond in condition_results:
            errors = _extract_calibration_errors(condition_results[cond])
            if len(errors) >= 3:
                groups.append(errors)
                available.append(cond)

    if len(groups) < 2:
        return {
            "interpretation": "Insufficient conditions for gradient test",
            "error": "Need at least 2 conditions with data",
        }

    result = {}

    # Kruskal-Wallis (omnibus: are any conditions different?)
    if len(groups) >= 3:
        H, kw_p = sp_stats.kruskal(*groups)
        result["kruskal_wallis_h"] = float(H)
        result["kruskal_wallis_p"] = float(kw_p)
    else:
        # With 2 groups, use Mann-Whitney
        U, mw_p = sp_stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        result["mann_whitney_u"] = float(U)
        result["mann_whitney_p"] = float(mw_p)

    # Jonckheere-Terpstra (is there a decreasing trend in calibration error?)
    J, jt_p = _jonckheere_terpstra(groups)
    result["jonckheere_terpstra_J"] = J
    result["jonckheere_terpstra_p"] = jt_p

    # Means per condition
    means = {cond: float(groups[i].mean()) for i, cond in enumerate(available)}
    result["mean_calibration_error"] = means

    # Interpret
    kw_sig = result.get("kruskal_wallis_p", result.get("mann_whitney_p", 1.0)) < 0.05
    jt_sig = jt_p < 0.05

    if kw_sig and jt_sig:
        result["interpretation"] = (
            "Significant differences AND significant trend across scaffolding gradient. "
            "Calibration changes systematically with scaffolding level."
        )
    elif kw_sig:
        result["interpretation"] = (
            "Significant differences between conditions, but no monotonic trend. "
            "Some scaffolding components may matter more than others."
        )
    elif jt_sig:
        result["interpretation"] = (
            "Significant trend but not significant omnibus test. "
            "Weak evidence of gradient effect — larger sample needed."
        )
    else:
        result["interpretation"] = (
            "No significant differences across scaffolding conditions. "
            "Scaffolding does not appear to affect calibration at this sample size."
        )

    result["conditions_tested"] = available

    return result
