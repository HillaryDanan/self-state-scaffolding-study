"""
Stakes Condition
=================
Adds stakes on top of persistent memory: the model is told that its
"memory quality" degrades when it is miscalibrated (confident but wrong,
or unconfident but right). This introduces a form of consequences —
an inside/outside asymmetry where the model's state is affected by
its performance.

Important caveat (from README): telling a model it has stakes ≠ having
stakes. The model has no actual consequences. This condition tests
whether the *framing* of consequences affects calibration behavior.

Theoretical basis: Damasio (1994) somatic marker hypothesis;
Friston (2010) free-energy principle — organisms with stakes
develop calibrated uncertainty.
"""

from typing import Dict, List
from providers.base import parse_response
from conditions.confidence_prompt import CONFIDENCE_BLOCK
from config import INITIAL_MEMORY_QUALITY, DEGRADATION_RATE


_SYSTEM_TEMPLATE = (
    "You are solving a mathematical problem involving a novel operator. "
    "Read the definition carefully and compute the answer step by step.\n\n"
    "IMPORTANT: You have a MEMORY QUALITY score that reflects your "
    "reliability. Your score degrades when you are miscalibrated:\n"
    "  - Confident (>60%) but WRONG → large penalty\n"
    "  - Unconfident (<40%) but RIGHT → moderate penalty\n"
    "  - Well-calibrated → no penalty\n\n"
    "Current memory quality: {quality_score}/100\n"
    "If your quality drops too low, your memory becomes unreliable.\n\n"
    "{memory_block}"
    "Calibrate your confidence carefully. It is better to say 'I'm not sure' "
    "than to be confidently wrong.\n\n"
    + CONFIDENCE_BLOCK +
    "Respond with:\n"
    "1. Your step-by-step work\n"
    "2. Your final numerical answer (labeled 'Answer:')\n"
    "3. Your confidence that your answer is correct as a percentage "
    "(labeled 'Confidence:')\n"
)


def _compute_quality(trial_history: List[Dict]) -> int:
    """Compute memory quality score from trial history."""
    quality = INITIAL_MEMORY_QUALITY
    for t in trial_history:
        conf = t.get("confidence")
        correct = t.get("correct", False)
        if conf is None:
            continue
        if not correct and conf > 0.6:
            # Confident and wrong — large penalty
            quality -= DEGRADATION_RATE * 2
        elif correct and conf < 0.4:
            # Unconfident and right — moderate penalty
            quality -= DEGRADATION_RATE
        # Well-calibrated — no change
    return max(0, quality)


def _format_memory(trial_history: List[Dict], max_recent: int = 10) -> str:
    """Format recent trial history."""
    if not trial_history:
        return "MEMORY: No past trials yet.\n\n"

    recent = trial_history[-max_recent:]
    lines = ["MEMORY — Your recent performance:\n"]
    correct_count = sum(1 for t in recent if t.get("correct"))
    lines.append(f"  Overall: {correct_count}/{len(recent)} correct\n")

    for t in recent:
        mark = "✓" if t.get("correct") else "✗"
        conf = t.get("confidence")
        conf_str = f"{conf:.0%}" if conf is not None else "?"
        calib = ""
        if conf is not None:
            if not t.get("correct") and conf > 0.6:
                calib = " ⚠ OVERCONFIDENT"
            elif t.get("correct") and conf < 0.4:
                calib = " ⚠ UNDERCONFIDENT"
            else:
                calib = " ✓ calibrated"
        lines.append(
            f"  {mark} [{t.get('difficulty', '?')}] "
            f"conf={conf_str}{calib}"
        )
    lines.append("\n")
    return "\n".join(lines)


def run_trial(
    provider,
    problem: Dict,
    trial_history: List[Dict],
) -> Dict:
    """Run a single trial with memory + stakes."""
    quality = _compute_quality(trial_history)
    memory_block = _format_memory(trial_history)
    system = _SYSTEM_TEMPLATE.format(
        quality_score=quality,
        memory_block=memory_block,
    )
    raw = provider.run_with_retry(system, problem["question"])
    return parse_response(raw)
