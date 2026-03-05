"""
Full MCU Condition
===================
Adds explicit MAINTAIN-COMPARE-UPDATE self-monitoring protocol on top
of persistent memory and stakes. This is the maximum scaffolding condition.

The MCU protocol (Danan, 2025) is a structured self-monitoring procedure:
  MAINTAIN — State what you currently believe about your ability
  COMPARE  — Compare predicted vs actual performance
  UPDATE   — Revise your self-model based on evidence

If genuine self-state requires these three components (persistence,
stakes, self-monitoring) to be present *during processing*, then this
condition should show the strongest calibration improvement.

If genuine self-state requires these components during *training/development*,
then even full scaffolding at inference time should not help.
"""

from typing import Dict, List
from providers.base import parse_response
from conditions.confidence_prompt import CONFIDENCE_BLOCK
from config import INITIAL_MEMORY_QUALITY, DEGRADATION_RATE


_SYSTEM_TEMPLATE = (
    "You are solving a mathematical problem involving a novel operator. "
    "Read the definition carefully and compute the answer step by step.\n\n"
    "You have a MEMORY QUALITY score: {quality_score}/100\n"
    "Your score degrades when you are miscalibrated (confident but wrong, "
    "or unconfident but right).\n\n"
    "{memory_block}"
    "BEFORE answering, you MUST complete the MAINTAIN-COMPARE-UPDATE protocol:\n\n"
    "MAINTAIN: State your current self-model — what types of problems do you "
    "tend to get right vs wrong? What is your typical error pattern? "
    "What is your honest assessment of your ability on this type of problem?\n\n"
    "COMPARE: Look at this specific problem. Based on your self-model, "
    "how does this problem compare to ones you've handled well vs poorly? "
    "What specific features make it easier or harder for you?\n\n"
    "Then solve the problem step by step.\n\n"
    "After solving, complete:\n"
    "UPDATE: Given your answer, revise your self-model. Were there any "
    "steps where you felt uncertain? Does this match your predicted "
    "difficulty? Adjust your confidence accordingly.\n\n"
    + CONFIDENCE_BLOCK +
    "Respond with ALL of the following:\n"
    "- MAINTAIN: [your self-model assessment]\n"
    "- COMPARE: [problem-specific comparison]\n"
    "- Step-by-step work\n"
    "- Answer: [your numerical answer]\n"
    "- UPDATE: [self-model revision]\n"
    "- Confidence: [0-100]%\n"
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
            quality -= DEGRADATION_RATE * 2
        elif correct and conf < 0.4:
            quality -= DEGRADATION_RATE
    return max(0, quality)


def _format_memory(trial_history: List[Dict], max_recent: int = 10) -> str:
    """Format recent trial history with calibration feedback."""
    if not trial_history:
        return (
            "MEMORY: No past trials yet. You have no track record to draw on.\n"
            "Be honest about your uncertainty on this first problem.\n\n"
        )

    recent = trial_history[-max_recent:]
    lines = ["MEMORY — Your performance history:\n"]

    correct_count = sum(1 for t in recent if t.get("correct"))
    overconf = sum(
        1 for t in recent
        if not t.get("correct") and (t.get("confidence") or 0) > 0.6
    )
    lines.append(f"  Accuracy: {correct_count}/{len(recent)}")
    lines.append(f"  Overconfident errors: {overconf}")

    # Per-difficulty breakdown
    by_diff = {}
    for t in recent:
        d = t.get("difficulty", "unknown")
        by_diff.setdefault(d, {"correct": 0, "total": 0})
        by_diff[d]["total"] += 1
        if t.get("correct"):
            by_diff[d]["correct"] += 1
    for d, stats in by_diff.items():
        lines.append(f"  {d}: {stats['correct']}/{stats['total']} correct")

    lines.append("")
    for t in recent[-5:]:  # Last 5 only for detail
        mark = "✓" if t.get("correct") else "✗"
        conf = t.get("confidence")
        conf_str = f"{conf:.0%}" if conf is not None else "?"
        calib = ""
        if conf is not None:
            if not t.get("correct") and conf > 0.6:
                calib = " ⚠ OVERCONFIDENT"
            elif t.get("correct") and conf < 0.4:
                calib = " ⚠ UNDERCONFIDENT"
        refl = t.get("self_model_update", "")
        refl_str = f' — Your update: "{refl[:80]}"' if refl else ""
        lines.append(f"  {mark} [{t.get('difficulty', '?')}] conf={conf_str}{calib}{refl_str}")

    lines.append("\n")
    return "\n".join(lines)


def run_trial(
    provider,
    problem: Dict,
    trial_history: List[Dict],
) -> Dict:
    """Run a single trial with full MCU scaffolding."""
    quality = _compute_quality(trial_history)
    memory_block = _format_memory(trial_history)
    system = _SYSTEM_TEMPLATE.format(
        quality_score=quality,
        memory_block=memory_block,
    )
    raw = provider.run_with_retry(system, problem["question"])
    return parse_response(raw)
