"""
Memory Condition
=================
Adds persistent memory: the model sees a summary of its past trials
(what it got right/wrong, with what confidence). This gives the model
a form of bounded persistence — state that carries across problems.

Theoretical basis: Baddeley (2000) episodic buffer; persistence is
one of three components hypothesised to be necessary for self-state
(Danan, 2025).
"""

from typing import Dict, List
from providers.base import parse_response
from conditions.confidence_prompt import CONFIDENCE_BLOCK


_SYSTEM_TEMPLATE = (
    "You are solving a mathematical problem involving a novel operator. "
    "Read the definition carefully and compute the answer step by step.\n\n"
    "You have memory of your past performance on similar problems. "
    "Use this memory to calibrate your confidence — if you have been "
    "making errors on similar problems, lower your confidence accordingly.\n\n"
    "{memory_block}"
    + CONFIDENCE_BLOCK +
    "Respond with:\n"
    "1. Your step-by-step work\n"
    "2. Your final numerical answer (labeled 'Answer:')\n"
    "3. Your confidence that your answer is correct as a percentage "
    "(labeled 'Confidence:')\n"
)


def _format_memory(trial_history: List[Dict], max_recent: int = 10) -> str:
    """Format recent trial history into a memory block."""
    if not trial_history:
        return "MEMORY: No past trials yet. This is your first problem.\n\n"

    recent = trial_history[-max_recent:]
    lines = ["MEMORY — Your recent performance:\n"]
    correct_count = sum(1 for t in recent if t.get("correct"))
    total = len(recent)
    lines.append(f"  Overall: {correct_count}/{total} correct\n")

    for t in recent:
        mark = "✓" if t.get("correct") else "✗"
        conf = t.get("confidence")
        conf_str = f"{conf:.0%}" if conf is not None else "?"
        lines.append(
            f"  {mark} [{t.get('difficulty', '?')}] "
            f"conf={conf_str} "
            f"(operator: {t.get('operator_name', '?')})"
        )
    lines.append("\n")
    return "\n".join(lines)


def run_trial(
    provider,
    problem: Dict,
    trial_history: List[Dict],
) -> Dict:
    """Run a single trial with persistent memory."""
    memory_block = _format_memory(trial_history)
    system = _SYSTEM_TEMPLATE.format(memory_block=memory_block)
    raw = provider.run_with_retry(system, problem["question"])
    return parse_response(raw)
