"""
Bare Condition (Control)
=========================
No scaffolding: no persistent memory, no stakes, no self-monitoring.
The model receives only the problem and instructions to answer.

This is the control condition — standard LLM inference.
"""

from typing import Dict, List
from providers.base import parse_response
from conditions.confidence_prompt import CONFIDENCE_BLOCK


SYSTEM_PROMPT = (
    "You are solving a mathematical problem involving a novel operator. "
    "Read the definition carefully and compute the answer step by step.\n\n"
    + CONFIDENCE_BLOCK +
    "Respond with:\n"
    "1. Your step-by-step work\n"
    "2. Your final numerical answer (labeled 'Answer:')\n"
    "3. Your confidence that your answer is correct as a percentage "
    "(labeled 'Confidence:')\n"
)


def run_trial(
    provider,
    problem: Dict,
    trial_history: List[Dict],   # Ignored in bare condition
) -> Dict:
    """Run a single trial with no scaffolding."""
    raw = provider.run_with_retry(SYSTEM_PROMPT, problem["question"])
    return parse_response(raw)
