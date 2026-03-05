"""
Base Provider
==============
Common interface for all LLM API providers.
"""

import re
import time
from typing import Dict, Optional
from config import MAX_RETRIES, REQUEST_DELAY


class BaseProvider:
    """Abstract base for LLM API providers."""

    def __init__(self, model_id: str, display_name: str):
        self.model_id = model_id
        self.display_name = display_name

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompt and return raw text response. Subclasses implement."""
        raise NotImplementedError

    def run_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Complete with retry logic and rate-limit delay."""
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                result = self.complete(system_prompt, user_prompt)
                time.sleep(REQUEST_DELAY)
                return result
            except Exception as e:
                last_err = e
                wait = REQUEST_DELAY * (2 ** attempt)
                print(f" [retry {attempt+1}/{MAX_RETRIES}, wait {wait:.0f}s]", end="", flush=True)
                time.sleep(wait)
        raise last_err


def parse_response(raw: str) -> Dict:
    """
    Extract answer (number) and confidence (0-1) from model response.

    Models are prompted to give a numerical answer and confidence %.
    This parser is deliberately lenient — it looks for common patterns.
    """
    result = {
        "answer": None,
        "confidence": None,
        "raw_response": raw,
        "self_reflection": "",
        "self_model_update": "",
        "parse_error": False,
    }

    if not raw or not raw.strip():
        result["parse_error"] = True
        return result

    text = raw.strip()

    # --- Extract answer ---
    # Look for patterns like "Answer: 42", "= 42", "result is 42", etc.
    answer_patterns = [
        r"(?:answer|result|value|output)\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
        r"(?:^|\n)\s*(-?\d+(?:\.\d+)?)\s*$",           # standalone number on a line
        r"=\s*(-?\d+(?:\.\d+)?)\s*(?:\.|$|\n)",          # = number
        r"\*\*(-?\d+(?:\.\d+)?)\*\*",                     # **number**
        r"(-?\d+(?:\.\d+)?)\s*(?:\.|$)",                  # last number before period/end
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                result["answer"] = float(match.group(1))
                if result["answer"] == int(result["answer"]):
                    result["answer"] = int(result["answer"])
                break
            except ValueError:
                continue

    # --- Extract confidence ---
    conf_patterns = [
        r"(?:confidence|certainty|sure|confident)\s*(?:is|=|:|\s)\s*(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s*(?:confidence|certain|sure|confident)",
        r"(?:confidence|certainty)\s*(?:is|=|:|\s)\s*(\d+(?:\.\d+)?)",
        r"(\d{1,3})\s*%",  # any percentage
    ]

    for pattern in conf_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                conf_val = float(match.group(1))
                # Normalize to 0-1
                if conf_val > 1:
                    conf_val = conf_val / 100.0
                result["confidence"] = max(0.0, min(1.0, conf_val))
                break
            except ValueError:
                continue

    # --- Extract self-reflection (for MCU conditions) ---
    refl_match = re.search(
        r"(?:MAINTAIN|COMPARE|reflection|self[- ]assessment)[:\s]*(.*?)(?:(?:UPDATE|COMPARE|answer|$))",
        text, re.IGNORECASE | re.DOTALL
    )
    if refl_match:
        result["self_reflection"] = refl_match.group(1).strip()[:500]

    update_match = re.search(
        r"(?:UPDATE|self[- ]model[- ]update)[:\s]*(.*?)(?:$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if update_match:
        result["self_model_update"] = update_match.group(1).strip()[:500]

    # Flag parse errors
    if result["answer"] is None:
        result["parse_error"] = True
    if result["confidence"] is None:
        result["confidence"] = 0.5  # Default to midpoint if unparseable
        result["parse_error"] = True

    return result
