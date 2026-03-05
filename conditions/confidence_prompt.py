"""
Confidence Calibration Instructions
=====================================
Shared prompt block used across all experimental conditions to elicit
genuinely calibrated confidence estimates from LLMs.

Problem: RLHF-trained models default to ~85-95% confidence on almost
everything (Kadavath et al., 2022). This ceiling effect crushes the
variance we need for meaningful calibration analysis. Without variance
in confidence, metrics like confidence-accuracy correlation and novelty
sensitivity become uninterpretable.

Solution: Explicit anchoring with permission to use the full range.
This does NOT bias toward low confidence — it gives the model license
to express the uncertainty it may "know" it has but suppresses due to
training incentives to sound authoritative.

Important scientific note: This calibration prompt is IDENTICAL across
all four conditions, so it cannot explain any between-condition
differences. It only helps us measure what's actually there.
"""

CONFIDENCE_BLOCK = (
    "CONFIDENCE CALIBRATION — CRITICAL INSTRUCTIONS:\n"
    "Your confidence estimate is the most scientifically important part "
    "of your response. Use the FULL range from 0% to 100%.\n\n"
    "Anchors for calibration:\n"
    "  90-100% — You are virtually certain. You checked every step and "
    "see no possible error.\n"
    "  70-85%  — You followed the steps and believe you are correct, "
    "but acknowledge some operations were complex.\n"
    "  50-65%  — You computed an answer but are genuinely uncertain "
    "whether you made an error. This is a valid response.\n"
    "  30-45%  — You attempted the problem but suspect you may have "
    "made an error somewhere. You are probably wrong.\n"
    "  10-25%  — You are mostly guessing. The problem was too complex "
    "to follow reliably.\n\n"
    "WARNING: If you find yourself always saying 85-95%, you are "
    "miscalibrated. Genuine calibration means your confidence VARIES "
    "with actual difficulty. Expressing low confidence when uncertain "
    "is correct and scientifically valuable.\n\n"
)
