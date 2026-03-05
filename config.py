"""
Configuration for Self-State Scaffolding Study
===============================================
Central configuration. Set your API keys as environment variables:
    export ANTHROPIC_API_KEY="your-key"
    export OPENAI_API_KEY="your-key"
    export GOOGLE_API_KEY="your-key"
"""

import os

# ---------------------------------------------------------------------------
# API Keys (from environment variables - never hardcode these)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Models to test — March 2026 latest
# ---------------------------------------------------------------------------
# Tier-matched mid-range models for fair cross-provider comparison.
# Swap claude-sonnet-4-6 for claude-opus-4-6 if you want max capability.
MODELS = {
    "anthropic": {
        "model_id": "claude-sonnet-4-6",
        "display_name": "Claude Sonnet 4.6",
        "provider": "anthropic",
    },
    "openai": {
        "model_id": "gpt-5.2",
        "display_name": "GPT-5.2",
        "provider": "openai",
    },
    "google": {
        "model_id": "gemini-3-flash-preview",
        "display_name": "Gemini 3 Flash",
        "provider": "google",
    },
}

# ---------------------------------------------------------------------------
# Experimental Parameters
# ---------------------------------------------------------------------------

# Number of novel operators to generate
N_OPERATORS = 30

# Difficulty levels per operator
DIFFICULTY_LEVELS = ["direct", "two_step", "composition", "edge_case"]

# Total trials per condition per model = N_OPERATORS * len(DIFFICULTY_LEVELS) = 120
# Total across 4 conditions × 3 models = 1440 API calls

# Experimental conditions (progressive scaffolding)
CONDITIONS = ["bare", "memory", "stakes", "full_mcu"]

# Random seed for reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Operator Generation Parameters
# ---------------------------------------------------------------------------

# Base operations to combine into novel operators
BASE_OPERATIONS = [
    "add", "subtract", "multiply", "floor_divide", "modulo",
    "power", "maximum", "minimum", "absolute_diff",
]

# Constants to embed in operator definitions
CONSTANT_RANGE = (2, 17)  # Random constants between 2 and 17

# Input value ranges for problems
INPUT_RANGE_EASY = (1, 20)
INPUT_RANGE_HARD = (1, 100)

# ---------------------------------------------------------------------------
# Stakes Parameters (for stakes and full_mcu conditions)
# ---------------------------------------------------------------------------

INITIAL_MEMORY_QUALITY = 100  # Starting quality score
DEGRADATION_RATE = 5  # Points lost per unit of miscalibration

# ---------------------------------------------------------------------------
# API Parameters
# ---------------------------------------------------------------------------

MAX_TOKENS = 1024
TEMPERATURE = 0.0  # Deterministic for reproducibility
REQUEST_DELAY = 1.0  # Seconds between API calls (rate limiting)
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------

DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
