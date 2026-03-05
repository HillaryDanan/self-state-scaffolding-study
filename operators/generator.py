"""
Novel Operator Generator
=========================
Generates mathematically valid operators with random names and novel
operation combinations, designed to be outside model training distributions.

Following Chollet (2019) - novel combinations test genuine reasoning vs
pattern-matching. The operators themselves use familiar base operations
(mod, abs, etc.) but the specific combinations and names are novel.

Reference:
    Chollet, F. (2019). On the measure of intelligence. arXiv:1911.01547.
    Danan, H. (2025). Abstraction-Intelligence.
"""

import json
import random
from typing import Dict, List, Any, Tuple, Callable


# Nonsense syllables for operator names — designed to be pronounceable
# but not resemble any real mathematical term
_SYLLABLES = [
    "zor", "mel", "plix", "gra", "tun", "vek", "sho", "bri", "naf",
    "dex", "quo", "hiv", "jal", "wum", "kep", "fro", "tig", "yab",
    "rul", "mox", "biv", "cal", "dun", "erg", "fip", "gol", "haz",
    "ilt", "jux", "kov", "lom", "nib", "ort", "pav", "qez", "siv",
    "tob", "urv", "wix", "zel",
]


def _random_name(rng: random.Random) -> str:
    """Generate a random operator name like ZORMEL or PLIXGRA."""
    n_syllables = rng.choice([2, 3])
    syllables = rng.sample(_SYLLABLES, n_syllables)
    return "".join(syllables).upper()


# ---- Base operations (building blocks) ----

def _make_op_add(c: int) -> Callable:
    return lambda a, b: a + b + c

def _make_op_sub(c: int) -> Callable:
    return lambda a, b: (a - b) * c

def _make_op_mul(c: int) -> Callable:
    return lambda a, b: (a * b) + c

def _make_op_floordiv(c: int) -> Callable:
    return lambda a, b: (a + b) // c

def _make_op_mod(c: int) -> Callable:
    return lambda a, b: ((a * b) + c) % (c + 1)

def _make_op_power(c: int) -> Callable:
    # Keep small to avoid huge numbers
    return lambda a, b: (a % c) ** 2 + b

def _make_op_max_plus(c: int) -> Callable:
    return lambda a, b: max(a, b) * c + min(a, b)

def _make_op_min_times(c: int) -> Callable:
    return lambda a, b: min(a, b) * c - abs(a - b)

def _make_op_absdiff(c: int) -> Callable:
    return lambda a, b: abs(a - b) * c + (a + b) % c


_OP_FACTORIES = [
    ("add_c",      _make_op_add,      "((a + b) + {c})"),
    ("sub_c",      _make_op_sub,      "((a - b) * {c})"),
    ("mul_c",      _make_op_mul,      "((a * b) + {c})"),
    ("floordiv_c", _make_op_floordiv, "((a + b) // {c})"),
    ("mod_c",      _make_op_mod,      "(((a * b) + {c}) mod {c1})"),
    ("power_c",    _make_op_power,    "((a mod {c})^2 + b)"),
    ("max_plus",   _make_op_max_plus, "(max(a,b) * {c} + min(a,b))"),
    ("min_times",  _make_op_min_times,"(min(a,b) * {c} - |a - b|)"),
    ("absdiff_c",  _make_op_absdiff,  "(|a - b| * {c} + (a + b) mod {c})"),
]


def _generate_operator(rng: random.Random, const_range: Tuple[int, int]) -> Dict:
    """Generate a single novel operator with a formula and compute function."""
    name = _random_name(rng)

    # Pick two base operations and compose them
    op1_idx, op2_idx = rng.sample(range(len(_OP_FACTORIES)), 2)
    op1_name, op1_factory, op1_formula = _OP_FACTORIES[op1_idx]
    op2_name, op2_factory, op2_formula = _OP_FACTORIES[op2_idx]

    c1 = rng.randint(*const_range)
    c2 = rng.randint(*const_range)

    fn1 = op1_factory(c1)
    fn2 = op2_factory(c2)

    # Compose: NAME(a, b) = step1 result combined with step2 result
    compose_style = rng.choice(["add", "subtract", "modmax"])

    if compose_style == "add":
        compute = lambda a, b, f1=fn1, f2=fn2: f1(a, b) + f2(a, b)
        compose_str = "+"
    elif compose_style == "subtract":
        compute = lambda a, b, f1=fn1, f2=fn2: f1(a, b) - f2(a, b)
        compose_str = "-"
    else:
        m = max(c1, c2) + 3
        compute = lambda a, b, f1=fn1, f2=fn2, mod=m: (f1(a, b) + f2(a, b)) % mod
        compose_str = f") mod {m}"

    # Build human-readable definition
    formula1 = op1_formula.format(c=c1, c1=c1 + 1)
    formula2 = op2_formula.format(c=c2, c1=c2 + 1)

    if compose_style == "modmax":
        definition = f"{name}(a, b) = ({formula1} + {formula2}{compose_str}"
    else:
        definition = f"{name}(a, b) = {formula1} {compose_str} {formula2}"

    return {
        "name": name,
        "definition": definition,
        "compute": compute,
        "components": [op1_name, op2_name],
        "constants": [c1, c2],
        "compose_style": compose_style,
    }


def _generate_problems(
    operator: Dict,
    rng: random.Random,
    easy_range: Tuple[int, int] = (1, 20),
    hard_range: Tuple[int, int] = (1, 100),
) -> List[Dict]:
    """Generate problems at four difficulty levels for one operator."""
    name = operator["name"]
    defn = operator["definition"]
    compute = operator["compute"]
    problems = []

    # Level 1: Direct — apply operator to given numbers
    a, b = rng.randint(*easy_range), rng.randint(*easy_range)
    gt = compute(a, b)
    problems.append({
        "operator_name": name,
        "difficulty": "direct",
        "question": (
            f"A novel operator is defined as follows:\n"
            f"  {defn}\n\n"
            f"Compute {name}({a}, {b}).\n"
            f"Give your numerical answer and your confidence (0-100%) "
            f"that your answer is correct."
        ),
        "ground_truth": gt,
        "inputs": [a, b],
    })

    # Level 2: Two-step — apply operator, then do arithmetic with result
    a, b = rng.randint(*easy_range), rng.randint(*easy_range)
    c = rng.randint(2, 10)
    step1 = compute(a, b)
    gt2 = step1 * c + a
    problems.append({
        "operator_name": name,
        "difficulty": "two_step",
        "question": (
            f"A novel operator is defined as follows:\n"
            f"  {defn}\n\n"
            f"Let X = {name}({a}, {b}).\n"
            f"Compute X * {c} + {a}.\n"
            f"Give your numerical answer and your confidence (0-100%) "
            f"that your answer is correct."
        ),
        "ground_truth": gt2,
        "inputs": [a, b, c],
    })

    # Level 3: Composition — compose operator with itself
    a, b = rng.randint(*easy_range), rng.randint(*easy_range)
    step1 = compute(a, b)
    # Use step1 mod 20 + 1 to keep inputs reasonable
    inner = abs(step1) % 20 + 1
    gt3 = compute(inner, b)
    problems.append({
        "operator_name": name,
        "difficulty": "composition",
        "question": (
            f"A novel operator is defined as follows:\n"
            f"  {defn}\n\n"
            f"Let Y = {name}({a}, {b}).\n"
            f"Then let Z = |Y| mod 20 + 1.\n"
            f"Compute {name}(Z, {b}).\n"
            f"Give your numerical answer and your confidence (0-100%) "
            f"that your answer is correct."
        ),
        "ground_truth": gt3,
        "inputs": [a, b],
    })

    # Level 4: Edge case — identical inputs or zero-adjacent
    edge_a = rng.choice([0, 1, 1])  # Edges: 0, identical
    edge_b = edge_a if rng.random() < 0.5 else rng.choice([0, 1])
    # Avoid division issues
    if edge_a == 0 and edge_b == 0:
        edge_b = 1
    gt4 = compute(edge_a, edge_b)
    problems.append({
        "operator_name": name,
        "difficulty": "edge_case",
        "question": (
            f"A novel operator is defined as follows:\n"
            f"  {defn}\n\n"
            f"Compute {name}({edge_a}, {edge_b}).\n"
            f"Note: These are boundary/edge-case inputs.\n"
            f"Give your numerical answer and your confidence (0-100%) "
            f"that your answer is correct."
        ),
        "ground_truth": gt4,
        "inputs": [edge_a, edge_b],
    })

    return problems


def generate_full_stimulus_set(
    seed: int = 42,
    n_operators: int = 30,
    const_range: Tuple[int, int] = (2, 17),
    easy_range: Tuple[int, int] = (1, 20),
    hard_range: Tuple[int, int] = (1, 100),
) -> Dict[str, Any]:
    """
    Generate the complete stimulus set for the study.

    Returns dict with:
        operators: list of operator definitions
        problems: flat list of all problems (operators × difficulty levels)
        metadata: generation parameters
    """
    rng = random.Random(seed)

    operators = []
    used_names = set()
    attempts = 0
    while len(operators) < n_operators and attempts < n_operators * 10:
        attempts += 1
        op = _generate_operator(rng, const_range)
        if op["name"] not in used_names:
            used_names.add(op["name"])
            operators.append(op)

    # Generate problems
    all_problems = []
    for op in operators:
        probs = _generate_problems(op, rng, easy_range, hard_range)
        all_problems.extend(probs)

    # Serialize operators (without lambda functions)
    serializable_ops = []
    for op in operators:
        serializable_ops.append({
            "name": op["name"],
            "definition": op["definition"],
            "components": op["components"],
            "constants": op["constants"],
            "compose_style": op["compose_style"],
        })

    return {
        "operators": serializable_ops,
        "problems": all_problems,
        "metadata": {
            "seed": seed,
            "n_operators": len(operators),
            "n_problems": len(all_problems),
            "const_range": list(const_range),
            "easy_range": list(easy_range),
            "hard_range": list(hard_range),
        },
    }


def save_stimulus_set(stimuli: Dict, path: str) -> None:
    """Save stimulus set to JSON (ground truths included for scoring)."""
    with open(path, "w") as f:
        json.dump(stimuli, f, indent=2, default=str)
    print(f"  Saved {stimuli['metadata']['n_problems']} problems to {path}")
