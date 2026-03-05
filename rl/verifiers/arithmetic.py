"""Arithmetic correctness verifier: checks that point calculations in
the thought field are numerically correct."""

from __future__ import annotations

import re

_CALC_RE = re.compile(
    r"(\d+)\s*[x×*]\s*(\d+)\s*=\s*(\d+)"
)


def check_arithmetic(
    thought: str,
    values: dict[str, int],
) -> tuple[bool, str]:
    """Scan *thought* for ``qty x pts = subtotal`` expressions and verify
    each one is correct.

    Two checks are applied:
    1. Multiplication correctness: ``a * b == c`` for every ``a x b = c``.
    2. Scenario-value consistency: each per-unit factor ``b`` must appear in
       ``values.values()``. This ensures the model is using the actual
       scenario point values rather than made-up numbers.

    Returns ``(valid, detail)``.
    """
    calcs = _CALC_RE.findall(thought)
    if not calcs:
        return False, "No arithmetic expressions found (at least one required)."

    known_pts = set(values.values())
    errors = []
    for a_str, b_str, c_str in calcs:
        a, b, c = int(a_str), int(b_str), int(c_str)
        if a * b != c:
            errors.append(f"{a} x {b} = {c} (expected {a * b})")
        elif known_pts and b not in known_pts:
            errors.append(f"{a} x {b} = {c} (pts/unit {b} not in scenario values {sorted(known_pts)})")

    if errors:
        return False, f"Arithmetic errors: {'; '.join(errors)}"

    return True, "ok"
