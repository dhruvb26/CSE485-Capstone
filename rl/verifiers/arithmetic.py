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

    *values* maps item names to their per-unit point value (used for
    contextual validation but not strictly required if the thought
    contains self-consistent arithmetic).

    Returns ``(valid, detail)``.
    """
    calcs = _CALC_RE.findall(thought)
    if not calcs:
        return True, "No arithmetic expressions found to verify."

    errors = []
    for a_str, b_str, c_str in calcs:
        a, b, c = int(a_str), int(b_str), int(c_str)
        if a * b != c:
            errors.append(f"{a} x {b} = {c} (expected {a * b})")

    if errors:
        return False, f"Arithmetic errors: {'; '.join(errors)}"

    return True, "ok"
