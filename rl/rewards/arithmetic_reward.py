"""Arithmetic correctness reward -- thin wrapper over verifiers.arithmetic."""

from __future__ import annotations

from rl.verifiers.arithmetic import check_arithmetic


def arithmetic_reward(thought: str, values: dict[str, int]) -> float:
    """Return 1.0 if all arithmetic in the thought field is correct, 0.0 otherwise."""
    ok, _ = check_arithmetic(thought, values)
    return 1.0 if ok else 0.0
