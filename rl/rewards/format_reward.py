"""Format validity reward -- thin wrapper over verifiers.format."""

from __future__ import annotations

from rl.verifiers.format import check_format


def format_reward(raw_text: str) -> float:
    """Return 1.0 if the output is valid format, 0.0 otherwise."""
    ok, _ = check_format(raw_text)
    return 1.0 if ok else 0.0
