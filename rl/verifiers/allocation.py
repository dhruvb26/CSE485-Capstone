"""Allocation validity verifier: checks that agent + partner allocations
do not exceed the available quantity for each item."""

from __future__ import annotations

_CASINO_ITEMS = ("food", "water", "firewood")


def check_allocation(
    action: dict,
    scenario: dict,
) -> tuple[bool, str]:
    """Verify that per-item allocations are non-negative and do not exceed
    the total available count.

    *action* is the parsed ``action`` field (e.g. ``{"type": "offer", "food": 2, ...}``).
    *scenario* must contain ``items`` (a dict mapping item name to total count).

    Returns ``(valid, detail)``.
    """
    items: dict[str, int] = scenario["items"]

    for item, total in items.items():
        allocated = action.get(item, 0)
        if not isinstance(allocated, (int, float)):
            return False, f"Allocation for '{item}' is not a number: {allocated}"
        allocated = int(allocated)
        if allocated < 0:
            return False, f"Negative allocation for '{item}': {allocated}"
        if allocated > total:
            return False, f"Allocation for '{item}' ({allocated}) exceeds available ({total})"

    return True, "ok"
