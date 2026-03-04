"""No-hallucination verifier: checks that the action field only references
items that exist in the negotiation scenario."""

from __future__ import annotations


def check_hallucination(
    action: dict,
    scenario: dict,
) -> tuple[bool, str]:
    """Verify that every item key in *action* (excluding ``type``) is a
    valid item in the scenario.

    *scenario* must contain ``items`` (a dict mapping item name to total count).

    Returns ``(valid, detail)``.
    """
    valid_items = set(scenario["items"].keys())
    action_items = {k for k in action if k != "type"}

    unknown = action_items - valid_items
    if unknown:
        return False, f"Action references items not in scenario: {sorted(unknown)}"

    return True, "ok"
