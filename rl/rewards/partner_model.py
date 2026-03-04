"""Partner model accuracy reward: compares the thought field's estimate of
partner priority against the ground-truth scenario values."""

from __future__ import annotations

import re

from rl.env.scenario import ITEMS, Scenario


def partner_model_reward(thought: str, scenario: Scenario) -> float:
    """Return 1.0 if the thought field correctly identifies the partner's
    highest-priority item, 0.0 otherwise.

    Looks for phrases like "partner's highest priority item is <item>" or
    "partner priority estimate: <item>" in the thought text.
    """
    partner_vals = scenario.partner_values
    true_highest = max(ITEMS, key=lambda i: partner_vals[i])

    lower = thought.lower()
    for item in ITEMS:
        patterns = [
            rf"partner.*(?:highest|most|top).*{item}",
            rf"{item}.*partner.*(?:highest|most|top)",
            rf"partner.*priority.*{item}",
        ]
        for pat in patterns:
            if re.search(pat, lower):
                return 1.0 if item == true_highest else 0.0

    return 0.0
