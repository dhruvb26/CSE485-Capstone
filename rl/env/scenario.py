"""CaSiNo-style scenario sampling for self-play training."""

from __future__ import annotations

import random
from dataclasses import dataclass

ITEMS = ("food", "water", "firewood")
UNITS_PER_ITEM = 3
POINT_VALUES = (3, 4, 5)


@dataclass(frozen=True)
class Scenario:
    """A negotiation scenario with per-agent point values.

    ``items`` maps each item name to the total available count (always 3).
    ``agent_values`` and ``partner_values`` map each item to its per-unit
    point value for the respective agent.
    """
    items: dict[str, int]
    agent_values: dict[str, int]
    partner_values: dict[str, int]

    @property
    def agent_max_points(self) -> int:
        return sum(self.items[k] * self.agent_values[k] for k in ITEMS)

    @property
    def partner_max_points(self) -> int:
        return sum(self.items[k] * self.partner_values[k] for k in ITEMS)


def sample_scenario(rng: random.Random) -> Scenario:
    """Sample a random CaSiNo-style scenario.

    Each agent gets a random permutation of (3, 4, 5) point values
    across the three items, ensuring asymmetric priorities so there
    is always a non-zero zone of possible agreement.
    """
    items = {item: UNITS_PER_ITEM for item in ITEMS}

    agent_perm = list(POINT_VALUES)
    rng.shuffle(agent_perm)

    partner_perm = list(POINT_VALUES)
    rng.shuffle(partner_perm)

    # Re-roll if both agents have the same highest-priority item
    while agent_perm.index(5) == partner_perm.index(5):
        rng.shuffle(partner_perm)

    agent_values = dict(zip(ITEMS, agent_perm))
    partner_values = dict(zip(ITEMS, partner_perm))

    return Scenario(
        items=items,
        agent_values=agent_values,
        partner_values=partner_values,
    )
