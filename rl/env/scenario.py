"""CaSiNo-style scenario sampling for self-play training."""

from __future__ import annotations

import random
from dataclasses import dataclass

ITEMS = ("food", "water", "firewood")
UNITS_PER_ITEM = 3
# Mirrors the CaSiNo low/medium/high priority point mapping.
POINT_VALUES = (3, 4, 5)


@dataclass(frozen=True)
class Scenario:
    """A negotiation scenario with per-agent point values.

    ``items`` maps each item name to the total available count (always 3).
    ``agent_values`` and ``partner_values`` map each item to its per-unit
    point value for the respective agent.

    Point values are a permutation of (3, 4, 5) — one per item — matching
    the CaSiNo low/medium/high priority encoding.
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

    @property
    def max_joint_points(self) -> int:
        """Upper bound on combined points if both agents received all items."""
        return self.agent_max_points + self.partner_max_points

    @property
    def is_competitive(self) -> bool:
        """True when both agents place highest value on the same item.

        These scenarios require more concession-making since the top item
        cannot be fully won by either side.
        """
        agent_top = max(self.agent_values, key=self.agent_values.__getitem__)
        partner_top = max(self.partner_values, key=self.partner_values.__getitem__)
        return agent_top == partner_top

    def describe(self) -> str:
        """Human-readable one-liner for logging."""
        av = ", ".join(f"{k}={v}" for k, v in self.agent_values.items())
        pv = ", ".join(f"{k}={v}" for k, v in self.partner_values.items())
        tag = " [competitive]" if self.is_competitive else ""
        return f"agent({av}) | partner({pv}){tag}"


def sample_scenario(rng: random.Random) -> Scenario:
    """Sample a random CaSiNo-style scenario.

    Both agents independently receive a random permutation of (3, 4, 5)
    point values across the three items.  All 36 priority combinations are
    reachable, including the 6 competitive cases where both agents place
    highest value on the same item.  Those cases are harder (require
    genuine concessions) but are present in the real CaSiNo dataset and
    important for robust training.
    """
    items = {item: UNITS_PER_ITEM for item in ITEMS}

    agent_perm = list(POINT_VALUES)
    rng.shuffle(agent_perm)

    partner_perm = list(POINT_VALUES)
    rng.shuffle(partner_perm)

    agent_values = dict(zip(ITEMS, agent_perm))
    partner_values = dict(zip(ITEMS, partner_perm))

    return Scenario(
        items=items,
        agent_values=agent_values,
        partner_values=partner_values,
    )
