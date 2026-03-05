"""Action types and structured action representation for negotiations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rl.verifiers.allocation import check_allocation
from rl.verifiers.hallucination import check_hallucination


class ActionType(str, Enum):
    OFFER = "offer"
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"
    PROPOSE = "propose"


@dataclass(frozen=True)
class Action:
    """A structured negotiation action.

    For ``offer`` and ``counter``, ``allocations`` maps item names to the
    quantity the agent is requesting for themselves.
    For ``accept`` and ``reject``, ``allocations`` may be empty.
    """
    type: ActionType
    allocations: dict[str, int]

    def validate(self, scenario_items: dict[str, int]) -> tuple[bool, str]:
        """Run allocation and hallucination checks against the scenario."""
        scenario = {"items": scenario_items}
        ok, detail = check_allocation(self.allocations, scenario)
        if not ok:
            return ok, detail
        ok, detail = check_hallucination(self.allocations, scenario)
        return ok, detail

    def to_dict(self) -> dict:
        return {"type": self.type.value, **self.allocations}

    @classmethod
    def from_dict(cls, d: dict) -> Action:
        action_type = ActionType(d["type"])
        allocs = {k: int(v) for k, v in d.items() if k != "type"}
        return cls(type=action_type, allocations=allocs)
