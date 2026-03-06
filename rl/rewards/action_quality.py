"""Action quality reward — scores the structural quality of the chosen action.

Gives GRPO candidate selection a signal about *what action the model chose*,
not just whether the output is well-formatted.  Without this, candidates that
emit a bare ``accept`` on turn 0 score identically to candidates that make
thoughtful offers, collapsing GRPO advantages to zero.
"""

from __future__ import annotations

from rl.env.actions import Action, ActionType
from rl.env.scenario import ITEMS, Scenario


def action_quality_reward(
    action: Action | None,
    scenario: Scenario,
    turn_index: int,
    valid: bool,
) -> float:
    """Return a scalar in roughly [-1, 1] reflecting action quality.

    * ``accept`` on turn 0 (no prior offer exists) -> strong penalty
    * ``reject`` on turn 0 -> moderate penalty
    * ``offer``/``counter`` -> scored by self-allocation AND partner
      fairness (deals that leave the partner with nothing are penalised
      because they will never be accepted)
    * ``accept`` on later turns -> small positive (negotiation concluded)
    """
    if not valid or action is None:
        return 0.0

    if action.type == ActionType.ACCEPT:
        if turn_index == 0:
            return -1.0
        return 0.3

    if action.type == ActionType.REJECT:
        if turn_index == 0:
            return -0.5
        return -0.1

    if action.type in (ActionType.OFFER, ActionType.COUNTER):
        if not action.allocations or all(v == 0 for v in action.allocations.values()):
            return -0.4

        own_pts = sum(
            action.allocations.get(item, 0) * scenario.agent_values[item]
            for item in ITEMS
        )
        partner_pts = sum(
            (scenario.items[item] - action.allocations.get(item, 0))
            * scenario.partner_values[item]
            for item in ITEMS
        )

        max_pts = scenario.agent_max_points
        partner_max = scenario.partner_max_points
        if max_pts <= 0:
            return 0.1

        ratio = own_pts / max_pts
        partner_ratio = partner_pts / partner_max if partner_max > 0 else 0.0

        if ratio > 0.85:
            base = -0.2
        elif ratio < 0.10:
            base = 0.1
        else:
            base = 0.2 + 0.6 * min(ratio / 0.65, 1.0)

        if partner_ratio >= 0.30:
            fairness = 0.2
        elif partner_ratio >= 0.15:
            fairness = 0.1
        else:
            fairness = -0.2

        return base + fairness

    return 0.0
