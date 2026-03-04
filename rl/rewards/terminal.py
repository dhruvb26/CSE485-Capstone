"""Terminal joint-payoff reward."""

from __future__ import annotations

from rl.env.scenario import Scenario


def terminal_reward(
    agent_pts: int,
    partner_pts: int,
    scenario: Scenario,
    terminal_lambda: float,
    walkaway_penalty: float,
    deal_reached: bool,
) -> float:
    """Compute the terminal reward for a completed negotiation.

    R = (own_pts / max_own_pts) + lambda * (partner_pts / max_partner_pts)
    Walkaway (no deal) gets a fixed negative penalty instead.
    """
    if not deal_reached:
        return walkaway_penalty

    max_agent = scenario.agent_max_points
    max_partner = scenario.partner_max_points

    own_ratio = agent_pts / max_agent if max_agent > 0 else 0.0
    partner_ratio = partner_pts / max_partner if max_partner > 0 else 0.0

    return own_ratio + terminal_lambda * partner_ratio
