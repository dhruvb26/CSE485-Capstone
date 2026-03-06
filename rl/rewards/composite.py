"""Composite multi-source reward with decay schedule."""

from __future__ import annotations

from rl.config import RewardConfig
from rl.env.negotiation import NegotiationEnv, Turn
from rl.env.scenario import Scenario
from rl.rewards.action_quality import action_quality_reward
from rl.rewards.arithmetic_reward import arithmetic_reward
from rl.rewards.format_reward import format_reward
from rl.rewards.partner_model import partner_model_reward
from rl.rewards.terminal import terminal_reward


class CompositeReward:
    """Weighted multi-source reward model with linear decay for the format
    component.

    All weights and thresholds are read from ``RewardConfig`` (YAML).
    """

    def __init__(
        self,
        config: RewardConfig,
        terminal_lambda: float,
        walkaway_penalty: float,
    ):
        self.config = config
        self.terminal_lambda = terminal_lambda
        self.walkaway_penalty = walkaway_penalty

    def _decayed_weight(self, base_weight: float, episode: int) -> float:
        """Linearly decay *base_weight* to 0 over the decay window."""
        if self.config.decay_window <= 0:
            return 0.0
        progress = min(episode / self.config.decay_window, 1.0)
        return base_weight * (1.0 - progress)

    def score_episode(
        self,
        env: NegotiationEnv,
        episode: int,
    ) -> float:
        """Score a completed negotiation episode."""
        r_terminal = terminal_reward(
            agent_pts=env.agent_points(),
            partner_pts=env.partner_points(),
            scenario=env.scenario,
            terminal_lambda=self.terminal_lambda,
            walkaway_penalty=self.walkaway_penalty,
            deal_reached=env.deal_reached,
        )

        learner_turns = [t for t in env.history if t.agent == "learner"]
        if not learner_turns:
            return self.config.terminal_weight * r_terminal

        fmt_scores = [format_reward(t.raw_output) for t in learner_turns]
        r_format = sum(fmt_scores) / len(fmt_scores)

        arith_scores = [
            arithmetic_reward(t.thought, env.scenario.agent_values)
            for t in learner_turns
        ]
        r_arithmetic = sum(arith_scores) / len(arith_scores)

        pm_scores = [
            partner_model_reward(t.thought, env.scenario)
            for t in learner_turns
        ]
        r_partner = sum(pm_scores) / len(pm_scores)

        aq_scores = [
            action_quality_reward(t.action, env.scenario, i, t.valid)
            for i, t in enumerate(learner_turns)
        ]
        r_action = sum(aq_scores) / len(aq_scores)

        w_terminal = self.config.terminal_weight
        w_format = self._decayed_weight(self.config.format_weight, episode)
        w_arithmetic = self.config.arithmetic_weight
        w_partner = self.config.partner_model_weight
        w_action = self.config.action_quality_weight

        total = (
            w_terminal * r_terminal
            + w_format * r_format
            + w_arithmetic * r_arithmetic
            + w_partner * r_partner
            + w_action * r_action
        )
        return total

    def score_turn(
        self,
        turn: Turn,
        turn_index: int,
        scenario: Scenario,
        episode: int,
    ) -> float:
        """Score a single learner turn (used for per-candidate scoring in GRPO)."""
        r_format = format_reward(turn.raw_output)
        r_arithmetic = arithmetic_reward(turn.thought, scenario.agent_values)
        r_partner = partner_model_reward(turn.thought, scenario)
        r_action = action_quality_reward(
            turn.action, scenario, turn_index, turn.valid,
        )

        w_format = self._decayed_weight(self.config.format_weight, episode)
        w_arithmetic = self.config.arithmetic_weight
        w_partner = self.config.partner_model_weight
        w_action = self.config.action_quality_weight

        return (
            w_format * r_format
            + w_arithmetic * r_arithmetic
            + w_partner * r_partner
            + w_action * r_action
        )
