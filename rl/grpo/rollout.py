"""Dialogue rollouts for final-turn lookahead (Phase 3).

At turns T-2 and T-1:
    1. Generate K candidate responses from the learner
    2. For each candidate, run the clone forward until dialogue concludes
    3. Score each complete trajectory with the terminal reward
    4. Select the candidate with highest expected terminal reward
"""

from __future__ import annotations

import logging

from rl.env.negotiation import NegotiationEnv
from rl.rewards.terminal import terminal_reward

logger = logging.getLogger(__name__)


def rollout_select(
    env: NegotiationEnv,
    candidates: list[str],
    clone_generate_fn,
    terminal_lambda: float,
    walkaway_penalty: float,
    max_rollout_turns: int,
) -> int:
    """Evaluate each candidate by rolling out the rest of the dialogue with
    the clone, then return the index of the best candidate.

    ``clone_generate_fn(prompt: str) -> str`` generates a single clone
    response given a prompt string.
    """
    if not candidates:
        raise ValueError("No candidates to evaluate.")

    best_idx = 0
    best_score = float("-inf")

    for i, candidate in enumerate(candidates):
        score = _simulate(
            env, candidate, clone_generate_fn,
            terminal_lambda, walkaway_penalty, max_rollout_turns,
        )
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _simulate(
    env: NegotiationEnv,
    candidate: str,
    clone_generate_fn,
    terminal_lambda: float,
    walkaway_penalty: float,
    max_rollout_turns: int,
) -> float:
    """Simulate a dialogue from the current env state, starting with
    *candidate* as the learner's next utterance, then alternating with
    the clone until the episode ends.

    Returns the terminal reward for the simulated trajectory.
    """
    sim = NegotiationEnv(
        scenario=env.scenario,
        persona=env.persona,
        max_turns=env.max_turns,
    )
    sim.history = list(env.history)
    sim._done = env._done
    sim._deal_reached = env._deal_reached
    sim._agent_deal_action = env._agent_deal_action
    sim._partner_deal_action = env._partner_deal_action

    sim.step(candidate)

    rollout_steps = 0
    while not sim.is_done and rollout_steps < max_rollout_turns:
        if sim.is_learner_turn:
            prompt = sim.build_learner_prompt()
            output = clone_generate_fn(prompt)
        else:
            prompt = sim.build_clone_prompt()
            output = clone_generate_fn(prompt)
        sim.step(output)
        rollout_steps += 1

    return terminal_reward(
        agent_pts=sim.agent_points(),
        partner_pts=sim.partner_points(),
        scenario=sim.scenario,
        terminal_lambda=terminal_lambda,
        walkaway_penalty=walkaway_penalty,
        deal_reached=sim.deal_reached,
    )
