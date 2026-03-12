"""Dataset preparation for GRPO negotiation training.

Handles loading CaSiNo scenarios and converting self-play episodes into
per-turn HF Datasets that GRPOTrainer can consume.
"""

from __future__ import annotations

import json
import logging
import re

from datasets import Dataset

from rl.grpo.rollout import Episode
from rl.sft.data import POINTS, load_all_conversations

log = logging.getLogger(__name__)

_SUBMIT_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)", re.IGNORECASE
)
MAX_POINTS = sum(3 * p for p in POINTS.values())  # 3*5 + 3*4 + 3*3 = 36


def load_scenarios(
    csv_path: str,
    max_instances: int | None = None,
) -> list[dict]:
    """Load CaSiNo scenarios from CSV.

    Each returned dict has ``participant_info`` and ``agent_ids`` --
    the ``chat_logs`` from the CSV are not needed for self-play.
    """
    all_convos = load_all_conversations(csv_path)
    if max_instances is not None:
        all_convos = all_convos[:max_instances]

    scenarios: list[dict] = []
    for _row_idx, _chat_logs, participant_info, agent_ids in all_convos:
        scenarios.append(
            {"participant_info": participant_info, "agent_ids": agent_ids}
        )

    log.warning("Loaded %d scenarios from %s", len(scenarios), csv_path)
    return scenarios


def _agent_point_map(participant_info: dict, agent_id: str) -> dict[str, int]:
    """Return ``{"food": pts, "water": pts, "firewood": pts}`` for an agent."""
    v2i = participant_info[agent_id]["value2issue"]
    result: dict[str, int] = {}
    for level in ("High", "Medium", "Low"):
        result[v2i[level].lower()] = POINTS[level]
    return result


def _find_last_opponent_offer(messages: list[dict]) -> dict[str, int] | None:
    """Scan user messages (opponent turns) for the most recent SUBMIT_DEAL."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        m = _SUBMIT_RE.search(content)
        if m:
            return {
                "food": int(m.group(1)),
                "water": int(m.group(2)),
                "firewood": int(m.group(3)),
            }
    return None


def episodes_to_dataset(episodes: list[Episode]) -> Dataset:
    """Convert self-play episodes into a per-turn HF Dataset.

    For each learner turn in each episode, emits one row with the
    conversation prefix as the ``prompt`` and flat metadata columns
    for the reward functions.
    """
    rows: list[dict] = []

    for ep in episodes:
        pts = _agent_point_map(ep.participant_info, ep.learner_agent_id)

        for turn_idx in ep.learner_turns:
            prompt = ep.learner_messages[:turn_idx]
            if not prompt or prompt[-1]["role"] == "system":
                continue

            opp_offer = _find_last_opponent_offer(prompt)

            rows.append(
                {
                    "prompt": prompt,
                    "food_points": pts.get("food", 3),
                    "water_points": pts.get("water", 3),
                    "firewood_points": pts.get("firewood", 3),
                    "max_points": MAX_POINTS,
                    "last_opponent_offer": json.dumps(opp_offer) if opp_offer else "null",
                    "episode_outcome": ep.outcome,
                    "episode_learner_points": ep.learner_points if ep.learner_points is not None else -1,
                    "opponent_persona": ep.persona,
                }
            )

    if not rows:
        raise ValueError("No training samples produced from episodes")

    log.warning(
        "Built GRPO dataset: %d per-turn samples from %d episodes",
        len(rows),
        len(episodes),
    )
    return Dataset.from_list(rows)
