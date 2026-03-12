"""Reward functions for GRPO negotiation training.

Three reward functions following TRL's custom reward function contract:
each receives ``completions`` plus dataset columns as ``**kwargs`` and
returns a ``list[float]``.
"""

from __future__ import annotations

import json
import re

_SUBMIT_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)", re.IGNORECASE
)
_VALID_ACTIONS = {"[TALK]", "[SUBMIT_DEAL]", "[ACCEPT_DEAL]", "[REJECT_DEAL]", "[WALK_AWAY]"}

def _get_text(completion) -> str:
    """Extract plain text from a completion (string or conversational)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        if completion:
            last = completion[-1]
            return last.get("content", "") if isinstance(last, dict) else str(last)
    return str(completion)


def _tag(text: str, tag_name: str) -> str | None:
    """Extract content between ``<tag>...</tag>``."""
    open_t = f"<{tag_name}>"
    close_t = f"</{tag_name}>"
    start = text.find(open_t)
    if start == -1:
        return None
    start += len(open_t)
    end = text.find(close_t, start)
    if end == -1:
        return None
    return text[start:end].strip()


def _parse_action(text: str) -> str | None:
    return _tag(text, "action")


def _parse_thought(text: str) -> str | None:
    return _tag(text, "thought")


def _parse_submit_deal(action_str: str) -> dict[str, int] | None:
    m = _SUBMIT_RE.search(action_str)
    if m is None:
        return None
    return {
        "food": int(m.group(1)),
        "water": int(m.group(2)),
        "firewood": int(m.group(3)),
    }


def _compute_points(
    alloc: dict[str, int],
    food_pts: int,
    water_pts: int,
    firewood_pts: int,
) -> int:
    return (
        alloc.get("food", 0) * food_pts
        + alloc.get("water", 0) * water_pts
        + alloc.get("firewood", 0) * firewood_pts
    )


def _load_opponent_offer(raw: str) -> dict[str, int] | None:
    """Deserialize the ``last_opponent_offer`` dataset column."""
    if not raw or raw == "null":
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {k: int(v) for k, v in obj.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _action_base(action_str: str) -> str:
    """Return just the action keyword, e.g. ``'[SUBMIT_DEAL]'``."""
    for a in _VALID_ACTIONS:
        if a in action_str:
            return a
    return action_str.strip()

def format_reward(completions, **kwargs) -> list[float]:
    """Per-turn format validity reward.

    +1.0  valid, clean action with thought
    -1.0  action tag missing
    -1.0  ACCEPT_DEAL contaminated with extra deal content
    -1.0  thought tag missing or empty
    -1.0  SUBMIT_DEAL item value outside [0, 3]
    """
    rewards: list[float] = []
    for comp in completions:
        text = _get_text(comp)
        r = 0.0

        action_str = _parse_action(text)
        thought_str = _parse_thought(text)

        if action_str is None:
            rewards.append(-1.0)
            continue

        base = _action_base(action_str)
        if base not in _VALID_ACTIONS:
            rewards.append(-1.0)
            continue

        r += 1.0

        if thought_str is None or len(thought_str) == 0:
            r -= 2.0

        if base == "[ACCEPT_DEAL]" and len(action_str.replace("[ACCEPT_DEAL]", "").strip()) > 0:
            r = -1.0
            rewards.append(max(r, -1.0))
            continue

        if base == "[SUBMIT_DEAL]":
            deal = _parse_submit_deal(action_str)
            if deal is None:
                r = -1.0
            else:
                for v in deal.values():
                    if v < 0 or v > 3:
                        r = -1.0
                        break

        rewards.append(max(min(r, 1.0), -1.0))
    return rewards

def _invert_alloc(alloc: dict[str, int]) -> dict[str, int]:
    """Given one side's allocation, return the other side's (3 - qty each)."""
    return {k: 3 - v for k, v in alloc.items()}


def offer_reward(
    completions,
    food_points=None,
    water_points=None,
    firewood_points=None,
    max_points=None,
    last_opponent_offer=None,
    **kwargs,
) -> list[float]:
    """Mid-episode offer quality reward (continuous).

    SUBMIT_DEAL  -> (own_pct - 0.5), ranging roughly -0.5 to +0.5
    ACCEPT_DEAL  -> (own_pct - 0.5) based on what the learner actually gets
    Others       -> 0.0
    """
    rewards: list[float] = []

    food_pts_list = food_points if food_points is not None else []
    water_pts_list = water_points if water_points is not None else []
    fw_pts_list = firewood_points if firewood_points is not None else []
    max_pts_list = max_points if max_points is not None else []
    opp_offer_list = last_opponent_offer if last_opponent_offer is not None else []

    for i, comp in enumerate(completions):
        text = _get_text(comp)
        action_str = _parse_action(text)

        fp = food_pts_list[i] if i < len(food_pts_list) else 5
        wp = water_pts_list[i] if i < len(water_pts_list) else 4
        fwp = fw_pts_list[i] if i < len(fw_pts_list) else 3
        mp = max_pts_list[i] if i < len(max_pts_list) else 36
        opp_raw = opp_offer_list[i] if i < len(opp_offer_list) else "null"

        if action_str is None:
            rewards.append(0.0)
            continue

        base = _action_base(action_str)

        if base == "[SUBMIT_DEAL]":
            deal = _parse_submit_deal(action_str)
            if deal is not None and mp > 0:
                own_pct = _compute_points(deal, fp, wp, fwp) / mp
                rewards.append(own_pct - 0.5)
            else:
                rewards.append(0.0)
            continue

        if base == "[ACCEPT_DEAL]":
            if len(action_str.replace("[ACCEPT_DEAL]", "").strip()) > 0:
                rewards.append(-0.5)
                continue
            opp_alloc = _load_opponent_offer(opp_raw)
            if opp_alloc is not None and mp > 0:
                learner_alloc = _invert_alloc(opp_alloc)
                own_pct = _compute_points(learner_alloc, fp, wp, fwp) / mp
                rewards.append(own_pct - 0.5)
            else:
                rewards.append(0.0)
            continue

        rewards.append(0.0)
    return rewards

def terminal_reward(
    completions,
    food_points=None,
    water_points=None,
    firewood_points=None,
    max_points=None,
    last_opponent_offer=None,
    episode_learner_points=None,
    prompts=None,
    **kwargs,
) -> list[float]:
    """Terminal episode reward.

    Action-specific scores (applied on top of the episode baseline):
        ACCEPT_DEAL  -> learner_pct  (normalized 0-1, using inverted allocation)
        SUBMIT_DEAL  -> own_pct      (what the learner would get if accepted)
        WALK_AWAY    -> -0.25
        reject_loop  -> -0.5
        Others       -> 0.0

    Episode baseline (applied to every turn):
        If ``episode_learner_points`` is available, adds a small bonus/penalty
        ``0.3 * (ep_pts / max_pts - 0.5)`` so all turns in good episodes get
        positive signal and all turns in bad episodes get negative signal.
    """
    rewards: list[float] = []

    food_pts_list = food_points if food_points is not None else []
    water_pts_list = water_points if water_points is not None else []
    fw_pts_list = firewood_points if firewood_points is not None else []
    max_pts_list = max_points if max_points is not None else []
    opp_offer_list = last_opponent_offer if last_opponent_offer is not None else []
    ep_pts_list = episode_learner_points if episode_learner_points is not None else []
    prompts_list = prompts if prompts is not None else []

    for i, comp in enumerate(completions):
        text = _get_text(comp)
        action_str = _parse_action(text)

        fp = food_pts_list[i] if i < len(food_pts_list) else 5
        wp = water_pts_list[i] if i < len(water_pts_list) else 4
        fwp = fw_pts_list[i] if i < len(fw_pts_list) else 3
        mp = max_pts_list[i] if i < len(max_pts_list) else 36
        opp_raw = opp_offer_list[i] if i < len(opp_offer_list) else "null"
        ep_pts = ep_pts_list[i] if i < len(ep_pts_list) else -1

        baseline = 0.0
        if ep_pts > 0 and mp > 0:
            baseline = 0.3 * (ep_pts / mp - 0.5)

        if action_str is None:
            rewards.append(baseline)
            continue

        base = _action_base(action_str)

        if base == "[ACCEPT_DEAL]":
            if len(action_str.replace("[ACCEPT_DEAL]", "").strip()) > 0:
                rewards.append(-0.5 + baseline)
                continue
            opp_alloc = _load_opponent_offer(opp_raw)
            if opp_alloc is not None and mp > 0:
                learner_alloc = _invert_alloc(opp_alloc)
                rewards.append(_compute_points(learner_alloc, fp, wp, fwp) / mp + baseline)
            else:
                rewards.append(baseline)
            continue

        if base == "[SUBMIT_DEAL]":
            deal = _parse_submit_deal(action_str)
            if deal is not None and mp > 0:
                own_pct = _compute_points(deal, fp, wp, fwp) / mp
                rewards.append(own_pct + baseline)
            else:
                rewards.append(baseline)
            continue

        if base == "[WALK_AWAY]":
            rewards.append(-0.25 + baseline)
            continue

        if base == "[REJECT_DEAL]":
            if _detect_reject_loop(prompts_list, i):
                rewards.append(-0.5 + baseline)
                continue

        rewards.append(baseline)
    return rewards


def _detect_reject_loop(prompts_list, idx: int, window: int = 3) -> bool:
    """Scan prompt messages for 3+ consecutive identical SUBMIT_DEAL actions."""
    if idx >= len(prompts_list):
        return False
    prompt = prompts_list[idx]
    if not isinstance(prompt, list):
        return False
    deals: list[str] = []
    for msg in reversed(prompt):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        action = _tag(content, "action")
        if action and "[SUBMIT_DEAL]" in action:
            deals.append(action.strip())
        if len(deals) >= window:
            break
    if len(deals) < window:
        return False
    return len(set(deals)) == 1
