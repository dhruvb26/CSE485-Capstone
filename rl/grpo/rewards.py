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


def format_reward(completions, **kwargs) -> list[float]:
    """Per-turn format validity reward.

    +1.0  valid single action with non-empty thought
    -1.0  otherwise (missing/unparseable tags, multiple actions, bad values)
    """
    rewards: list[float] = []
    for comp in completions:
        text = _get_text(comp)
        action_str = _tag(text, "action")
        thought_str = _tag(text, "thought")

        if action_str is None or not thought_str:
            rewards.append(-1.0)
            continue

        found = [a for a in _VALID_ACTIONS if a in action_str]
        if len(found) != 1:
            rewards.append(-1.0)
            continue
        base = found[0]

        if base == "[ACCEPT_DEAL]" and action_str.replace("[ACCEPT_DEAL]", "").strip():
            rewards.append(-1.0)
            continue

        if base == "[SUBMIT_DEAL]":
            m = _SUBMIT_RE.search(action_str)
            if m is None or any(not (0 <= int(v) <= 3) for v in m.groups()):
                rewards.append(-1.0)
                continue

        rewards.append(1.0)
    return rewards


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
    food_pts_list = food_points or []
    water_pts_list = water_points or []
    fw_pts_list = firewood_points or []
    max_pts_list = max_points or []
    opp_offer_list = last_opponent_offer or []

    for i, comp in enumerate(completions):
        text = _get_text(comp)
        action_str = _tag(text, "action")

        fp = food_pts_list[i] if i < len(food_pts_list) else 5
        wp = water_pts_list[i] if i < len(water_pts_list) else 4
        fwp = fw_pts_list[i] if i < len(fw_pts_list) else 3
        mp = max_pts_list[i] if i < len(max_pts_list) else 36
        opp_raw = opp_offer_list[i] if i < len(opp_offer_list) else "null"

        if action_str is None:
            rewards.append(0.0)
            continue

        found = [a for a in _VALID_ACTIONS if a in action_str]
        base = found[0] if len(found) == 1 else None

        if base == "[SUBMIT_DEAL]":
            m = _SUBMIT_RE.search(action_str)
            if m is not None and mp > 0:
                pts = int(m.group(1)) * fp + int(m.group(2)) * wp + int(m.group(3)) * fwp
                rewards.append(pts / mp - 0.5)
            else:
                rewards.append(0.0)
            continue

        if base == "[ACCEPT_DEAL]":
            if action_str.replace("[ACCEPT_DEAL]", "").strip():
                rewards.append(-0.5)
                continue
            opp_alloc = None
            if opp_raw and opp_raw != "null":
                try:
                    obj = json.loads(opp_raw)
                    if isinstance(obj, dict):
                        opp_alloc = {k: int(v) for k, v in obj.items()}
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            if opp_alloc is not None and mp > 0:
                pts = (3 - opp_alloc.get("food", 0)) * fp \
                    + (3 - opp_alloc.get("water", 0)) * wp \
                    + (3 - opp_alloc.get("firewood", 0)) * fwp
                rewards.append(pts / mp - 0.5)
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
        ``0.3 * (episode_learner_points / max_pts - 0.5)`` so all turns in
        good episodes get positive signal, bad episodes get negative.
    """
    rewards: list[float] = []
    food_pts_list = food_points or []
    water_pts_list = water_points or []
    fw_pts_list = firewood_points or []
    max_pts_list = max_points or []
    opp_offer_list = last_opponent_offer or []
    ep_pts_list = episode_learner_points or []
    prompts_list = prompts or []

    for i, comp in enumerate(completions):
        text = _get_text(comp)
        action_str = _tag(text, "action")

        fp = food_pts_list[i] if i < len(food_pts_list) else 5
        wp = water_pts_list[i] if i < len(water_pts_list) else 4
        fwp = fw_pts_list[i] if i < len(fw_pts_list) else 3
        mp = max_pts_list[i] if i < len(max_pts_list) else 36
        opp_raw = opp_offer_list[i] if i < len(opp_offer_list) else "null"
        ep_pts = ep_pts_list[i] if i < len(ep_pts_list) else -1

        baseline = 0.3 * (ep_pts / mp - 0.5) if ep_pts > 0 and mp > 0 else 0.0

        if action_str is None:
            rewards.append(baseline)
            continue

        found = [a for a in _VALID_ACTIONS if a in action_str]
        base = found[0] if len(found) == 1 else None

        if base == "[ACCEPT_DEAL]":
            if action_str.replace("[ACCEPT_DEAL]", "").strip():
                rewards.append(-0.5 + baseline)
                continue
            opp_alloc = None
            if opp_raw and opp_raw != "null":
                try:
                    obj = json.loads(opp_raw)
                    if isinstance(obj, dict):
                        opp_alloc = {k: int(v) for k, v in obj.items()}
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            if opp_alloc is not None and mp > 0:
                pts = (3 - opp_alloc.get("food", 0)) * fp \
                    + (3 - opp_alloc.get("water", 0)) * wp \
                    + (3 - opp_alloc.get("firewood", 0)) * fwp
                rewards.append(pts / mp + baseline)
            else:
                rewards.append(baseline)
            continue

        if base == "[SUBMIT_DEAL]":
            m = _SUBMIT_RE.search(action_str)
            if m is not None and mp > 0:
                pts = int(m.group(1)) * fp + int(m.group(2)) * wp + int(m.group(3)) * fwp
                rewards.append(pts / mp + baseline)
            else:
                rewards.append(baseline)
            continue

        if base == "[WALK_AWAY]":
            rewards.append(-0.25 + baseline)
            continue

        if base == "[REJECT_DEAL]":
            # 3+ consecutive identical SUBMIT_DEAL in prompt history = degenerate loop
            if i < len(prompts_list) and isinstance(prompts_list[i], list):
                deals: list[str] = []
                for msg in reversed(prompts_list[i]):
                    if isinstance(msg, dict):
                        act = _tag(msg.get("content", ""), "action")
                        if act and "[SUBMIT_DEAL]" in act:
                            deals.append(act.strip())
                    if len(deals) >= 3:
                        break
                if len(deals) >= 3 and len(set(deals)) == 1:
                    rewards.append(-0.5 + baseline)
                    continue

        rewards.append(baseline)
    return rewards
