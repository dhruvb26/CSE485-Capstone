from __future__ import annotations

from rl.handlers.casino.dataset import agent_points, get_partner


def total_items(_inst: dict) -> str:
    return "There are 3 food packages + 3 water packages + 3 firewood packages = 9 items total."


def max_points(inst: dict, agent: str) -> str:
    pts = agent_points(inst, agent)
    lines = [f"If I take all 3 {item} I get 3 × {p} = {3*p} pts." for item, p in pts.items()]
    total = sum(3 * p for p in pts.values())
    return " ".join(lines) + f" Total = {total} points."


def point_values(inst: dict, agent: str) -> str:
    pts = agent_points(inst, agent)
    return "My point values are: " + ", ".join(f"{k} = {v}" for k, v in pts.items()) + "."


def priority(inst: dict, agent: str, level: str) -> str:
    pts = agent_points(inst, agent)
    target = inst["participant_info"][agent]["value2issue"][level].lower()
    val = pts[target]
    label = "highest" if level == "High" else "lowest"
    return (
        f"Comparing my values: {', '.join(f'{k}={v}' for k, v in pts.items())}. "
        f"The {label} is {target} at {val} points per package."
    )


def mid_priority(inst: dict, agent: str, level: str) -> str:
    return (
        "Even with the dialogue context, my own point values are fixed and given. "
        + priority(inst, agent, level)
    )


def partner_priority(inst: dict, agent: str, level: str) -> str:
    partner = get_partner(agent)
    item = inst["participant_info"][partner]["value2issue"][level].lower()
    label = "highest" if level == "High" else "lowest"
    return (
        f"Reading the dialogue for signals about the partner's priorities: "
        f"what items they push for, concede on, or mention most urgently. "
        f"Based on these dialogue cues, the partner's {label} priority item is {item}."
    )


def strategy(labels: frozenset[str]) -> str:
    if not labels:
        return "This utterance does not employ a strategic move."
    return f"The utterance employs the following strategies: {', '.join(sorted(labels))}."
