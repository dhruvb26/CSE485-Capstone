"""Deterministic thought templates for DealOrNoDeal SFT tasks."""

from __future__ import annotations

from rl.handlers.dnd.dataset import agent_input

_ITEMS = ["books", "hats", "balls"]


def total_items(inst: dict, agent: str) -> str:
    inp = agent_input(inst, agent)
    counts = inp["count"]
    parts = [f"{counts[i]} {_ITEMS[i]}" for i in range(3)]
    return f"There are {' + '.join(parts)} = {sum(counts)} items total."


def max_points(inst: dict, agent: str) -> str:
    inp = agent_input(inst, agent)
    counts = inp["count"]
    values = inp["value"]
    parts = []
    for i in range(3):
        sub = counts[i] * values[i]
        parts.append(f"If I take all {counts[i]} {_ITEMS[i]} I get {counts[i]} x {values[i]} = {sub} pts.")
    total = sum(counts[i] * values[i] for i in range(3))
    return " ".join(parts) + f" Total = {total} points."


def point_values(inst: dict, agent: str) -> str:
    inp = agent_input(inst, agent)
    values = inp["value"]
    return "My point values are: " + ", ".join(
        f"{_ITEMS[i]} = {values[i]}" for i in range(3)
    ) + "."
