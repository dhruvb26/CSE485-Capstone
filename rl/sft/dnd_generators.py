"""SFT example generators for DealOrNoDeal start-stage tasks."""

from __future__ import annotations

import json
from typing import Any

from rl.handlers.dnd.dataset import agent_input, build_prompt
from rl.sft import dnd_thoughts

_AGENT = "YOU"
_Row = dict[str, Any]


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def sta_total_item_count(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        inp = agent_input(inst, _AGENT)
        total = sum(inp["count"])
        prompt = build_prompt(
            inst, _AGENT,
            "What is the total number of items being negotiated over? "
            "(Count all items of all types combined.)",
            'Inside <answer> put a JSON object with one key: "total_item_count" whose value '
            "is a plain integer. "
            "Do not write math expressions -- evaluate the sum and write the final number.",
        )
        rows.append({
            "task": "sta_total_item_count_dnd",
            "prompt": prompt,
            "answer_json": _json({"total_item_count": total}),
            "det_thought": dnd_thoughts.total_items(inst, _AGENT),
        })
    return rows


def sta_max_points(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        inp = agent_input(inst, _AGENT)
        max_pts = sum(c * v for c, v in zip(inp["count"], inp["value"]))
        prompt = build_prompt(
            inst, _AGENT,
            "What is the maximum number of points you could get if you received ALL "
            "available items?",
            'Inside <answer> put a JSON object with one key: "max_points" whose value is a '
            "plain integer. "
            "Do not write math expressions -- evaluate the total and write the final number.",
        )
        rows.append({
            "task": "sta_max_points_dnd",
            "prompt": prompt,
            "answer_json": _json({"max_points": max_pts}),
            "det_thought": dnd_thoughts.max_points(inst, _AGENT),
        })
    return rows


def sta_point_values(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        inp = agent_input(inst, _AGENT)
        values = inp["value"]
        gt = {"books": str(values[0]), "hats": str(values[1]), "balls": str(values[2])}
        prompt = build_prompt(
            inst, _AGENT,
            "How many points is one item of each type worth to you?",
            'Inside <answer> put a JSON object with keys "books", "hats", "balls" and '
            "values as the point counts (numbers) for each.",
        )
        rows.append({
            "task": "sta_ask_point_values_dnd",
            "prompt": prompt,
            "answer_json": _json(gt),
            "det_thought": dnd_thoughts.point_values(inst, _AGENT),
        })
    return rows


DND_TASK_GENERATORS: dict[str, Any] = {
    "sta_total_item_count_dnd": sta_total_item_count,
    "sta_max_points_dnd": sta_max_points,
    "sta_ask_point_values_dnd": sta_point_values,
}

DND_ALL_TASKS: list[str] = list(DND_TASK_GENERATORS.keys())
