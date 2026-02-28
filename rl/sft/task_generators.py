from __future__ import annotations

import json
from typing import Any

from rl.handlers.casino.dataset import (
    STRATEGY_LABEL_MAP,
    STRATEGY_LABELS,
    _META_TURNS,
    agent_points,
    build_mid_prompt,
    build_prompt,
    get_partner,
    sanitize_unicode,
)
from rl.handlers.casino.tasks import (
    _mid_dialogue_cut,
    _PRIORITY_OUTPUT_SPEC,
    _STRATEGY_OUTPUT_SPEC,
)
from rl.sft import thoughts

_AGENT = "mturk_agent_1"

_Row = dict[str, Any]


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _normalise_label(raw: str) -> str:
    return STRATEGY_LABEL_MAP.get(raw.strip(), raw.strip())


def sta_total_item_count(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        prompt = build_prompt(
            inst, _AGENT,
            "What is the total number of items being negotiated over? "
            "(Count all packages of all types combined.)",
            'Inside <answer> put a JSON object with one key: "total_item_count" whose value '
            'is a plain integer (e.g. {"total_item_count": 9}). '
            "Do not write math expressions — evaluate the sum and write the final number.",
        )
        rows.append({
            "task": "sta_total_item_count_ca",
            "prompt": prompt,
            "answer_json": _json({"total_item_count": 9}),
            "det_thought": thoughts.total_items(inst),
        })
    return rows


def sta_max_points(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        pts = agent_points(inst, _AGENT)
        max_pts = sum(3 * p for p in pts.values())
        prompt = build_prompt(
            inst, _AGENT,
            "What is the maximum number of points you could get if you received ALL available "
            "packages (all 3 food, all 3 water, and all 3 firewood)?",
            'Inside <answer> put a JSON object with one key: "max_points" whose value is a '
            'plain integer (e.g. {"max_points": 36}). '
            "Do not write math expressions — evaluate the total and write the final number.",
        )
        rows.append({
            "task": "sta_max_points_ca",
            "prompt": prompt,
            "answer_json": _json({"max_points": max_pts}),
            "det_thought": thoughts.max_points(inst, _AGENT),
        })
    return rows


def sta_point_values(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        pts = agent_points(inst, _AGENT)
        gt = {k: str(v) for k, v in pts.items()}
        prompt = build_prompt(
            inst, _AGENT,
            "How many points is one package of each issue worth to you?",
            'Inside <answer> put a JSON object with keys "food", "water", "firewood" and '
            "values as the point counts (numbers) for each.",
        )
        rows.append({
            "task": "sta_ask_point_values_ca",
            "prompt": prompt,
            "answer_json": _json(gt),
            "det_thought": thoughts.point_values(inst, _AGENT),
        })
    return rows


def sta_high_priority(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        item = inst["participant_info"][_AGENT]["value2issue"]["High"].lower()
        prompt = build_prompt(
            inst, _AGENT,
            "Which single item (food, water, or firewood) is worth the MOST points per "
            "package to you? That is your highest priority issue.",
            'Inside <answer> put a JSON object with one key: "item" and value exactly one of '
            '"food", "water", or "firewood" — whichever has the highest points per package.',
        )
        rows.append({
            "task": "sta_ask_high_priority_ca",
            "prompt": prompt,
            "answer_json": _json({"item": item}),
            "det_thought": thoughts.priority(inst, _AGENT, "High"),
        })
    return rows


def sta_low_priority(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        item = inst["participant_info"][_AGENT]["value2issue"]["Low"].lower()
        prompt = build_prompt(
            inst, _AGENT,
            "Which single item (food, water, or firewood) is worth the FEWEST points per "
            "package to you? That is your lowest priority issue.",
            'Inside <answer> put a JSON object with one key: "item" and value exactly one of '
            '"food", "water", or "firewood" — whichever has the fewest points per package.',
        )
        rows.append({
            "task": "sta_ask_low_priority_ca",
            "prompt": prompt,
            "answer_json": _json({"item": item}),
            "det_thought": thoughts.priority(inst, _AGENT, "Low"),
        })
    return rows


def mid_high_priority(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        item = inst["participant_info"][_AGENT]["value2issue"]["High"].lower()
        prompt = build_mid_prompt(
            inst, _AGENT, _mid_dialogue_cut(inst["chat_logs"]),
            "Which single item (food, water, or firewood) is worth the MOST points per "
            "package to you? That is your highest priority issue.",
            _PRIORITY_OUTPUT_SPEC,
        )
        rows.append({
            "task": "mid_ask_high_priority_ca",
            "prompt": prompt,
            "answer_json": _json({"item": item}),
            "det_thought": thoughts.mid_priority(inst, _AGENT, "High"),
        })
    return rows


def mid_low_priority(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        item = inst["participant_info"][_AGENT]["value2issue"]["Low"].lower()
        prompt = build_mid_prompt(
            inst, _AGENT, _mid_dialogue_cut(inst["chat_logs"]),
            "Which single item (food, water, or firewood) is worth the FEWEST points per "
            "package to you? That is your lowest priority issue.",
            _PRIORITY_OUTPUT_SPEC,
        )
        rows.append({
            "task": "mid_ask_low_priority_ca",
            "prompt": prompt,
            "answer_json": _json({"item": item}),
            "det_thought": thoughts.mid_priority(inst, _AGENT, "Low"),
        })
    return rows


def mid_partner_high(instances: list[dict]) -> list[_Row]:
    rows = []
    partner = get_partner(_AGENT)
    for inst in instances:
        item = inst["participant_info"][partner]["value2issue"]["High"].lower()
        prompt = build_mid_prompt(
            inst, _AGENT, _mid_dialogue_cut(inst["chat_logs"]),
            "Based on the dialogue so far, which single item (food, water, or firewood) "
            "do you think is your PARTNER's highest priority?",
            _PRIORITY_OUTPUT_SPEC,
        )
        rows.append({
            "task": "mid_partner_ask_high_priority_ca",
            "prompt": prompt,
            "answer_json": _json({"item": item}),
            "det_thought": thoughts.partner_priority(inst, _AGENT, "High"),
        })
    return rows


def mid_partner_low(instances: list[dict]) -> list[_Row]:
    rows = []
    partner = get_partner(_AGENT)
    for inst in instances:
        item = inst["participant_info"][partner]["value2issue"]["Low"].lower()
        prompt = build_mid_prompt(
            inst, _AGENT, _mid_dialogue_cut(inst["chat_logs"]),
            "Based on the dialogue so far, which single item (food, water, or firewood) "
            "do you think is your PARTNER's lowest priority?",
            _PRIORITY_OUTPUT_SPEC,
        )
        rows.append({
            "task": "mid_partner_ask_low_priority_ca",
            "prompt": prompt,
            "answer_json": _json({"item": item}),
            "det_thought": thoughts.partner_priority(inst, _AGENT, "Low"),
        })
    return rows


def mid_strategy(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        chat_logs = inst["chat_logs"]
        annotations = inst["annotations"]
        for turn_idx, (log_turn, ann) in enumerate(zip(chat_logs, annotations)):
            if log_turn["text"] in _META_TURNS:
                continue
            if "non-strategic" in ann[1]:
                continue
            labels = frozenset(
                _normalise_label(s)
                for s in ann[1].split(",")
                if _normalise_label(s) in STRATEGY_LABELS
            )
            if not labels:
                continue
            context = chat_logs[:turn_idx]
            target_text = sanitize_unicode(log_turn["text"])
            question = (
                "What negotiation strategies are used in the following target utterance?\n"
                f'Target utterance: "{target_text}"'
            )
            prompt = build_mid_prompt(inst, _AGENT, context, question, _STRATEGY_OUTPUT_SPEC)
            rows.append({
                "task": "mid_strategy_ca",
                "prompt": prompt,
                "answer_json": _json(sorted(labels)),
                "det_thought": thoughts.strategy(labels),
            })
    return rows


TASK_GENERATORS: dict[str, Any] = {
    "sta_total_item_count_ca": sta_total_item_count,
    "sta_max_points_ca": sta_max_points,
    "sta_ask_point_values_ca": sta_point_values,
    "sta_ask_high_priority_ca": sta_high_priority,
    "sta_ask_low_priority_ca": sta_low_priority,
    "mid_ask_high_priority_ca": mid_high_priority,
    "mid_ask_low_priority_ca": mid_low_priority,
    "mid_partner_ask_high_priority_ca": mid_partner_high,
    "mid_partner_ask_low_priority_ca": mid_partner_low,
    "mid_strategy_ca": mid_strategy,
}

ALL_TASKS: list[str] = list(TASK_GENERATORS.keys())
