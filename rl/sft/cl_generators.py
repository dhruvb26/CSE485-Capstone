"""SFT example generators for CraigslistBargain tasks."""

from __future__ import annotations

import json
from typing import Any

from rl.handlers.craigslist.dataset import (
    build_prompt,
    extract_prices,
    infer_action,
    parse_turns,
)
from rl.sft import cl_thoughts

_Row = dict[str, Any]


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def mid_action_inference(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        output_turns = parse_turns(inst["output"])
        if not output_turns:
            continue

        all_turns = parse_turns(inst["input"]) + output_turns
        dialogue = "\n".join(f"{t['role'].title()}: {t['text']}" for t in all_turns)

        last = output_turns[-1]
        successful = inst.get("metadata", {}).get("successful", False)
        action = infer_action(last["text"], True, successful)

        prompt = build_prompt(
            inst,
            "What type of negotiation action does the LAST utterance represent? "
            "Choose one of: propose, counter, accept, reject.",
            'Inside <answer> put a JSON object with one key: "action" whose value is '
            'exactly one of "propose", "counter", "accept", or "reject".',
            dialogue,
        )
        rows.append({
            "task": "mid_action_inference_cl",
            "prompt": prompt,
            "answer_json": _json({"action": action}),
            "det_thought": cl_thoughts.action_inference(action, last["text"]),
        })
    return rows


def mid_price_reasoning(instances: list[dict]) -> list[_Row]:
    rows = []
    for inst in instances:
        output_turns = parse_turns(inst["output"])
        if not output_turns:
            continue

        all_text = " ".join(t["text"] for t in output_turns)
        prices = extract_prices(all_text)
        if not prices:
            continue

        last_price = prices[-1]
        parsed = inst["_parsed"]

        all_turns = parse_turns(inst["input"]) + output_turns
        dialogue = "\n".join(f"{t['role'].title()}: {t['text']}" for t in all_turns)

        prompt = build_prompt(
            inst,
            "What is the last price mentioned by the agent (the one making offers) in "
            "the dialogue?",
            'Inside <answer> put a JSON object with one key: "price" whose value is '
            "a number (the dollar amount).",
            dialogue,
        )
        rows.append({
            "task": "mid_price_reasoning_cl",
            "prompt": prompt,
            "answer_json": _json({"price": last_price}),
            "det_thought": cl_thoughts.price_reasoning(
                last_price,
                parsed["listing_price"],
                parsed["role"],
                output_turns[-1]["text"],
            ),
        })
    return rows


CL_TASK_GENERATORS: dict[str, Any] = {
    "mid_action_inference_cl": mid_action_inference,
    "mid_price_reasoning_cl": mid_price_reasoning,
}

CL_ALL_TASKS: list[str] = list(CL_TASK_GENERATORS.keys())
