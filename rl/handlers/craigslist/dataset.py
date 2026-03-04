"""
CraigslistBargain dataset handler and shared prompt-building utilities.

Data format (craigslist_bargains_alpaca.jsonl):
    instruction  -- role + product name + price + category + description
    input        -- opponent turns (newline-separated, prefixed with role)
    output       -- agent turns
    metadata     -- uuid, category, successful, perspective
"""

from __future__ import annotations

import json
import re

from rl.handlers.base import BaseDatasetHandler

_PRICE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


class CraigslistDatasetHandler(BaseDatasetHandler):
    """Loads craigslist_bargains_alpaca.jsonl and parses each line."""

    def load(self):
        self.dataset = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_parsed"] = _parse_instruction(row["instruction"])
                self.dataset.append(row)


def _parse_instruction(instruction: str) -> dict:
    """Extract role, product name, listing price, and category from the instruction string."""
    role_match = re.search(r"You are a (\w+)", instruction)
    role = role_match.group(1).lower() if role_match else "unknown"

    price_match = _PRICE_RE.search(instruction)
    listing_price = float(price_match.group(1).replace(",", "")) if price_match else 0.0

    cat_match = re.search(r"\(\$[\d,.]+,\s*(\w[\w\s-]*)\)", instruction)
    category = cat_match.group(1).strip() if cat_match else "unknown"

    name_match = re.search(r"for:\s*(.+?)\s*\(\$", instruction)
    product_name = name_match.group(1).strip() if name_match else "unknown product"

    return {
        "role": role,
        "listing_price": listing_price,
        "category": category,
        "product_name": product_name,
    }


def extract_prices(text: str) -> list[float]:
    """Extract all dollar amounts from a dialogue text."""
    return [float(m.replace(",", "")) for m in _PRICE_RE.findall(text) if m.replace(",", "")]


def parse_turns(text: str) -> list[dict]:
    """Parse 'Role: message' lines into a list of {role, text} dicts."""
    turns = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        colon = line.find(":")
        if colon > 0:
            role = line[:colon].strip().lower()
            msg = line[colon + 1:].strip()
            turns.append({"role": role, "text": msg})
    return turns


def infer_action(turn_text: str, is_last_turn: bool, successful: bool) -> str:
    """Heuristically infer the action type from a single turn's text."""
    lower = turn_text.lower()
    if is_last_turn and successful:
        return "accept"
    if any(w in lower for w in ("deal", "agree", "accept", "sold", "sounds good", "you got it")):
        return "accept"
    if any(w in lower for w in ("no", "sorry", "can't", "cannot", "too low", "too high", "pass")):
        if _PRICE_RE.search(turn_text):
            return "counter"
        return "reject"
    if _PRICE_RE.search(turn_text):
        return "propose"
    return "propose"


_PROMPT_HEADER = (
    "Task Description: You are observing a negotiation on an online marketplace.\n\n"
    "Product: {product_name}\n"
    "Listing price: ${listing_price:.2f}\n"
    "Category: {category}\n\n"
    "Here is the dialogue so far:\n"
    "<dialogue>\n{dialogue}\n</dialogue>\n\n"
    "Question: {question}"
)


def build_prompt(instance: dict, question: str, output_spec: str, dialogue: str) -> str:
    p = instance["_parsed"]
    body = _PROMPT_HEADER.format(
        product_name=p["product_name"],
        listing_price=p["listing_price"],
        category=p["category"],
        dialogue=dialogue,
        question=question,
    )
    return body + " " + output_spec
