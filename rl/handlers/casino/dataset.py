"""
CaSiNo dataset handler and shared prompt-building utilities.

Priority → point value mapping (same as SysEval benchmark):
    High → 5 pts  |  Medium → 4 pts  |  Low → 3 pts

Item counts are fixed across all CA instances (3 food, 3 water, 3 firewood).
"""

import ast
import copy

import pandas as pd

from rl.handlers.base import BaseDatasetHandler

PRIORITY_TO_POINTS: dict[str, int] = {"Low": 3, "Medium": 4, "High": 5}

_PROMPT_HEADER = (
    "Task Description: You are negotiating with your campsite neighbor over extra supply "
    "of food, water, and firewood for your camping trip. Different types of packages are "
    "worth different amount of points to each one of you. You'll be provided with "
    "information about the negotiation. Then, you'll answer a question."
    "\n\nHere are the number of food, water, and firewood packages available in the "
    "negotiation, contained in <count> tags.\n<count>\nFood Packages: 3\n"
    "Water Packages: 3\nFirewood Packages: 3\n</count>"
    "\n\nHere are the number of points you get for each type of package, contained in "
    "<value> tags.\n<value>\nEach Food Package: {food} points\n"
    "Each Water Package: {water} points\nEach Firewood Package: {firewood} points\n</value>"
    "\n\nQuestion: {question}"
)

OUTPUT_FORMAT_INSTRUCTION = (
    "Response format: First, complete ALL your reasoning inside <thought>...</thought>. "
    "Only after your reasoning is done, output your final answer inside <answer>...</answer>. "
    "Do NOT output <answer> before your reasoning is complete."
)


class CasinoDatasetHandler(BaseDatasetHandler):
    """Loads ca.test.csv (or ca.train.csv), filtering out Walk-Away conversations."""

    def load(self):
        df = pd.read_csv(self.data_path)
        self.dataset = []
        for inst in df.to_dict(orient="records"):
            if "Walk-Away" in inst["chat_logs"]:
                continue
            inst2 = copy.deepcopy(inst)
            inst2["annotations"] = ast.literal_eval(inst["annotations"])
            inst2["chat_logs"] = ast.literal_eval(inst["chat_logs"])
            inst2["participant_info"] = ast.literal_eval(inst["participant_info"])
            self.dataset.append(inst2)


def agent_points(instance: dict, agent: str) -> dict[str, int]:
    """Return {item_lower: points} for the given agent derived from their value2issue map."""
    v2i: dict[str, str] = instance["participant_info"][agent]["value2issue"]
    return {item.lower(): PRIORITY_TO_POINTS[level] for level, item in v2i.items()}


def build_prompt(instance: dict, agent: str, question: str, output_spec: str) -> str:
    pts = agent_points(instance, agent)
    body = _PROMPT_HEADER.format(
        food=pts.get("food", "?"),
        water=pts.get("water", "?"),
        firewood=pts.get("firewood", "?"),
        question=question,
    )
    return f"{body}\n\n{OUTPUT_FORMAT_INSTRUCTION}\n\n{output_spec}"
