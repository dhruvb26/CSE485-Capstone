"""
CaSiNo dataset handler and shared prompt-building utilities.

Priority → point value mapping (same as SysEval benchmark):
    High → 5 pts  |  Medium → 4 pts  |  Low → 3 pts

Item counts are fixed across all CA instances (3 food, 3 water, 3 firewood).
"""

import ast
import copy
import re

import pandas as pd

from rl.handlers.base import BaseDatasetHandler

PRIORITY_TO_POINTS: dict[str, int] = {"Low": 3, "Medium": 4, "High": 5}

# Atomic strategy labels used by the SysEval benchmark (paper table).
# "promote-coordination" → "coordination", "showing-empathy" → "empathy" (reference normalisation).
# "non-strategic" is excluded: those turns are skipped entirely in mid_strategy_ca.
STRATEGY_LABELS: frozenset[str] = frozenset([
    "small-talk", "empathy", "vouch-fair", "elicit-pref",
    "self-need", "other-need", "no-need", "coordination",
    "uv-part",
])

# Map raw annotation strings to the normalised label vocabulary
STRATEGY_LABEL_MAP: dict[str, str] = {
    "promote-coordination": "coordination",
    "showing-empathy": "empathy",
}

# Turns that are interface artefacts, not real dialogue
_META_TURNS: frozenset[str] = frozenset(["Submit-Deal", "Accept-Deal"])

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


def get_partner(agent: str) -> str:
    """Return the agent key for the other participant."""
    return "mturk_agent_2" if agent == "mturk_agent_1" else "mturk_agent_1"


def sanitize_unicode(s: str) -> str:
    """Remove Unicode surrogates so text is valid UTF-8 for API requests."""
    return re.sub(r"[\uD800-\uDFFF]", "", s)


def format_dialogue(turns: list[dict], agent: str) -> str:
    """Format chat_log turns into a <dialogue> block, skipping meta-turns.

    Each turn is labelled 'You' or 'Partner' relative to the focal agent.
    Dialogue text is sanitized to avoid surrogate characters (invalid UTF-8).
    """
    lines = []
    for t in turns:
        if t["text"] in _META_TURNS:
            continue
        speaker = "You" if t["id"] == agent else "Partner"
        lines.append(f"{speaker}: {sanitize_unicode(t['text'])}")
    return "<dialogue>\n" + "\n".join(lines) + "\n</dialogue>"


def build_prompt(instance: dict, agent: str, question: str, output_spec: str) -> str:
    pts = agent_points(instance, agent)
    body = _PROMPT_HEADER.format(
        food=pts.get("food", "?"),
        water=pts.get("water", "?"),
        firewood=pts.get("firewood", "?"),
        question=question,
    )
    return f"{body}\n\n{OUTPUT_FORMAT_INSTRUCTION}\n\n{output_spec}"


def build_mid_prompt(
    instance: dict,
    agent: str,
    dialogue_turns: list[dict],
    question: str,
    output_spec: str,
) -> str:
    """Like build_prompt but inserts a <dialogue> section before the question."""
    pts = agent_points(instance, agent)
    body = _PROMPT_HEADER.format(
        food=pts.get("food", "?"),
        water=pts.get("water", "?"),
        firewood=pts.get("firewood", "?"),
        question=question,
    )
    dialogue_block = format_dialogue(dialogue_turns, agent)
    return (
        f"{body}\n\n"
        f"Here is the dialogue so far, contained in <dialogue> tags.\n"
        f"{dialogue_block}\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTION}\n\n{output_spec}"
    )
