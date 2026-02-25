"""
Stage-0 / Stage-1 start tasks for the CaSiNo dataset (CA tasks 1-5).

Each class implements the BaseTaskHandler interface:
    build_prompt  → fills the prompt template for a single instance
    ground_truth  → computes the correct answer programmatically (no model needed)
    parse_output  → extracts a structured answer from raw model text
    score         → binary exact-match (0.0 or 1.0)
"""

import re

from rl.handlers.base import BaseTaskHandler
from rl.handlers.casino.dataset import agent_points, build_prompt


class TotalItemCountCA(BaseTaskHandler):
    """
    sta_total_item_count_ca
    Ground truth is always "9" — 3 food + 3 water + 3 firewood.
    """

    task_id = "sta_total_item_count_ca"

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is the total number of items being negotiated over?",
            "Present your answer as a single number with no additional text.",
        )

    def ground_truth(self, instance: dict, agent: str) -> str:
        return "9"

    def parse_output(self, text: str) -> str | None:
        return self.extract_number(text)

    def score(self, prediction: str, truth: str) -> float:
        return 1.0 if prediction == truth else 0.0


class MaxPointsCA(BaseTaskHandler):
    """
    sta_max_points_ca
    Ground truth is always "36" — 3×5 + 3×4 + 3×3 (take everything).
    """

    task_id = "sta_max_points_ca"

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is the maximum number of points that you can possibly get in any deal?",
            "Present your answer as a single number with no additional text.",
        )

    def ground_truth(self, instance: dict, agent: str) -> str:
        return "36"

    def parse_output(self, text: str) -> str | None:
        return self.extract_number(text)

    def score(self, prediction: str, truth: str) -> float:
        return 1.0 if prediction == truth else 0.0


class PointValuesCA(BaseTaskHandler):
    """
    sta_ask_point_values_ca
    Ground truth: {"food": "3/4/5", "water": "3/4/5", "firewood": "3/4/5"}.
    All three keys must match exactly to score 1.0.
    """

    task_id = "sta_ask_point_values_ca"
    _keys = ("food", "water", "firewood")

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "How many points is one package of each issue worth to you?",
            "Present your answer as a json within <answer> </answer> tags with keys as "
            "issues (food, water, and firewood) and values as the corresponding answers.",
        )

    def ground_truth(self, instance: dict, agent: str) -> dict[str, str]:
        return {k: str(v) for k, v in agent_points(instance, agent).items()}

    def parse_output(self, text: str) -> dict[str, str] | None:
        data = self.extract_json(text)
        if data is None:
            return None
        return {k.lower(): str(v) for k, v in data.items() if k.lower() in self._keys}

    def score(self, prediction: dict, truth: dict) -> float:
        if not isinstance(prediction, dict):
            return 0.0
        return 1.0 if all(prediction.get(k) == truth.get(k) for k in self._keys) else 0.0


class HighPriorityCA(BaseTaskHandler):
    """
    sta_ask_high_priority_ca
    Ground truth: lowercase name of the agent's highest-value item.
    """

    task_id = "sta_ask_high_priority_ca"
    _options = ("food", "water", "firewood")
    _mc = {"a": "food", "b": "water", "c": "firewood"}

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is your highest priority issue?",
            "Present your answer as one of the following multiple choice options. "
            "You must select an option.\nA: food\nB: water\nC: firewood",
        )

    def ground_truth(self, instance: dict, agent: str) -> str:
        return instance["participant_info"][agent]["value2issue"]["High"].lower()

    def parse_output(self, text: str) -> str | None:
        t = text.lower().strip()
        for letter, item in self._mc.items():
            if re.search(rf"\b{letter}\b", t):
                return item
        for item in self._options:
            if item in t:
                return item
        return None

    def score(self, prediction: str, truth: str) -> float:
        return 1.0 if prediction == truth else 0.0


class LowPriorityCA(BaseTaskHandler):
    """
    sta_ask_low_priority_ca
    Ground truth: lowercase name of the agent's lowest-value item.
    """

    task_id = "sta_ask_low_priority_ca"
    _options = ("food", "water", "firewood")
    _mc = {"a": "food", "b": "water", "c": "firewood"}

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is your lowest priority issue?",
            "Present your answer as one of the following multiple choice options. "
            "You must select an option.\nA: food\nB: water\nC: firewood",
        )

    def ground_truth(self, instance: dict, agent: str) -> str:
        return instance["participant_info"][agent]["value2issue"]["Low"].lower()

    def parse_output(self, text: str) -> str | None:
        t = text.lower().strip()
        for letter, item in self._mc.items():
            if re.search(rf"\b{letter}\b", t):
                return item
        for item in self._options:
            if item in t:
                return item
        return None

    def score(self, prediction: str, truth: str) -> float:
        return 1.0 if prediction == truth else 0.0


CA_TASK_REGISTRY: dict[str, type[BaseTaskHandler]] = {
    "sta_total_item_count_ca": TotalItemCountCA,
    "sta_max_points_ca": MaxPointsCA,
    "sta_ask_point_values_ca": PointValuesCA,
    "sta_ask_high_priority_ca": HighPriorityCA,
    "sta_ask_low_priority_ca": LowPriorityCA,
}
