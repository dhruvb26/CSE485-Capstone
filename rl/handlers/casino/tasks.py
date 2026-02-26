"""
Stage-0 / Stage-1 start tasks for the CaSiNo dataset (CA tasks 1-5).

Each class implements the BaseTaskHandler interface:
    build_prompt  → fills the prompt template for a single instance
    ground_truth  → computes the correct answer programmatically (no model needed)
    parse_output  → extracts a structured answer from raw model text
    score         → binary exact-match (0.0 or 1.0)

All tasks expect a JSON object inside <answer>; verification is on structured output only.
"""

from rl.handlers.base import BaseTaskHandler
from rl.handlers.casino.dataset import agent_points, build_prompt


class TotalItemCountCA(BaseTaskHandler):
    """
    sta_total_item_count_ca
    Ground truth is always {"total_item_count": 9} — 3 food + 3 water + 3 firewood.
    """

    task_id = "sta_total_item_count_ca"
    _key = "total_item_count"

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is the total number of items being negotiated over?",
            'Inside <answer> put a JSON object with one key: "total_item_count" (number).',
        )

    def ground_truth(self, instance: dict, agent: str) -> dict[str, int]:
        return {self._key: 9}

    def parse_output(self, text: str) -> dict[str, int] | None:
        data = self.extract_json(text)
        if not data or self._key not in data:
            return None
        try:
            return {self._key: int(data[self._key])}
        except (TypeError, ValueError):
            return None

    def score(self, prediction: dict | None, truth: dict) -> float:
        return 1.0 if prediction and prediction.get(self._key) == truth.get(self._key) else 0.0


class MaxPointsCA(BaseTaskHandler):
    """
    sta_max_points_ca
    Ground truth is always {"max_points": 36} — 3×5 + 3×4 + 3×3 (take everything).
    """

    task_id = "sta_max_points_ca"
    _key = "max_points"

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is the maximum number of points that you can possibly get in any deal?",
            'Inside <answer> put a JSON object with one key: "max_points" (number).',
        )

    def ground_truth(self, instance: dict, agent: str) -> dict[str, int]:
        return {self._key: 36}

    def parse_output(self, text: str) -> dict[str, int] | None:
        data = self.extract_json(text)
        if not data or self._key not in data:
            return None
        try:
            return {self._key: int(data[self._key])}
        except (TypeError, ValueError):
            return None

    def score(self, prediction: dict | None, truth: dict) -> float:
        return 1.0 if prediction and prediction.get(self._key) == truth.get(self._key) else 0.0


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
            'Inside <answer> put a JSON object with keys "food", "water", "firewood" and '
            "values as the point counts (numbers) for each.",
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
    Ground truth: {"item": "food"|"water"|"firewood"} for the agent's highest-value item.
    """

    task_id = "sta_ask_high_priority_ca"
    _key = "item"
    _options = ("food", "water", "firewood")

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is your highest priority issue?",
            'Inside <answer> put a JSON object with one key: "item" and value one of '
            '"food", "water", "firewood".',
        )

    def ground_truth(self, instance: dict, agent: str) -> dict[str, str]:
        item = instance["participant_info"][agent]["value2issue"]["High"].lower()
        return {self._key: item}

    def parse_output(self, text: str) -> dict[str, str] | None:
        data = self.extract_json(text)
        if not data or self._key not in data:
            return None
        raw = data[self._key]
        item = str(raw).lower().strip() if raw is not None else None
        return {self._key: item} if item in self._options else None

    def score(self, prediction: dict | None, truth: dict) -> float:
        return 1.0 if prediction and prediction.get(self._key) == truth.get(self._key) else 0.0


class LowPriorityCA(BaseTaskHandler):
    """
    sta_ask_low_priority_ca
    Ground truth: {"item": "food"|"water"|"firewood"} for the agent's lowest-value item.
    """

    task_id = "sta_ask_low_priority_ca"
    _key = "item"
    _options = ("food", "water", "firewood")

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is your lowest priority issue?",
            'Inside <answer> put a JSON object with one key: "item" and value one of '
            '"food", "water", "firewood".',
        )

    def ground_truth(self, instance: dict, agent: str) -> dict[str, str]:
        item = instance["participant_info"][agent]["value2issue"]["Low"].lower()
        return {self._key: item}

    def parse_output(self, text: str) -> dict[str, str] | None:
        data = self.extract_json(text)
        if not data or self._key not in data:
            return None
        raw = data[self._key]
        item = str(raw).lower().strip() if raw is not None else None
        return {self._key: item} if item in self._options else None

    def score(self, prediction: dict | None, truth: dict) -> float:
        return 1.0 if prediction and prediction.get(self._key) == truth.get(self._key) else 0.0


CA_TASK_REGISTRY: dict[str, type[BaseTaskHandler]] = {
    "sta_total_item_count_ca": TotalItemCountCA,
    "sta_max_points_ca": MaxPointsCA,
    "sta_ask_point_values_ca": PointValuesCA,
    "sta_ask_high_priority_ca": HighPriorityCA,
    "sta_ask_low_priority_ca": LowPriorityCA,
}
