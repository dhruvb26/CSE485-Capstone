"""
Stage-0 / Stage-1 start tasks for the DealOrNoDeal dataset (DND tasks 1-3).

Each class implements the BaseTaskHandler interface:
    build_prompt  → fills the prompt template for a single instance
    ground_truth  → computes the correct answer programmatically (no model needed)
    parse_output  → extracts a structured answer from raw model text
    score         → binary exact-match (0.0 or 1.0)
"""

from rl.handlers.base import BaseTaskHandler
from rl.handlers.dnd.dataset import agent_input, build_prompt


class TotalItemCountDND(BaseTaskHandler):
    """
    sta_total_item_count_dnd
    Ground truth: sum of counts (varies per instance).
    """

    task_id = "sta_total_item_count_dnd"

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is the total number of items being negotiated over?",
            "Present your answer as a single number with no additional text.",
        )

    def ground_truth(self, instance: dict, agent: str) -> str:
        return str(sum(agent_input(instance, agent)["count"]))

    def parse_output(self, text: str) -> str | None:
        return self.extract_number(text)

    def score(self, prediction: str, truth: str) -> float:
        return 1.0 if prediction == truth else 0.0


class MaxPointsDND(BaseTaskHandler):
    """
    sta_max_points_dnd
    Ground truth is always "10" — the DND dataset guarantees max achievable = 10.
    """

    task_id = "sta_max_points_dnd"

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "What is the maximum number of points that you can possibly get in any deal?",
            "Present your answer as a single number with no additional text.",
        )

    def ground_truth(self, instance: dict, agent: str) -> str:
        return "10"

    def parse_output(self, text: str) -> str | None:
        return self.extract_number(text)

    def score(self, prediction: str, truth: str) -> float:
        return 1.0 if prediction == truth else 0.0


class PointValuesDND(BaseTaskHandler):
    """
    sta_ask_point_values_dnd
    Ground truth: {"books": "N", "hats": "N", "balls": "N"}.
    All three keys must match exactly to score 1.0.
    """

    task_id = "sta_ask_point_values_dnd"
    _keys = ("books", "hats", "balls")

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_prompt(
            instance,
            agent,
            "How many points is one item of each issue worth to you?",
            "Present your answer as a json within <answer> </answer> tags with keys as "
            "issues (books, hats, and balls) and values as the corresponding answers.",
        )

    def ground_truth(self, instance: dict, agent: str) -> dict[str, str]:
        values = agent_input(instance, agent)["value"]
        return {"books": str(values[0]), "hats": str(values[1]), "balls": str(values[2])}

    def parse_output(self, text: str) -> dict[str, str] | None:
        data = self.extract_json(text)
        if data is None:
            return None
        return {k.lower(): str(v) for k, v in data.items() if k.lower() in self._keys}

    def score(self, prediction: dict, truth: dict) -> float:
        if not isinstance(prediction, dict):
            return 0.0
        return 1.0 if all(prediction.get(k) == truth.get(k) for k in self._keys) else 0.0


DND_TASK_REGISTRY: dict[str, type[BaseTaskHandler]] = {
    "sta_total_item_count_dnd": TotalItemCountDND,
    "sta_max_points_dnd": MaxPointsDND,
    "sta_ask_point_values_dnd": PointValuesDND,
}
