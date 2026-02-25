import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDatasetHandler(ABC):
    def __init__(self, data_path: Path | str):
        self.data_path = Path(data_path)
        self.dataset: list[dict] = []
        self.load()

    @abstractmethod
    def load(self): ...

    def get_instances(self, n: int | None = None) -> list[dict]:
        return self.dataset if n is None else self.dataset[:n]


class BaseTaskHandler(ABC):
    task_id: str

    @abstractmethod
    def build_prompt(self, instance: dict, agent: str) -> str: ...

    @abstractmethod
    def ground_truth(self, instance: dict, agent: str) -> Any: ...

    @abstractmethod
    def parse_output(self, text: str) -> Any: ...

    @abstractmethod
    def score(self, prediction: Any, truth: Any) -> float: ...

    def extract_answer_tag(self, text: str) -> str | None:
        """Extract content between <answer> … </answer> tags."""
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def extract_number(self, text: str) -> str | None:
        """Return the first integer (including negative) found in text."""
        m = re.search(r"-?\d+", text)
        return m.group(0) if m else None

    def extract_json(self, text: str) -> dict | None:
        """Try to parse a JSON object from text; checks <answer> tags first."""
        # 1. inside <answer> tags
        tagged = self.extract_answer_tag(text)
        if tagged:
            try:
                return json.loads(tagged)
            except Exception:
                pass

        # 2. whole text
        try:
            return json.loads(text.strip())
        except Exception:
            pass

        # 3. first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return None

    def evaluate(
        self,
        dataset_handler: BaseDatasetHandler,
        model,
        n: int | None = None,
        agent: str = "mturk_agent_1",
    ) -> dict:
        instances = dataset_handler.get_instances(n)
        results = []

        for instance in instances:
            prompt = self.build_prompt(instance, agent)
            truth = self.ground_truth(instance, agent)
            raw = model.generate(prompt)
            prediction = self.parse_output(raw)
            s = self.score(prediction, truth) if prediction is not None else 0.0
            results.append(
                {
                    "prompt": prompt,
                    "raw_output": raw,
                    "prediction": prediction,
                    "ground_truth": truth,
                    "score": s,
                }
            )

        accuracy = sum(r["score"] for r in results) / len(results) if results else 0.0
        return {
            "task": self.task_id,
            "n": len(results),
            "accuracy": accuracy,
            "results": results,
        }
