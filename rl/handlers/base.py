import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


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
        """Extract content between <answer> … </answer> tags (first occurrence)."""
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def extract_thought_tag(self, text: str) -> str | None:
        """Extract content between <thought> … </thought> tags (first occurrence)."""
        m = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def extract_number(self, text: str) -> str | None:
        """Return the first integer (including negative) found in text."""
        m = re.search(r"-?\d+", text)
        return m.group(0) if m else None

    @staticmethod
    def _resolve_arith(text: str) -> str:
        """Evaluate a simple integer arithmetic expression (e.g. '3 + 3 + 3' → '9').

        Only allows digits, spaces, and the four basic operators to prevent eval abuse.
        Falls back to the original text if it cannot be evaluated.
        """
        stripped = text.strip()
        if re.fullmatch(r"[\d\s\+\-\*\/\(\)]+", stripped):
            try:
                result = eval(stripped, {"__builtins__": {}})  # noqa: S307
                if isinstance(result, (int, float)):
                    return str(int(result))
            except Exception:
                pass
        return stripped

    def extract_json(self, text: str) -> dict | None:
        """Try to parse a JSON object from text; checks <answer> tags first.

        Numeric JSON values that are simple arithmetic expressions (e.g. "3 + 3 + 3")
        are evaluated before parsing so that model outputs like
        ``{"total_item_count": 3 + 3 + 3}`` are handled correctly.
        """
        def _try_parse(blob: str) -> dict | None:
            blob = blob.strip()
            try:
                return json.loads(blob)
            except Exception:
                pass
            # Replace arithmetic expressions in JSON values before retrying.
            fixed = re.sub(
                r':\s*([\d\s\+\-\*\/\(\)]+?)(?=[,\}])',
                lambda m: ': ' + self._resolve_arith(m.group(1)),
                blob,
            )
            try:
                return json.loads(fixed)
            except Exception:
                return None

        # 1. inside <answer> tags (preferred)
        tagged = self.extract_answer_tag(text)
        if tagged:
            result = _try_parse(tagged)
            if result is not None:
                return result

        # 2. whole text
        result = _try_parse(text)
        if result is not None:
            return result

        # 3. first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return _try_parse(m.group(0))

        return None

    def evaluate(
        self,
        dataset_handler: BaseDatasetHandler,
        model,
        n: int | None = None,
        agent: str = "mturk_agent_1",
        run_dir: Path | None = None,
    ) -> dict:
        instances = dataset_handler.get_instances(n)
        results = []

        run_log_path: Path | None = None
        if run_dir is not None:
            dataset = "ca" if self.task_id.endswith("_ca") else "dnd"
            task_log_dir = run_dir / dataset / self.task_id
            task_log_dir.mkdir(parents=True, exist_ok=True)
            run_log_path = task_log_dir / "model_io.jsonl"

        for idx, instance in enumerate(tqdm(instances, desc=self.task_id, unit="instance")):
            prompt = self.build_prompt(instance, agent)
            truth = self.ground_truth(instance, agent)
            raw = model.generate(prompt)
            thought = self.extract_thought_tag(raw)
            answer = self.extract_answer_tag(raw)
            prediction = self.parse_output(raw)
            s = self.score(prediction, truth) if prediction is not None else 0.0
            results.append(
                {
                    "prompt": prompt,
                    "raw_output": raw,
                    "thought": thought,
                    "answer": answer,
                    "prediction": prediction,
                    "ground_truth": truth,
                    "score": s,
                }
            )
            if run_log_path is not None:
                log_line = {
                    "task_id": self.task_id,
                    "instance_idx": idx,
                    "prompt": prompt,
                    "raw_output": raw,
                    "thought": thought,
                    "answer": answer,
                    "prediction": prediction,
                    "ground_truth": truth,
                    "score": s,
                }
                with run_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

        accuracy = sum(r["score"] for r in results) / len(results) if results else 0.0
        return {
            "task": self.task_id,
            "n": len(results),
            "accuracy": accuracy,
            "results": results,
        }
