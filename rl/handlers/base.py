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
        """Return the first integer (including negative) found in text.

        If *text* is a ``{thought, talk, action}`` turn-schema JSON, the
        ``thought`` field is searched as well.
        """
        m = re.search(r"-?\d+", text)
        if m:
            return m.group(0)
        turn = self._parse_turn_json(text)
        if turn:
            m = re.search(r"-?\d+", turn.get("thought", ""))
            if m:
                return m.group(0)
        return None

    def _parse_turn_json(self, text: str) -> dict | None:
        """If *text* is a ``{thought, talk, action}`` turn-schema JSON,
        return the parsed dict.  Otherwise ``None``."""
        raw = self._extract_json_raw(text)
        if (
            isinstance(raw, dict)
            and "thought" in raw
            and "action" in raw
        ):
            return raw
        return None

    def extract_from_turn_thought(self, text: str) -> dict:
        """Parse structured facts out of a turn-schema ``thought`` string.

        The deterministic thought format produced by turn_generators is::

            My values: food=Xpts, water=Ypts, firewood=Zpts. Max possible = Npts.
            My allocation: ... Partner priority estimate: ITEM. Decision: TYPE.

        Returns a dict with any of the following keys that could be extracted:
        ``values``, ``max_points``, ``partner_priority``, ``my_priority``,
        ``my_low_priority``, ``decision``.
        """
        turn = self._parse_turn_json(text)
        thought = turn.get("thought", "") if turn else ""
        if not thought:
            return {}

        info: dict = {}

        # Point values:  "food=5pts, water=4pts, firewood=3pts"
        val_matches = re.findall(r"(\w+)=(\d+)pts", thought)
        if val_matches:
            info["values"] = {k.lower(): str(v) for k, v in val_matches}
            pts = {k.lower(): int(v) for k, v in val_matches}
            if pts:
                info["my_priority"] = max(pts, key=pts.get)
                info["my_low_priority"] = min(pts, key=pts.get)

        # Max possible points
        m = re.search(r"Max possible\s*=\s*(\d+)", thought)
        if m:
            info["max_points"] = int(m.group(1))

        # Partner priority
        m = re.search(r"Partner priority estimate:\s*(\w+)", thought)
        if m and m.group(1).lower() != "unknown":
            info["partner_priority"] = m.group(1).lower()

        # Decision / action type
        m = re.search(r"Decision:\s*(\w+)", thought)
        if m:
            info["decision"] = m.group(1).lower()

        return info
    
    def extract_json(self, text: str) -> dict | None:
        """Try to parse a JSON object from text.

        Strategy:
        1. Content inside ``<answer>…</answer>`` tags
        2. Whole text as JSON
        3. First ``{…}`` block
        4. If the result is a turn-schema, return the ``action`` dict
        """
        raw = self._extract_json_raw(text)
        if raw is None:
            return None
        if isinstance(raw, dict) and "thought" in raw and "action" in raw:
            action = raw.get("action")
            return action if isinstance(action, dict) else raw
        return raw

    def _extract_json_raw(self, text: str) -> dict | None:
        """Inner helper: parse JSON from text without unwrapping."""
        tagged = self.extract_answer_tag(text)
        if tagged:
            try:
                return json.loads(tagged)
            except Exception:
                pass

        try:
            return json.loads(text.strip())
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return None

    def log_prediction(self, prediction) -> str | dict | None:
        """Convert a parsed prediction to a SysEval-compatible log value.

        Override per task when the prediction is a dict with a single key.
        Default: str(prediction).
        """
        return str(prediction) if prediction is not None else None

    def log_ground_truth(self, truth) -> str | dict:
        """Convert a ground truth to a SysEval-compatible log value.

        Override per task when the truth is a dict with a single key.
        Default: str(truth).
        """
        return str(truth)

    @staticmethod
    def _remove_duplicates(
        prompts: list[str], ground_truth: list
    ) -> tuple[list[str], list]:
        """Return (unique_prompts, unique_gts) keeping the first occurrence of each prompt."""
        seen: set[str] = set()
        uniq_p, uniq_gt = [], []
        for p, gt in zip(prompts, ground_truth):
            if p not in seen:
                seen.add(p)
                uniq_p.append(p)
                uniq_gt.append(gt)
        return uniq_p, uniq_gt

    def evaluate(
        self,
        dataset_handler: BaseDatasetHandler,
        model,
        n: int | None = None,
        agent: str = "mturk_agent_1",
        run_dir: Path | None = None,
    ) -> dict:
        instances = dataset_handler.get_instances(n)

        all_prompts: list[str] = []
        all_gts: list = []
        for instance in instances:
            all_prompts.append(self.build_prompt(instance, agent))
            all_gts.append(self.ground_truth(instance, agent))

        uniq_prompts, uniq_gts = self._remove_duplicates(all_prompts, all_gts)

        outputs_dict: dict[str, str] = {}
        for prompt in tqdm(uniq_prompts, desc=self.task_id, unit="prompt"):
            outputs_dict[prompt] = model.generate(prompt)

        final_prompts: list[str] = []
        final_raw_responses: list[str] = []
        final_preds_log: list = []
        final_gts_log: list = []
        scores: list[float] = []

        for prompt, gt in zip(uniq_prompts, uniq_gts):
            raw = outputs_dict[prompt]
            prediction = self.parse_output(raw)
            if prediction is not None:
                scores.append(self.score(prediction, gt))
                final_prompts.append(prompt)
                final_raw_responses.append(raw)
                final_preds_log.append(self.log_prediction(prediction))
                final_gts_log.append(self.log_ground_truth(gt))

        accuracy = sum(scores) / len(scores) if scores else 0.0

        stats = {
            "total": len(all_prompts),
            "unique": len(uniq_prompts),
            "valid": len(final_prompts),
            "accuracy": round(accuracy, 4),
        }

        log_data = {
            "stats": stats,
            "ground truth": final_gts_log,
            "predictions": final_preds_log,
            "raw_responses": final_raw_responses,
            "prompts": final_prompts,
            "outputs_dict": outputs_dict,
        }

        if run_dir is not None:
            suffix = self.task_id.rsplit("_", 1)[-1]
            dataset = {"ca": "ca", "dnd": "dnd", "cl": "cl"}.get(suffix, suffix)
            task_log_dir = run_dir / dataset / self.task_id
            task_log_dir.mkdir(parents=True, exist_ok=True)
            model_id = getattr(model, "model_id", "model")
            fname = f"{dataset}_{model_id}_{self.task_id}_{len(all_prompts)}.json"
            with (task_log_dir / fname).open("w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

        return {
            "task": self.task_id,
            "n": len(all_prompts),
            "accuracy": accuracy,
        }
