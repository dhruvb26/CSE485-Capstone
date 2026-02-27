"""
Start and mid-stage tasks for the CaSiNo dataset (CA tasks 1-10).

Each class implements the BaseTaskHandler interface:
    build_prompt  → fills the prompt template for a single instance
    ground_truth  → computes the correct answer programmatically (no model needed)
    parse_output  → extracts a structured answer from raw model text
    score         → binary exact-match (0.0 or 1.0)

All tasks expect a JSON object (or array for StrategyCA) inside <answer>.
Mid-stage tasks (mid_*) include partial dialogue history in the prompt.
"""

import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

from rl.handlers.base import BaseTaskHandler
from rl.handlers.casino.dataset import (
    STRATEGY_LABEL_MAP,
    STRATEGY_LABELS,
    _META_TURNS,
    agent_points,
    build_mid_prompt,
    build_prompt,
    get_partner,
    sanitize_unicode,
)

logger = logging.getLogger(__name__)


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
            "What is the total number of items being negotiated over? "
            "(Count all packages of all types combined.)",
            'Inside <answer> put a JSON object with one key: "total_item_count" whose value '
            "is a plain integer (e.g. {\"total_item_count\": 9}). "
            "Do not write math expressions — evaluate the sum and write the final number.",
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

    def log_prediction(self, prediction) -> str | None:
        return str(prediction[self._key]) if prediction else None

    def log_ground_truth(self, truth) -> str:
        return str(truth[self._key])


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
            "What is the maximum number of points you could get if you received ALL available "
            "packages (all 3 food, all 3 water, and all 3 firewood)?",
            'Inside <answer> put a JSON object with one key: "max_points" whose value is a '
            "plain integer (e.g. {\"max_points\": 36}). "
            "Do not write math expressions — evaluate the total and write the final number.",
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

    def log_prediction(self, prediction) -> str | None:
        return str(prediction[self._key]) if prediction else None

    def log_ground_truth(self, truth) -> str:
        return str(truth[self._key])


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

    def log_prediction(self, prediction) -> dict | None:
        return prediction  # already {food: str, water: str, firewood: str}

    def log_ground_truth(self, truth) -> dict:
        return truth  # already {food: str, water: str, firewood: str}


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
            "Which single item (food, water, or firewood) is worth the MOST points per "
            "package to you? That is your highest priority issue.",
            'Inside <answer> put a JSON object with one key: "item" and value exactly one of '
            '"food", "water", or "firewood" — whichever has the highest points per package.',
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

    def log_prediction(self, prediction) -> str | None:
        return prediction.get(self._key) if prediction else None

    def log_ground_truth(self, truth) -> str:
        return truth.get(self._key, "")


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
            "Which single item (food, water, or firewood) is worth the FEWEST points per "
            "package to you? That is your lowest priority issue.",
            'Inside <answer> put a JSON object with one key: "item" and value exactly one of '
            '"food", "water", or "firewood" — whichever has the fewest points per package.',
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

    def log_prediction(self, prediction) -> str | None:
        return prediction.get(self._key) if prediction else None

    def log_ground_truth(self, truth) -> str:
        return truth.get(self._key, "")


def _mid_dialogue_cut(chat_logs: list[dict]) -> list[dict]:
    """Return the first half of real (non-meta) turns for mid-stage prompts."""
    real = [t for t in chat_logs if t["text"] not in _META_TURNS]
    cut = max(1, len(real) // 2)
    return real[:cut]


_PRIORITY_OUTPUT_SPEC = (
    'Inside <answer> put a JSON object with one key: "item" and value exactly one of '
    '"food", "water", or "firewood".'
)

_STRATEGY_LABEL_LIST = ", ".join(sorted(STRATEGY_LABELS))

_STRATEGY_OUTPUT_SPEC = (
    "Inside <answer> put a JSON array of strategy labels that apply to the target "
    f"utterance. Choose ONLY from: [{_STRATEGY_LABEL_LIST}]. "
    'Example: <answer>["self-need", "coordination"]</answer>.'
)


def _macro_f1(pred_sets: list[frozenset], gt_sets: list[frozenset]) -> float:
    """Compute Macro-F1 over all strategy labels (same method as reference eval_metric.py)."""
    all_labels = STRATEGY_LABELS
    f1_scores: list[float] = []
    for label in all_labels:
        tp = sum(1 for p, g in zip(pred_sets, gt_sets) if label in p and label in g)
        fp = sum(1 for p, g in zip(pred_sets, gt_sets) if label in p and label not in g)
        fn = sum(1 for p, g in zip(pred_sets, gt_sets) if label not in p and label in g)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


class StrategyCA(BaseTaskHandler):
    """
    mid_strategy_ca
    Classify which negotiation strategies appear in a given utterance.

    Per paper (SysEval):
    - Metric: Macro-F1 across the 9 strategy labels (not exact-set accuracy).
    - Non-strategic turns are SKIPPED — they carry no useful signal.
    - Label normalisation: "promote-coordination" → "coordination",
      "showing-empathy" → "empathy" (matches reference possible_outputs).
    """

    task_id = "mid_strategy_ca"

    # build_prompt is unused externally; evaluate() drives its own loop
    def build_prompt(self, instance: dict, agent: str) -> str:
        raise NotImplementedError("Call evaluate() directly — StrategyCA fans out per turn.")

    def _build_turn_prompt(
        self, instance: dict, agent: str, turn_idx: int, context_turns: list[dict]
    ) -> str:
        target_text = sanitize_unicode(instance["chat_logs"][turn_idx]["text"])
        question = (
            "What negotiation strategies are used in the following target utterance?\n"
            f'Target utterance: "{target_text}"'
        )
        return build_mid_prompt(instance, agent, context_turns, question, _STRATEGY_OUTPUT_SPEC)

    def ground_truth(self, instance: dict, agent: str) -> frozenset[str]:
        raise NotImplementedError("Use _turn_ground_truth() inside evaluate().")

    def _normalise_label(self, raw: str) -> str:
        return STRATEGY_LABEL_MAP.get(raw.strip(), raw.strip())

    def _turn_ground_truth(self, annotation_label: str) -> frozenset[str]:
        return frozenset(
            self._normalise_label(s)
            for s in annotation_label.split(",")
            if self._normalise_label(s) in STRATEGY_LABELS
        )

    def parse_output(self, text: str) -> frozenset[str] | None:
        tagged = self.extract_answer_tag(text)
        if not tagged:
            return None
        try:
            raw = json.loads(tagged)
        except Exception:
            return None
        if not isinstance(raw, list):
            return None
        labels = frozenset(
            self._normalise_label(str(s))
            for s in raw
            if self._normalise_label(str(s)) in STRATEGY_LABELS
        )
        # Empty prediction is valid (the model says no strategy applies)
        return labels

    def score(self, prediction: frozenset | None, truth: frozenset) -> float:
        """Exact-set match per turn — used for inline logging only, not the main metric."""
        return 1.0 if prediction == truth else 0.0

    def log_prediction(self, prediction) -> list | None:
        return sorted(prediction) if prediction is not None else None

    def log_ground_truth(self, truth) -> list:
        return sorted(truth)

    def evaluate(
        self,
        dataset_handler,
        model,
        n: int | None = None,
        agent: str = "mturk_agent_1",
        run_dir: Path | None = None,
    ) -> dict:
        instances = dataset_handler.get_instances(n)

        all_prompts: list[str] = []
        all_gts: list[frozenset] = []

        for instance in instances:
            chat_logs = instance["chat_logs"]
            annotations = instance["annotations"]
            for turn_idx, (log_turn, ann) in enumerate(zip(chat_logs, annotations)):
                if log_turn["text"] in _META_TURNS:
                    continue
                # Skip non-strategic turns (reference does the same: "if 'non-strategic' in annotations[index][-1]: continue")
                if "non-strategic" in ann[1]:
                    continue
                context = chat_logs[:turn_idx]
                prompt = self._build_turn_prompt(instance, agent, turn_idx, context)
                gt = self._turn_ground_truth(ann[1])
                all_prompts.append(prompt)
                all_gts.append(gt)

        uniq_prompts, uniq_gts = self._remove_duplicates(all_prompts, all_gts)

        outputs_dict: dict[str, str] = {}
        for prompt in tqdm(uniq_prompts, desc=self.task_id, unit="prompt"):
            outputs_dict[prompt] = model.generate(prompt)

        final_prompts: list[str] = []
        final_preds_raw: list[frozenset] = []
        final_gts_raw: list[frozenset] = []
        final_preds_log: list = []
        final_gts_log: list = []
        exact_scores: list[float] = []

        for prompt, gt in zip(uniq_prompts, uniq_gts):
            raw = outputs_dict[prompt]
            prediction = self.parse_output(raw)
            if prediction is not None:
                final_prompts.append(prompt)
                final_preds_raw.append(prediction)
                final_gts_raw.append(gt)
                exact_scores.append(self.score(prediction, gt))
                final_preds_log.append(self.log_prediction(prediction))
                final_gts_log.append(self.log_ground_truth(gt))

        exact_accuracy = sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
        macro_f1 = _macro_f1(final_preds_raw, final_gts_raw) if final_preds_raw else 0.0

        stats = {
            "total": len(all_prompts),
            "unique": len(uniq_prompts),
            "valid": len(final_prompts),
            "accuracy": round(exact_accuracy, 4),
            "macro_f1": round(macro_f1, 4),
        }

        log_data = {
            "stats": stats,
            "ground truth": final_gts_log,
            "predictions": final_preds_log,
            "prompts": final_prompts,
            "outputs_dict": outputs_dict,
        }

        if run_dir is not None:
            task_log_dir = run_dir / "ca" / self.task_id
            task_log_dir.mkdir(parents=True, exist_ok=True)
            model_id = getattr(model, "model_id", "model")
            fname = f"ca_{model_id}_{self.task_id}_{len(all_prompts)}.json"
            with (task_log_dir / fname).open("w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

        logger.info("%s: macro_f1=%.4f  exact=%.4f", self.task_id, macro_f1, exact_accuracy)
        return {"task": self.task_id, "n": len(all_prompts), "accuracy": macro_f1, "macro_f1": macro_f1}


class MidHighPriorityCA(BaseTaskHandler):
    """
    mid_ask_high_priority_ca
    Same as sta_ask_high_priority_ca but the prompt includes the first half of the dialogue.
    Tests whether the model stays anchored to its own point values under dialogue pressure.
    """

    task_id = "mid_ask_high_priority_ca"
    _key = "item"
    _options = ("food", "water", "firewood")

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_mid_prompt(
            instance,
            agent,
            _mid_dialogue_cut(instance["chat_logs"]),
            "Which single item (food, water, or firewood) is worth the MOST points per "
            "package to you? That is your highest priority issue.",
            _PRIORITY_OUTPUT_SPEC,
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

    def log_prediction(self, prediction) -> str | None:
        return prediction.get(self._key) if prediction else None

    def log_ground_truth(self, truth) -> str:
        return truth.get(self._key, "")


class MidLowPriorityCA(BaseTaskHandler):
    """
    mid_ask_low_priority_ca
    Same as sta_ask_low_priority_ca but with partial dialogue history.
    """

    task_id = "mid_ask_low_priority_ca"
    _key = "item"
    _options = ("food", "water", "firewood")

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_mid_prompt(
            instance,
            agent,
            _mid_dialogue_cut(instance["chat_logs"]),
            "Which single item (food, water, or firewood) is worth the FEWEST points per "
            "package to you? That is your lowest priority issue.",
            _PRIORITY_OUTPUT_SPEC,
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

    def log_prediction(self, prediction) -> str | None:
        return prediction.get(self._key) if prediction else None

    def log_ground_truth(self, truth) -> str:
        return truth.get(self._key, "")


class MidPartnerHighPriorityCA(BaseTaskHandler):
    """
    mid_partner_ask_high_priority_ca
    Infer the PARTNER's highest-priority item from partial dialogue.
    The agent's own point values are shown (but not the partner's).
    Ground truth: partner's value2issue["High"].
    """

    task_id = "mid_partner_ask_high_priority_ca"
    _key = "item"
    _options = ("food", "water", "firewood")

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_mid_prompt(
            instance,
            agent,
            _mid_dialogue_cut(instance["chat_logs"]),
            "Based on the dialogue so far, which single item (food, water, or firewood) "
            "do you think is your PARTNER's highest priority?",
            _PRIORITY_OUTPUT_SPEC,
        )

    def ground_truth(self, instance: dict, agent: str) -> dict[str, str]:
        partner = get_partner(agent)
        item = instance["participant_info"][partner]["value2issue"]["High"].lower()
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

    def log_prediction(self, prediction) -> str | None:
        return prediction.get(self._key) if prediction else None

    def log_ground_truth(self, truth) -> str:
        return truth.get(self._key, "")


class MidPartnerLowPriorityCA(BaseTaskHandler):
    """
    mid_partner_ask_low_priority_ca
    Infer the PARTNER's lowest-priority item from partial dialogue.
    """

    task_id = "mid_partner_ask_low_priority_ca"
    _key = "item"
    _options = ("food", "water", "firewood")

    def build_prompt(self, instance: dict, agent: str) -> str:
        return build_mid_prompt(
            instance,
            agent,
            _mid_dialogue_cut(instance["chat_logs"]),
            "Based on the dialogue so far, which single item (food, water, or firewood) "
            "do you think is your PARTNER's lowest priority?",
            _PRIORITY_OUTPUT_SPEC,
        )

    def ground_truth(self, instance: dict, agent: str) -> dict[str, str]:
        partner = get_partner(agent)
        item = instance["participant_info"][partner]["value2issue"]["Low"].lower()
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

    def log_prediction(self, prediction) -> str | None:
        return prediction.get(self._key) if prediction else None

    def log_ground_truth(self, truth) -> str:
        return truth.get(self._key, "")


CA_TASK_REGISTRY: dict[str, type[BaseTaskHandler]] = {
    "sta_total_item_count_ca": TotalItemCountCA,
    "sta_max_points_ca": MaxPointsCA,
    "sta_ask_point_values_ca": PointValuesCA,
    "sta_ask_high_priority_ca": HighPriorityCA,
    "sta_ask_low_priority_ca": LowPriorityCA,
    "mid_strategy_ca": StrategyCA,
    "mid_ask_high_priority_ca": MidHighPriorityCA,
    "mid_ask_low_priority_ca": MidLowPriorityCA,
    "mid_partner_ask_high_priority_ca": MidPartnerHighPriorityCA,
    "mid_partner_ask_low_priority_ca": MidPartnerLowPriorityCA,
}
