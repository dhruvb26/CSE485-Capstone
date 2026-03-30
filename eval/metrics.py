"""
Evaluation metrics for negotiation tasks.

Implements the metrics from:
  "Are LLMs Effective Negotiators?" (EMNLP 2024 Findings)

Metrics:
  - accuracy: comprehension / partner modeling / regression tasks
  - elementwise_accuracy: proposal tasks (dict outputs)
  - f1_per_class: annotation tasks (dialog acts, strategies)
  - bleu_rouge: generation tasks
"""

import os
import numpy as np
from typing import List
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix

from eval.utils import json_loader


class EvaluationMetrics:
    def __init__(self):
        pass

    def compute_metric(self, preds: List, gt: List, metric: str, quiet: bool = False):
        if metric == "accuracy":
            if not quiet:
                print(classification_report(gt, preds, zero_division=0))
                print("Count information of ground-truth:", Counter(gt))
                print("Count information of predictions:", Counter(preds))
            return accuracy_score(gt, preds)
        elif metric == "f1_per_class":
            return self.f1_per_class_score(gt, preds)
        elif metric == "elementwise_accuracy":
            return self.elementwise_accuracy_score(gt, preds)
        elif metric == "bleu_rouge":
            return self.bleu_rouge_score(gt, preds)
        else:
            raise NotImplementedError(f"{metric} is not implemented")

    def bleu_rouge_score(self, gt: List, preds: List):
        try:
            import evaluate
        except ImportError:
            raise ImportError(
                "The 'evaluate' library is required for BLEU/ROUGE scoring. "
                "Install it with: pip install evaluate"
            )
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")

        preds_clean = [str(p).strip() or "." for p in preds]
        gt_clean = [str(g).strip() or "." for g in gt]

        rouge_result = rouge_metric.compute(predictions=preds_clean, references=gt_clean)
        rouge_result = {k: round(v, 2) for k, v in rouge_result.items()}

        bleu_result = bleu_metric.compute(predictions=preds_clean, references=gt_clean, max_order=1)
        bleu_result = {k: v for k, v in bleu_result.items()}

        return f"BLEU({bleu_result['bleu']})/Rouge({rouge_result['rouge1']})"

    def f1_per_class_score(self, gt: List, preds: List):
        mlb = MultiLabelBinarizer()
        mlb.fit(gt + preds)
        label_indices = {index: label for index, label in enumerate(mlb.classes_)}
        ground_truth_binary = mlb.transform(gt)
        predictions_binary = mlb.transform(preds)
        confusion_matrix = multilabel_confusion_matrix(
            ground_truth_binary, predictions_binary
        )

        label_f1_scores = {}
        for index in range(confusion_matrix.shape[0]):
            _tn = confusion_matrix[index, 0, 0]
            fp = confusion_matrix[index, 0, 1]
            fn = confusion_matrix[index, 1, 0]
            tp = confusion_matrix[index, 1, 1]
            label = label_indices[index]
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = (2 * precision * recall) / (precision + recall + 1e-10)
            label_f1_scores[label] = f1

        average_f1 = sum(label_f1_scores.values()) / len(label_f1_scores)
        return average_f1

    def elementwise_accuracy_score(self, gt: List, preds: List):
        assert len(gt) == len(preds), "length of ground truth and prediction should be same"
        return np.mean(
            [
                str(val) == str(_pred.get(item, "None"))
                for _gt, _pred in zip(gt, preds)
                for item, val in _gt.items()
            ]
        )

    @staticmethod
    def get_eval_method_by_task():
        """Return a dict mapping task_name -> metric_name."""
        tasktype_path = os.path.join(os.path.dirname(__file__), "tasks", "TASKTYPE.json")
        task_class = json_loader(tasktype_path)["T5"]

        evaluation_method = {}

        # Proposal tasks -> elementwise_accuracy
        if "multi_outputs" in task_class and "proposal" in task_class["multi_outputs"]:
            for t in task_class["multi_outputs"]["proposal"]:
                evaluation_method[t] = "elementwise_accuracy"

        # Strategy tasks -> f1_per_class
        if "multi_outputs" in task_class and "strategy" in task_class["multi_outputs"]:
            for t in task_class["multi_outputs"]["strategy"]:
                evaluation_method[t] = "f1_per_class"

        # Dialog act (multi-output) -> f1_per_class
        if "multi_outputs" in task_class and "dialog_act" in task_class["multi_outputs"]:
            for t in task_class["multi_outputs"]["dialog_act"]:
                evaluation_method[t] = "f1_per_class"

        # Classification tasks -> accuracy
        if "classification" in task_class:
            for subcategory in task_class["classification"].values():
                for t in subcategory:
                    evaluation_method[t] = "accuracy"

        # Dialog act (classification-level) -> f1_per_class for DND/JI, accuracy for CRA
        if "classification" in task_class and "dialog_act" in task_class["classification"]:
            for t in task_class["classification"]["dialog_act"]:
                if "ji" in t or "dnd" in t:
                    evaluation_method[t] = "f1_per_class"

        # Regression tasks -> accuracy
        if "regression" in task_class:
            for subcategory in task_class["regression"].values():
                for t in subcategory:
                    evaluation_method[t] = "accuracy"

        # Generation tasks -> bleu_rouge
        if "generation" in task_class:
            for subcategory in task_class["generation"].values():
                for t in subcategory:
                    evaluation_method[t] = "bleu_rouge"

        # Start-stage tasks not in TASKTYPE.json
        evaluation_method.setdefault("sta_ask_point_values_dnd", "elementwise_accuracy")
        evaluation_method.setdefault("sta_ask_point_values_ca", "elementwise_accuracy")
        for t in [
            "sta_total_item_count_dnd", "sta_max_points_dnd",
            "sta_total_item_count_ca", "sta_max_points_ca",
            "sta_ask_high_priority_ca", "sta_ask_low_priority_ca",
            "sta_ask_high_priority_ji_w", "sta_ask_low_priority_ji_w",
            "mid_ask_high_priority_ca", "mid_ask_low_priority_ca",
            "mid_ask_high_priority_ji_w", "mid_ask_low_priority_ji_w",
        ]:
            evaluation_method.setdefault(t, "accuracy")

        return evaluation_method
