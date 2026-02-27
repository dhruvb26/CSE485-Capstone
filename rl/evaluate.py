"""
Offline evaluation utilities for SysEval-format log files.

Log files are written by BaseTaskHandler.evaluate() in the format:
    {
        "stats":      {"total": N, "unique": N, "valid": N, "accuracy": 0.0},
        "ground truth": [...],   # aligned with "predictions"
        "predictions":  [...],   # only valid (parsed) entries
        "prompts":      [...],   # corresponding unique prompts
        "outputs_dict": {...}    # unique prompt → raw model output
    }

For dict-valued tasks (sta_ask_point_values_ca), ground truth and predictions
are dicts like {"food": "5", "water": "3", "firewood": "4"}.  Per-key accuracy
is computed in addition to overall (all-keys-match) accuracy.

Usage:
    from rl.evaluate import task_accuracy, run_accuracy

    # single log file
    result = task_accuracy("runs/2026-02-26T.../ca/sta_ask_high_priority_ca/casino_model_task_121.json")

    # entire run directory
    results = run_accuracy("runs/2026-02-26T...")
    for task, r in results.items():
        print(f"{task}: {r['accuracy']:.2%}  ({r['valid']}/{r['total']})")
"""

from __future__ import annotations

import json
from pathlib import Path


def task_accuracy(log_path: str | Path) -> dict:
    """Compute accuracy metrics from a single SysEval-format log file.

    Returns a dict with at minimum:
        accuracy  – correct / valid  (0.0 if valid == 0)
        valid     – number of parseable predictions
        unique    – number of unique prompts evaluated
        total     – total instances (before dedup)

    For dict-valued tasks (point_values) also includes:
        per_key_accuracy  – {key: float} accuracy per key
    """
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)

    stats = data.get("stats", {})
    gt_list: list = data.get("ground truth", [])
    pred_list: list = data.get("predictions", [])

    base = {
        "valid": stats.get("valid", len(pred_list)),
        "unique": stats.get("unique", len(pred_list)),
        "total": stats.get("total", len(pred_list)),
    }

    if not pred_list:
        return {**base, "accuracy": 0.0}

    # Dict-valued task (e.g. sta_ask_point_values_ca)
    if isinstance(gt_list[0], dict):
        keys = list(gt_list[0].keys())
        per_key: dict[str, int] = {k: 0 for k in keys}
        overall_correct = 0
        n = len(gt_list)
        for gt, pred in zip(gt_list, pred_list):
            if not isinstance(pred, dict):
                continue
            all_match = True
            for k in keys:
                if pred.get(k) == gt.get(k):
                    per_key[k] += 1
                else:
                    all_match = False
            if all_match:
                overall_correct += 1
        return {
            **base,
            "accuracy": overall_correct / n,
            "correct": overall_correct,
            "per_key_accuracy": {k: per_key[k] / n for k in keys},
        }

    # Multi-label task (strategy): ground truth and predictions are lists of labels
    if pred_list and isinstance(gt_list[0], list):
        macro_f1 = stats.get("macro_f1")
        exact_match = stats.get("accuracy")
        return {
            **base,
            "accuracy": macro_f1 if macro_f1 is not None else 0.0,
            "macro_f1": macro_f1,
            "exact_match": exact_match,
        }

    # String-valued task (classification / numeric)
    correct = sum(p == g for p, g in zip(pred_list, gt_list))
    n = len(gt_list)
    return {
        **base,
        "accuracy": correct / n,
        "correct": correct,
    }


def run_accuracy(run_dir: str | Path) -> dict[str, dict]:
    """Compute accuracy for every SysEval-format log file found under run_dir.

    Returns a dict mapping task name (file stem) → accuracy result dict.
    config.json files are skipped.
    """
    run_dir = Path(run_dir)
    results: dict[str, dict] = {}
    for log_file in sorted(run_dir.rglob("*.json")):
        if log_file.name == "config.json":
            continue
        try:
            results[log_file.stem] = task_accuracy(log_file)
        except Exception as exc:
            results[log_file.stem] = {"error": str(exc)}
    return results


def print_run_summary(run_dir: str | Path) -> None:
    """Print a formatted accuracy table for all tasks in a run directory."""
    results = run_accuracy(run_dir)
    if not results:
        print("No log files found.")
        return
    header = f"{'Task':<55} {'Acc':>6}  {'Valid':>5}/{'Uniq':<5} {'Total':>6}"
    print(header)
    print("-" * len(header))
    for name, r in sorted(results.items()):
        if "error" in r:
            print(f"{name:<55}  ERROR: {r['error']}")
            continue
        acc = r.get("accuracy", 0.0)
        valid = r.get("valid", 0)
        unique = r.get("unique", valid)
        total = r.get("total", 0)
        metric_label = "F1 " if r.get("macro_f1") is not None else "Acc"
        print(f"{name:<55} {metric_label}{acc:>6.1%}  {valid:>5}/{unique:<5} {total:>6}")
        if "per_key_accuracy" in r:
            for k, v in r["per_key_accuracy"].items():
                print(f"  {k:<53} {v:>6.1%}")
        if r.get("exact_match") is not None:
            print(f"  {'exact-match':<53} {r['exact_match']:>6.1%}")

if __name__ == "__main__":
    print_run_summary("../runs/2026-02-26T22-15-09.997Z")