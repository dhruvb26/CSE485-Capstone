from __future__ import annotations

import json
from pathlib import Path


def task_accuracy(log_path: str | Path) -> dict:
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

    if isinstance(gt_list[0], list):
        macro_f1 = stats.get("macro_f1")
        return {
            **base,
            "accuracy": macro_f1 if macro_f1 is not None else 0.0,
            "macro_f1": macro_f1,
            "exact_match": stats.get("accuracy"),
        }

    correct = sum(p == g for p, g in zip(pred_list, gt_list))
    n = len(gt_list)
    return {**base, "accuracy": correct / n, "correct": correct}


def run_accuracy(run_dir: str | Path) -> dict[str, dict]:
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
