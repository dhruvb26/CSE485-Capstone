"""
Main entry point for the SysEval negotiation LLM evaluation framework.

Usage:
    python -m eval                              # uses eval/config.yaml
    python -m eval --config path/to/config.yaml # custom config
    python -m eval --evaluate-only              # just score existing logs
    python -m eval --list-tasks                 # show available tasks
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace

from eval.registry import (
    SUPPORTED_CONFIGS, CLS_NAME2PATHS, TASK_TO_DATASET,
    get_tasks_for_dataset, get_all_task_names,
)
from eval.utils import dynamic_import, get_output_path
from eval.metrics import EvaluationMetrics


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_tasks(task_list):
    """Expand shorthand task specs like 'all', 'all_dnd', etc."""
    resolved = []
    for t in task_list:
        if t == "all":
            resolved.extend(get_all_task_names())
        elif t.startswith("all_"):
            dataset = t[4:]
            resolved.extend(get_tasks_for_dataset(dataset))
        else:
            resolved.append(t)
    return list(dict.fromkeys(resolved))


def build_args(config, model_cfg):
    """Build a SimpleNamespace mimicking the original argparse args."""
    args = SimpleNamespace(
        num_instances=config.get("num_instances", 200),
        max_num_instances=config.get("max_num_instances", 200),
        use_cot=config.get("use_cot", False),
        num_multishot=config.get("num_multishot", 0),
        num_prior_utts=config.get("num_prior_utts", 0),
        num_utts_partial_dial=config.get("num_utts_partial_dial", -1),
        storage_dir=config.get("storage_dir", "./logs/eval"),
        openai_model_str=model_cfg.get("model_str", "gpt-4o-mini-2024-07-18"),
        hf_model_str=model_cfg.get("model_str", ""),
        # local_model fields
        model_path=model_cfg.get("model_path", ""),
        base_model=model_cfg.get("base_model", ""),
        max_new_tokens=model_cfg.get("max_new_tokens", 256),
        token_limit=model_cfg.get("token_limit", 4096),
        label=model_cfg.get("label", ""),
        # vllm_model fields
        base_url=model_cfg.get("base_url", "http://localhost:8000/v1"),
        max_tokens=model_cfg.get("max_tokens", 512),
    )
    return args


def validate_config(config, tasks, model_cfgs):
    """Validate that all (dataset, model, task) combos are supported."""
    errors = []
    for model_cfg in model_cfgs:
        model_type = model_cfg["type"]
        for task_name in tasks:
            dataset_name = TASK_TO_DATASET.get(task_name)
            if dataset_name is None:
                errors.append(f"Unknown task: {task_name}")
                continue
            if (dataset_name, model_type, task_name) not in SUPPORTED_CONFIGS:
                errors.append(
                    f"Unsupported config: dataset={dataset_name}, model={model_type}, task={task_name}"
                )
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


def _resolve_model_display_name(model_type, model_cfg):
    """Get a human-readable model name for logs and file paths."""
    if model_type == "local_model":
        label = model_cfg.get("label", "")
        if label:
            return label
        path = model_cfg.get("model_path", "local")
        return os.path.basename(path.rstrip("/"))
    return model_cfg.get("model_str", model_type)


def run_evaluation(config):
    """Run full evaluation: model inference + scoring."""
    model_cfgs = config.get("models", [])
    tasks = resolve_tasks(config.get("tasks", []))
    storage_dir = config.get("storage_dir", "./logs/eval")

    if not tasks:
        print("No tasks specified in config.", file=sys.stderr)
        sys.exit(1)
    if not model_cfgs:
        print("No models specified in config.", file=sys.stderr)
        sys.exit(1)

    validate_config(config, tasks, model_cfgs)

    os.makedirs(storage_dir, exist_ok=True)

    # Group tasks by dataset to avoid loading the same dataset multiple times
    dataset_tasks = {}
    for task_name in tasks:
        ds = TASK_TO_DATASET[task_name]
        dataset_tasks.setdefault(ds, []).append(task_name)

    for model_cfg in model_cfgs:
        model_type = model_cfg["type"]
        model_display = _resolve_model_display_name(model_type, model_cfg)
        print(f"Model: {model_type} ({model_display})")

        args = build_args(config, model_cfg)

        # Initialize model once per model config
        model_cls = dynamic_import(CLS_NAME2PATHS["models"][model_type])
        model_handler = model_cls(model_type, args)

        for dataset_name, task_names in dataset_tasks.items():
            # Check this model supports these tasks
            valid_tasks = [
                t for t in task_names
                if (dataset_name, model_type, t) in SUPPORTED_CONFIGS
            ]
            if not valid_tasks:
                continue

            print(f"\n  Dataset: {dataset_name}")
            print(f"  {'-'*50}")

            # Initialize dataset once per dataset
            dataset_cls = dynamic_import(CLS_NAME2PATHS["datasets"][dataset_name])
            dataset_handler = dataset_cls(dataset_name, args)

            for tix, task_name in enumerate(valid_tasks):
                # Check if output already exists
                mname = _resolve_model_display_name(model_type, model_cfg)
                if model_type != "open_ai":
                    mname = mname.replace("/", "_")
                out_path = get_output_path(
                    storage_dir, dataset_name, mname,
                    task_name, args.num_instances, args=args
                )
                if os.path.exists(out_path):
                    print(f"    [{tix+1}/{len(valid_tasks)}] {task_name} -- already exists, skipping")
                    continue

                print(f"    [{tix+1}/{len(valid_tasks)}] {task_name} -- running...")

                task_cls = dynamic_import(CLS_NAME2PATHS["tasks"][task_name])
                task_handler = task_cls(task_name, args)
                task_handler.evaluate(dataset_handler, model_handler)

                print(f"    [{tix+1}/{len(valid_tasks)}] {task_name} -- done")

    print(f"\n{'='*70}")
    print("Inference complete. Logs saved to:", storage_dir)
    print(f"{'='*70}")

    # Score the logs
    score_logs(storage_dir)


def score_logs(storage_dir):
    """Score all log files in the storage directory."""
    import json

    log_dir = Path(storage_dir)
    if not log_dir.is_dir():
        print(f"No logs directory found at {log_dir}", file=sys.stderr)
        return

    task_to_metric = EvaluationMetrics.get_eval_method_by_task()
    evaluator = EvaluationMetrics()

    log_files = sorted(log_dir.rglob("*.json"))
    if not log_files:
        print("No log files found to evaluate.", file=sys.stderr)
        return

    known_tasks = set(task_to_metric.keys())

    results = []
    for path in log_files:
        stem = path.stem
        task = _parse_task(stem, known_tasks)
        if not task:
            continue

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        gt = data.get("ground truth")
        preds = data.get("predictions")
        if gt is None or preds is None or len(gt) != len(preds):
            continue

        metric = task_to_metric.get(task)
        if not metric:
            continue

        preds, gt = _ensure_format(preds, gt, metric)

        try:
            score = evaluator.compute_metric(preds=preds, gt=gt, metric=metric, quiet=True)
        except Exception:
            continue

        # Parse model from filename
        for suffix in (f"_{task}_200", f"_{task}_10"):
            if stem.endswith(suffix):
                rest = stem[:-len(suffix)]
                break
        else:
            rest = stem
        parts = rest.split("_", 1)
        dataset = parts[0] if parts else ""
        model = parts[1] if len(parts) > 1 else ""

        results.append({
            "dataset": dataset,
            "model": model,
            "task": task,
            "metric": metric,
            "score": score,
            "n": len(gt),
        })

    if not results:
        print("No evaluable logs found.")
        return

    by_model = {}
    for r in results:
        key = (r["dataset"], r["model"])
        by_model.setdefault(key, []).append(r)

    for (dataset, model), rows in sorted(by_model.items()):
        print(f"\n  {dataset} | {model}")
        print(f"  {'-'*60}")
        for r in sorted(rows, key=lambda x: x["task"]):
            score_str = f"{r['score']:.4f}" if isinstance(r["score"], (int, float)) else str(r["score"])
            print(f"    {r['task']:<45} {r['metric']:<22} {score_str}  (n={r['n']})")


def _parse_task(filename_stem, known_tasks):
    for task in sorted(known_tasks, key=len, reverse=True):
        if filename_stem.endswith("_" + task + "_10") or filename_stem.endswith("_" + task + "_200"):
            return task
    return None


def _flatten_labels(x):
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(_flatten_labels(item))
        return out
    return [str(x)]


def _ensure_format(preds, gt, metric):
    if metric == "elementwise_accuracy":
        preds = [p if isinstance(p, dict) else {} for p in preds]
        gt = [g if isinstance(g, dict) else {} for g in gt]
        return preds, gt
    if metric == "f1_per_class":
        preds = [_flatten_labels(p) for p in preds]
        gt = [_flatten_labels(g) for g in gt]
        return preds, gt
    return preds, gt


def list_available_tasks():
    """Print all available tasks grouped by dataset."""
    print("\nAvailable evaluation tasks:")
    print("=" * 60)

    datasets = {}
    for (ds, model, task) in SUPPORTED_CONFIGS:
        datasets.setdefault(ds, set()).add(task)

    for ds in sorted(datasets.keys()):
        print(f"\n  {ds}:")
        for task in sorted(datasets[ds]):
            print(f"    - {task}")


def main():
    parser = argparse.ArgumentParser(
        description="SysEval: Systematic Evaluation of LLM Negotiation Capabilities"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Only score existing logs (skip model inference)",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List all available evaluation tasks and exit",
    )
    args = parser.parse_args()

    if args.list_tasks:
        list_available_tasks()
        return

    config = load_config(args.config)

    if args.evaluate_only or config.get("evaluate_only", False):
        storage_dir = config.get("storage_dir", "./logs/eval")
        score_logs(storage_dir)
    else:
        run_evaluation(config)


if __name__ == "__main__":
    main()
