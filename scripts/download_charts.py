"""Fetch metrics from Trackio and generate chart PNGs.

Writes PNGs and ``charts_data.json`` under ``raw/charts/`` (gitignored).

Usage:
    uv run scripts/download_charts.py
    uv run scripts/download_charts.py --run grpo-annotated
    uv run scripts/download_charts.py --data-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"


PROJECT = "negotiation-agent"
SPACE = "dhruvb26/negotiation-agent"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "raw" / "charts"
MAX_WORKERS = 12


GRPO_METRICS = {
    "entropy",
    "epoch",
    "frac_reward_zero_std",
    "grad_norm",
    "kl",
    "learning_rate",
    "loss",
    "num_tokens",
    "reward",
    "reward_std",
    "step_time",
}
GRPO_PREFIXES = ("clip_ratio/", "completions/", "rewards/")

SKIP = {
    "mean_token_accuracy",
    "total_flos",
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
}

RUN_GROUPS = {
    "grpo-annotated": {
        "runs": ["grpo-annotated-0411-0808", "grpo-annotated-0411-1746"],
        "extra_prefixes": ("turn/",),
        "merge": True,
    },
    "grpo-selfplay": {
        "runs": ["grpo-self_play-0413-1759"],
        "extra_skip": {"completions/clipped_ratio"},
        "skip_prefixes": ("turn/",),
    },
}

THEME = {
    "bg": "#FAFAF8",
    "text": "#1a1a1a",
    "grid": 0.2,
    "spine": "#ddd",
    "legend_bg": "#ffffff",
    "legend_edge": "#e0e0e0",
    "colors": ["#eb5600", "#d44e00", "#ff7a26", "#b34300", "#993a00"],
}


def _keep_metric(name: str, group: dict) -> bool:
    if name in SKIP or name in group.get("extra_skip", set()):
        return False
    if any(name.startswith(p) for p in group.get("skip_prefixes", ())):
        return False
    if name in GRPO_METRICS or any(name.startswith(p) for p in GRPO_PREFIXES):
        return True
    return any(name.startswith(p) for p in group.get("extra_prefixes", ()))


def trackio_cmd(*args: str) -> dict:
    result = subprocess.run(
        ["trackio", *args, "--space", SPACE, "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def list_metrics(run: str) -> list[str]:
    return trackio_cmd("list", "metrics", "--project", PROJECT, "--run", run).get(
        "metrics", []
    )


def get_metric(run: str, metric: str) -> dict:
    data = trackio_cmd(
        "get", "metric", "--project", PROJECT, "--run", run, "--metric", metric
    )
    steps, values = [], []
    for entry in data.get("values", []):
        steps.append(entry["step"])
        values.append(entry["value"])
    return {"run": run, "metric": metric, "steps": steps, "values": values}


def smooth(values: list[float], weight: float = 0.9) -> np.ndarray:
    arr = np.array(values, dtype=float)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = weight * out[i - 1] + (1 - weight) * arr[i]
    return out


def merge_runs(series_list: list[dict]) -> list[dict]:
    """Merge multiple runs of the same metric into a single continuous series.

    When runs overlap (e.g. run1 ends at step 900, run2 starts at 810),
    prefer the later run's values in the overlap region.
    """
    if len(series_list) <= 1:
        return series_list

    # Sort by first step
    series_list.sort(key=lambda s: s["steps"][0] if s["steps"] else 0)

    merged_steps: list[int] = []
    merged_values: list[float] = []
    step_to_value: dict[int, float] = {}

    # Earlier runs first, later runs overwrite in overlap
    for s in series_list:
        for step, val in zip(s["steps"], s["values"]):
            step_to_value[step] = val

    for step in sorted(step_to_value):
        merged_steps.append(step)
        merged_values.append(step_to_value[step])

    return [
        {
            "run": " + ".join(s["run"] for s in series_list),
            "metric": series_list[0]["metric"],
            "steps": merged_steps,
            "values": merged_values,
        }
    ]


def plot_metric(series_list: list[dict], out_path: Path) -> None:
    t = THEME
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])

    metric_name = series_list[0]["metric"]

    for i, s in enumerate(series_list):
        color = t["colors"][i % len(t["colors"])]
        ax.plot(s["steps"], s["values"], color=color, alpha=0.3, linewidth=0.8)
        ax.plot(
            s["steps"], smooth(s["values"]), color=color, linewidth=2, label=s["run"]
        )

    ax.set_xlabel("step", color=t["text"])
    ax.set_ylabel(metric_name.split("/")[-1], color=t["text"])
    ax.set_title(metric_name, color=t["text"], fontsize=13, fontweight="bold")
    ax.tick_params(colors=t["text"])
    ax.grid(True, alpha=t["grid"])
    for spine in ax.spines.values():
        spine.set_color(t["spine"])
    ax.legend(
        facecolor=t["legend_bg"],
        edgecolor=t["legend_edge"],
        labelcolor=t["text"],
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def fetch_all(runs: list[str], metrics: set[str]) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {m: [] for m in metrics}
    tasks = [(run, metric) for metric in metrics for run in runs]
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(get_metric, r, m): (r, m) for r, m in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            series = future.result()
            if series["steps"]:
                results[series["metric"]].append(series)
            print(f"\r  fetching {done}/{total}", end="", flush=True)
    print()
    return results


def run_group(name: str, group: dict, *, data_only: bool) -> dict:
    runs = group["runs"]
    print(f"\n{name}")

    all_metrics: set[str] = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(list_metrics, r): r for r in runs}
        for future in as_completed(futures):
            all_metrics.update(m for m in future.result() if _keep_metric(m, group))

    print(f"  {len(all_metrics)} metrics x {len(runs)} runs")
    by_metric = fetch_all(runs, all_metrics)

    out_dir = OUTPUT_ROOT / name
    saved = 0
    group_data: dict = {}
    for metric in sorted(all_metrics):
        series_list = by_metric[metric]
        if not series_list:
            continue
        series_list.sort(key=lambda s: runs.index(s["run"]) if s["run"] in runs else 0)
        if len(series_list) > 1 and group.get("merge", False):
            series_list = merge_runs(series_list)
        if not data_only:
            fname = metric.replace("/", "-").replace("_", "-") + ".png"
            plot_metric(series_list, out_dir / fname)
        group_data[metric] = [
            {"run": s["run"], "steps": s["steps"], "values": s["values"]}
            for s in series_list
        ]
        saved += 1

    print(f"  {saved} {'metrics fetched' if data_only else 'charts saved'}")
    return group_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Only process groups containing this run name")
    parser.add_argument("--data-only", action="store_true", help="Skip chart PNGs")
    args = parser.parse_args()

    groups = RUN_GROUPS
    if args.run:
        groups = {
            k: v for k, v in RUN_GROUPS.items() if any(args.run in r for r in v["runs"])
        }
        if not groups:
            print(f"no group found for '{args.run}'")
            sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    out_json = OUTPUT_ROOT / "charts_data.json"
    all_data: dict = json.loads(out_json.read_text()) if out_json.exists() else {}

    for group_name, group in groups.items():
        all_data[group_name] = run_group(group_name, group, data_only=args.data_only)
    out_json.write_text(json.dumps(all_data, indent=2))
    print(f"\nraw data saved to {out_json.relative_to(REPO_ROOT)}")

    print("done")


if __name__ == "__main__":
    main()
