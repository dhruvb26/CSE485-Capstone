"""Fetch metrics from the Trackio HF Space and generate chart PNGs.

Usage:
    uv run scripts/download_charts.py
    uv run scripts/download_charts.py --run sft-0328-0839
    uv run scripts/download_charts.py --light
    uv run scripts/download_charts.py --data-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

PROJECT = "negotiation-agent"
SPACE = "dhruvb26/negotiation-agent"
ASSETS = Path(__file__).resolve().parent.parent / "assets"
MAX_WORKERS = 12

FONT_FAMILY = "Arial"
try:
    fm.findfont(fm.FontProperties(family=FONT_FAMILY), fallback_to_default=False)
except ValueError:
    FONT_FAMILY = "Helvetica"
plt.rcParams["font.family"] = FONT_FAMILY

SFT_METRICS = {
    "entropy",
    "epoch",
    "grad_norm",
    "learning_rate",
    "loss",
    "mean_token_accuracy",
    "num_tokens",
}

GRPO_METRICS = {
    "entropy",
    "epoch",
    "grad_norm",
    "kl",
    "learning_rate",
    "loss",
    "num_tokens",
    "reward",
    "reward_std",
    "step_time",
}
GRPO_PREFIXES = {"clip_ratio/", "completions/", "rewards/"}

GRPO_SELFPLAY_SKIP = {"completions/clipped_ratio", "mean_token_accuracy"}
GRPO_SELFPLAY_SKIP_PREFIXES = {"turn/"}

GRPO_ANNOTATED_SKIP = {"mean_token_accuracy"}

GLOBAL_SKIP = {
    "frac_reward_zero_std",
    "total_flos",
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
}


def _grpo_match(name: str) -> bool:
    return name in GRPO_METRICS or any(name.startswith(p) for p in GRPO_PREFIXES)


def _selfplay_keep(name: str) -> bool:
    if name in GLOBAL_SKIP or name in GRPO_SELFPLAY_SKIP:
        return False
    if any(name.startswith(p) for p in GRPO_SELFPLAY_SKIP_PREFIXES):
        return False
    return _grpo_match(name)


def _annotated_keep(name: str) -> bool:
    if name in GLOBAL_SKIP or name in GRPO_ANNOTATED_SKIP:
        return False
    return _grpo_match(name) or name.startswith("turn/")


def _sft_keep(name: str) -> bool:
    return name in SFT_METRICS


RUN_GROUPS: dict[str, dict] = {
    "sft": {
        "runs": ["sft-0328-0839"],
        "keep": _sft_keep,
    },
    "grpo-selfplay": {
        "runs": ["grpo-self_play-0329-2116"],
        "keep": _selfplay_keep,
    },
    "grpo-annotated-combined": {
        "runs": ["grpo-annotated-0328-0853", "grpo-annotated-0328-1711"],
        "keep": _annotated_keep,
    },
}

THEME = {
    "dark": {
        "bg": "#1a1a1a",
        "text": "#e8e8e8",
        "grid": 0.12,
        "spine": "#333",
        "legend_bg": "#252525",
        "legend_edge": "#333",
        "colors": ["#eb5600", "#f4924d", "#c44800", "#ff8533", "#a33c00"],
    },
    "light": {
        "bg": "#FAFAF8",
        "text": "#1a1a1a",
        "grid": 0.2,
        "spine": "#ddd",
        "legend_bg": "#ffffff",
        "legend_edge": "#e0e0e0",
        "colors": ["#eb5600", "#d44e00", "#ff7a26", "#b34300", "#993a00"],
    },
}

theme: dict = THEME["dark"]


@dataclass
class MetricSeries:
    run: str
    metric: str
    steps: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


def trackio_cmd(*args: str) -> dict:
    cmd = ["trackio", *args, "--space", SPACE, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def list_metrics(run: str) -> list[str]:
    return trackio_cmd("list", "metrics", "--project", PROJECT, "--run", run).get(
        "metrics", []
    )


def get_metric(run: str, metric: str) -> MetricSeries:
    data = trackio_cmd(
        "get", "metric", "--project", PROJECT, "--run", run, "--metric", metric
    )
    series = MetricSeries(run=run, metric=metric)
    for entry in data.get("values", []):
        series.steps.append(entry["step"])
        series.values.append(entry["value"])
    return series


def smooth(values: list[float], weight: float = 0.9) -> np.ndarray:
    arr = np.array(values, dtype=float)
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * arr[i]
    return smoothed


def metric_to_filename(metric: str) -> str:
    return metric.replace("/", "-").replace("_", "-") + ".png"


def plot_metric(series_list: list[MetricSeries], out_path: Path) -> None:
    t = theme
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])

    metric_name = series_list[0].metric
    colors = t["colors"]

    for i, s in enumerate(series_list):
        color = colors[i % len(colors)]
        ax.plot(s.steps, s.values, color=color, alpha=0.3, linewidth=0.8)
        ax.plot(s.steps, smooth(s.values), color=color, linewidth=2, label=s.run)

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


def fetch_all_series(
    runs: list[str], metrics: set[str]
) -> dict[str, list[MetricSeries]]:
    results: dict[str, list[MetricSeries]] = {m: [] for m in metrics}
    tasks = [(run, metric) for metric in metrics for run in runs]
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(get_metric, r, m): (r, m) for r, m in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            series = future.result()
            if series.steps:
                results[series.metric].append(series)
            print(f"\r  fetching {done}/{total}", end="", flush=True)

    print()
    return results


def run_group(group_name: str, group: dict, *, data_only: bool = False) -> dict:
    runs = group["runs"]
    keep_fn = group["keep"]

    print(f"\n{group_name}")

    all_metrics: set[str] = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(list_metrics, r): r for r in runs}
        for future in as_completed(futures):
            kept = [m for m in future.result() if keep_fn(m)]
            all_metrics.update(kept)

    print(f"  {len(all_metrics)} metrics x {len(runs)} runs")
    by_metric = fetch_all_series(runs, all_metrics)

    out_dir = ASSETS / group_name
    saved = 0
    group_data: dict[str, list[dict]] = {}
    for metric in sorted(all_metrics):
        series_list = by_metric[metric]
        if not series_list:
            continue
        series_list.sort(key=lambda s: runs.index(s.run) if s.run in runs else 0)
        if not data_only:
            plot_metric(series_list, out_dir / metric_to_filename(metric))
        group_data[metric] = [
            {"run": s.run, "steps": s.steps, "values": s.values} for s in series_list
        ]
        saved += 1

    print(f"  {saved} {'metrics fetched' if data_only else 'charts saved'}")
    return group_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Only process groups containing this run name")
    parser.add_argument("--light", action="store_true", help="Light mode charts")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only fetch raw data JSON, skip chart PNGs",
    )
    args = parser.parse_args()

    global theme
    if not args.data_only:
        theme = THEME["light"] if args.light else THEME["dark"]

    if args.run:
        groups = {
            k: v for k, v in RUN_GROUPS.items() if any(args.run in r for r in v["runs"])
        }
        if not groups:
            print(f"no group found for '{args.run}'")
            sys.exit(1)
    else:
        groups = RUN_GROUPS

    all_data: dict[str, dict] = {}
    for group_name, group in groups.items():
        all_data[group_name] = run_group(group_name, group, data_only=args.data_only)

    out_json = ASSETS / "charts_data.json"
    out_json.write_text(json.dumps(all_data, indent=2))
    print(f"\nraw data saved to {out_json.relative_to(ASSETS.parent)}")

    print("done")


if __name__ == "__main__":
    main()
