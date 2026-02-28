from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from rl.handlers.casino.dataset import CasinoDatasetHandler
from rl.sft.llm_thoughts import LLMThoughtGenerator
from rl.sft.task_generators import ALL_TASKS, TASK_GENERATORS

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("rl/sft.config.yaml")

_THOUGHT_TAG = {"llm": "thought", "deterministic": "reasoning"}


@dataclass
class TaskSpec:
    llm_ratio: float
    weight: float


@dataclass
class SFTConfig:
    data: str
    out: str
    seed: int
    dedup: bool
    max_total: int | None
    llm_model: str
    llm_api_key_env: str
    default_llm_ratio: float
    tasks: dict[str, TaskSpec]

    @classmethod
    def from_dict(cls, d: dict) -> SFTConfig:
        default_ratio = d["default_llm_ratio"]
        llm_ratios: dict[str, float] = d.get("llm_ratios") or {}
        weights: dict[str, float] = d.get("weights") or {}
        return cls(
            data=d["data"],
            out=d["out"],
            seed=d["seed"],
            dedup=d["dedup"],
            max_total=d.get("max_total"),
            llm_model=d["llm_model"],
            llm_api_key_env=d["llm_api_key_env"],
            default_llm_ratio=default_ratio,
            tasks={
                task_id: TaskSpec(
                    llm_ratio=llm_ratios.get(task_id, default_ratio),
                    weight=weights.get(task_id, 1.0),
                )
                for task_id in d["tasks"]
            },
        )


def load_config(path: Path = _CONFIG_PATH) -> SFTConfig:
    with open(path, encoding="utf-8") as f:
        return SFTConfig.from_dict(yaml.safe_load(f))


def _make_completion(thought: str, answer_json: str, source: str) -> str:
    tag = _THOUGHT_TAG.get(source, "reasoning")
    return f"<{tag}>{thought}</{tag}>\n<answer>{answer_json}</answer>"


def _compute_per_task_caps(
    tasks: list[str],
    task_weights: dict[str, float],
    available: dict[str, int],
    max_total: int,
) -> dict[str, int]:
    weights = {t: task_weights.get(t, 1.0) for t in tasks}
    total_weight = sum(weights.values())
    quotas = {t: (weights[t] / total_weight) * max_total for t in tasks}
    remaining = max_total
    caps: dict[str, int] = {}
    for t in tasks:
        caps[t] = min(int(quotas[t]), available.get(t, 0))
        remaining -= caps[t]
    if remaining > 0:
        eligible = sorted(
            [t for t in tasks if available.get(t, 0) > caps[t]],
            key=lambda t: -weights[t],
        )
        for t in eligible:
            extra = min(remaining, available.get(t, 0) - caps[t])
            caps[t] += extra
            remaining -= extra
            if remaining == 0:
                break
    return caps


def generate(cfg: SFTConfig) -> None:
    tasks = list(cfg.tasks.keys())

    unknown = set(tasks) - set(TASK_GENERATORS)
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Valid: {ALL_TASKS}")

    out_path = Path(cfg.out)
    effective_ratios = {t: cfg.tasks[t].llm_ratio for t in tasks}
    task_weights = {t: cfg.tasks[t].weight for t in tasks}

    llm_gen: LLMThoughtGenerator | None = None
    if any(r > 0.0 for r in effective_ratios.values()):
        llm_gen = LLMThoughtGenerator(model=cfg.llm_model, api_key_env=cfg.llm_api_key_env)

    rng = random.Random(cfg.seed)

    handler = CasinoDatasetHandler(cfg.data)
    instances = handler.get_instances()
    logger.info("Loaded %d instances from %s", len(instances), cfg.data)

    raw_rows_by_task: dict[str, list] = {}
    for task_id in tqdm(tasks, desc="Tasks", unit="task"):
        rows = TASK_GENERATORS[task_id](instances)
        if cfg.dedup:
            seen: set[str] = set()
            deduped = []
            for row in rows:
                if row["prompt"] not in seen:
                    seen.add(row["prompt"])
                    deduped.append(row)
            logger.debug("%s dedup: %d → %d", task_id, len(rows), len(deduped))
            rows = deduped
        raw_rows_by_task[task_id] = rows

    raw_rows: list[dict] = []
    for task_id in tasks:
        raw_rows.extend(raw_rows_by_task[task_id])
    logger.info("Total after dedup: %d examples", len(raw_rows))

    if cfg.max_total is not None and cfg.max_total < len(raw_rows):
        available = {t: len(raw_rows_by_task[t]) for t in tasks}
        caps = _compute_per_task_caps(tasks, task_weights, available, cfg.max_total)
        sampled: list = []
        for task_id in tasks:
            pool = raw_rows_by_task[task_id]
            cap = caps[task_id]
            if len(pool) > cap:
                pool = rng.sample(pool, cap)
            sampled.extend(pool)
            logger.info("%s: %d → %d (cap %d)", task_id, available[task_id], len(pool), cap)
        rng.shuffle(sampled)
        raw_rows = sampled
        logger.info("After cap (%d): %d examples", cfg.max_total, len(raw_rows))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for row in tqdm(raw_rows, desc="Generating thoughts", unit="ex"):
                task_id = row["task"]
                ratio = effective_ratios.get(task_id, cfg.default_llm_ratio)
                use_llm_here = ratio >= 1.0 or (ratio > 0.0 and rng.random() < ratio)

                if use_llm_here and llm_gen is not None:
                    thought, source = llm_gen.generate(
                        row["prompt"], row["answer_json"], row["det_thought"]
                    )
                else:
                    thought, source = row["det_thought"], "deterministic"

                f.write(json.dumps({
                    "task": task_id,
                    "prompt": row["prompt"],
                    "completion": _make_completion(thought, row["answer_json"], source),
                    "thought_source": source,
                }, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
    except KeyboardInterrupt:
        logger.info("Interrupted — %d examples saved to %s", written, out_path)
        return

    logger.info("Wrote %d examples to %s", written, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config()
    generate(cfg)
