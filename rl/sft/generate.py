from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from rl.handlers.casino.dataset import CasinoDatasetHandler
from rl.handlers.dnd.dataset import DNDDatasetHandler
from rl.sft.dnd_generators import DND_TASK_GENERATORS
from rl.sft.llm_thoughts import AsyncLLMThoughtGenerator
from rl.sft.task_generators import TASK_GENERATORS as CA_TASK_GENERATORS
from rl.sft.turn_generators import TURN_GENERATORS

logger = logging.getLogger(__name__)

ALL_TASK_GENERATORS = {
    **CA_TASK_GENERATORS,
    **DND_TASK_GENERATORS,
    **TURN_GENERATORS,
}
ALL_TASKS: list[str] = list(ALL_TASK_GENERATORS.keys())

_THOUGHT_TAG = {"llm": "thought", "deterministic": "reasoning"}

DATASET_HANDLERS: dict[str, type] = {
    "casino": CasinoDatasetHandler,
    "dnd": DNDDatasetHandler,
}


@dataclass
class TaskSpec:
    llm_ratio: float
    weight: float


@dataclass
class DatasetSpec:
    path: str
    tasks: list[str]


@dataclass
class SFTConfig:
    datasets: dict[str, DatasetSpec]
    out: str
    seed: int
    dedup: bool
    max_total: int | None
    llm_model: str
    llm_api_key_env: str
    tasks: dict[str, TaskSpec]
    max_concurrency: int

    @classmethod
    def from_dict(cls, d: dict) -> SFTConfig:
        llm_ratios: dict[str, float] = d["llm_ratios"]
        weights: dict[str, float] = d["weights"]

        datasets = {}
        for name, spec in d["datasets"].items():
            datasets[name] = DatasetSpec(path=spec["path"], tasks=spec["tasks"])

        all_task_ids = [tid for ds in datasets.values() for tid in ds.tasks]
        task_specs = {
            tid: TaskSpec(llm_ratio=llm_ratios[tid], weight=weights[tid])
            for tid in all_task_ids
        }

        return cls(
            datasets=datasets,
            out=d["out"],
            seed=d["seed"],
            dedup=d["dedup"],
            max_total=d["max_total"],
            llm_model=d["llm_model"],
            llm_api_key_env=d["llm_api_key_env"],
            tasks=task_specs,
            max_concurrency=d.get("max_concurrency", 30),
        )


def load_config(path: Path) -> SFTConfig:
    with open(path, encoding="utf-8") as f:
        return SFTConfig.from_dict(yaml.safe_load(f))


def _make_eval_completion(thought: str, answer_json: str, source: str) -> str:
    tag = _THOUGHT_TAG.get(source, "reasoning")
    return f"<{tag}>{thought}</{tag}>\n<answer>{answer_json}</answer>"


def _make_turn_completion(thought: str, talk: str, action: dict) -> str:
    action_json = json.dumps(action, ensure_ascii=False)
    return (
        f"<thought>{thought}</thought>\n"
        f"<talk>{talk}</talk>\n"
        f"<action>{action_json}</action>"
    )


def _is_turn_row(row: dict) -> bool:
    return "talk" in row and "action" in row


def _compute_per_task_caps(
    tasks: list[str],
    task_weights: dict[str, float],
    available: dict[str, int],
    max_total: int,
) -> dict[str, int]:
    weights = {t: task_weights[t] for t in tasks}
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
    all_task_ids = list(cfg.tasks.keys())
    unknown = set(all_task_ids) - set(ALL_TASK_GENERATORS)
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Valid: {ALL_TASKS}")

    out_path = Path(cfg.out)
    effective_ratios = {t: cfg.tasks[t].llm_ratio for t in all_task_ids}
    task_weights = {t: cfg.tasks[t].weight for t in all_task_ids}

    rng = random.Random(cfg.seed)

    raw_rows_by_task: dict[str, list] = {}

    for ds_name, ds_spec in cfg.datasets.items():
        if ds_name not in DATASET_HANDLERS:
            raise ValueError(f"Unknown dataset: {ds_name}. Valid: {list(DATASET_HANDLERS)}")
        handler = DATASET_HANDLERS[ds_name](ds_spec.path)
        instances = handler.get_instances()
        logger.info("Loaded %d instances from %s (%s)", len(instances), ds_spec.path, ds_name)

        for task_id in tqdm(ds_spec.tasks, desc=f"Tasks ({ds_name})", unit="task"):
            rows = ALL_TASK_GENERATORS[task_id](instances)
            if cfg.dedup:
                seen: set[str] = set()
                deduped = []
                for row in rows:
                    key = row["prompt"]
                    if key not in seen:
                        seen.add(key)
                        deduped.append(row)
                logger.debug("%s dedup: %d -> %d", task_id, len(rows), len(deduped))
                rows = deduped
            raw_rows_by_task[task_id] = rows

    raw_rows: list[dict] = []
    for task_id in all_task_ids:
        raw_rows.extend(raw_rows_by_task.get(task_id, []))
    logger.info("Total after dedup: %d examples", len(raw_rows))

    if cfg.max_total is not None and cfg.max_total < len(raw_rows):
        available = {t: len(raw_rows_by_task.get(t, [])) for t in all_task_ids}
        caps = _compute_per_task_caps(all_task_ids, task_weights, available, cfg.max_total)
        sampled: list = []
        for task_id in all_task_ids:
            pool = raw_rows_by_task.get(task_id, [])
            cap = caps[task_id]
            if len(pool) > cap:
                pool = rng.sample(pool, cap)
            sampled.extend(pool)
            logger.info("%s: %d -> %d (cap %d)", task_id, available[task_id], len(pool), cap)
        rng.shuffle(sampled)
        raw_rows = sampled
        logger.info("After cap (%d): %d examples", cfg.max_total, len(raw_rows))

    for row in raw_rows:
        ratio = effective_ratios[row["task"]]
        row["_use_llm"] = ratio >= 1.0 or (ratio > 0.0 and rng.random() < ratio)

    rows_needing_llm = [r for r in raw_rows if r["_use_llm"]]
    logger.info(
        "%d / %d rows selected for LLM thought generation",
        len(rows_needing_llm), len(raw_rows),
    )

    if rows_needing_llm:
        try:
            llm_gen = AsyncLLMThoughtGenerator(
                model=cfg.llm_model,
                api_key_env=cfg.llm_api_key_env,
                max_concurrency=cfg.max_concurrency,
            )
            results = asyncio.run(llm_gen.generate_batch(rows_needing_llm))
            for row, (thought, source) in zip(rows_needing_llm, results):
                row["_llm_thought"] = thought
                row["_llm_source"] = source
            logger.info("LLM batch complete: %d thoughts generated", len(results))
        except EnvironmentError:
            logger.warning("OpenAI API key not set; all rows will use deterministic thoughts.")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(raw_rows, desc="Writing output", unit="ex"):
            if row.get("_llm_thought"):
                thought = row["_llm_thought"]
                source = row.get("_llm_source", "llm")
            else:
                thought = row["det_thought"]
                source = "deterministic"

            if _is_turn_row(row):
                completion = _make_turn_completion(thought, row["talk"], row["action"])
            else:
                completion = _make_eval_completion(thought, row["answer_json"], source)

            wrapped_prompt = "<|im_start|>user\n" + row["prompt"].strip() + "<|im_end|>\n<|im_start|>assistant\n"
            f.write(json.dumps({
                "task": row["task"],
                "prompt": wrapped_prompt,
                "completion": completion,
                "thought_source": source,
            }, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d examples to %s", written, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    cfg = load_config(Path("rl/configs/sft_generate.yaml"))
    generate(cfg)
