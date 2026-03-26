from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from rl.generator import SyntheticDataGenerator, SyntheticDataGeneratorConfig


async def run_generate(cfg: dict) -> None:
    data_path: str = cfg["data_path"]
    dataset: str = cfg["dataset"]
    concurrency: int = cfg["concurrency"]
    batch_size: int = cfg.get("batch_size", concurrency * 2)
    max_rows: int | None = cfg.get("max_rows")

    output_path = Path(data_path).with_suffix(".jsonl")

    gen_config = SyntheticDataGeneratorConfig(
        model=cfg["model"],
        temperature=cfg["temperature"],
        base_url=cfg.get("base_url"),
        api_key_env=cfg.get("api_key_env"),
    )
    generator = SyntheticDataGenerator(gen_config)
    semaphore = asyncio.Semaphore(concurrency)

    ds = load_dataset("csv", data_files=data_path)["train"]
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    completed: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    if completed:
        logger.info(
            "Resuming — {} rows already in {}", len(completed) // 2, output_path
        )

    logger.info(
        "Processing {} rows from {} | output={} concurrency={} batch_size={}",
        len(ds),
        data_path,
        output_path,
        concurrency,
        batch_size,
    )

    pbar = tqdm(total=len(ds), desc="Rows processed", unit="row")
    total_failed = 0
    file_lock = asyncio.Lock()

    async def _bounded_row(row_idx: int, row: dict) -> Exception | None:
        nonlocal total_failed
        participant_info = json.loads(row["participant_info"])
        row_prefix = f"{dataset}_{row_idx}_"
        if all(f"{row_prefix}{aid}" in completed for aid in participant_info):
            pbar.update(1)
            return None

        async with semaphore:
            try:
                chat_logs = json.loads(row["chat_logs"])
                results = await asyncio.gather(
                    *[
                        generator.make_request(
                            chat_logs,
                            participant_info,
                            aid,
                            row_id=f"{dataset}_{row_idx}_{aid}",
                            dataset=dataset,
                            row_idx=row_idx,
                        )
                        for aid in participant_info
                    ]
                )
            except Exception as exc:
                total_failed += 1
                logger.error("Row {} error: {}", row_idx, exc)
                return exc

        async with file_lock:
            with open(output_path, "a") as f:
                for record in results:
                    if record:
                        f.write(json.dumps(record) + "\n")
        pbar.update(1)
        return None

    try:
        for batch_start in range(0, len(ds), batch_size):
            batch_end = min(batch_start + batch_size, len(ds))
            tasks = [
                _bounded_row(idx, ds[idx]) for idx in range(batch_start, batch_end)
            ]
            await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("Interrupted — progress saved to {}", output_path)
    finally:
        pbar.close()

    if total_failed:
        logger.error("{} rows failed out of {}", total_failed, len(ds))
    else:
        logger.info("All {} rows completed successfully", len(ds))


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%m%d-%H%M")


def run_sft(cfg: dict) -> None:
    import trackio

    from rl.config import LoRAConfig, ModelConfig, SFTTrainerConfig
    from rl.trainers import AnnotatedSFTTrainer

    model_cfg = ModelConfig(**cfg["model"])
    lora_cfg = LoRAConfig(**cfg.get("lora", {}))
    sft_params = {k: v for k, v in cfg.get("sft", {}).items()}
    sft_params["lora"] = lora_cfg
    sft_config = SFTTrainerConfig(**sft_params)

    trackio.init(
        project=f"sft-{_timestamp()}",
        space_id="dhruvb26/capstone-trackio",
        config={
            "model": model_cfg.name,
            "learning_rate": sft_config.learning_rate,
            "epochs": sft_config.num_train_epochs,
            "batch_size": sft_config.per_device_train_batch_size,
            "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
            "lora_r": lora_cfg.r,
        },
    )

    try:
        AnnotatedSFTTrainer.run(
            model_config=model_cfg,
            sft_config=sft_config,
            resume_from=cfg.get("resume_from"),
        )
    finally:
        trackio.finish()


def run_grpo(cfg: dict) -> None:
    import trackio

    from rl.config import (
        GRPOTrainerConfig,
        JudgeConfig,
        LoRAConfig,
        ModelConfig,
        SelfPlayConfig,
    )
    from rl.trainers import AnnotatedGRPOTrainer, SelfPlayGRPOTrainer

    mode = cfg.get("mode", "self_play")
    model_cfg = ModelConfig(**cfg["model"])
    lora_cfg = LoRAConfig(**cfg.get("lora", {}))
    judge_cfg = JudgeConfig(**cfg.get("judge", {}))
    grpo_params = {k: v for k, v in cfg.get("grpo", {}).items()}
    grpo_params["lora"] = lora_cfg
    grpo_params["judge"] = judge_cfg

    trackio.init(
        project=f"grpo-{mode}-{_timestamp()}",
        space_id="dhruvb26/capstone-trackio",
        config={
            "model": model_cfg.name,
            "mode": mode,
            "learning_rate": grpo_params.get("learning_rate"),
            "batch_size": grpo_params.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": grpo_params.get("gradient_accumulation_steps"),
            "num_generations": grpo_params.get("num_generations"),
            "beta": grpo_params.get("beta"),
            "loss_type": grpo_params.get("extra_kwargs", {}).get("loss_type"),
            "lora_r": lora_cfg.r,
        },
    )

    try:
        if mode == "self_play":
            SelfPlayGRPOTrainer.run(
                model_config=model_cfg,
                config=SelfPlayConfig(**grpo_params),
                resume_from=cfg.get("resume_from"),
            )
        elif mode == "annotated":
            AnnotatedGRPOTrainer.run(
                model_config=model_cfg,
                grpo_config=GRPOTrainerConfig(**grpo_params),
                resume_from=cfg.get("resume_from"),
            )
        else:
            raise ValueError(
                f"Unknown grpo mode: {mode!r} (expected 'self_play' or 'annotated')"
            )
    finally:
        trackio.finish()


def main() -> None:
    commands = {
        "generate": run_generate,
        "train": run_sft,
        "grpo": run_grpo,
    }

    parser = argparse.ArgumentParser(description="RL pipeline runner")
    parser.add_argument("command", choices=commands, help="Pipeline stage to run")
    args = parser.parse_args()

    configs_dir = Path(__file__).parent / "configs"

    path = configs_dir / f"{args.command}.yaml"
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}

    handler = commands[args.command]
    if asyncio.iscoroutinefunction(handler):
        asyncio.run(handler(cfg))
    else:
        handler(cfg)


if __name__ == "__main__":
    main()
