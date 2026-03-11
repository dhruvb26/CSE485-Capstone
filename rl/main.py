import argparse
import asyncio
import json
import logging
import os
import signal

from dotenv import load_dotenv
from tqdm import tqdm

from rl.config import GenerateConfig, load_generate_config, load_training_config
from rl.sft.data import load_all_conversations
from rl.sft.generate import annotate_agent, create_openai_client
from rl.sft.train import (
    build_trainer,
    load_model_and_tokenizer,
    load_sft_dataset,
    run_training,
)

load_dotenv(override=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


async def run_annotation(config: GenerateConfig):
    """Annotate all datasets concurrently and write ordered JSONL.

    Handles SIGINT gracefully: in-flight API calls are allowed to finish,
    no new tasks are started, and all completed results are saved.
    """
    client = create_openai_client(config)
    os.makedirs(os.path.dirname(config.output_jsonl) or ".", exist_ok=True)

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_handler = signal.getsignal(signal.SIGINT)

    def _request_shutdown():
        if shutdown_event.is_set():
            log.warning("Second interrupt — forcing exit")
            raise KeyboardInterrupt
        log.warning("Interrupt received — finishing in-flight calls then saving…")
        shutdown_event.set()

    loop.add_signal_handler(signal.SIGINT, _request_shutdown)

    tasks: list[tuple[int, str, int, list[dict], dict, str]] = []
    conv_count = 0
    task_idx = 0

    for ds_name, ds_cfg in config.datasets.items():
        all_conversations = load_all_conversations(ds_cfg.path)
        if config.max_instances is not None:
            remaining = config.max_instances - conv_count
            conversations = all_conversations[:remaining]
        else:
            conversations = all_conversations

        for row_idx, chat_logs, participant_info, agent_ids in conversations:
            for agent_id in agent_ids:
                tasks.append(
                    (task_idx, ds_name, row_idx, chat_logs, participant_info, agent_id)
                )
                task_idx += 1
            conv_count += 1

        if config.max_instances is not None and conv_count >= config.max_instances:
            break

    agent_semaphore = asyncio.Semaphore(config.max_concurrent)
    pbar = tqdm(total=len(tasks), desc="Processing", unit="agent")

    async def _process(idx, ds_name, row_idx, chat_logs, participant_info, agent_id):
        sample_id = f"{ds_name}_{row_idx}_{agent_id}"

        if shutdown_event.is_set():
            pbar.update(1)
            return None

        async with agent_semaphore:
            if shutdown_event.is_set():
                pbar.update(1)
                return None

            try:
                sft_messages = await annotate_agent(
                    client,
                    chat_logs,
                    participant_info,
                    agent_id,
                    config,
                    sample_id=sample_id,
                    shutdown_event=shutdown_event,
                )
            except Exception:
                log.warning(
                    "Annotation failed for row %d, agent %s — skipping",
                    row_idx,
                    agent_id,
                )
                return None
            finally:
                pbar.update(1)

        return (
            idx,
            {
                "id": sample_id,
                "dataset": ds_name,
                "row_idx": row_idx,
                "agent_id": agent_id,
                "messages": sft_messages,
            },
        )

    try:
        results = await asyncio.gather(*[_process(*t) for t in tasks])
    except (asyncio.CancelledError, KeyboardInterrupt):
        results = []
    finally:
        pbar.close()
        loop.remove_signal_handler(signal.SIGINT)
        signal.signal(signal.SIGINT, original_handler)

    records = sorted([r for r in results if r is not None], key=lambda r: r[0])

    with open(config.output_jsonl, "w") as out:
        for _, record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(records)
    total = len(tasks)
    if shutdown_event.is_set():
        log.warning(
            "Interrupted — saved %d/%d completed samples to %s",
            n,
            total,
            config.output_jsonl,
        )
    else:
        log.warning("Annotations saved to %s (%d samples)", config.output_jsonl, n)


def main_generate():
    config = load_generate_config("rl/configs/generate.yaml")
    asyncio.run(run_annotation(config))


def main_train():
    config = load_training_config("rl/configs/training.yaml")

    full_dataset = load_sft_dataset(config.data_path)

    eval_dataset = None
    if config.val_split and config.val_split > 0:
        split = full_dataset.train_test_split(test_size=config.val_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        log.warning(
            "Split dataset: %d train, %d val (%.0f%% val)",
            len(train_dataset),
            len(eval_dataset),
            config.val_split * 100,
        )
    else:
        train_dataset = full_dataset

    model, tokenizer = load_model_and_tokenizer(config)
    trainer = build_trainer(model, tokenizer, config, train_dataset, eval_dataset)
    run_training(trainer, resume_from=config.resume_from)


def main():
    parser = argparse.ArgumentParser(description="RL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="Run GPT annotation to produce SFT data")
    subparsers.add_parser(
        "train", help="Run SFT training with LoRA on the generated data"
    )
    args = parser.parse_args()

    if args.command == "generate":
        main_generate()
    elif args.command == "train":
        main_train()


if __name__ == "__main__":
    main()
