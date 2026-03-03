from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from rl.config import SFTTrainConfig, load_train_config

logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path, tasks: list[str] | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if tasks is None or row.get("task") in tasks:
                rows.append(row)
    return rows


def tokenise_dataset(rows: list[dict], tokenizer, max_seq_length: int):
    def _tokenise(row: dict) -> dict:
        full = row["prompt"].rstrip() + "\n" + row["completion"]
        prompt_only = row["prompt"].rstrip() + "\n"

        full_ids = tokenizer(
            full,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        prompt_ids = tokenizer(
            prompt_only,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        prompt_len = len(prompt_ids)
        labels = ([-100] * prompt_len + full_ids[prompt_len:])[:max_seq_length]
        return {"input_ids": full_ids[:max_seq_length], "labels": labels}

    return Dataset.from_list([_tokenise(r) for r in rows])


def train(cfg: SFTTrainConfig) -> None:
    t = cfg.training
    
    # Configure quantization if needed
    quantization_config = None
    if t.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load base model — model_name always points to the base pretrained model.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Tokenizer comes from the base model regardless of whether an adapter is used.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if t.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    if cfg.adapter_path:
        logger.info("Loading existing LoRA adapter from %s", cfg.adapter_path)
        saved_lora_config = LoraConfig.from_pretrained(cfg.adapter_path)
        saved_lora_config.inference_mode = False
        if cfg.lora.rank != saved_lora_config.r:
            logger.warning(
                "lora.rank=%d in config is ignored when resuming from an adapter; "
                "using rank=%d from the loaded checkpoint.",
                cfg.lora.rank,
                saved_lora_config.r,
            )
        model = get_peft_model(model, saved_lora_config)
        adapter_weights = load_safetensors(
            str(Path(cfg.adapter_path) / "adapter_model.safetensors")
        )
        set_peft_model_state_dict(model, adapter_weights)
        logger.info(
            "Resuming LoRA (rank=%d, alpha=%d)",
            saved_lora_config.r,
            saved_lora_config.lora_alpha,
        )
    else:
        logger.info("Initialising fresh LoRA (rank=%d, alpha=%d)", cfg.lora.rank, cfg.lora.alpha)
        peft_config = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    rows = load_jsonl(cfg.data, tasks=cfg.tasks)
    if not rows:
        raise ValueError(f"No examples loaded from {cfg.data} (tasks={cfg.tasks})")
    logger.info("Training on %d examples", len(rows))

    dataset = tokenise_dataset(rows, tokenizer, t.max_seq_length)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=t.epochs,
            per_device_train_batch_size=t.batch_size,
            gradient_accumulation_steps=t.grad_accum,
            gradient_checkpointing=False,
            learning_rate=t.lr,
            warmup_ratio=t.warmup_ratio,
            lr_scheduler_type="cosine",
            fp16=True,
            bf16=False,
            logging_steps=10,
            save_steps=t.save_steps,
            save_total_limit=2,
            seed=cfg.seed,
            report_to="none",
            remove_unused_columns=False,
        ),
    )

    trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("LoRA adapter saved to %s", adapter_dir)

    peft_cfg = model.peft_config["default"]
    with (out_dir / "train_meta.json").open("w") as f:
        json.dump(
            {
                "model_name": cfg.model_name,
                "adapter_path": cfg.adapter_path,
                "tasks": cfg.tasks,
                "n_examples": len(rows),
                "lora_rank": peft_cfg.r,
                "lora_alpha": peft_cfg.lora_alpha,
                "epochs": t.epochs,
                "lr": t.lr,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    train(load_train_config())
