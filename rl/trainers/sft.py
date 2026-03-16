from __future__ import annotations

import json
import os

from datasets import Dataset
from loguru import logger
from trl import SFTConfig, SFTTrainer

from rl.trainers.base import BaseTrainer
from rl.config import ModelConfig, SFTTrainerConfig


class AnnotatedSFTTrainer(BaseTrainer):
    """SFT trainer for annotated JSONL conversations."""

    def __init__(self, model_config: ModelConfig, sft_config: SFTTrainerConfig):
        super().__init__(model_config)
        self.sft_config = sft_config

    def prepare_dataset(self) -> Dataset:
        rows: list[dict] = []

        try:
            with open(self.sft_config.data_path, encoding="utf-8") as file:
                for line in file:
                    record = json.loads(line)
                    rows.append({"messages": record["messages"]})
        except Exception:
            logger.exception(
                f"Failed to load annotated SFT data from {self.sft_config.data_path}"
            )
            raise

        if not rows:
            raise ValueError(f"No examples found in {self.sft_config.data_path}")

        logger.info(
            f"Loaded {len(rows)} SFT examples from {self.sft_config.data_path}"
        )
        return Dataset.from_list(rows)

    def build_trainer(self, train_dataset: Dataset) -> SFTTrainer:
        cfg = self.sft_config

        if cfg.assistant_only_loss:
            self._patch_chat_template()

        training_args = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            assistant_only_loss=cfg.assistant_only_loss,
            gradient_checkpointing=cfg.gradient_checkpointing,
            bf16=cfg.bf16,
            report_to=cfg.report_to,
            **cfg.extra_kwargs,
        )
        peft_config = self._build_peft_config(cfg.lora)

        os.makedirs(training_args.output_dir, exist_ok=True)

        return SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

    @classmethod
    def run(
        cls,
        model_config: ModelConfig,
        sft_config: SFTTrainerConfig,
        resume_from: str | None = None,
    ) -> None:
        trainer = cls(model_config, sft_config)
        trainer.train(resume_from=resume_from)
