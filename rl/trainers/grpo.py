from __future__ import annotations

import json
import os

from datasets import Dataset
from loguru import logger
from trl import GRPOConfig, GRPOTrainer

from rl.trainers.base import BaseTrainer
from rl.config import GRPOTrainerConfig, ModelConfig
from rl.rewards import arithmetic_reward, format_reward, length_reward, thought_judge_reward


class AnnotatedGRPOTrainer(BaseTrainer):
    """GRPO trainer that uses annotated conversations as ground-truth context."""

    def __init__(self, model_config: ModelConfig, grpo_config: GRPOTrainerConfig):
        super().__init__(model_config)
        self.grpo_config = grpo_config

    def load_model(self):
        super().load_model()

        sft_checkpoint = self.grpo_config.sft_checkpoint
        if sft_checkpoint and os.path.isdir(sft_checkpoint):
            self.load_checkpoint(sft_checkpoint)

        return self.model, self.tokenizer

    def prepare_dataset(self) -> Dataset:
        """Create GRPO prompts from annotated conversations.

        Args:
            None.

        Returns:
            A dataset with one ``prompt`` row per assistant turn after the split.

        Raises:
            ValueError: If no prompts can be generated from the JSONL file.
            Exception: If the JSONL file cannot be read or parsed.
        """
        cfg = self.grpo_config
        prompts: list[dict] = []

        try:
            with open(cfg.data_path, encoding="utf-8") as file:
                for line in file:
                    messages = json.loads(line)["messages"]
                    assistant_indices = [
                        index
                        for index, message in enumerate(messages)
                        if message["role"] == "assistant"
                    ]
                    if not assistant_indices:
                        continue

                    system_prompt = messages[0]["content"] if messages else ""
                    split_at = max(1, int(len(assistant_indices) * cfg.prompt_split))
                    for assistant_index in assistant_indices[split_at:]:
                        prompt = messages[:assistant_index]
                        if not prompt or prompt[-1]["role"] != "user":
                            continue
                        prompts.append({
                            "prompt": prompt,
                            "system_prompt": system_prompt,
                        })
        except Exception:
            logger.exception(
                f"Failed to build annotated GRPO prompts from {cfg.data_path}"
            )
            raise

        if not prompts:
            raise ValueError(f"No GRPO prompts generated from {cfg.data_path}")

        logger.info(f"Built {len(prompts)} GRPO prompts from {cfg.data_path}")
        return Dataset.from_list(prompts)

    def build_trainer(self, train_dataset: Dataset) -> GRPOTrainer:
        cfg = self.grpo_config
        self._patch_chat_template()

        training_args = GRPOConfig(
            output_dir=cfg.output_dir,
            num_generations=cfg.num_generations,
            beta=cfg.beta,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            gradient_checkpointing=cfg.gradient_checkpointing,
            bf16=cfg.bf16,
            report_to=cfg.report_to,
            **cfg.extra_kwargs,
        )
        peft_config = self._build_peft_config(cfg.lora)

        os.makedirs(training_args.output_dir, exist_ok=True)

        return GRPOTrainer(
            model=self.model,
            args=training_args,
            reward_funcs=[length_reward, thought_judge_reward, format_reward, arithmetic_reward],
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

    @classmethod
    def run(
        cls,
        model_config: ModelConfig,
        grpo_config: GRPOTrainerConfig,
        resume_from: str | None = None,
    ) -> None:
        trainer = cls(model_config, grpo_config)
        trainer.train(resume_from=resume_from)
