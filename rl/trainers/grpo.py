from __future__ import annotations

import json
import os
from collections import defaultdict

from datasets import Dataset
from loguru import logger
from trl import GRPOConfig, GRPOTrainer

from rl.callbacks import TrackioLoggingCallback
from rl.config import GRPOTrainerConfig, ModelConfig
from rl.rewards import (
    arithmetic_reward,
    configure_judge,
    format_reward,
    length_reward,
    thought_judge_reward,
)
from rl.trainers.base import BaseTrainer

_turn_reward_buffer: dict[str, list[tuple[float, float]]] = defaultdict(list)


def _wrap_reward_with_turn_tracking(reward_fn):
    """Wrap a reward function to capture (normalised turn position, reward)."""

    def wrapper(completions, **kwargs):
        rewards = reward_fn(completions, **kwargs)
        turn_indices = kwargs.get("turn_index", [])
        total_turns_list = kwargs.get("total_turns", [])
        name = reward_fn.__name__
        for i, r in enumerate(rewards):
            ti = turn_indices[i] if i < len(turn_indices) else 0
            tt = total_turns_list[i] if i < len(total_turns_list) else 1
            position = ti / max(tt - 1, 1)
            _turn_reward_buffer[name].append((position, r))
        return rewards

    wrapper.__name__ = reward_fn.__name__
    return wrapper


class TurnTrackingGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that injects per-turn-position reward metrics into logs.

    Early = first half of conversation turns (position < 0.5).
    Late  = second half (position >= 0.5), where SUBMIT_DEAL decisions
    happen and arithmetic correctness becomes critical.
    """

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if _turn_reward_buffer:
            all_positions: list[float] = []
            for reward_name, entries in _turn_reward_buffer.items():
                all_positions.extend(p for p, _ in entries)
                early = [r for p, r in entries if p < 0.5]
                late = [r for p, r in entries if p >= 0.5]
                if early:
                    logs[f"turn/{reward_name}/early_mean"] = sum(early) / len(early)
                if late:
                    logs[f"turn/{reward_name}/late_mean"] = sum(late) / len(late)
            if all_positions:
                logs["turn/mean_position"] = sum(all_positions) / len(all_positions)
            _turn_reward_buffer.clear()
        super().log(logs, *args, **kwargs)


class AnnotatedGRPOTrainer(BaseTrainer):
    """GRPO trainer that uses annotated conversations as ground-truth context."""

    def __init__(self, model_config: ModelConfig, grpo_config: GRPOTrainerConfig):
        super().__init__(model_config)
        self.grpo_config = grpo_config

    def load_model(self):
        super().load_model()

        checkpoint = self.grpo_config.checkpoint
        if checkpoint and os.path.isdir(checkpoint):
            self.load_checkpoint(checkpoint)

        return self.model, self.tokenizer

    def prepare_dataset(self) -> Dataset:
        """Create GRPO prompts from annotated conversations.

        Each row includes ``turn_index`` (0-based assistant-turn position) and
        ``total_turns`` so reward analytics can split early vs. late turns.

        Returns:
            A dataset with one ``prompt`` row per assistant turn after the split.
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

                    total_turns = len(assistant_indices)
                    system_prompt = messages[0]["content"] if messages else ""
                    split_at = max(1, int(total_turns * cfg.prompt_split))

                    for turn_pos, assistant_index in enumerate(
                        assistant_indices[split_at:], start=split_at
                    ):
                        prompt = messages[:assistant_index]
                        if not prompt or prompt[-1]["role"] != "user":
                            continue
                        prompts.append(
                            {
                                "prompt": prompt,
                                "system_prompt": system_prompt,
                                "turn_index": turn_pos,
                                "total_turns": total_turns,
                            }
                        )
        except Exception:
            logger.exception(
                f"Failed to build annotated GRPO prompts from {cfg.data_path}"
            )
            raise

        if not prompts:
            raise ValueError(f"No GRPO prompts generated from {cfg.data_path}")

        logger.info(f"Built {len(prompts)} GRPO prompts from {cfg.data_path}")
        return Dataset.from_list(prompts)

    def build_trainer(self, train_dataset: Dataset) -> TurnTrackingGRPOTrainer:
        cfg = self.grpo_config
        self._patch_chat_template()

        training_args = GRPOConfig(
            output_dir=cfg.output_dir,
            num_generations=cfg.num_generations,
            beta=cfg.beta,
            num_train_epochs=cfg.num_train_epochs,
            max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            gradient_checkpointing=cfg.gradient_checkpointing,
            bf16=cfg.bf16,
            report_to=cfg.report_to,
            reward_weights=[0.5, 1.0, 0.5, 2.0],
            **cfg.extra_kwargs,
        )
        peft_config = self._build_peft_config(cfg.lora)

        os.makedirs(training_args.output_dir, exist_ok=True)

        reward_funcs = [
            _wrap_reward_with_turn_tracking(fn)
            for fn in [
                length_reward,
                thought_judge_reward,
                format_reward,
                arithmetic_reward,
            ]
        ]

        return TurnTrackingGRPOTrainer(
            model=self.model,
            args=training_args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
            callbacks=[TrackioLoggingCallback()],
        )

    @classmethod
    def run(
        cls,
        model_config: ModelConfig,
        grpo_config: GRPOTrainerConfig,
        resume_from: str | None = None,
    ) -> None:
        configure_judge(
            model=grpo_config.judge.model,
            base_url=grpo_config.judge.base_url,
            api_key_env=grpo_config.judge.api_key_env,
        )
        trainer = cls(model_config, grpo_config)
        trainer.train(resume_from=resume_from)
