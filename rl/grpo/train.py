from __future__ import annotations

import json
import logging
import os

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.grpo.rewards import format_reward, offer_reward, terminal_reward
from rl.sft.train import (
    DTYPE_MAP,
    METRICS_FILENAME,
    JSONLLoggingCallback,
    _build_peft_config,
    _patch_chat_template_for_generation,
)

log = logging.getLogger(__name__)

COMPLETIONS_FILENAME = "completions.jsonl"


class CompletionLogger:
    """Wraps reward functions to periodically log completions and all reward scores.

    TRL GRPOTrainer calls each reward function in order with the same
    ``(completions, **kwargs)`` batch.  This class wraps all of them,
    captures the completions on the *first* call, accumulates each
    function's scores, and writes a combined record after the *last*
    function returns.
    """

    def __init__(self, reward_fns, reward_names, log_path, log_every=10):
        self._fns = list(zip(reward_fns, reward_names))
        self._path = log_path
        self._every = log_every
        self._step = 0
        self._batch_completions = None
        self._batch_kwargs = None
        self._batch_rewards: dict[str, list[float]] = {}

    def wrapped_fns(self):
        return [
            self._wrap(fn, name, idx)
            for idx, (fn, name) in enumerate(self._fns)
        ]

    def _wrap(self, fn, name, idx):
        is_first = idx == 0
        is_last = idx == len(self._fns) - 1

        def wrapper(completions, **kwargs):
            rewards = fn(completions, **kwargs)
            if is_first:
                self._step += 1
                self._batch_completions = completions
                self._batch_kwargs = kwargs
                self._batch_rewards = {}
            self._batch_rewards[name] = rewards
            if is_last and self._step % self._every == 0:
                self._flush()
            return rewards

        return wrapper

    @staticmethod
    def _extract_text(comp) -> str:
        if isinstance(comp, str):
            return comp
        if isinstance(comp, list):
            for msg in comp:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            if comp:
                last = comp[-1]
                return last.get("content", "") if isinstance(last, dict) else str(last)
        return str(comp)

    @staticmethod
    def _prompt_preview(prompt, max_len=200) -> str:
        if isinstance(prompt, list):
            for msg in reversed(prompt):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")[:max_len]
        if isinstance(prompt, str):
            return prompt[-max_len:]
        return ""

    def _flush(self):
        completions = self._batch_completions or []
        prompts = self._batch_kwargs.get("prompts", [])

        with open(self._path, "a") as f:
            for i in range(len(completions)):
                record: dict = {
                    "step": self._step,
                    "idx": i,
                    "prompt_preview": self._prompt_preview(
                        prompts[i] if i < len(prompts) else ""
                    ),
                    "completion": self._extract_text(completions[i]),
                }
                for name, rewards in self._batch_rewards.items():
                    record[name] = rewards[i] if i < len(rewards) else None
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_grpo_model_and_tokenizer(config):
    """Load the base model with the SFT LoRA adapter for GRPO training.

    If ``config.sft_checkpoint`` points to a PEFT adapter directory the
    base model is loaded first, then the adapter is merged and unloaded
    so GRPOTrainer can attach a fresh LoRA for the GRPO phase.

    If the checkpoint is already a full model (merged), it is loaded
    directly.
    """
    model_cfg = config.model
    if model_cfg.hf_home:
        os.environ["HF_HOME"] = model_cfg.hf_home

    dtype = DTYPE_MAP.get(model_cfg.dtype, torch.float16)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)
    tokenizer.padding_side = "left"

    sft_ckpt = config.sft_checkpoint
    if sft_ckpt and os.path.isdir(sft_ckpt):
        adapter_config_path = os.path.join(sft_ckpt, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            log.warning("Loading SFT LoRA adapter from %s", sft_ckpt)
            base_model = PeftModel.from_pretrained(base_model, sft_ckpt)
            base_model = base_model.merge_and_unload()
            log.warning("Merged SFT adapter into base model")
        else:
            log.warning(
                "SFT checkpoint %s has no adapter_config.json — "
                "assuming already-merged model weights",
                sft_ckpt,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                sft_ckpt,
                torch_dtype=dtype,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(sft_ckpt)
            tokenizer.padding_side = "left"

    return base_model, tokenizer


def build_grpo_trainer(model, tokenizer, config, train_dataset: Dataset):
    """Construct a TRL GRPOTrainer from the GRPO training config."""
    from trl import GRPOConfig, GRPOTrainer

    _patch_chat_template_for_generation(tokenizer)

    grpo_kwargs = dict(config.grpo.raw)
    grpo_kwargs["reward_weights"] = config.reward_weights

    training_args = GRPOConfig(**grpo_kwargs)
    peft_config = _build_peft_config(config.lora)

    os.makedirs(training_args.output_dir, exist_ok=True)
    metrics_path = os.path.join(training_args.output_dir, METRICS_FILENAME)
    completions_path = os.path.join(training_args.output_dir, COMPLETIONS_FILENAME)

    comp_logger = CompletionLogger(
        reward_fns=[format_reward, offer_reward, terminal_reward],
        reward_names=["format_reward", "offer_reward", "terminal_reward"],
        log_path=completions_path,
        log_every=training_args.logging_steps,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=comp_logger.wrapped_fns(),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[JSONLLoggingCallback(metrics_path)],
    )
    return trainer


def run_grpo_training(trainer, resume_from: str | None = None):
    """Execute the GRPO training loop, optionally resuming from a checkpoint."""
    if resume_from == "latest":
        resume_from_checkpoint = True
    elif resume_from:
        resume_from_checkpoint = resume_from
    else:
        resume_from_checkpoint = None

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception:
        log.error("GRPO training failed")
        raise

    trainer.save_model()
    log.info("GRPO model saved to %s", trainer.args.output_dir)
