from __future__ import annotations

import os
from abc import ABC, abstractmethod

from datasets import Dataset
from loguru import logger
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.config import LoRAConfig, ModelConfig


class BaseTrainer(ABC):
    """Abstract base for SFT and GRPO trainers sharing Qwen + LoRA boilerplate."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None

    def load_model(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the base model and tokenizer used for training.

        Args:
            None.

        Returns:
            The loaded model and tokenizer.

        Raises:
            Exception: If the model or tokenizer cannot be loaded.
        """
        cfg = self.model_config
        if cfg.hf_home:
            os.environ["HF_HOME"] = os.path.expandvars(cfg.hf_home)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.name,
                dtype=cfg.dtype,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(cfg.name)
        except Exception:
            logger.exception(f"Failed to load base model from {cfg.name}")
            raise

        tokenizer.padding_side = "left"

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def load_checkpoint(
        self, checkpoint_path: str
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a training checkpoint as either a merged LoRA adapter or full model.

        Args:
            checkpoint_path: Path to a checkpoint directory.

        Returns:
            The model and tokenizer after loading the checkpoint.

        Raises:
            Exception: If checkpoint loading fails.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")

        try:
            if os.path.exists(adapter_config_path):
                logger.info(f"Loading LoRA adapter from {checkpoint_path}")
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
                self.model = self.model.merge_and_unload()
                logger.info(f"Merged LoRA adapter from {checkpoint_path}")
            else:
                logger.info(f"Loading full model weights from {checkpoint_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    dtype=self.model_config.dtype,
                    device_map="auto",
                )
                self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                self.tokenizer.padding_side = "left"
        except Exception:
            logger.exception(f"Failed to load checkpoint from {checkpoint_path}")
            raise

        return self.model, self.tokenizer

    @staticmethod
    def _build_peft_config(lora: LoRAConfig) -> PeftLoraConfig:
        return PeftLoraConfig(
            r=lora.r,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            bias=lora.bias,
            target_modules=lora.target_modules,
            task_type=lora.task_type,
        )

    def _patch_chat_template(self) -> None:
        if self.tokenizer is None:
            return
        template = self.tokenizer.chat_template or ""
        if "{% generation %}" not in template:
            self.tokenizer.chat_template = (
                "{%- for message in messages %}"
                "{%- if message['role'] == 'system' %}"
                "{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                "{%- elif message['role'] == 'user' %}"
                "{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                "{%- elif message['role'] == 'assistant' %}"
                "{{- '<|im_start|>assistant\n' }}"
                "{% generation %}"
                "{{ message['content'] }}"
                "{% endgeneration %}"
                "{{- '<|im_end|>\n' }}"
                "{%- endif %}"
                "{%- endfor %}"
                "{%- if add_generation_prompt %}"
                "{{- '<|im_start|>assistant\n' }}"
                "{%- endif %}"
            )
            logger.info("Patched chat template with generation markers")

    @abstractmethod
    def build_trainer(self, train_dataset: Dataset): ...

    @abstractmethod
    def prepare_dataset(self) -> Dataset: ...

    def train(self, resume_from: str | None = None) -> None:
        """Run the full trainer lifecycle.

        Args:
            resume_from: Checkpoint path to resume from, or ``"latest"`` to let the
                trainer choose the most recent checkpoint.

        Returns:
            None.

        Raises:
            Exception: If model loading, dataset preparation, trainer creation,
                training, or saving fails.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            dataset = self.prepare_dataset()
            trainer = self.build_trainer(dataset)
        except Exception:
            logger.exception("Failed to prepare training components")
            raise

        if resume_from == "latest":
            resume_from_checkpoint = True
        elif resume_from:
            resume_from_checkpoint = resume_from
        else:
            resume_from_checkpoint = None

        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except Exception:
            logger.exception("Training failed")
            raise

        try:
            trainer.save_model()
        except Exception:
            logger.exception("Failed to save trained model")
            raise

        logger.info(f"Model saved to {trainer.args.output_dir}")
