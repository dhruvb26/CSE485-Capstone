"""Model loading, device placement, and clone synchronisation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rl.config import GRPOConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    cfg: GRPOConfig,
    device_map: dict | str = {"": 0},
    gradient_checkpointing: bool = False,
):
    """Load base model with optional 4-bit quantisation and LoRA.

    Args:
        device_map: Passed directly to ``from_pretrained``. Use ``{"": 0}`` /
            ``{"": 1}`` to pin to a specific GPU.
        gradient_checkpointing: Recompute activations during backprop to trade
            compute for VRAM. Enable for the learner; leave False for the clone.
    """
    t = cfg.training
    quant_config = None
    if t.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = PeftLoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if cfg.sft_adapter_path:
        from safetensors.torch import load_file as load_safetensors
        from peft import set_peft_model_state_dict
        adapter_weights = load_safetensors(
            str(Path(cfg.sft_adapter_path) / "adapter_model.safetensors")
        )
        set_peft_model_state_dict(model, adapter_weights)
        logger.info("Loaded SFT adapter from %s", cfg.sft_adapter_path)

    return model, tokenizer


def sync_clone(learner_model, clone_model) -> None:
    """Copy learner's LoRA weights to the clone."""
    clone_device = next(clone_model.parameters()).device
    learner_state = {
        k: v.clone().to(clone_device) for k, v in learner_model.state_dict().items()
        if "lora" in k.lower()
    }
    clone_state = clone_model.state_dict()
    clone_state.update(learner_state)
    clone_model.load_state_dict(clone_state)
    logger.info("Synced clone weights from learner")
