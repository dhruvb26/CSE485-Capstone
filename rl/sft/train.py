import json
import logging
import os
from dataclasses import asdict

import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig as TrlSFTConfig
from trl import SFTTrainer

from rl.config import TrainingConfig

log = logging.getLogger(__name__)

METRICS_FILENAME = "train_metrics.jsonl"

class JSONLLoggingCallback(TrainerCallback):
    """Append each training log entry as a JSON line to a file."""

    def __init__(self, log_path: str):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"global_step": state.global_step, **logs}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_sft_dataset(jsonl_path: str) -> Dataset:
    """Load a JSONL file produced by the annotation pipeline into an HF Dataset.

    Each line must contain a ``messages`` field holding an OpenAI-style list
    of ``{"role": ..., "content": ...}`` dicts.  Only the ``messages`` column
    is kept so SFTTrainer auto-detects the conversational format.
    """
    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rows.append({"messages": record["messages"]})
    if not rows:
        raise ValueError(f"No examples found in {jsonl_path}")
    log.info("Loaded %d SFT examples from %s", len(rows), jsonl_path)
    return Dataset.from_list(rows)


def load_model_and_tokenizer(
    config: TrainingConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base causal LM and its tokenizer from the training config."""
    model_cfg = config.model
    if model_cfg.hf_home:
        os.environ["HF_HOME"] = model_cfg.hf_home

    dtype = DTYPE_MAP.get(model_cfg.dtype, torch.float16)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_cfg.name,
            torch_dtype=dtype,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_cfg.name,
        )
    except Exception:
        log.error("Failed to load model or tokenizer: %s", model_cfg.name)
        raise

    return model, tokenizer


QWEN_CHAT_TEMPLATE_WITH_GENERATION = (
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


def _patch_chat_template_for_generation(tokenizer: AutoTokenizer) -> None:
    """Replace the chat template with one that includes ``{% generation %}`` markers.

    Qwen 2.5's default template lacks the ``{% generation %}`` /
    ``{% endgeneration %}`` keywords that TRL requires for
    ``assistant_only_loss=True``.
    """
    template = tokenizer.chat_template or ""
    if "{% generation %}" not in template:
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE_WITH_GENERATION
        log.info("Set chat template with {%% generation %%} markers for assistant_only_loss")


def build_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: TrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
) -> SFTTrainer:
    """Construct an SFTTrainer from the training config.

    Reads SFT hyperparameters (including ``assistant_only_loss`` and
    ``eos_token`` for Qwen-style chat templates) and LoRA settings from
    the typed config and returns a ready-to-train SFTTrainer instance.
    """
    sft_kwargs = asdict(config.sft)
    eos_token = sft_kwargs.pop("eos_token", None)

    if eos_token:
        tokenizer.eos_token = eos_token

    if sft_kwargs.get("assistant_only_loss"):
        _patch_chat_template_for_generation(tokenizer)

    training_args = TrlSFTConfig(**sft_kwargs)
    peft_config = PeftLoraConfig(**asdict(config.lora))

    os.makedirs(training_args.output_dir, exist_ok=True)
    metrics_path = os.path.join(training_args.output_dir, METRICS_FILENAME)

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[JSONLLoggingCallback(metrics_path)],
    )


def run_training(trainer: SFTTrainer, resume_from: str | None = None):
    """Execute the training loop, optionally resuming from a checkpoint.

    Args:
        trainer: A fully configured SFTTrainer.
        resume_from: ``None`` to train from scratch, ``"latest"`` to
            auto-detect the most recent checkpoint in ``output_dir``,
            or an explicit checkpoint directory path.
    """
    if resume_from == "latest":
        resume_from_checkpoint = True
    elif resume_from:
        resume_from_checkpoint = resume_from
    else:
        resume_from_checkpoint = None

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception:
        log.error("Training failed")
        raise
