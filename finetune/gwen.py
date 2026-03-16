"""LoRA fine-tuning script for Qwen on local Craigslist Bargains data.

Loads a local Alpaca-formatted JSONL dataset, applies LoRA adapters to a
Qwen causal LM, trains, and optionally runs a smoke test comparing the
base model against the fine-tuned adapter.

Example:
    Train mode::

        python gwen.py --model Qwen/Qwen2.5-7B-Instruct --mode train

    Test mode (compare base vs fine-tuned)::

        python gwen.py --mode test --compare-base
"""

import argparse
import logging
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class LoRAConfig:
    """Bundles model, dataset, LoRA, and training hyper-parameters.

    All output paths are derived from the script's location so the adapter
    artifacts land inside ``finetune/gwen_adapter_v0/``.

    Args:
        model_name: HuggingFace model identifier.
        dataset_name: Unused directly; kept for CLI compatibility.
        num_epochs: Total training epochs.
        batch_size: Per-device training batch size.
        learning_rate: Peak learning rate for the optimizer.

    Attributes:
        dataset: The loaded HuggingFace ``Dataset`` ready for tokenization.
        lora_config: PEFT ``LoraConfig`` applied to q/v projection layers.
        training_args: HuggingFace ``TrainingArguments`` for the ``Trainer``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        dataset_name: str = "yahma/alpaca-cleaned",
        num_epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.clean_model_name = model_name.split("/")[-1].replace("-", "_")

        self.model_base_path = os.path.join(script_dir, "gwen_adapter_v0")
        self.output_dir = os.path.join(
            self.model_base_path, "training_output", f"{self.clean_model_name}-lora"
        )
        self.model_save_path = os.path.join(
            self.model_base_path, "pretrained_models", f"{self.clean_model_name}-lora"
        )
        self.tokenizer_save_path = os.path.join(
            self.model_base_path, "tokenizers", f"{self.clean_model_name}-lora"
        )

        # Local Alpaca-formatted Craigslist Bargains data
        dataset_path = os.path.join(
            script_dir, "formatted_data/craigslist_bargains_alpaca.jsonl"
        )
        logger.info(f"Loading local dataset: {dataset_path}")
        full_dataset = load_dataset("json", data_files=dataset_path)["train"]
        self.dataset = full_dataset

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )


def load_model_tokenizer(
    config: LoRAConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base causal LM and its tokenizer in bf16.

    Args:
        config: Configuration holding the model name.

    Returns:
        A ``(model, tokenizer)`` tuple ready for LoRA wrapping.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    return model, tokenizer


def format_example(
    example: dict, tokenizer: AutoTokenizer
) -> dict:
    """Tokenize a single Alpaca-style example for causal LM training.

    Concatenates ``instruction``, ``input``, and ``output`` fields into one
    sequence and copies ``input_ids`` into ``labels`` so the full text is
    used as the training target.

    Args:
        example: Dict with ``instruction``, ``input``, and ``output`` keys.
        tokenizer: Tokenizer used to encode the concatenated text.

    Returns:
        Dict of tokenized tensors with an added ``labels`` key.
    """
    prompt = (
        f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse:"
    )
    text = f"{prompt} {example['output']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train(config: LoRAConfig) -> None:
    """Run LoRA fine-tuning and persist the adapter + tokenizer.

    Args:
        config: Full training configuration (model, data, hyper-params).
    """
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.tokenizer_save_path, exist_ok=True)

    logger.info(f"Loading model and tokenizer: {config.model_name}")
    model, tokenizer = load_model_tokenizer(config)

    model = get_peft_model(model, config.lora_config)
    model.print_trainable_parameters()

    tokenized_dataset = config.dataset.map(
        lambda ex: format_example(ex, tokenizer), batched=False
    )

    trainer = Trainer(
        model=model,
        args=config.training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Persist adapter weights and tokenizer separately
    model.save_pretrained(config.model_save_path)
    tokenizer.save_pretrained(config.tokenizer_save_path)
    logger.info("Training completed successfully")


def test(config: LoRAConfig, compare_base: bool = True) -> None:
    """Smoke-test the fine-tuned adapter on negotiation prompts.

    Loads the saved tokenizer and adapter, generates responses for a handful
    of hard-coded prompts, and optionally compares them against the base
    model to verify the adapter had an effect.

    Args:
        config: Configuration holding model name and artifact paths.
        compare_base: When True, also generate with the base model and
            log whether outputs diverge.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_save_path)

    test_prompts = [
        "Partner: I want the tent and sleeping bag.",
        "Partner: How about $50 for the water filter?",
        "Partner: I need all three items urgently.",
    ]

    for prompt in test_prompts:
        if compare_base:
            logger.info("Base Model Response:")
            base = AutoModelForCausalLM.from_pretrained(
                config.model_name, device_map="auto", dtype=torch.bfloat16
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            base_outputs = base.generate(**inputs, max_new_tokens=100)
            base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            logger.info(f"  {base_response}")

            del base
            torch.cuda.empty_cache()

        # Load base model again and merge the LoRA adapter on top
        logger.info("Fine-tuned Model Response:")
        base_for_lora = AutoModelForCausalLM.from_pretrained(
            config.model_name, device_map="auto", dtype=torch.bfloat16
        )
        lora_model = PeftModel.from_pretrained(base_for_lora, config.model_save_path)
        lora_model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        lora_outputs = lora_model.generate(**inputs, max_new_tokens=100)
        lora_response = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        logger.info(f"  {lora_response}")

        if compare_base and base_response != lora_response:
            logger.info("Outputs differ (fine-tuning had an effect).")
        elif compare_base:
            logger.warning("Outputs identical (may need more training)!")

        del base_for_lora, lora_model
        torch.cuda.empty_cache()

    logger.info("Smoke test complete.")


def main() -> None:
    """Parse CLI arguments and dispatch to train / test."""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Causal Language Models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="yahma/alpaca-cleaned",
        help="HuggingFace dataset name (default: yahma/alpaca-cleaned)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "both"],
        default="test",
        help="Run mode: train, test, or both (default: test)",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Compare base model vs fine-tuned model outputs during testing",
    )

    args = parser.parse_args()
    config = LoRAConfig(model_name=args.model, dataset_name=args.dataset)

    if args.mode in ["train", "both"]:
        train(config)

    if args.mode in ["test", "both"]:
        test(config, compare_base=args.compare_base)


if __name__ == "__main__":
    main()
