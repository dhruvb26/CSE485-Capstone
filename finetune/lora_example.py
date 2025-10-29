import argparse
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

logger = logging.getLogger(__name__)


class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Chat",
        dataset_path: str = "data/casino/conversations.json",
        num_epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Derive paths
        self.clean_model_name = model_name.split("/")[-1].replace("-", "_")
        self.output_dir = f"./training_output/{self.clean_model_name}-lora"
        self.model_save_path = f"./pretrained_models/{self.clean_model_name}-lora"
        self.tokenizer_save_path = f"./tokenizers/{self.clean_model_name}-lora"

        # Load dataset
        self.dataset = load_dataset("json", data_files=dataset_path)["train"]

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Training arguments
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
    """Load model and tokenizer with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
    )

    return model, tokenizer


def format_example(example, tokenizer):
    """Format training examples for instruction tuning."""
    prompt = (
        f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse:"
    )
    text = f"{prompt} {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=1024)


def train(config: LoRAConfig):
    """Train the model with LoRA fine-tuning."""
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

    model.save_pretrained(config.model_save_path)
    tokenizer.save_pretrained(config.tokenizer_save_path)


def test(config: LoRAConfig, compare_base: bool = True):
    """Smoke test to verify model loads and generates properly.

    Args:
        config: LoRA configuration object
        compare_base: If True, compare base model vs fine-tuned model outputs
    """

    tokenizer_path = config.tokenizer_save_path
    adapter_path = config.model_save_path

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Test prompts
    test_prompts = [
        "Partner: I want the tent and sleeping bag.",
        "Partner: How about $50 for the water filter?",
        "Partner: I need all three items urgently.",
    ]

    for prompt in test_prompts:
        if compare_base:
            # Base model output
            logger.info("Base Model Response:")
            base = AutoModelForCausalLM.from_pretrained(
                config.model_name, device_map="auto", torch_dtype=torch.bfloat16
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            base_outputs = base.generate(**inputs, max_new_tokens=100)
            base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            logger.info(f"  {base_response}")

            # Clean up
            del base
            torch.cuda.empty_cache()

        # Fine-tuned model output
        logger.info("Fine-tuned Model Response:")
        base_for_lora = AutoModelForCausalLM.from_pretrained(
            config.model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        lora_model = PeftModel.from_pretrained(base_for_lora, adapter_path)
        lora_model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        lora_outputs = lora_model.generate(**inputs, max_new_tokens=100)
        lora_response = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        logger.info(f"  {lora_response}")

        if compare_base and base_response != lora_response:
            logger.info("✅ Outputs differ (fine-tuning had an effect).")
        elif compare_base:
            logger.warning("⚠️  Outputs identical (may need more training)!")

        # Clean up
        del base_for_lora, lora_model
        torch.cuda.empty_cache()

    logger.info("✓ Smoke test complete.")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Causal Language Models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Chat",
        help="Model name or path (default: Qwen/Qwen2.5-7B-Chat)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/casino/conversations.json",
        help="Path to training dataset (default: data/casino/conversations.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "both"],
        default="test",
        help="Run mode: train, test, or both (default: test)",
    )

    args = parser.parse_args()
    config = LoRAConfig(model_name=args.model, dataset_path=args.dataset)

    if args.mode in ["train", "both"]:
        train(config)

    if args.mode in ["test", "both"]:
        test(config, compare_base=args.compare_base)


if __name__ == "__main__":
    main()
