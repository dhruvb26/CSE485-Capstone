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
    BitsAndBytesConfig,
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
    """Configuration for LoRA fine-tuning."""

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
        
        # Path to save model adapter weights
        self.model_base_path = os.path.join(script_dir, "gwen_adapter_v0")
        self.output_dir = os.path.join(self.model_base_path, "training_output", f"{self.clean_model_name}-lora")
        self.model_save_path = os.path.join(self.model_base_path, "pretrained_models", f"{self.clean_model_name}-lora")
        self.tokenizer_save_path = os.path.join(self.model_base_path, "tokenizers", f"{self.clean_model_name}-lora")


        # Load dataset (using a small subset for testing)
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load local Alpaca-style dataset
        dataset_path = os.path.join(script_dir, "formatted_data/craigslist_bargains_alpaca.jsonl")
        logger.info(f"Loading local dataset: {dataset_path}")
        full_dataset = load_dataset("json", data_files=dataset_path)["train"]
        self.dataset = full_dataset

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
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

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     config.model_name,
    #     device_map="auto",
    #     quantization_config=bnb_config,
    #     dtype=torch.bfloat16,
    # )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,   # or comment this out to load full precision
    )
    
    return model, tokenizer


def format_example(example, tokenizer):
    """Format training examples for instruction tuning."""
    prompt = (
        f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse:"
    )
    text = f"{prompt} {example['output']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
    
def train(config: LoRAConfig):
    """Train the model with LoRA fine-tuning."""
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.tokenizer_save_path, exist_ok=True)

    logger.info("Starting LoRA fine-tuning training")
    logger.info(f"Loading model and tokenizer: {config.model_name}")
    model, tokenizer = load_model_tokenizer(config)

    logger.info("Applying LoRA configuration to model")
    model = get_peft_model(model, config.lora_config)
    model.print_trainable_parameters()

    logger.info("Tokenizing dataset")
    tokenized_dataset = config.dataset.map(
        lambda ex: format_example(ex, tokenizer), batched=False
    )

    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=config.training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training process")
    trainer.train()

    logger.info(f"Saving fine-tuned model to: {config.model_save_path}")
    model.save_pretrained(config.model_save_path)
    logger.info(f"Saving tokenizer to: {config.tokenizer_save_path}")
    tokenizer.save_pretrained(config.tokenizer_save_path)
    logger.info("Training completed successfully")


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
                config.model_name, device_map="auto", dtype=torch.bfloat16
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
            config.model_name, device_map="auto", dtype=torch.bfloat16
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
