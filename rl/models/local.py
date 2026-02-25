import torch
from transformers import StoppingCriteriaList, StopStringCriteria

from rl.models.base import BaseModel

_REQUIRED = (
    "model_name",
    "max_seq_length",
    "lora_rank",
    "load_in_4bit",
    "temperature",
    "max_new_tokens",
    "gpu_memory_utilization",
    "stop_strings",
)


class LocalModel(BaseModel):
    """Unsloth-backed local inference model."""

    def __init__(self, config: dict):
        missing = [k for k in _REQUIRED if k not in config]
        if missing:
            raise KeyError(f"Missing required keys in config.model.local: {missing}")

        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            config["model_name"],
            config["max_seq_length"],
            load_in_4bit=config["load_in_4bit"],
            fast_inference=False,
            max_lora_rank=config["lora_rank"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
        )
        FastLanguageModel.for_inference(self.model)

        self.max_new_tokens: int = config["max_new_tokens"]
        self.temperature: float = config["temperature"]
        self.stopping_criteria = StoppingCriteriaList(
            [
                StopStringCriteria(
                    tokenizer=self.tokenizer, stop_strings=config["stop_strings"]
                )
            ]
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=1.0,
                do_sample=self.temperature > 0,
                stopping_criteria=self.stopping_criteria,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
