import logging

import torch
from transformers import StoppingCriteriaList, StopStringCriteria
from unsloth import FastLanguageModel

from rl.config import LocalModelConfig
from rl.models.base import BaseModel

logger = logging.getLogger(__name__)


class LocalModel(BaseModel):
    def __init__(self, cfg: LocalModelConfig):
        self._model_name = cfg.model_name
        self._adapter_path = cfg.adapter_path

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            cfg.model_name,
            cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )

        if cfg.adapter_path:
            logger.info("Loading LoRA adapter from %s", cfg.adapter_path)
            self.model.load_adapter(cfg.adapter_path)

        FastLanguageModel.for_inference(self.model)

        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.stopping_criteria = StoppingCriteriaList(
            [StopStringCriteria(tokenizer=self.tokenizer, stop_strings=cfg.stop_strings)]
        )

    @property
    def model_id(self) -> str:
        if self._adapter_path:
            # Use adapter directory name to distinguish runs in log filenames
            return self._adapter_path.rstrip("/").split("/")[-1]
        return self._model_name.replace("/", "_")

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
