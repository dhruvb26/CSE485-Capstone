import os

from openai import OpenAI

from rl.config import OpenAIModelConfig
from rl.models.base import BaseModel


class OpenAIModel(BaseModel):
    def __init__(self, cfg: OpenAIModelConfig):
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{cfg.api_key_env}' is not set."
            )
        self.client = OpenAI(api_key=api_key)
        self._model_name = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

    @property
    def model_id(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
