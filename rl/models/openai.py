import os

from openai import OpenAI

from rl.models.base import BaseModel

_REQUIRED = ("model", "api_key_env", "temperature", "max_tokens")


class OpenAIModel(BaseModel):
    """OpenAI-hosted inference."""

    def __init__(self, config: dict):
        missing = [k for k in _REQUIRED if k not in config]
        if missing:
            raise KeyError(
                f"Missing required keys in config.model.openai: {missing}"
            )

        api_key_env: str = config["api_key_env"]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                "Export it before running the script."
            )

        self.client = OpenAI(api_key=api_key)
        self._model_name: str = config["model"]
        self.temperature: float = config["temperature"]
        self.max_tokens: int = config["max_tokens"]

    @property
    def model_id(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
