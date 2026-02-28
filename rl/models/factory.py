from rl.config import TrainConfig
from rl.models.base import BaseModel


def get_model(cfg: TrainConfig) -> BaseModel:
    if cfg.model.type == "openai":
        from rl.models.openai import OpenAIModel
        return OpenAIModel(cfg.model.openai)
    elif cfg.model.type == "local":
        from rl.models.local import LocalModel
        return LocalModel(cfg.model.local)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type!r}. Choose 'openai' or 'local'.")
