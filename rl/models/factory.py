from rl.models.base import BaseModel


def get_model(config: dict) -> BaseModel:
    model_type = config["model"]["type"]
    if model_type == "openai":
        from rl.models.openai import OpenAIModel
        return OpenAIModel(config["model"]["openai"])
    elif model_type == "local":
        from rl.models.local import LocalModel
        return LocalModel(config["model"]["local"])
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Choose 'openai' or 'local'.")
