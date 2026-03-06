from rl.handlers.casino import CA_TASK_REGISTRY, CasinoDatasetHandler

TASK_REGISTRY = {**CA_TASK_REGISTRY}

__all__ = [
    "CasinoDatasetHandler",
    "CA_TASK_REGISTRY",
    "TASK_REGISTRY",
]
