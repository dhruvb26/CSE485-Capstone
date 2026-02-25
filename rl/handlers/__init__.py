from rl.handlers.casino import CA_TASK_REGISTRY, CasinoDatasetHandler
from rl.handlers.dnd import DND_TASK_REGISTRY, DNDDatasetHandler

TASK_REGISTRY = {**CA_TASK_REGISTRY, **DND_TASK_REGISTRY}

__all__ = [
    "CasinoDatasetHandler",
    "DNDDatasetHandler",
    "CA_TASK_REGISTRY",
    "DND_TASK_REGISTRY",
    "TASK_REGISTRY",
]
