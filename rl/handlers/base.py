from abc import ABC, abstractmethod
from pathlib import Path


class BaseDatasetHandler(ABC):
    def __init__(self, data_path: Path | str):
        self.data_path = Path(data_path)
        self.dataset: list[dict] = []
        self.load()

    @abstractmethod
    def load(self): ...

    def get_instances(self, n: int | None = None) -> list[dict]:
        return self.dataset if n is None else self.dataset[:n]
