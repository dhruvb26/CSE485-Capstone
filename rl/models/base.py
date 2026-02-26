from abc import ABC, abstractmethod


class BaseModel(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def generate(self, prompt: str) -> str: ...
