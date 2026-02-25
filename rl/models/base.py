from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
