from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path("rl/config.yaml")
TRAIN_CONFIG_PATH = Path("rl/train.config.yaml")


@dataclass
class LoggingConfig:
    level: str


@dataclass
class OpenAIModelConfig:
    model: str
    api_key_env: str
    temperature: float
    max_tokens: int


@dataclass
class LocalModelConfig:
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    temperature: float
    max_new_tokens: int
    gpu_memory_utilization: float
    stop_strings: list[str]
    adapter_path: str | None = None


@dataclass
class ModelConfig:
    type: str
    openai: OpenAIModelConfig
    local: LocalModelConfig


@dataclass
class DatasetPaths:
    test: str
    train: str


@dataclass
class DataConfig:
    base_dir: str
    casino: DatasetPaths
    dnd: DatasetPaths


@dataclass
class EvalConfig:
    n_instances: int
    tasks: dict[str, list[str]]


@dataclass
class TrainConfig:
    logging: LoggingConfig
    model: ModelConfig
    data: DataConfig
    eval: EvalConfig

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        m = d["model"]
        return cls(
            logging=LoggingConfig(**d["logging"]),
            model=ModelConfig(
                type=m["type"],
                openai=OpenAIModelConfig(**m["openai"]),
                local=LocalModelConfig(**m["local"]),
            ),
            data=DataConfig(
                base_dir=d["data"]["base_dir"],
                casino=DatasetPaths(**d["data"]["casino"]),
                dnd=DatasetPaths(**d["data"]["dnd"]),
            ),
            eval=EvalConfig(
                n_instances=d["eval"]["n_instances"],
                tasks=d["eval"]["tasks"],
            ),
        )


def load_config(path: Path = CONFIG_PATH) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        return TrainConfig.from_dict(yaml.safe_load(f))


@dataclass
class LoraConfig:
    rank: int
    alpha: int
    dropout: float


@dataclass
class TrainingConfig:
    max_seq_length: int
    load_in_4bit: bool
    lr: float
    epochs: int
    batch_size: int
    grad_accum: int
    warmup_ratio: float
    save_steps: int


@dataclass
class SFTTrainConfig:
    data: str
    tasks: list[str] | None
    model_name: str
    out_dir: str
    seed: int
    lora: LoraConfig
    training: TrainingConfig
    # Path to an existing LoRA adapter to resume training from.
    # When set, model_name must be the base model and the adapter is loaded on top.
    # When null, a fresh LoRA is applied to the base model.
    adapter_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> SFTTrainConfig:
        return cls(
            data=d["data"],
            tasks=d.get("tasks"),
            model_name=d["model_name"],
            out_dir=d["out_dir"],
            seed=d["seed"],
            lora=LoraConfig(**d["lora"]),
            training=TrainingConfig(**d["training"]),
            adapter_path=d.get("adapter_path"),
        )


def load_train_config(path: Path = TRAIN_CONFIG_PATH) -> SFTTrainConfig:
    with open(path, encoding="utf-8") as f:
        return SFTTrainConfig.from_dict(yaml.safe_load(f))
