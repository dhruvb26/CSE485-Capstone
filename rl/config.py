from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
    adapter_path: str | None


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
    craigslist: DatasetPaths | None


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
                craigslist=DatasetPaths(**d["data"]["craigslist"])
                if "craigslist" in d["data"]
                else None,
            ),
            eval=EvalConfig(
                n_instances=d["eval"]["n_instances"],
                tasks=d["eval"]["tasks"],
            ),
        )


def load_config(path: Path) -> TrainConfig:
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
    adapter_path: str | None

    @classmethod
    def from_dict(cls, d: dict) -> SFTTrainConfig:
        return cls(
            data=d["data"],
            tasks=d["tasks"],
            model_name=d["model_name"],
            out_dir=d["out_dir"],
            seed=d["seed"],
            lora=LoraConfig(**d["lora"]),
            training=TrainingConfig(**d["training"]),
            adapter_path=d["adapter_path"],
        )


def load_train_config(path: Path) -> SFTTrainConfig:
    with open(path, encoding="utf-8") as f:
        return SFTTrainConfig.from_dict(yaml.safe_load(f))


@dataclass
class GRPOConfig:
    candidates_per_turn: int
    max_turns: int
    clone_sync_interval: int
    kl_coeff: float
    terminal_lambda: float
    walkaway_penalty: float
    lora: LoraConfig
    training: TrainingConfig
    base_model: str
    sft_adapter_path: str
    out_dir: str
    seed: int
    temperature: float = 1.0
    resume_from: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> GRPOConfig:
        return cls(
            candidates_per_turn=d["candidates_per_turn"],
            max_turns=d["max_turns"],
            clone_sync_interval=d["clone_sync_interval"],
            kl_coeff=d["kl_coeff"],
            terminal_lambda=d["terminal_lambda"],
            walkaway_penalty=d["walkaway_penalty"],
            lora=LoraConfig(**d["lora"]),
            training=TrainingConfig(**d["training"]),
            base_model=d["base_model"],
            sft_adapter_path=d["sft_adapter_path"],
            out_dir=d["out_dir"],
            seed=d["seed"],
            temperature=d.get("temperature", 1.0),
            resume_from=d.get("resume_from"),
        )


@dataclass
class RewardConfig:
    terminal_weight: float
    format_weight: float
    arithmetic_weight: float
    strategy_weight: float
    partner_model_weight: float
    action_quality_weight: float
    decay_window: int
    strategy_classifier_path: str

    @classmethod
    def from_dict(cls, d: dict) -> RewardConfig:
        return cls(
            terminal_weight=d["terminal_weight"],
            format_weight=d["format_weight"],
            arithmetic_weight=d["arithmetic_weight"],
            strategy_weight=d["strategy_weight"],
            partner_model_weight=d["partner_model_weight"],
            action_quality_weight=d.get("action_quality_weight", 0.20),
            decay_window=d["decay_window"],
            strategy_classifier_path=d["strategy_classifier_path"],
        )


def load_grpo_config(path: Path) -> tuple[GRPOConfig, RewardConfig]:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return GRPOConfig.from_dict(d["grpo"]), RewardConfig.from_dict(d["reward"])
