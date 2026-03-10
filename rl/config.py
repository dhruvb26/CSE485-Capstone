from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

log = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    path: str

    @classmethod
    def from_dict(cls, d: dict) -> DatasetConfig:
        return cls(path=d["path"])


@dataclass
class GenerateConfig:
    model: str
    temperature: float
    base_url: str | None
    api_key_env: str | None
    max_retries: int
    retry_min_wait: int
    retry_max_wait: int
    max_concurrent: int
    max_instances: int | None
    datasets: dict[str, DatasetConfig]
    output_jsonl: str

    @classmethod
    def from_dict(cls, d: dict) -> GenerateConfig:
        mode = d.get("mode", "api")
        mode_cfg = d.get(mode)
        if mode_cfg is None:
            raise ValueError(
                f"Mode '{mode}' not found in generate config. "
                f"Available modes: {[k for k in d if isinstance(d[k], dict) and k != 'datasets']}"
            )

        datasets = {
            name: DatasetConfig.from_dict(ds) for name, ds in d["datasets"].items()
        }
        return cls(
            model=mode_cfg["model"],
            temperature=d["temperature"],
            base_url=mode_cfg.get("base_url"),
            api_key_env=mode_cfg.get("api_key_env"),
            max_retries=mode_cfg["max_retries"],
            retry_min_wait=mode_cfg["retry_min_wait"],
            retry_max_wait=mode_cfg["retry_max_wait"],
            max_concurrent=mode_cfg["max_concurrent"],
            max_instances=d.get("max_instances"),
            datasets=datasets,
            output_jsonl=d["output_jsonl"],
        )


@dataclass
class ModelConfig:
    name: str
    hf_home: str
    dtype: str

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(
            name=d["name"],
            hf_home=d["hf_home"],
            dtype=d["dtype"],
        )


@dataclass
class SFTConfig:
    output_dir: str
    max_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    eval_strategy: str
    eval_steps: int
    max_length: int
    assistant_only_loss: bool
    eos_token: str

    @classmethod
    def from_dict(cls, d: dict) -> SFTConfig:
        return cls(
            output_dir=d["output_dir"],
            max_steps=d["max_steps"],
            per_device_train_batch_size=d["per_device_train_batch_size"],
            per_device_eval_batch_size=d.get("per_device_eval_batch_size", 1),
            learning_rate=d["learning_rate"],
            logging_steps=d["logging_steps"],
            save_steps=d["save_steps"],
            eval_strategy=d["eval_strategy"],
            eval_steps=d["eval_steps"],
            max_length=d["max_length"],
            assistant_only_loss=d.get("assistant_only_loss", False),
            eos_token=d.get("eos_token", ""),
        )


@dataclass
class LoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: str
    task_type: str

    @classmethod
    def from_dict(cls, d: dict) -> LoraConfig:
        return cls(
            r=d["r"],
            lora_alpha=d["lora_alpha"],
            lora_dropout=d["lora_dropout"],
            bias=d["bias"],
            target_modules=d["target_modules"],
            task_type=d["task_type"],
        )


@dataclass
class TrainingConfig:
    model: ModelConfig
    sft: SFTConfig
    lora: LoraConfig
    data_path: str
    val_split: float
    resume_from: str | None

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        return cls(
            model=ModelConfig.from_dict(d["model"]),
            sft=SFTConfig.from_dict(d["sft"]),
            lora=LoraConfig.from_dict(d["lora"]),
            data_path=d["data_path"],
            val_split=d.get("val_split", 0.0),
            resume_from=d.get("resume_from"),
        )


def load_generate_config(path: str) -> GenerateConfig:
    """Load generate configuration from a YAML file and return a typed config."""
    try:
        with open(path, "r") as f:
            return GenerateConfig.from_dict(yaml.safe_load(f))
    except FileNotFoundError:
        log.error("Generate config not found: %s", path)
        raise
    except Exception:
        log.error("Failed to parse generate config: %s", path)
        raise


def load_training_config(path: str) -> TrainingConfig:
    """Load training configuration from a YAML file and return a typed config."""
    try:
        with open(path, "r") as f:
            return TrainingConfig.from_dict(yaml.safe_load(f))
    except FileNotFoundError:
        log.error("Training config not found: %s", path)
        raise
    except Exception:
        log.error("Failed to parse training config: %s", path)
        raise
