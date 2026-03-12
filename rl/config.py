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
class SFTRawConfig:
    """Thin wrapper around the raw ``sft:`` YAML section.

    All keys are forwarded directly to ``trl.SFTConfig`` (which extends
    ``transformers.TrainingArguments``), so any field supported by TRL or
    HF TrainingArguments can be specified in the YAML without changing
    this class.
    """

    raw: dict

    @classmethod
    def from_dict(cls, d: dict) -> SFTRawConfig:
        return cls(raw=dict(d))


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
    sft: SFTRawConfig
    lora: LoraConfig
    data_path: str
    val_split: float
    resume_from: str | None

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        return cls(
            model=ModelConfig.from_dict(d["model"]),
            sft=SFTRawConfig.from_dict(d["sft"]),
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

@dataclass
class RolloutConfig:
    max_turns: int
    temperature: float
    top_p: float
    persona_weights: dict[str, float]

    @classmethod
    def from_dict(cls, d: dict) -> RolloutConfig:
        return cls(
            max_turns=d.get("max_turns", 10),
            temperature=d.get("temperature", 0.7),
            top_p=d.get("top_p", 0.9),
            persona_weights=d.get(
                "persona_weights",
                {
                    "uncompromising": 0.25,
                    "selfish": 0.25,
                    "anchoring": 0.25,
                    "cooperative": 0.25,
                },
            ),
        )


@dataclass
class GRPORawConfig:
    """Thin wrapper around the raw ``grpo:`` YAML section.

    All keys are forwarded directly to ``trl.GRPOConfig``.
    """

    raw: dict

    @classmethod
    def from_dict(cls, d: dict) -> GRPORawConfig:
        return cls(raw=dict(d))


@dataclass
class GRPOTrainingConfig:
    model: ModelConfig
    rollout: RolloutConfig
    grpo: GRPORawConfig
    lora: LoraConfig
    data_path: str
    sft_checkpoint: str
    reward_weights: list[float]
    max_episodes: int | None
    resume_from: str | None

    @classmethod
    def from_dict(cls, d: dict) -> GRPOTrainingConfig:
        return cls(
            model=ModelConfig.from_dict(d["model"]),
            rollout=RolloutConfig.from_dict(d.get("rollout", {})),
            grpo=GRPORawConfig.from_dict(d["grpo"]),
            lora=LoraConfig.from_dict(d["lora"]),
            data_path=d["data_path"],
            sft_checkpoint=d.get("sft_checkpoint", ""),
            reward_weights=d.get("reward_weights", [0.2, 0.3, 0.5]),
            max_episodes=d.get("max_episodes"),
            resume_from=d.get("resume_from"),
        )


def load_grpo_config(path: str) -> GRPOTrainingConfig:
    """Load GRPO training configuration from a YAML file."""
    try:
        with open(path, "r") as f:
            return GRPOTrainingConfig.from_dict(yaml.safe_load(f))
    except FileNotFoundError:
        log.error("GRPO config not found: %s", path)
        raise
    except Exception:
        log.error("Failed to parse GRPO config: %s", path)
        raise
