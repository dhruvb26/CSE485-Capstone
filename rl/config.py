from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str
    dtype: str = "bfloat16"
    hf_home: str = ""


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: str = "all-linear"
    task_type: str = "CAUSAL_LM"


@dataclass
class SFTTrainerConfig:
    data_path: str
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    output_dir: str = "checkpoints/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 8e-6
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_strategy: str = "epoch"
    assistant_only_loss: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    report_to: str = "trackio"
    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class GRPOTrainerConfig:
    sft_checkpoint: str
    data_path: str
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    judge: "JudgeConfig" = field(default_factory=lambda: JudgeConfig())
    output_dir: str = "checkpoints/grpo"
    prompt_split: float = 0.5
    num_generations: int = 8
    beta: float = 0.04
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-7
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_strategy: str = "epoch"
    save_steps: int = 200
    save_total_limit: int | None = None
    gradient_checkpointing: bool = True
    bf16: bool = True
    report_to: str = "trackio"
    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class JudgeConfig:
    model: str = ""
    base_url: str = ""
    api_key_env: str = "OPENROUTER_API_KEY"


@dataclass
class SelfPlayConfig:
    sft_checkpoint: str
    csv_path: str
    opponent_checkpoint: str = ""
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    output_dir: str = "checkpoints/grpo-selfplay"
    prompt_split: float = 0.5
    num_episodes: int = 50
    max_turns: int = 18
    temperature: float = 0.7
    top_p: float = 0.9
    persona_weights: dict[str, float] = field(default_factory=lambda: {
        "uncompromising": 0.25,
        "selfish": 0.25,
        "anchoring": 0.25,
        "cooperative": 0.25,
    })
    num_generations: int = 8
    beta: float = 0.04
    num_train_epochs: int = 1
    num_online_iterations: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-7
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_strategy: str = "epoch"
    gradient_checkpointing: bool = True
    bf16: bool = True
    report_to: str = "trackio"
    extra_kwargs: dict = field(default_factory=dict)
