__all__ = [
    "DatasetConfig",
    "GenerateConfig",
    "TrainingConfig",
    "ModelConfig",
    "SFTConfig",
    "LoraConfig",
    "load_generate_config",
    "load_training_config",
    "build_sft_messages",
    "build_annotation_requests",
    "merge_annotations",
    "count_conversations",
    "load_conversation",
    "load_all_conversations",
    "annotate_agent",
    "create_openai_client",
    "load_model_and_tokenizer",
    "load_sft_dataset",
    "build_trainer",
    "run_training",
]

_CONFIG = {
    "DatasetConfig",
    "GenerateConfig",
    "LoraConfig",
    "ModelConfig",
    "SFTConfig",
    "TrainingConfig",
    "load_generate_config",
    "load_training_config",
}
_SFT = set(__all__) - _CONFIG


def __getattr__(name: str):
    if name in _CONFIG:
        from rl import config
        return getattr(config, name)
    if name in _SFT:
        from rl import sft
        return getattr(sft, name)
    raise AttributeError(f"module 'rl' has no attribute {name!r}")
