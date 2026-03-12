__all__ = [
    "DatasetConfig",
    "GenerateConfig",
    "TrainingConfig",
    "ModelConfig",
    "SFTRawConfig",
    "LoraConfig",
    "RolloutConfig",
    "GRPORawConfig",
    "GRPOTrainingConfig",
    "load_generate_config",
    "load_training_config",
    "load_grpo_config",
    "build_sft_messages",
    "build_annotation_context",
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
    "Episode",
    "run_self_play",
    "format_reward",
    "offer_reward",
    "terminal_reward",
    "load_scenarios",
    "episodes_to_dataset",
    "load_grpo_model_and_tokenizer",
    "build_grpo_trainer",
    "run_grpo_training",
]

_CONFIG = {
    "DatasetConfig",
    "GenerateConfig",
    "LoraConfig",
    "ModelConfig",
    "SFTRawConfig",
    "TrainingConfig",
    "RolloutConfig",
    "GRPORawConfig",
    "GRPOTrainingConfig",
    "load_generate_config",
    "load_training_config",
    "load_grpo_config",
}
_SFT = {
    "build_sft_messages",
    "build_annotation_context",
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
}
_GRPO = {
    "Episode",
    "run_self_play",
    "format_reward",
    "offer_reward",
    "terminal_reward",
    "load_scenarios",
    "episodes_to_dataset",
    "load_grpo_model_and_tokenizer",
    "build_grpo_trainer",
    "run_grpo_training",
}


def __getattr__(name: str):
    if name in _CONFIG:
        from rl import config

        return getattr(config, name)
    if name in _SFT:
        from rl import sft

        return getattr(sft, name)
    if name in _GRPO:
        from rl import grpo

        return getattr(grpo, name)
    raise AttributeError(f"module 'rl' has no attribute {name!r}")
