__all__ = [
    "build_sft_messages",
    "build_annotation_requests",
    "merge_annotations",
    "count_conversations",
    "load_conversation",
    "load_all_conversations",
    "annotate_agent",
    "create_openai_client",
    "build_trainer",
    "load_model_and_tokenizer",
    "load_sft_dataset",
    "run_training",
]

_DATA = {
    "build_sft_messages",
    "build_annotation_requests",
    "merge_annotations",
    "count_conversations",
    "load_conversation",
    "load_all_conversations",
}
_GENERATE = {"annotate_agent", "create_openai_client"}
_TRAIN = {"build_trainer", "load_model_and_tokenizer", "load_sft_dataset", "run_training"}


def __getattr__(name: str):
    if name in _DATA:
        from rl.sft import data
        return getattr(data, name)
    if name in _GENERATE:
        from rl.sft import generate
        return getattr(generate, name)
    if name in _TRAIN:
        from rl.sft import train
        return getattr(train, name)
    raise AttributeError(f"module 'rl.sft' has no attribute {name!r}")
