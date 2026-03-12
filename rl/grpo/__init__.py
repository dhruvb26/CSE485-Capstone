__all__ = [
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

_ROLLOUT = {"Episode", "run_self_play"}
_REWARDS = {"format_reward", "offer_reward", "terminal_reward"}
_DATA = {"load_scenarios", "episodes_to_dataset"}
_TRAIN = {"load_grpo_model_and_tokenizer", "build_grpo_trainer", "run_grpo_training"}


def __getattr__(name: str):
    if name in _ROLLOUT:
        from rl.grpo import rollout

        return getattr(rollout, name)
    if name in _REWARDS:
        from rl.grpo import rewards

        return getattr(rewards, name)
    if name in _DATA:
        from rl.grpo import data

        return getattr(data, name)
    if name in _TRAIN:
        from rl.grpo import train

        return getattr(train, name)
    raise AttributeError(f"module 'rl.grpo' has no attribute {name!r}")
