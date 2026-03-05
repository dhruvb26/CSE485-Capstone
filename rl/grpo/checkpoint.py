"""Checkpoint save / load for GRPO training state."""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CKPT_FILENAME = "training_state.pt"


def save_checkpoint(
    out_dir: Path,
    tag: str,
    learner_model,
    tokenizer,
    optimizer,
    episode: int,
    rng: random.Random,
    recent_rewards: deque,
    recent_deals: deque,
) -> Path:
    """Persist adapter weights, optimizer state, and RNG state to disk."""
    ckpt_dir = out_dir / f"adapter-ep{tag}"
    learner_model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    state = {
        "episode": episode,
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "recent_rewards": list(recent_rewards),
        "recent_deals": list(recent_deals),
    }
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()

    torch.save(state, ckpt_dir / CKPT_FILENAME)
    logger.info("Saved checkpoint to %s (episode %d)", ckpt_dir, episode)
    return ckpt_dir


def load_checkpoint(
    ckpt_dir: Path,
    learner_model,
    optimizer,
    rng: random.Random,
    rolling_window: int,
) -> tuple[int, deque, deque]:
    """Restore training state from a checkpoint directory.

    Returns ``(start_episode, recent_rewards, recent_deals)``.
    """
    from safetensors.torch import load_file as load_safetensors
    from peft import set_peft_model_state_dict

    adapter_path = ckpt_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        weights = load_safetensors(str(adapter_path))
        set_peft_model_state_dict(learner_model, weights)
        logger.info("Restored adapter weights from %s", ckpt_dir)

    state_path = ckpt_dir / CKPT_FILENAME
    if not state_path.exists():
        logger.warning("No %s found in %s — starting fresh", CKPT_FILENAME, ckpt_dir)
        return 0, deque(maxlen=rolling_window), deque(maxlen=rolling_window)

    state = torch.load(state_path, weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    rng.setstate(state["rng_state"])
    torch.random.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_states" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])

    start_episode = state["episode"] + 1

    recent_rewards: deque[float] = deque(state["recent_rewards"], maxlen=rolling_window)
    recent_deals: deque[bool] = deque(state["recent_deals"], maxlen=rolling_window)

    logger.info("Resumed from episode %d (checkpoint %s)", start_episode, ckpt_dir)
    return start_episode, recent_rewards, recent_deals
