"""GRPO policy gradient utilities — candidate generation, advantage
computation, per-turn and episode-level updates."""

from __future__ import annotations

import logging
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 2048

@torch.no_grad()
def generate_candidates(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Sample *n_candidates* completions from the model for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n_candidates,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.15,
    )

    prompt_len = inputs["input_ids"].shape[1]
    candidates = []
    for seq in outputs:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        candidates.append(text)
    return candidates


def clone_generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str, max_new_tokens: int) -> str:
    """Generate a single response from the clone model."""
    candidates = generate_candidates(model, tokenizer, prompt, 1, max_new_tokens, 0.7)
    return candidates[0] if candidates else ""

def compute_grpo_advantages(scores: list[float]) -> list[float]:
    """Normalise scores into GRPO advantages: A_i = (r_i - mean) / std."""
    if len(scores) <= 1:
        return [0.0] * len(scores)
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(s - mean) / std for s in scores]

def _clip_and_step(model: PreTrainedModel, optimizer: torch.optim.Optimizer, label: str = "") -> None:
    """Shared grad-norm clip + NaN guard + optimizer step."""
    raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    prefix = f"  {label} " if label else "  "
    logger.debug("%sgrad norm before clip: %.4f", prefix, raw_grad_norm.item())
    if raw_grad_norm.item() != raw_grad_norm.item():  # NaN check
        logger.warning("%sNaN gradient detected — skipping optimizer step", prefix)
        optimizer.zero_grad()
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def _kl_weighted_loss(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    inputs: dict,
    prompt_len: int,
    advantage: float,
    kl_coeff: float,
    divisor: int,
) -> bool:
    """Compute and backward a single (candidate, advantage) contribution.

    Shared by both per-turn GRPO and episode-level REINFORCE.
    """
    ref_device = next(ref_model.parameters()).device

    outputs = model(**inputs)
    logits = outputs.logits[:, prompt_len - 1:-1, :]
    target_ids = inputs["input_ids"][:, prompt_len:]

    if target_ids.shape[1] == 0:
        return False

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    mean_log_prob = token_log_probs.mean()

    with torch.no_grad():
        ref_input_ids = inputs["input_ids"].to(ref_device)
        ref_kw = {"input_ids": ref_input_ids}
        if "attention_mask" in inputs:
            ref_kw["attention_mask"] = inputs["attention_mask"].to(ref_device)
        ref_logits = ref_model(**ref_kw).logits[:, prompt_len - 1:-1, :].to(model.device)
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        ref_token_log_probs = ref_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        ref_mean_log_prob = ref_token_log_probs.mean()

    kl = mean_log_prob - ref_mean_log_prob
    loss = (-(advantage * mean_log_prob) + kl_coeff * kl) / divisor
    loss.backward()
    return True


def policy_gradient_step(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    prompt: str,
    candidates: list[str],
    advantages: list[float],
    kl_coeff: float,
) -> None:
    """Single GRPO policy gradient update step.

    Increases log-probability of candidates with positive advantage,
    decreases for negative advantage.

    Each candidate is backward-ed immediately so only one computation graph
    lives in memory at a time.  Loss is divided by the number of active
    candidates so the effective gradient magnitude is independent of G.

    KL divergence is computed token-level against *ref_model* (frozen SFT
    checkpoint), keeping the policy from drifting too far from its starting
    point.
    """
    n_active = sum(1 for a in advantages if abs(a) >= 1e-8)
    if n_active == 0:
        return

    model.train()
    optimizer.zero_grad()

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    prompt_len = prompt_ids["input_ids"].shape[1]

    for candidate, advantage in zip(candidates, advantages):
        if abs(advantage) < 1e-8:
            continue

        full_text = prompt + candidate
        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        _kl_weighted_loss(
            model, ref_model, tokenizer, inputs,
            prompt_len, advantage, kl_coeff, n_active,
        )

    _clip_and_step(model, optimizer)


def episode_reinforce_step(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    episode_turns: list[dict],
    ep_reward: float,
    baseline: float,
    kl_coeff: float,
) -> None:
    """REINFORCE-style update using the episode-level terminal reward.

    For each learner turn that produced a chosen response, compute a
    policy gradient weighted by (ep_reward - baseline).  This feeds the
    terminal outcome signal back into the per-turn policy.
    """
    advantage = ep_reward - baseline
    if abs(advantage) < 1e-8:
        return

    model.train()
    optimizer.zero_grad()

    n_turns = sum(1 for t in episode_turns if t.get("agent") == "learner")
    if n_turns == 0:
        return

    for turn_record in episode_turns:
        if turn_record.get("agent") != "learner":
            continue
        prompt = turn_record["prompt"]
        response = turn_record["best_response"]

        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        prompt_len = prompt_ids["input_ids"].shape[1]

        full_text = prompt + response
        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        _kl_weighted_loss(
            model, ref_model, tokenizer, inputs,
            prompt_len, advantage, kl_coeff, n_turns,
        )

    _clip_and_step(model, optimizer, label="episode reinforce")
