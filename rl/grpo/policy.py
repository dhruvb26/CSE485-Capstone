"""GRPO policy gradient utilities — candidate generation, advantage
computation, per-turn and episode-level updates.

Per-turn GRPO gradients are masked to <action>…</action> tokens only,
keeping thought/talk formatting stable against noisy per-candidate
advantages.  The episode-level REINFORCE step is unmasked so the
terminal outcome signal can improve strategic reasoning in <thought>
tokens (regularised by KL against the frozen SFT reference).
"""

from __future__ import annotations

import logging
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 2048

ACTION_OPEN_TAG = "<action>"
ACTION_CLOSE_TAG = "</action>"


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
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    )
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


def clone_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
) -> str:
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


def _compute_prompt_len(tokenizer, prompt: str, full_text: str, max_length: int) -> int:
    """Return the number of tokens in the full_text encoding that belong to the prompt.

    Uses ``return_offsets_mapping`` (fast tokenizers) to find the first token
    whose character offset starts at or after the prompt/response boundary,
    falling back to prompt-only tokenization length for slow tokenizers.
    """
    prompt_char_end = len(prompt)
    try:
        enc = tokenizer(
            full_text, truncation=True, max_length=max_length,
            return_offsets_mapping=True, return_tensors=None,
        )
        offsets = enc["offset_mapping"]
        for i, (start, _end) in enumerate(offsets):
            if start >= prompt_char_end:
                return max(i, 1)
        return len(enc["input_ids"])
    except Exception:
        prompt_ids = tokenizer(
            prompt, truncation=True, max_length=max_length, return_tensors=None,
        )["input_ids"]
        return len(prompt_ids)


def build_action_mask(
    tokenizer: PreTrainedTokenizerBase,
    full_text: str,
    prompt_len: int,
    max_length: int,
    prompt_char_len: int = 0,
) -> torch.Tensor | None:
    """Build a binary mask over response tokens that is 1 inside <action>…</action>.

    Locates the action span via plain string search on the raw text, then
    maps character offsets to token positions through the tokenizer's offset
    mapping.  This is immune to the BPE-boundary mismatches that break the
    earlier "tokenise the tag in isolation and search for that subsequence"
    approach.

    *prompt_char_len* is the character length of the prompt prefix so that the
    search skips any ``<action>`` tags that appear inside the prompt template.

    Returns a 1-D bool tensor of shape (n_response_tokens,), or None if the
    action span cannot be located (caller should skip this candidate).
    """
    open_char = full_text.find(ACTION_OPEN_TAG, prompt_char_len)
    close_char = full_text.find(ACTION_CLOSE_TAG, prompt_char_len)
    if open_char == -1 or close_char == -1 or close_char <= open_char:
        return None

    content_start_char = open_char + len(ACTION_OPEN_TAG)
    content_end_char = close_char

    try:
        enc = tokenizer(
            full_text, truncation=True, max_length=max_length,
            return_offsets_mapping=True, return_tensors=None,
        )
    except Exception:
        logger.warning("Tokenizer does not support offset mapping; cannot build action mask")
        return None

    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    response_ids = full_ids[prompt_len:]
    n_resp = len(response_ids)
    if n_resp == 0:
        return None

    response_offsets = offsets[prompt_len:]

    mask = torch.zeros(n_resp, dtype=torch.bool)
    for i, (tok_start, tok_end) in enumerate(response_offsets):
        if tok_start < content_end_char and tok_end > content_start_char:
            mask[i] = True

    if not mask.any():
        return None

    return mask


def _clip_and_step(
    model: PreTrainedModel, optimizer: torch.optim.Optimizer, label: str = ""
) -> None:
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
    action_mask: torch.Tensor | None = None,
) -> bool:
    """Compute and backward a single (candidate, advantage) contribution.

    Shared by both per-turn GRPO and episode-level REINFORCE.

    When *action_mask* is provided (a 1-D bool tensor over response tokens),
    only the masked positions contribute to the policy-gradient and KL terms.
    This restricts RL signal to <action> tokens only.
    """
    ref_device = next(ref_model.parameters()).device

    outputs = model(**inputs)
    logits = outputs.logits[:, prompt_len - 1 : -1, :]
    target_ids = inputs["input_ids"][:, prompt_len:]

    if target_ids.shape[1] == 0:
        return False

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        ref_input_ids = inputs["input_ids"].to(ref_device)
        ref_kw = {"input_ids": ref_input_ids}
        if "attention_mask" in inputs:
            ref_kw["attention_mask"] = inputs["attention_mask"].to(ref_device)
        ref_logits = (
            ref_model(**ref_kw).logits[:, prompt_len - 1 : -1, :].to(model.device)
        )
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        ref_token_log_probs = ref_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(
            -1
        )

    if action_mask is not None:
        mask = action_mask.to(token_log_probs.device)
        if mask.shape[0] != token_log_probs.shape[1]:
            mask = mask[: token_log_probs.shape[1]]
        mask_f = mask.float().unsqueeze(0)
        n_masked = mask_f.sum().clamp(min=1)
        mean_log_prob = (token_log_probs * mask_f).sum() / n_masked
        ref_mean_log_prob = (ref_token_log_probs * mask_f).sum() / n_masked
    else:
        mean_log_prob = token_log_probs.mean()
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
    accumulate_only: bool = False,
) -> None:
    """Single GRPO policy gradient update step (action-masked).

    RL gradients only flow through <action>…</action> tokens.  Candidates
    whose action span cannot be located are skipped entirely.

    When *accumulate_only* is True, gradients are accumulated but the
    optimizer step is deferred (caller manages zero_grad / step).
    """
    n_active = sum(1 for a in advantages if abs(a) >= 1e-8)
    if n_active == 0:
        return

    model.train()
    if not accumulate_only:
        optimizer.zero_grad()

    for candidate, advantage in zip(candidates, advantages):
        if abs(advantage) < 1e-8:
            continue

        full_text = prompt + candidate
        prompt_len = _compute_prompt_len(tokenizer, prompt, full_text, MAX_SEQ_LEN)

        action_mask = build_action_mask(
            tokenizer, full_text, prompt_len, MAX_SEQ_LEN,
            prompt_char_len=len(prompt),
        )
        if action_mask is None:
            logger.debug("  skipping candidate — no <action> span found")
            continue

        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        _kl_weighted_loss(
            model,
            ref_model,
            inputs,
            prompt_len,
            advantage,
            kl_coeff,
            n_active,
            action_mask=action_mask,
        )

    if not accumulate_only:
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
    accumulate_only: bool = False,
) -> None:
    """REINFORCE-style update using the episode-level terminal reward.

    Unlike the per-turn GRPO step, gradients are **unmasked** — they flow
    through all response tokens (thought, talk, and action).  This allows
    the episode-level outcome signal to improve strategic reasoning in
    <thought> while the KL penalty against the frozen SFT reference
    prevents thought formatting from drifting.

    When *accumulate_only* is True, gradients are accumulated but the
    optimizer step is deferred (caller manages zero_grad / step).
    """
    advantage = ep_reward - baseline
    if abs(advantage) < 1e-8:
        return

    model.train()
    if not accumulate_only:
        optimizer.zero_grad()

    n_turns = sum(1 for t in episode_turns if t.get("agent") == "learner")
    if n_turns == 0:
        return

    for turn_record in episode_turns:
        if turn_record.get("agent") != "learner":
            continue
        prompt = turn_record["prompt"]
        response = turn_record["best_response"]

        full_text = prompt + response
        prompt_len = _compute_prompt_len(tokenizer, prompt, full_text, MAX_SEQ_LEN)

        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        _kl_weighted_loss(
            model,
            ref_model,
            inputs,
            prompt_len,
            advantage,
            kl_coeff,
            n_turns,
        )

    if not accumulate_only:
        _clip_and_step(model, optimizer, label="episode reinforce")
