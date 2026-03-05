"""GRPO training loop for negotiation self-play.

For each episode:
    1. Sample scenario + assign clone persona
    2. Run negotiation (max turns from config)
       - At each learner turn: sample G candidates
       - Score each with CompositeReward
       - Compute GRPO advantage: A_i = (r_i - mean) / std
       - Update policy to increase log-prob of positive-advantage candidates
    3. Every N episodes: sync clone weights from learner

Usage:
    python -m rl.grpo.trainer
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rl.config import GRPOConfig, RewardConfig, load_grpo_config
from rl.env.negotiation import NegotiationEnv, Turn, _parse_agent_output
from rl.env.personas import PERSONAS
from rl.env.scenario import sample_scenario
from rl.rewards.composite import CompositeReward

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
RUNS_DIR = Path("runs")


def setup_logging() -> str:
    """Configure root logger to write to both stderr and a timestamped log file.

    Returns the timestamp string so callers can reuse it for the run directory.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"grpo_train_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    logger.info("Logging to %s", log_file.resolve())
    return timestamp


def _device_map_is_cpu(device_map) -> bool:
    """Return True if device_map routes the entire model to CPU."""
    if isinstance(device_map, str):
        return device_map == "cpu"
    if isinstance(device_map, dict):
        return bool(device_map) and all(v == "cpu" for v in device_map.values())
    return False


def _load_model_and_tokenizer(
    cfg: GRPOConfig,
    device_map: dict | str = {"": 0},
    gradient_checkpointing: bool = False,
):
    """Load base model with optional 4-bit quantisation and LoRA.

    Args:
        device_map: Passed directly to ``from_pretrained``. Use ``{"": 0}`` /
            ``{"": 1}`` to pin to a specific GPU, or ``{"": "cpu"}`` to force
            CPU placement (single-GPU mode for the clone).
        gradient_checkpointing: Recompute activations during backprop to trade
            compute for VRAM. Enable for the learner; leave False for the clone.
    """
    t = cfg.training
    # bitsandbytes 4-bit is CUDA-only; fall back to fp16 for CPU placement.
    on_gpu = not _device_map_is_cpu(device_map)
    quant_config = None
    if t.load_in_4bit and on_gpu:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = PeftLoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if cfg.sft_adapter_path:
        from safetensors.torch import load_file as load_safetensors
        from peft import set_peft_model_state_dict
        adapter_weights = load_safetensors(
            str(Path(cfg.sft_adapter_path) / "adapter_model.safetensors")
        )
        set_peft_model_state_dict(model, adapter_weights)
        logger.info("Loaded SFT adapter from %s", cfg.sft_adapter_path)

    return model, tokenizer


@torch.no_grad()
def _generate_candidates(
    model,
    tokenizer,
    prompt: str,
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Sample *n_candidates* completions from the model for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=max(temperature, 0.01),
        num_return_sequences=n_candidates,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    candidates = []
    for seq in outputs:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        candidates.append(text)
    return candidates


def _compute_grpo_advantages(scores: list[float]) -> list[float]:
    """Normalise scores into GRPO advantages: A_i = (r_i - mean) / std."""
    if len(scores) <= 1:
        return [0.0] * len(scores)
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(s - mean) / std for s in scores]


def _policy_gradient_step(
    model,
    ref_model,
    tokenizer,
    optimizer,
    prompt: str,
    candidates: list[str],
    advantages: list[float],
    kl_coeff: float,
):
    """Single GRPO policy gradient update step.

    Increases log-probability of candidates with positive advantage,
    decreases for negative advantage.

    Each candidate is backward-ed immediately so only one computation graph
    lives in memory at a time. Loss is divided by the number of active
    candidates so the effective gradient magnitude is independent of G.

    KL divergence is computed token-level against ref_model (the clone),
    keeping the policy from drifting too far from the SFT checkpoint.
    """
    n_active = sum(1 for a in advantages if abs(a) >= 1e-8)
    if n_active == 0:
        return

    ref_device = next(ref_model.parameters()).device

    model.train()
    optimizer.zero_grad()

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    prompt_len = prompt_ids["input_ids"].shape[1]

    for candidate, advantage in zip(candidates, advantages):
        if abs(advantage) < 1e-8:
            continue

        full_text = prompt + candidate
        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits[:, prompt_len - 1:-1, :]
        target_ids = inputs["input_ids"][:, prompt_len:]

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
        loss = (-(advantage * mean_log_prob) + kl_coeff * kl) / n_active
        loss.backward()

    raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    logger.debug("  grad norm before clip: %.4f", raw_grad_norm.item())
    if raw_grad_norm.item() != raw_grad_norm.item():  # NaN check
        logger.warning("  NaN gradient detected — skipping optimizer step")
        optimizer.zero_grad()
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def _clone_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Generate a single response from the clone model."""
    candidates = _generate_candidates(model, tokenizer, prompt, 1, max_new_tokens, 0.7)
    return candidates[0] if candidates else ""


def _sync_clone(learner_model, clone_model) -> None:
    """Copy learner's LoRA weights to the clone."""
    clone_device = next(clone_model.parameters()).device
    learner_state = {
        k: v.clone().to(clone_device) for k, v in learner_model.state_dict().items()
        if "lora" in k.lower()
    }
    clone_state = clone_model.state_dict()
    clone_state.update(learner_state)
    clone_model.load_state_dict(clone_state)
    logger.info("Synced clone weights from learner")


def grpo_train(grpo_cfg: GRPOConfig, reward_cfg: RewardConfig, run_timestamp: str | None = None) -> None:
    """Main GRPO training loop."""
    rng = random.Random(grpo_cfg.seed)
    torch.manual_seed(grpo_cfg.seed)

    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"grpo_train_{timestamp}"
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir.resolve())

    # n_episodes = grpo_cfg.training.epochs * 1000
    n_episodes = 10 # for testing
    max_new_tokens = 512

    n_gpus = torch.cuda.device_count()
    logger.info("Detected %d GPU(s)", n_gpus)
    if n_gpus >= 2:
        learner_device_map: dict | str = {"": 0}
        clone_device_map: dict | str = {"": 1}
        logger.info("Multi-GPU: learner → cuda:0, clone → cuda:1")
    else:
        learner_device_map = {"": 0}
        clone_device_map = {"": "cpu"}
        logger.info(
            "Single-GPU: learner → cuda:0 (4-bit), clone → cpu (fp16). "
            "Clone generation will be slow; allocate 2 GPUs to avoid this."
        )

    logger.info("Loading learner model...")
    learner_model, tokenizer = _load_model_and_tokenizer(
        grpo_cfg, device_map=learner_device_map, gradient_checkpointing=False
    )

    logger.info("Loading clone model...")
    clone_model, _ = _load_model_and_tokenizer(
        grpo_cfg, device_map=clone_device_map, gradient_checkpointing=False
    )
    clone_model.eval()

    reward = CompositeReward(
        config=reward_cfg,
        terminal_lambda=grpo_cfg.terminal_lambda,
        walkaway_penalty=grpo_cfg.walkaway_penalty,
    )

    optimizer = torch.optim.AdamW(
        [p for p in learner_model.parameters() if p.requires_grad],
        lr=grpo_cfg.training.lr,
    )

    persona_names = list(PERSONAS.keys())
    out_dir = Path(grpo_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "grpo_log.jsonl"

    ROLLING_WINDOW = min(50, max(n_episodes // 5, 1))
    recent_rewards: deque[float] = deque(maxlen=ROLLING_WINDOW)
    recent_deals: deque[bool] = deque(maxlen=ROLLING_WINDOW)
    train_start = time.time()

    with (run_dir / "run_meta.json").open("w") as _f:
        json.dump(
            {
                "timestamp": timestamp,
                "base_model": grpo_cfg.base_model,
                "sft_adapter_path": grpo_cfg.sft_adapter_path,
                "n_episodes": n_episodes,
                "candidates_per_turn": grpo_cfg.candidates_per_turn,
                "max_turns": grpo_cfg.max_turns,
                "kl_coeff": grpo_cfg.kl_coeff,
                "seed": grpo_cfg.seed,
                "lora": {"rank": grpo_cfg.lora.rank, "alpha": grpo_cfg.lora.alpha, "dropout": grpo_cfg.lora.dropout},
                "reward": {
                    "terminal_weight": reward_cfg.terminal_weight,
                    "format_weight": reward_cfg.format_weight,
                    "arithmetic_weight": reward_cfg.arithmetic_weight,
                    "strategy_weight": reward_cfg.strategy_weight,
                    "partner_model_weight": reward_cfg.partner_model_weight,
                },
            },
            _f,
            indent=2,
        )

    for episode in range(n_episodes):
        ep_start = time.time()
        scenario = sample_scenario(rng)
        persona = PERSONAS[rng.choice(persona_names)]

        logger.debug(
            "Episode %d/%d: persona=%s scenario_items=%s",
            episode + 1, n_episodes, persona.name, scenario,
        )

        env = NegotiationEnv(
            scenario=scenario,
            persona=persona,
            max_turns=grpo_cfg.max_turns,
        )

        episode_rewards = []
        learner_turn_idx = 0
        ep_turns: list[dict] = []

        while not env.is_done:
            if env.is_learner_turn:
                prompt = env.build_learner_prompt()
                candidates = _generate_candidates(
                    learner_model, tokenizer, prompt,
                    grpo_cfg.candidates_per_turn,
                    max_new_tokens, 0.8,
                )

                turn_scores = []
                valid_count = 0
                for cand in candidates:
                    thought, talk, action, valid = _parse_agent_output(cand, scenario)
                    if valid:
                        valid_count += 1
                    mock_turn = Turn(
                        agent="learner",
                        raw_output=cand,
                        thought=thought,
                        talk=talk,
                        action=action,
                        valid=valid,
                    )
                    score = reward.score_turn(
                        mock_turn, env.current_turn // 2, scenario, episode,
                    )
                    turn_scores.append(score)

                advantages = _compute_grpo_advantages(turn_scores)

                best_idx = max(range(len(turn_scores)), key=lambda i: turn_scores[i])
                logger.debug(
                    "  learner turn %d: valid=%d/%d  scores=[%s]  best=%d (%.3f)",
                    learner_turn_idx, valid_count, len(candidates),
                    ", ".join(f"{s:.3f}" for s in turn_scores),
                    best_idx, turn_scores[best_idx],
                )
                logger.debug(
                    "  best candidate (first 300 chars): %s",
                    candidates[best_idx][:300].replace("\n", "\\n"),
                )

                ep_turns.append({
                    "turn": env.current_turn,
                    "agent": "learner",
                    "prompt": prompt,
                    "candidates": candidates,
                    "scores": turn_scores,
                    "advantages": advantages,
                    "best_idx": best_idx,
                    "best_response": candidates[best_idx],
                })

                _policy_gradient_step(
                    learner_model, clone_model, tokenizer, optimizer,
                    prompt, candidates, advantages,
                    grpo_cfg.kl_coeff,
                )

                env.step(candidates[best_idx])
                episode_rewards.append(turn_scores[best_idx])
                learner_turn_idx += 1

            else:
                clone_prompt = env.build_clone_prompt()
                clone_output = _clone_generate(
                    clone_model, tokenizer, clone_prompt, max_new_tokens,
                )
                logger.debug(
                    "  clone turn (first 300 chars): %s",
                    clone_output[:300].replace("\n", "\\n"),
                )
                ep_turns.append({
                    "turn": env.current_turn,
                    "agent": "clone",
                    "prompt": clone_prompt,
                    "response": clone_output,
                })
                env.step(clone_output)

        ep_reward = reward.score_episode(env, episode)
        ep_elapsed = time.time() - ep_start

        recent_rewards.append(ep_reward)
        recent_deals.append(bool(env.deal_reached))
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        deal_rate = sum(recent_deals) / len(recent_deals)

        ep_file = episodes_dir / f"ep_{episode:04d}.jsonl"
        with ep_file.open("w") as _ef:
            for turn_record in ep_turns:
                _ef.write(json.dumps(turn_record) + "\n")
            _ef.write(json.dumps({
                "episode_summary": True,
                "episode": episode,
                "persona": persona.name,
                "deal": bool(env.deal_reached),
                "turns": env.current_turn,
                "reward": ep_reward,
                "elapsed_s": round(ep_elapsed, 2),
            }) + "\n")

        with log_path.open("a") as f:
            f.write(json.dumps({
                "episode": episode,
                "reward": ep_reward,
                "deal": env.deal_reached,
                "turns": env.current_turn,
                "persona": persona.name,
                "avg_reward": round(avg_reward, 4),
                "deal_rate": round(deal_rate, 4),
                "elapsed_s": round(ep_elapsed, 2),
            }) + "\n")

        logger.info(
            "Ep %d/%d  reward=%.4f  avg(%d)=%.4f  deal=%s  rate=%.0f%%  "
            "turns=%d  persona=%-15s  %.1fs",
            episode + 1, n_episodes, ep_reward,
            len(recent_rewards), avg_reward,
            env.deal_reached, deal_rate * 100,
            env.current_turn, persona.name, ep_elapsed,
        )

        if (episode + 1) % grpo_cfg.clone_sync_interval == 0:
            _sync_clone(learner_model, clone_model)

        if (episode + 1) % grpo_cfg.training.save_steps == 0:
            adapter_dir = out_dir / f"adapter-ep{episode + 1}"
            learner_model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            logger.info("Saved checkpoint to %s", adapter_dir)

    total_elapsed = time.time() - train_start
    logger.info("Training complete")
    logger.info(
        "  %d episodes in %.1f min (%.1f s/ep avg)",
        n_episodes, total_elapsed / 60, total_elapsed / max(n_episodes, 1),
    )
    if recent_rewards:
        logger.info(
            "  final avg(%d) reward=%.4f  deal_rate=%.0f%%",
            len(recent_rewards), avg_reward, deal_rate * 100,
        )

    final_dir = out_dir / "adapter"
    learner_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Final adapter saved to %s", final_dir)


if __name__ == "__main__":
    _timestamp = setup_logging()
    grpo_cfg, reward_cfg = load_grpo_config(Path("rl/grpo.config.yaml"))
    grpo_train(grpo_cfg, reward_cfg, run_timestamp=_timestamp)
