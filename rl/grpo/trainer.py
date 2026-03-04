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
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rl.config import GRPOConfig, RewardConfig, load_grpo_config
from rl.env.negotiation import NegotiationEnv
from rl.env.personas import PERSONAS
from rl.env.scenario import sample_scenario
from rl.rewards.composite import CompositeReward

logger = logging.getLogger(__name__)


def _load_model_and_tokenizer(cfg: GRPOConfig):
    """Load base model with optional 4-bit quantisation and LoRA."""
    t = cfg.training
    quant_config = None
    if t.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if t.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
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
    """
    model.train()
    total_loss = torch.tensor(0.0, device=model.device)

    for candidate, advantage in zip(candidates, advantages):
        if abs(advantage) < 1e-8:
            continue

        full_text = prompt + candidate
        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        prompt_len = prompt_ids["input_ids"].shape[1]

        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits[:, prompt_len - 1:-1, :]
        target_ids = inputs["input_ids"][:, prompt_len:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        mean_log_prob = token_log_probs.mean()

        loss = -advantage * mean_log_prob + kl_coeff * (-mean_log_prob)
        total_loss = total_loss + loss

    if total_loss.requires_grad:
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _clone_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Generate a single response from the clone model."""
    candidates = _generate_candidates(model, tokenizer, prompt, 1, max_new_tokens, 0.7)
    return candidates[0] if candidates else ""


def _sync_clone(learner_model, clone_model) -> None:
    """Copy learner's LoRA weights to the clone."""
    learner_state = {
        k: v.clone() for k, v in learner_model.state_dict().items()
        if "lora" in k.lower()
    }
    clone_state = clone_model.state_dict()
    clone_state.update(learner_state)
    clone_model.load_state_dict(clone_state)
    logger.info("Synced clone weights from learner")


def grpo_train(grpo_cfg: GRPOConfig, reward_cfg: RewardConfig) -> None:
    """Main GRPO training loop."""
    rng = random.Random(grpo_cfg.seed)
    torch.manual_seed(grpo_cfg.seed)

    logger.info("Loading learner model...")
    learner_model, tokenizer = _load_model_and_tokenizer(grpo_cfg)

    logger.info("Loading clone model...")
    clone_model, _ = _load_model_and_tokenizer(grpo_cfg)
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

    n_episodes = grpo_cfg.training.epochs * 1000
    max_new_tokens = 256

    for episode in range(n_episodes):
        scenario = sample_scenario(rng)
        persona = PERSONAS[rng.choice(persona_names)]

        env = NegotiationEnv(
            scenario=scenario,
            persona=persona,
            max_turns=grpo_cfg.max_turns,
        )

        episode_rewards = []

        while not env.is_done:
            if env.is_learner_turn:
                prompt = env.build_learner_prompt()
                candidates = _generate_candidates(
                    learner_model, tokenizer, prompt,
                    grpo_cfg.candidates_per_turn,
                    max_new_tokens, 0.8,
                )

                turn_scores = []
                for cand in candidates:
                    temp_turn = env.step.__func__  # noqa: peek without stepping
                    from rl.env.negotiation import _parse_agent_output
                    thought, talk, action, valid = _parse_agent_output(cand, scenario)
                    from rl.env.negotiation import Turn
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

                _policy_gradient_step(
                    learner_model, tokenizer, optimizer,
                    prompt, candidates, advantages,
                    grpo_cfg.kl_coeff,
                )

                best_idx = max(range(len(turn_scores)), key=lambda i: turn_scores[i])
                env.step(candidates[best_idx])
                episode_rewards.append(turn_scores[best_idx])

            else:
                clone_prompt = env.build_clone_prompt()
                clone_output = _clone_generate(
                    clone_model, tokenizer, clone_prompt, max_new_tokens,
                )
                env.step(clone_output)

        ep_reward = reward.score_episode(env, episode)

        with log_path.open("a") as f:
            f.write(json.dumps({
                "episode": episode,
                "reward": ep_reward,
                "deal": env.deal_reached,
                "turns": env.current_turn,
                "persona": persona.name,
            }) + "\n")

        if (episode + 1) % 50 == 0:
            logger.info(
                "Episode %d: reward=%.4f deal=%s turns=%d persona=%s",
                episode + 1, ep_reward, env.deal_reached, env.current_turn, persona.name,
            )

        if (episode + 1) % grpo_cfg.clone_sync_interval == 0:
            _sync_clone(learner_model, clone_model)

        if (episode + 1) % grpo_cfg.training.save_steps == 0:
            adapter_dir = out_dir / f"adapter-ep{episode + 1}"
            learner_model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            logger.info("Saved checkpoint to %s", adapter_dir)

    final_dir = out_dir / "adapter"
    learner_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Training complete. Final adapter saved to %s", final_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    grpo_cfg, reward_cfg = load_grpo_config(Path("rl/grpo.config.yaml"))
    grpo_train(grpo_cfg, reward_cfg)
