"""GRPO training loop for negotiation self-play.

For each episode:
    1. Sample scenario + assign clone persona
    2. Run negotiation (max turns from config)
       - At each learner turn: sample G candidates
       - Score each with CompositeReward
       - Compute GRPO advantage: A_i = (r_i - mean) / std
       - Update policy to increase log-prob of positive-advantage candidates
    3. After the episode: run an episode-level REINFORCE step using the
       terminal reward to retroactively reinforce/penalise chosen responses
    4. Every N episodes: sync clone weights from learner

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
from torch.optim.lr_scheduler import LambdaLR

from rl.config import GRPOConfig, RewardConfig, load_grpo_config
from rl.env.negotiation import NegotiationEnv, Turn, _parse_agent_output
from rl.env.personas import PERSONAS
from rl.env.scenario import sample_scenario
from rl.grpo.checkpoint import load_checkpoint, save_checkpoint
from rl.grpo.models import load_model_and_tokenizer, sync_clone
from rl.grpo.policy import (
    _clip_and_step,
    clone_generate,
    compute_grpo_advantages,
    episode_reinforce_step,
    generate_candidates,
    policy_gradient_step,
)
from rl.rewards.composite import CompositeReward

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
RUNS_DIR = Path("runs")

_CLONE_MAX_RETRIES = 2


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


def grpo_train(
    grpo_cfg: GRPOConfig, reward_cfg: RewardConfig, run_timestamp: str
) -> None:
    """Main GRPO training loop."""
    rng = random.Random(grpo_cfg.seed)
    torch.manual_seed(grpo_cfg.seed)

    run_dir = RUNS_DIR / f"grpo_train_{run_timestamp}"
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir.resolve())

    n_episodes = 50  # for testing
    max_new_tokens = 512
    temperature = grpo_cfg.temperature

    n_gpus = torch.cuda.device_count()
    logger.info("Detected %d GPU(s)", n_gpus)
    if n_gpus >= 2:
        learner_device_map: dict | str = {"": 0}
        clone_device_map: dict | str = {"": 1}
        ref_device_map: dict | str = {"": 1}
        logger.info("Multi-GPU: learner → cuda:0, clone+ref → cuda:1")
    else:
        raise SystemExit("At least 2 GPUs are required for training.")

    logger.info("Loading learner model...")
    learner_model, tokenizer = load_model_and_tokenizer(
        grpo_cfg, device_map=learner_device_map
    )

    logger.info("Loading clone model...")
    clone_model, _ = load_model_and_tokenizer(grpo_cfg, device_map=clone_device_map)
    clone_model.eval()
    for param in clone_model.parameters():
        param.requires_grad = False

    logger.info("Loading frozen KL reference model...")
    ref_model, _ = load_model_and_tokenizer(grpo_cfg, device_map=ref_device_map)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    logger.info("KL reference model frozen (SFT weights, never updated)")

    reward = CompositeReward(
        config=reward_cfg,
        terminal_lambda=grpo_cfg.terminal_lambda,  # for the final payoff across the two agents
        walkaway_penalty=grpo_cfg.walkaway_penalty,
    )

    optimizer = torch.optim.AdamW(
        [p for p in learner_model.parameters() if p.requires_grad],
        lr=grpo_cfg.training.lr,
        weight_decay=0.0,
    )

    grad_accum = grpo_cfg.training.grad_accum
    warmup_ratio = grpo_cfg.training.warmup_ratio
    total_steps = n_episodes // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    logger.info(
        "LR schedule: %d warmup steps, %d total steps (grad_accum=%d)",
        warmup_steps, total_steps, grad_accum,
    )

    persona_names = list(PERSONAS.keys())
    out_dir = Path(grpo_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "grpo_log.jsonl"

    ROLLING_WINDOW = min(50, max(n_episodes // 5, 1))
    recent_rewards: deque[float] = deque(maxlen=ROLLING_WINDOW)
    recent_deals: deque[bool] = deque(maxlen=ROLLING_WINDOW)

    start_episode = 0
    if grpo_cfg.resume_from:
        resume_dir = Path(grpo_cfg.resume_from)
        if resume_dir.exists():
            start_episode, recent_rewards, recent_deals = load_checkpoint(
                resume_dir,
                learner_model,
                optimizer,
                rng,
                ROLLING_WINDOW,
            )
            sync_clone(learner_model, clone_model)
        else:
            logger.warning("resume_from path %s not found — starting fresh", resume_dir)

    train_start = time.time()

    with (run_dir / "run_meta.json").open("w") as _f:
        json.dump(
            {
                "timestamp": run_timestamp,
                "base_model": grpo_cfg.base_model,
                "sft_adapter_path": grpo_cfg.sft_adapter_path,
                "n_episodes": n_episodes,
                "candidates_per_turn": grpo_cfg.candidates_per_turn,
                "max_turns": grpo_cfg.max_turns,
                "kl_coeff": grpo_cfg.kl_coeff,
                "temperature": temperature,
                "seed": grpo_cfg.seed,
                "resumed_from": grpo_cfg.resume_from,
                "start_episode": start_episode,
                "lora": {
                    "rank": grpo_cfg.lora.rank,
                    "alpha": grpo_cfg.lora.alpha,
                    "dropout": grpo_cfg.lora.dropout,
                },
                "reward": {
                    "terminal_weight": reward_cfg.terminal_weight,
                    "format_weight": reward_cfg.format_weight,
                    "arithmetic_weight": reward_cfg.arithmetic_weight,
                    "partner_model_weight": reward_cfg.partner_model_weight,
                    "action_quality_weight": reward_cfg.action_quality_weight,
                },
            },
            _f,
            indent=2,
        )

    avg_reward = 0.0
    deal_rate = 0.0
    optimizer.zero_grad()

    for episode in range(start_episode, n_episodes):
        ep_start = time.time()
        scenario = sample_scenario(rng)
        persona = PERSONAS[rng.choice(persona_names)]

        logger.debug(
            "Episode %d/%d: persona=%s scenario_items=%s",
            episode + 1,
            n_episodes,
            persona.name,
            scenario,
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
                candidates = generate_candidates(
                    learner_model,
                    tokenizer,
                    prompt,
                    grpo_cfg.candidates_per_turn,
                    max_new_tokens,
                    temperature,
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
                        mock_turn,
                        env.current_turn // 2,
                        scenario,
                        episode,
                    )
                    turn_scores.append(score)

                advantages = compute_grpo_advantages(turn_scores)

                best_idx = max(range(len(turn_scores)), key=lambda i: turn_scores[i])
                logger.debug(
                    "  learner turn %d: valid=%d/%d  scores=[%s]  best=%d (%.3f)",
                    learner_turn_idx,
                    valid_count,
                    len(candidates),
                    ", ".join(f"{s:.3f}" for s in turn_scores),
                    best_idx,
                    turn_scores[best_idx],
                )
                logger.debug(
                    "  best candidate (first 300 chars): %s",
                    candidates[best_idx][:300].replace("\n", "\\n"),
                )

                ep_turns.append(
                    {
                        "turn": env.current_turn,
                        "agent": "learner",
                        "prompt": prompt,
                        "candidates": candidates,
                        "scores": turn_scores,
                        "advantages": advantages,
                        "best_idx": best_idx,
                        "best_response": candidates[best_idx],
                    }
                )

                policy_gradient_step(
                    learner_model,
                    ref_model,
                    tokenizer,
                    optimizer,
                    prompt,
                    candidates,
                    advantages,
                    grpo_cfg.kl_coeff,
                    accumulate_only=True,
                )

                env.step(candidates[best_idx])
                episode_rewards.append(turn_scores[best_idx])
                learner_turn_idx += 1

            else:
                clone_prompt = env.build_clone_prompt()
                clone_output = clone_generate(
                    clone_model,
                    tokenizer,
                    clone_prompt,
                    max_new_tokens,
                )

                _, _, _, clone_valid = _parse_agent_output(clone_output, scenario)
                if not clone_valid:
                    for _retry in range(_CLONE_MAX_RETRIES - 1):
                        logger.debug("  clone output invalid — retry %d", _retry + 1)
                        clone_output = clone_generate(
                            clone_model,
                            tokenizer,
                            clone_prompt,
                            max_new_tokens,
                        )
                        _, _, _, clone_valid = _parse_agent_output(
                            clone_output, scenario
                        )
                        if clone_valid:
                            break

                logger.debug(
                    "  clone turn (first 300 chars): %s",
                    clone_output[:300].replace("\n", "\\n"),
                )
                ep_turns.append(
                    {
                        "turn": env.current_turn,
                        "agent": "clone",
                        "prompt": clone_prompt,
                        "response": clone_output,
                    }
                )
                env.step(clone_output)

        ep_reward = reward.score_episode(env, episode)
        ep_elapsed = time.time() - ep_start

        baseline = avg_reward if recent_rewards else 0.0
        scaled_ep_reward = ep_reward * grpo_cfg.episode_weight
        scaled_baseline = baseline * grpo_cfg.episode_weight
        episode_reinforce_step(
            learner_model,
            ref_model,
            tokenizer,
            optimizer,
            ep_turns,
            scaled_ep_reward,
            scaled_baseline,
            grpo_cfg.kl_coeff,
            accumulate_only=True,
        )

        if (episode + 1) % grad_accum == 0:
            _clip_and_step(learner_model, optimizer)
            optimizer.zero_grad()
            scheduler.step()

        recent_rewards.append(ep_reward)
        recent_deals.append(bool(env.deal_reached))
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        deal_rate = sum(recent_deals) / len(recent_deals)

        ep_file = episodes_dir / f"ep_{episode:04d}.jsonl"
        with ep_file.open("w") as _ef:
            for turn_record in ep_turns:
                _ef.write(json.dumps(turn_record) + "\n")
            _ef.write(
                json.dumps(
                    {
                        "episode_summary": True,
                        "episode": episode,
                        "scenario": {
                            "items": scenario.items,
                            "agent_values": scenario.agent_values,
                            "partner_values": scenario.partner_values,
                        },
                        "persona": persona.name,
                        "deal": bool(env.deal_reached),
                        "turns": env.current_turn,
                        "reward": ep_reward,
                        "elapsed_s": round(ep_elapsed, 2),
                    }
                )
                + "\n"
            )

        with log_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "episode": episode,
                        "reward": ep_reward,
                        "deal": env.deal_reached,
                        "turns": env.current_turn,
                        "persona": persona.name,
                        "avg_reward": round(avg_reward, 4),
                        "deal_rate": round(deal_rate, 4),
                        "elapsed_s": round(ep_elapsed, 2),
                    }
                )
                + "\n"
            )

        logger.info(
            "Ep %d/%d  reward=%.4f  avg(%d)=%.4f  deal=%s  rate=%.0f%%  "
            "turns=%d  persona=%-15s  %.1fs",
            episode + 1,
            n_episodes,
            ep_reward,
            len(recent_rewards),
            avg_reward,
            env.deal_reached,
            deal_rate * 100,
            env.current_turn,
            persona.name,
            ep_elapsed,
        )

        if (episode + 1) % grpo_cfg.clone_sync_interval == 0:
            sync_clone(learner_model, clone_model)

        if (episode + 1) % grpo_cfg.training.save_steps == 0:
            save_checkpoint(
                out_dir,
                str(episode + 1),
                learner_model,
                tokenizer,
                optimizer,
                episode,
                rng,
                recent_rewards,
                recent_deals,
            )

    total_elapsed = time.time() - train_start
    logger.info("Training complete")
    logger.info(
        "  %d episodes in %.1f min (%.1f s/ep avg)",
        n_episodes - start_episode,
        total_elapsed / 60,
        total_elapsed / max(n_episodes - start_episode, 1),
    )
    if recent_rewards:
        logger.info(
            "  final avg(%d) reward=%.4f  deal_rate=%.0f%%",
            len(recent_rewards),
            avg_reward,
            deal_rate * 100,
        )

    final_dir = out_dir / "adapter"
    learner_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Final adapter saved to %s", final_dir)


if __name__ == "__main__":
    _timestamp = setup_logging()
    grpo_cfg, reward_cfg = load_grpo_config(Path("rl/configs/grpo.yaml"))
    grpo_train(grpo_cfg, reward_cfg, run_timestamp=_timestamp)
