from __future__ import annotations

import argparse
import ast
import csv
import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import trackio
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)

POINTS = {"High": 5, "Medium": 4, "Low": 3}
MAX_POINTS = 3 * 5 + 3 * 4 + 3 * 3

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

SYSTEM_PROMPT = """You are negotiating with your campsite neighbor over 3 packages each of food, water, and firewood. Each item allocation must be 0-3 and both sides must sum to 3 per item.

Your priorities:

{items_block}

Your reply must include these 3 parts in order:

Thought: your private strategic reasoning (not shown to the neighbor).
Talk: what you say to the neighbor. Be concise.
Action: exactly one of the following:

1. [SUBMIT_DEAL] food:F water:W firewood:FW if you want to propose a deal. F, W, FW are YOUR allocations (0-3 each). Your neighbor receives 3 minus each value. When your neighbor proposes a [SUBMIT_DEAL], the values shown are what YOU would receive.
2. [ACCEPT_DEAL] if you agree to the neighbor's most recent [SUBMIT_DEAL]. This ends the negotiation and closes the deal.
3. [REJECT_DEAL] if you want to reject the neighbor's most recent offer and await a new offer.
4. [WALK_AWAY] if you believe a good deal cannot be reached. This ends the negotiation with no deal."""


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config dtype string (``"bf16"``, ``"float32"``, …) to a torch dtype."""
    key = (name or "bfloat16").lower()
    if key not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype {name!r}; choose one of {sorted(DTYPE_MAP)}"
        )
    return DTYPE_MAP[key]


def flip_deal_perspective(text: str) -> str:
    """Convert [SUBMIT_DEAL] values from submitter's to receiver's perspective."""

    def _flip(m: re.Match) -> str:
        f, w, fw = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"[SUBMIT_DEAL] food:{3 - f} water:{3 - w} firewood:{3 - fw}"

    return re.sub(
        r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)",
        _flip,
        text,
        flags=re.IGNORECASE,
    )


def parse_submit_deal(text: str) -> dict[str, int] | None:
    """Extract a ``{food, water, firewood}`` allocation from a ``[SUBMIT_DEAL]`` action."""
    m = re.search(
        r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if m is None:
        return None
    return {
        "food": int(m.group(1)),
        "water": int(m.group(2)),
        "firewood": int(m.group(3)),
    }


def extract_section(text: str, label: str) -> str | None:
    """Extract content after a Thought:/Talk:/Action: label."""
    m = re.search(
        rf"^{label}:\s*(.+?)(?=^(?:Thought|Talk|Action):|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def strip_thought(text: str) -> str:
    """Remove the private ``Thought:`` section before showing a turn to the opponent."""
    return re.sub(
        r"^Thought:.*?(?=^(?:Talk|Action):|\Z)",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    ).strip()


def compute_points_for_deal(deal: dict[str, int], value2issue: dict[str, str]) -> int:
    """Score a learner allocation against that participant's High/Medium/Low priorities."""
    issue2points = {v.lower(): POINTS[k] for k, v in value2issue.items()}
    return sum(units * issue2points.get(item, 0) for item, units in deal.items())


def build_system_prompt(participant_info: dict, agent_id: str) -> str:
    """Render the negotiation system prompt with that agent's priority table."""
    value2issue = participant_info[agent_id]["value2issue"]
    value2reason = participant_info[agent_id]["value2reason"]
    items_block = "\n  ".join(
        f"{value2issue[p]} ({POINTS[p]} points) - {value2reason[p]}" for p in POINTS
    )
    return SYSTEM_PROMPT.format(items_block=items_block)


@dataclass
class Scenario:
    """One CaSiNo negotiation: both participants' priorities and agent ids."""

    participant_info: dict
    agent_ids: list[str]


@dataclass
class LearnerTurn:
    """Token ids for one learner turn, split for the GRPO forward pass."""

    prompt_ids: list[int]
    completion_ids: list[int]


@dataclass
class Episode:
    """A single negotiation rollout: the learner's turns plus the final outcome."""

    turns: list[LearnerTurn] = field(default_factory=list)
    reward: float = 0.0
    outcome: str = "max_turns"
    learner_points: int | None = None
    num_turns: int = 0


def load_scenarios(csv_path: str) -> list[Scenario]:
    """Parse a CaSiNo CSV (``chat_logs,participant_info,annotations``) into scenarios."""
    scenarios: list[Scenario] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            participant_info = ast.literal_eval(row["participant_info"])
            scenarios.append(
                Scenario(
                    participant_info=participant_info,
                    agent_ids=list(participant_info.keys()),
                )
            )
    return scenarios


def load_learner(
    model_name: str,
    dtype: str = "bfloat16",
    lora_cfg: dict | None = None,
    gradient_checkpointing: bool = True,
    max_memory: dict | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model + tokenizer and (optionally) wrap with a LoRA adapter.

    This model is used for the GRPO training forward/backward pass.
    Generation is handled separately by the vLLM engine.
    """
    torch_dtype = resolve_dtype(dtype)
    logger.info(f"Loading learner {model_name} (dtype={torch_dtype})")

    kwargs: dict = {
        "dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if max_memory:
        kwargs["max_memory"] = {
            (int(k) if str(k).isdigit() else k): v for k, v in max_memory.items()
        }

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if lora_cfg:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=lora_cfg.get("target_modules", "all-linear"),
            task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, tokenizer


def create_vllm_engine(
    model_name: str,
    dtype: str = "bfloat16",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.45,
    max_lora_rank: int = 64,
    tensor_parallel_size: int = 1,
) -> LLM:
    """Create a vLLM engine with LoRA support for fast generation."""
    logger.info(
        "Creating vLLM engine (tp=%d, gpu_mem=%.2f)",
        tensor_parallel_size,
        gpu_memory_utilization,
    )
    return LLM(
        model=model_name,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )


def generate_batch(
    engine: LLM,
    tokenizer: AutoTokenizer,
    messages_batch: list[list[dict]],
    temperature: float,
    max_new_tokens: int,
    lora_request: LoRARequest | None = None,
) -> list[tuple[str, list[int], list[int]]]:
    """Batched chat generation via vLLM; returns ``(text, prompt_ids, completion_ids)`` per input."""
    prompt_texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    outputs = engine.generate(
        prompt_texts,
        sampling_params,
        lora_request=lora_request,
    )
    results: list[tuple[str, list[int], list[int]]] = []
    for output in outputs:
        prompt_ids = list(output.prompt_token_ids)
        completion_ids = list(output.outputs[0].token_ids)
        text = output.outputs[0].text.strip()
        results.append((text, prompt_ids, completion_ids))
    return results


def sync_lora_weights(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    adapter_path: str,
    lora_id: int = 1,
) -> LoRARequest:
    """Save the current LoRA adapter to disk and return a fresh :class:`LoRARequest`.

    vLLM loads the adapter from the filesystem, so we write the updated weights
    after each GRPO step and bump ``lora_id`` to force a reload.
    """
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    return LoRARequest("learner_lora", lora_id, adapter_path)


def run_episode_group(
    engine: LLM,
    tokenizer: AutoTokenizer,
    scenario: Scenario,
    group_size: int,
    max_turns: int,
    max_new_tokens: int,
    learner_temperature: float = 1.0,
    opponent_temperature: float = 0.7,
    lora_request: LoRARequest | None = None,
) -> list[Episode]:
    """Run ``group_size`` negotiation rollouts in parallel on one scenario.

    The learner generates with the LoRA adapter (via *lora_request*) and the
    opponent generates with the base model (no LoRA).
    """
    ids = scenario.agent_ids
    learner_id, opponent_id = (
        (ids[0], ids[1]) if random.random() < 0.5 else (ids[1], ids[0])
    )
    learner_goes_first = random.random() < 0.5

    learner_system = build_system_prompt(scenario.participant_info, learner_id)
    opponent_system = build_system_prompt(scenario.participant_info, opponent_id)

    episodes = [Episode() for _ in range(group_size)]
    learner_msgs: list[list[dict]] = [
        [{"role": "system", "content": learner_system}] for _ in range(group_size)
    ]
    opponent_msgs: list[list[dict]] = [
        [{"role": "system", "content": opponent_system}] for _ in range(group_size)
    ]
    alive = [True] * group_size
    last_submit_deal: list[dict[str, int] | None] = [None] * group_size
    last_submit_by: list[str | None] = [None] * group_size
    recent_submits: list[list[str]] = [[] for _ in range(group_size)]

    for turn in range(max_turns):
        active = [i for i, a in enumerate(alive) if a]
        if not active:
            break

        is_learner_turn = (turn % 2 == 0) == learner_goes_first
        side = "learner" if is_learner_turn else "opponent"

        if is_learner_turn:
            batch_msgs = []
            for i in active:
                if len(learner_msgs[i]) == 1:
                    learner_msgs[i].append(
                        {"role": "user", "content": "Begin the negotiation."}
                    )
                batch_msgs.append(learner_msgs[i])
            results = generate_batch(
                engine,
                tokenizer,
                batch_msgs,
                learner_temperature,
                max_new_tokens,
                lora_request=lora_request,
            )
        else:
            batch_msgs = []
            for i in active:
                if len(opponent_msgs[i]) == 1:
                    opponent_msgs[i].append(
                        {"role": "user", "content": "Begin the negotiation."}
                    )
                batch_msgs.append(opponent_msgs[i])
            results = generate_batch(
                engine,
                tokenizer,
                batch_msgs,
                opponent_temperature,
                max_new_tokens,
                lora_request=None,
            )

        for pos, i in enumerate(active):
            text, prompt_ids, completion_ids = results[pos]
            ep = episodes[i]

            if is_learner_turn:
                learner_msgs[i].append({"role": "assistant", "content": text})
                opponent_msgs[i].append(
                    {
                        "role": "user",
                        "content": flip_deal_perspective(strip_thought(text)),
                    }
                )
                ep.turns.append(
                    LearnerTurn(prompt_ids=prompt_ids, completion_ids=completion_ids)
                )

                if not (
                    extract_section(text, "Thought")
                    and extract_section(text, "Talk")
                    and extract_section(text, "Action")
                ):
                    ep.outcome = "format_violation"
                    ep.reward = -1.0
                    ep.num_turns = turn + 1
                    alive[i] = False
                    continue
            else:
                opponent_msgs[i].append({"role": "assistant", "content": text})
                learner_msgs[i].append(
                    {
                        "role": "user",
                        "content": flip_deal_perspective(strip_thought(text)),
                    }
                )

            action = extract_section(text, "Action") or ""
            action_upper = action.strip().upper()

            parsed = parse_submit_deal(action)
            if parsed is not None:
                last_submit_deal[i] = parsed
                last_submit_by[i] = side
                recent_submits[i].append(action.strip())
                if len(recent_submits[i]) > 3:
                    recent_submits[i].pop(0)

            if (
                "[ACCEPT_DEAL]" in action_upper
                and last_submit_deal[i] is not None
                and last_submit_by[i] is not None
                and last_submit_by[i] != side
            ):
                ep.outcome = "deal"
                if is_learner_turn:
                    learner_alloc = {k: 3 - v for k, v in last_submit_deal[i].items()}
                else:
                    learner_alloc = last_submit_deal[i]
                ep.learner_points = compute_points_for_deal(
                    learner_alloc, scenario.participant_info[learner_id]["value2issue"]
                )
                ep.reward = ep.learner_points / MAX_POINTS
                ep.num_turns = turn + 1
                alive[i] = False
                continue

            if "[WALK_AWAY]" in action_upper:
                ep.outcome = "walk_away"
                ep.num_turns = turn + 1
                alive[i] = False
                continue

            if len(recent_submits[i]) >= 3 and len(set(recent_submits[i])) == 1:
                ep.outcome = "reject_loop"
                ep.num_turns = turn + 1
                alive[i] = False
                continue

    for i, ep in enumerate(episodes):
        if alive[i]:
            ep.outcome = "max_turns"
            ep.num_turns = max_turns

    return episodes


def grpo_update(
    model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    episode_groups: list[list[Episode]],
    kl_beta: float = 0.0,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """One GRPO optimizer step over a list of episode groups.

    Per-token loss is ``-A * log pi(a|s) + kl_beta * KL(pi || pi_ref)`` with the
    DeepSeek k3 unbiased KL estimator. The reference is the same model with
    LoRA disabled; ``kl_beta=0`` skips the reference pass entirely. Logits are
    cast to float32 before ``log_softmax`` to avoid bf16 underflow on the large
    Qwen vocabulary.
    """
    model.train()
    optimizer.zero_grad()
    device = next(model.parameters()).device
    use_kl = kl_beta > 0.0

    total_policy_loss = 0.0
    total_kl = 0.0
    total_tokens = 0
    all_rewards: list[float] = []
    all_advantages: list[float] = []

    for group in episode_groups:
        rewards = torch.tensor([ep.reward for ep in group], dtype=torch.float32)
        all_rewards.extend(rewards.tolist())

        if rewards.std().item() < 1e-8:
            continue

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        all_advantages.extend(advantages.tolist())

        for ep_index, ep in enumerate(group):
            advantage = advantages[ep_index].item()
            if abs(advantage) < 1e-8 and not use_kl:
                continue

            for turn in ep.turns:
                if not turn.completion_ids:
                    continue

                full_ids = torch.tensor(
                    turn.prompt_ids + turn.completion_ids,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                logits = model(input_ids=full_ids, use_cache=False).logits.float()

                prompt_len = len(turn.prompt_ids)
                completion_len = len(turn.completion_ids)
                completion_logits = logits[
                    0, prompt_len - 1 : prompt_len + completion_len - 1
                ]
                completion_targets = full_ids[
                    0, prompt_len : prompt_len + completion_len
                ]

                log_probs = F.log_softmax(completion_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    1, completion_targets.unsqueeze(1)
                ).squeeze(1)

                policy_loss = -(advantage * token_log_probs).sum()

                if use_kl:
                    with torch.no_grad(), model.disable_adapter():
                        ref_logits = model(
                            input_ids=full_ids, use_cache=False
                        ).logits.float()
                        ref_completion_logits = ref_logits[
                            0, prompt_len - 1 : prompt_len + completion_len - 1
                        ]
                        ref_log_probs = F.log_softmax(ref_completion_logits, dim=-1)
                        ref_token_log_probs = ref_log_probs.gather(
                            1, completion_targets.unsqueeze(1)
                        ).squeeze(1)

                    log_ratio = ref_token_log_probs - token_log_probs
                    kl_per_token = torch.exp(log_ratio) - log_ratio - 1.0
                    kl_term = kl_beta * kl_per_token.sum()
                else:
                    kl_term = torch.zeros((), device=device, dtype=policy_loss.dtype)

                turn_loss = policy_loss + kl_term
                turn_loss.backward()

                total_policy_loss += policy_loss.item()
                total_kl += float(kl_term.item())
                total_tokens += completion_len

    if total_tokens > 0:
        scale = 1.0 / float(total_tokens)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)

    optimizer.zero_grad()

    denom = max(total_tokens, 1)
    return {
        "train/loss": (total_policy_loss + total_kl) / denom,
        "train/policy_loss": total_policy_loss / denom,
        "train/kl": total_kl / denom,
        "train/kl_beta": float(kl_beta),
        "train/mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
        "train/mean_advantage": sum(all_advantages) / max(len(all_advantages), 1),
        "train/total_tokens": float(total_tokens),
        "train/num_groups": float(len(episode_groups)),
        "train/grad_norm": float(grad_norm),
    }


def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    step: int,
) -> None:
    """Write adapter weights and tokenizer to ``{output_dir}/checkpoint-{step}``."""
    path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info(f"Saved checkpoint to {path}")


def train(cfg: dict) -> None:
    """Run online GRPO training end-to-end from a parsed YAML config."""
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora")
    tcfg = cfg["training"]

    random.seed(42)
    torch.manual_seed(42)

    output_dir = tcfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    scenarios = load_scenarios(tcfg["csv_path"])
    logger.info(f"Loaded {len(scenarios)} scenarios from {tcfg['csv_path']}")

    learner, tokenizer = load_learner(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        lora_cfg=lora_cfg if tcfg.get("use_lora", True) else None,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        max_memory=model_cfg.get("max_memory"),
    )

    vllm_cfg = cfg.get("vllm", {})
    adapter_path = os.path.join(output_dir, "vllm-lora")
    lora_request = sync_lora_weights(learner, tokenizer, adapter_path, lora_id=1)
    engine = create_vllm_engine(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_model_len=int(vllm_cfg.get("max_model_len", 4096)),
        gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.45)),
        max_lora_rank=int(vllm_cfg.get("max_lora_rank", 64)),
        tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
    )
    lora_id = 1

    trainable = [p for p in learner.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(tcfg["learning_rate"]),
        betas=(float(tcfg.get("adam_beta1", 0.9)), float(tcfg.get("adam_beta2", 0.95))),
        eps=float(tcfg.get("adam_epsilon", 1e-8)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    trackio.init(
        project="agent-rlvr",
        name=f"online-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        space_id="dhruvb26/agent-rlvr",
        config=cfg,
    )

    num_iterations = int(tcfg["num_iterations"])
    groups_per_iter = int(tcfg["groups_per_iteration"])
    group_size = int(tcfg["group_size"])
    max_turns = int(tcfg["max_turns"])
    max_new_tokens = int(tcfg["max_new_tokens"])
    log_every = int(tcfg["log_every"])
    save_every = int(tcfg["save_every"])
    max_grad_norm = float(tcfg["max_grad_norm"])
    kl_beta = float(tcfg.get("kl_beta", 0.0))
    learner_temp = float(tcfg["learner_temperature"])
    opponent_temp = float(tcfg["opponent_temperature"])

    try:
        for iteration in range(1, num_iterations + 1):
            learner.eval()
            episode_groups: list[list[Episode]] = []
            for _ in range(groups_per_iter):
                scenario = random.choice(scenarios)
                group = run_episode_group(
                    engine=engine,
                    tokenizer=tokenizer,
                    scenario=scenario,
                    group_size=group_size,
                    max_turns=max_turns,
                    max_new_tokens=max_new_tokens,
                    learner_temperature=learner_temp,
                    opponent_temperature=opponent_temp,
                    lora_request=lora_request,
                )
                episode_groups.append(group)

            metrics = grpo_update(
                model=learner,
                optimizer=optimizer,
                episode_groups=episode_groups,
                kl_beta=kl_beta,
                max_grad_norm=max_grad_norm,
            )

            lora_id += 1
            lora_request = sync_lora_weights(
                learner,
                tokenizer,
                adapter_path,
                lora_id=lora_id,
            )

            outcomes: dict[str, int] = {}
            for grp in episode_groups:
                for ep in grp:
                    outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1
            metrics["train/iteration"] = float(iteration)
            for name, count in outcomes.items():
                metrics[f"outcomes/{name}"] = float(count)

            all_episodes = [ep for grp in episode_groups for ep in grp]
            n = max(len(all_episodes), 1)
            deal_eps = [
                ep
                for ep in all_episodes
                if ep.outcome == "deal" and ep.learner_points is not None
            ]

            metrics["episode/deal_rate"] = (
                sum(1 for ep in all_episodes if ep.outcome == "deal") / n
            )
            metrics["episode/format_violation_rate"] = (
                outcomes.get("format_violation", 0) / n
            )
            metrics["episode/walk_away_rate"] = outcomes.get("walk_away", 0) / n
            metrics["episode/avg_turns"] = sum(ep.num_turns for ep in all_episodes) / n
            metrics["episode/count"] = float(n)

            if deal_eps:
                deal_points = [ep.learner_points for ep in deal_eps]
                avg_points = sum(deal_points) / len(deal_points)
                metrics["episode/avg_points"] = avg_points
                metrics["episode/points_std"] = (
                    sum((p - avg_points) ** 2 for p in deal_points)
                    / max(len(deal_points) - 1, 1)
                ) ** 0.5
                metrics["episode/points_min"] = float(min(deal_points))
                metrics["episode/points_max"] = float(max(deal_points))
            else:
                metrics["episode/avg_points"] = 0.0

            groups_with_signal = sum(
                1
                for grp in episode_groups
                if torch.tensor([ep.reward for ep in grp]).std() > 1e-8
            )
            metrics["train/signal_rate"] = groups_with_signal / max(
                len(episode_groups), 1
            )

            if iteration % log_every == 0:
                logger.info(
                    "iter %d | loss=%.4f reward=%.3f adv=%.3f "
                    "grad_norm=%.3f tokens=%.0f outcomes=%s",
                    iteration,
                    metrics["train/loss"],
                    metrics["train/mean_reward"],
                    metrics["train/mean_advantage"],
                    metrics["train/grad_norm"],
                    metrics["train/total_tokens"],
                    outcomes,
                )
                trackio.log(metrics, step=iteration)

            if save_every > 0 and iteration % save_every == 0:
                save_checkpoint(learner, tokenizer, output_dir, iteration)
    finally:
        save_checkpoint(learner, tokenizer, output_dir, iteration)
        trackio.finish()


def main() -> None:
    """CLI entry point: parse ``--config`` and hand off to :func:`train`."""
    parser = argparse.ArgumentParser(description="RLVR Negotiation")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "online_grpo.yaml"),
        help="Path to the YAML training config",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
