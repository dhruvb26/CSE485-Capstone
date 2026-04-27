"""Self-play negotiation evaluation harness.

Supports CaSiNo and DND datasets via pluggable NegotiateEnv.
Supports API (OpenAI-compatible, including Tinker) and local HF models.

Usage:
    python -m eval negotiate --config eval/configs/negotiate.yaml
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from eval.display import print_comparison_table, print_matchup_report
from eval.envs import NegotiateEnv, get_env

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None


def extract_section(text: str, section: str) -> str | None:
    """Extract content after 'Section:' up to the next section or end of text."""
    pat = re.compile(
        rf"(?:^|\n)\s*{section}\s*:\s*(.*?)(?=\n\s*(?:Thought|Talk|Action)\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(text)
    return m.group(1).strip() if m else None


def strip_thought(text: str) -> str:
    return re.sub(
        r"(?:^|\n)\s*Thought\s*:.*?(?=\n\s*(?:Talk|Action)\s*:|\Z)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def last_opponent_action_is_submit(prompt: list[dict]) -> bool:
    for msg in reversed(prompt):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return bool(
                re.search(r"\[SUBMIT_DEAL\]", msg.get("content", ""), re.IGNORECASE)
            )
    return False


PERSONAS = {
    "uncompromising": (
        "You are a tough negotiator. You insist on getting most of your "
        "highest-value items and rarely make concessions. Only accept a "
        "deal that gives you strong value."
    ),
    "selfish": (
        "You are a self-interested negotiator. Always try to claim most of "
        "your highest-value items. Make small concessions only if the "
        "negotiation is about to collapse."
    ),
    "anchoring": (
        "You are a strategic negotiator. Open with an extreme offer claiming "
        "most of the highest-value items. Move slowly and never concede "
        "more than 1 unit per turn."
    ),
    "cooperative": (
        "You are a friendly negotiator who values reaching a deal. You are "
        "willing to split items fairly and respond positively to reasonable "
        "offers."
    ),
}


@dataclass
class EpisodeResult:
    episode_id: int
    learner_label: str
    opponent_label: str
    persona: str
    outcome: str  # deal | walk_away | reject_loop | max_turns
    learner_points: int | None = None
    opponent_points: int | None = None
    num_turns: int = 0
    learner_messages: list[dict] = field(default_factory=list)
    opponent_messages: list[dict] = field(default_factory=list)
    learner_agent_id: str = ""
    opponent_agent_id: str = ""
    who_terminated: str = ""
    learner_total_turns: int = 0
    learner_format_ok: int = 0
    learner_malformed_deals: int = 0
    opponent_total_turns: int = 0
    opponent_format_ok: int = 0
    opponent_malformed_deals: int = 0


@dataclass
class AggregateMetrics:
    total_episodes: int = 0
    deal_count: int = 0
    walk_away_count: int = 0
    reject_loop_count: int = 0
    max_turns_count: int = 0

    learner_points_on_deals: list[int] = field(default_factory=list)
    opponent_points_on_deals: list[int] = field(default_factory=list)
    turns_on_deals: list[int] = field(default_factory=list)
    all_turns: list[int] = field(default_factory=list)

    learner_total_turns: int = 0
    learner_format_ok: int = 0
    learner_malformed_deals: int = 0
    opponent_total_turns: int = 0
    opponent_format_ok: int = 0
    opponent_malformed_deals: int = 0

    def add(self, ep: EpisodeResult):
        self.total_episodes += 1
        self.all_turns.append(ep.num_turns)

        self.learner_total_turns += ep.learner_total_turns
        self.learner_format_ok += ep.learner_format_ok
        self.learner_malformed_deals += ep.learner_malformed_deals
        self.opponent_total_turns += ep.opponent_total_turns
        self.opponent_format_ok += ep.opponent_format_ok
        self.opponent_malformed_deals += ep.opponent_malformed_deals

        if ep.outcome == "deal":
            self.deal_count += 1
            if ep.learner_points is not None:
                self.learner_points_on_deals.append(ep.learner_points)
            if ep.opponent_points is not None:
                self.opponent_points_on_deals.append(ep.opponent_points)
            self.turns_on_deals.append(ep.num_turns)
        elif ep.outcome == "walk_away":
            self.walk_away_count += 1
        elif ep.outcome == "reject_loop":
            self.reject_loop_count += 1
        else:
            self.max_turns_count += 1

    def summary(self) -> dict:
        import math

        n = max(self.total_episodes, 1)
        deals = self.learner_points_on_deals
        opp_deals = self.opponent_points_on_deals

        avg_lp = sum(deals) / len(deals) if deals else None
        avg_op = sum(opp_deals) / len(opp_deals) if opp_deals else None

        joint_scores = [lp + op for lp, op in zip(deals, opp_deals)]
        avg_joint = (
            round(sum(joint_scores) / len(joint_scores), 2) if joint_scores else None
        )

        score_ratios = [
            lp / (lp + op) if (lp + op) > 0 else 0.5 for lp, op in zip(deals, opp_deals)
        ]
        avg_score_ratio = (
            round(sum(score_ratios) / len(score_ratios), 3) if score_ratios else None
        )

        if len(deals) >= 2:
            mean_lp = sum(deals) / len(deals)
            var_lp = sum((x - mean_lp) ** 2 for x in deals) / (len(deals) - 1)
            std_learner_points = round(math.sqrt(var_lp), 2)
        else:
            std_learner_points = None

        avg_turns_deal = (
            sum(self.turns_on_deals) / len(self.turns_on_deals)
            if self.turns_on_deals
            else None
        )
        avg_turns_all = (
            sum(self.all_turns) / len(self.all_turns) if self.all_turns else None
        )
        ppt = avg_lp / avg_turns_deal if (avg_lp and avg_turns_deal) else None

        lt = max(self.learner_total_turns, 1)
        ot = max(self.opponent_total_turns, 1)

        return {
            "total_episodes": self.total_episodes,
            "deal_rate": self.deal_count / n,
            "walk_away_rate": self.walk_away_count / n,
            "reject_loop_rate": self.reject_loop_count / n,
            "max_turns_rate": self.max_turns_count / n,
            "avg_learner_points": round(avg_lp, 2) if avg_lp is not None else None,
            "avg_opponent_points": round(avg_op, 2) if avg_op is not None else None,
            "std_learner_points": std_learner_points,
            "avg_joint_score": avg_joint,
            "avg_score_ratio": avg_score_ratio,
            "avg_turns_to_deal": round(avg_turns_deal, 2)
            if avg_turns_deal is not None
            else None,
            "avg_turns_all": round(avg_turns_all, 2)
            if avg_turns_all is not None
            else None,
            "points_per_turn": round(ppt, 3) if ppt is not None else None,
            "deal_count": self.deal_count,
            "walk_away_count": self.walk_away_count,
            "reject_loop_count": self.reject_loop_count,
            "max_turns_count": self.max_turns_count,
            "learner_format_rate": round(self.learner_format_ok / lt, 3),
            "learner_malformed_deal_rate": round(self.learner_malformed_deals / lt, 3),
            "opponent_format_rate": round(self.opponent_format_ok / ot, 3),
            "opponent_malformed_deal_rate": round(
                self.opponent_malformed_deals / ot, 3
            ),
            "learner_total_turns": self.learner_total_turns,
            "opponent_total_turns": self.opponent_total_turns,
        }


def load_model_and_tokenizer(
    model_path: str,
    base_model: str | None = None,
    dtype: str = "bfloat16",
) -> tuple:
    adapter_config = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.isdir(model_path) and os.path.exists(adapter_config)

    if is_lora:
        if not base_model:
            raise ValueError(
                f"LoRA adapter at {model_path} requires base_model to be set"
            )
        logger.info(f"Loading base model {base_model} + LoRA from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_turn(
    model,
    tokenizer,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


OPENAI_BASE_URL = "https://api.openai.com/v1"
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


class APIGenerator:
    """Generates responses via an OpenAI-compatible chat API."""

    def __init__(self, model: str, base_url: str, api_key: str, max_tokens: int = 512):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def __call__(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return resp.choices[0].message.content.strip()


class LocalGenerator:
    """Generates responses via a locally loaded HuggingFace model."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def __call__(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        return generate_turn(
            self.model,
            self.tokenizer,
            messages,
            temperature,
            top_p,
            self.max_new_tokens,
        )


def make_generator(
    agent_cfg: dict,
    default_base_url: str = OPENAI_BASE_URL,
    default_api_key_env: str = "OPENAI_API_KEY",
) -> APIGenerator | LocalGenerator:
    """Factory: build a generator from a matchup agent config block.

    Top-level ``base_url`` / ``api_key_env`` from the YAML are passed as
    *default_** kwargs so individual agents only need to override when they
    differ from the global setting.
    """
    if agent_cfg.get("type", "local") == "api":
        api_key_env = agent_cfg.get("api_key_env", default_api_key_env)
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise ValueError(f"{api_key_env} env var required for api-type models")
        base_url = agent_cfg.get("base_url", default_base_url)
        return APIGenerator(
            model=agent_cfg["model"],
            base_url=base_url,
            api_key=api_key,
            max_tokens=agent_cfg.get("max_tokens", 512),
        )
    model, tok = load_model_and_tokenizer(
        agent_cfg["model_path"],
        agent_cfg.get("base_model"),
        agent_cfg.get("dtype", "bfloat16"),
    )
    return LocalGenerator(model, tok, max_new_tokens=agent_cfg.get("max_tokens", 512))


def run_episode(
    env: NegotiateEnv,
    learner_gen: APIGenerator | LocalGenerator,
    opponent_gen: APIGenerator | LocalGenerator,
    scenario: dict,
    agent_ids: list[str],
    persona: str,
    episode_id: int,
    learner_label: str,
    opponent_label: str,
    max_turns: int = 18,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> EpisodeResult:
    learner_id, opponent_id = agent_ids[0], agent_ids[1]
    if random.random() < 0.5:
        learner_id, opponent_id = opponent_id, learner_id

    learner_system = env.build_system_prompt(scenario, learner_id)
    base_opponent_system = env.build_system_prompt(scenario, opponent_id)
    if persona and persona != "none" and persona in PERSONAS:
        opponent_system = PERSONAS[persona] + "\n\n" + base_opponent_system
    else:
        opponent_system = base_opponent_system

    learner_msgs: list[dict] = [{"role": "system", "content": learner_system}]
    opponent_msgs: list[dict] = [{"role": "system", "content": opponent_system}]

    result = EpisodeResult(
        episode_id=episode_id,
        learner_label=learner_label,
        opponent_label=opponent_label,
        persona=persona,
        outcome="max_turns",
        learner_agent_id=learner_id,
        opponent_agent_id=opponent_id,
    )

    learner_goes_first = random.random() < 0.5
    last_submit_deal: dict[str, int] | None = None

    for turn_index in range(max_turns):
        is_learner_turn = (turn_index % 2 == 0) == learner_goes_first

        if is_learner_turn:
            if len(learner_msgs) == 1:
                learner_msgs.append(
                    {"role": "user", "content": "Begin the negotiation."}
                )
            response = learner_gen(learner_msgs, temperature, top_p)
            learner_msgs.append({"role": "assistant", "content": response})
            opponent_msgs.append(
                {
                    "role": "user",
                    "content": env.flip_deal(strip_thought(response), scenario),
                }
            )
        else:
            if len(opponent_msgs) == 1:
                opponent_msgs.append(
                    {"role": "user", "content": "Begin the negotiation."}
                )
            response = opponent_gen(opponent_msgs, temperature, top_p)
            opponent_msgs.append({"role": "assistant", "content": response})
            learner_msgs.append(
                {
                    "role": "user",
                    "content": env.flip_deal(strip_thought(response), scenario),
                }
            )

        thought = extract_section(response, "Thought")
        talk = extract_section(response, "Talk")
        action = extract_section(response, "Action")

        format_ok = False
        malformed_deal = False
        if thought is not None and talk is not None and action is not None:
            try:
                order_ok = (
                    re.search(
                        r"(?:^|\n)\s*Thought\s*:", response, re.IGNORECASE
                    ).start()
                    < re.search(r"(?:^|\n)\s*Talk\s*:", response, re.IGNORECASE).start()
                    < re.search(
                        r"(?:^|\n)\s*Action\s*:", response, re.IGNORECASE
                    ).start()
                )
            except (ValueError, AttributeError):
                order_ok = False

            if order_ok:
                if re.search(r"\[SUBMIT_DEAL\]", action, re.IGNORECASE):
                    deal = env.parse_deal(action)
                    format_ok = env.validate_deal(deal, scenario)
                    if not format_ok:
                        malformed_deal = True
                elif re.fullmatch(
                    r"\s*\[(TALK|ACCEPT_DEAL|REJECT_DEAL|WALK_AWAY)\]\s*",
                    action,
                    re.IGNORECASE,
                ):
                    if re.search(r"\[ACCEPT_DEAL\]", action, re.IGNORECASE):
                        msgs = learner_msgs if is_learner_turn else opponent_msgs
                        format_ok = last_opponent_action_is_submit(msgs)
                    else:
                        format_ok = True

        if is_learner_turn:
            result.learner_total_turns += 1
            if format_ok:
                result.learner_format_ok += 1
            if malformed_deal:
                result.learner_malformed_deals += 1
        else:
            result.opponent_total_turns += 1
            if format_ok:
                result.opponent_format_ok += 1
            if malformed_deal:
                result.opponent_malformed_deals += 1

        if action:
            parsed_deal = env.parse_deal(action)
            if parsed_deal is not None:
                last_submit_deal = parsed_deal

        if action and "[ACCEPT_DEAL]" in action:
            result.outcome = "deal"
            result.who_terminated = "learner" if is_learner_turn else "opponent"
            if last_submit_deal is not None:
                learner_alloc = (
                    env.invert_alloc(last_submit_deal, scenario)
                    if is_learner_turn
                    else last_submit_deal
                )
                opponent_alloc = (
                    last_submit_deal
                    if is_learner_turn
                    else env.invert_alloc(last_submit_deal, scenario)
                )
                result.learner_points = env.compute_points(
                    learner_alloc, scenario, learner_id
                )
                result.opponent_points = env.compute_points(
                    opponent_alloc, scenario, opponent_id
                )
            result.num_turns = turn_index + 1
            break

        if action and "[WALK_AWAY]" in action:
            result.outcome = "walk_away"
            result.who_terminated = "learner" if is_learner_turn else "opponent"
            result.num_turns = turn_index + 1
            break

        recent_deals: list[str] = []
        for msg in reversed(learner_msgs):
            if msg["role"] != "assistant":
                continue
            a = extract_section(msg["content"], "Action")
            if a and "[SUBMIT_DEAL]" in a:
                recent_deals.append(a)
            if len(recent_deals) >= 3:
                break
        if len(recent_deals) >= 3 and len(set(recent_deals)) == 1:
            result.outcome = "reject_loop"
            result.who_terminated = "learner"
            result.num_turns = turn_index + 1
            break
    else:
        result.num_turns = max_turns

    result.learner_messages = learner_msgs
    result.opponent_messages = opponent_msgs
    return result


def format_dialogue(ep: EpisodeResult) -> str:
    lines = [
        f"{'=' * 70}",
        f"Episode {ep.episode_id}  |  Persona: {ep.persona}  |  Outcome: {ep.outcome}",
        f"Learner: {ep.learner_label} (as {ep.learner_agent_id})",
        f"Opponent: {ep.opponent_label} (as {ep.opponent_agent_id})",
        f"Turns: {ep.num_turns}  |  Terminated by: {ep.who_terminated}",
    ]
    if ep.outcome == "deal":
        lines.append(
            f"Learner pts: {ep.learner_points}  |  Opponent pts: {ep.opponent_points}"
        )
    lines.append(f"{'=' * 70}")

    for msg in ep.learner_messages:
        role = msg["role"].upper()
        if role == "SYSTEM":
            lines.append(f"\n[SYSTEM — Learner]\n{msg['content'][:200]}...\n")
            continue
        speaker = "LEARNER" if role == "ASSISTANT" else "OPPONENT→LEARNER"
        lines.append(f"\n--- {speaker} ---")
        lines.append(msg["content"])

    lines.append(f"\n{'=' * 70}\n")
    return "\n".join(lines)


def run_evaluation(cfg: dict) -> dict:
    dataset = cfg.get("dataset", "casino")
    env = get_env(dataset)
    logger.info(f"Negotiation dataset: {dataset}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.get("output_dir", "logs/negotiate")) / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = env.load_scenarios(cfg["csv_path"])
    logger.info(f"Loaded {len(scenarios)} scenarios from {cfg['csv_path']}")

    num_episodes = cfg.get("num_episodes", 200)
    max_turns = cfg.get("max_turns", 18)
    temperature = cfg.get("temperature", 0.7)
    top_p = cfg.get("top_p", 0.9)
    seed = cfg.get("seed", 42)
    random.seed(seed)

    pw = cfg.get("persona_weights") or {}
    persona_names = list(pw.keys()) if pw else ["none"]
    persona_weights = list(pw.values()) if pw else [1.0]

    default_base_url = cfg.get("base_url", OPENAI_BASE_URL)
    default_api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")

    matchups = cfg.get("matchups", [])
    all_results: dict[str, list[EpisodeResult]] = {}
    all_summaries: dict[str, dict] = {}

    for matchup in matchups:
        learner_cfg = matchup["learner"]
        opponent_cfg = matchup["opponent"]
        matchup_name = f"{learner_cfg['label']}_vs_{opponent_cfg['label']}"
        logger.info(f"\n{'#' * 60}\nMatchup: {matchup_name}\n{'#' * 60}")

        learner_gen = make_generator(learner_cfg, default_base_url, default_api_key_env)

        both_local = (
            learner_cfg.get("type", "local") == "local"
            and opponent_cfg.get("type", "local") == "local"
        )
        same_local_model = both_local and opponent_cfg.get(
            "model_path"
        ) == learner_cfg.get("model_path")
        if same_local_model:
            opponent_gen = learner_gen
            logger.info("Opponent is same model as learner — sharing weights")
        else:
            opponent_gen = make_generator(
                opponent_cfg, default_base_url, default_api_key_env
            )

        sampled_scenarios = random.choices(scenarios, k=num_episodes)
        sampled_personas = random.choices(
            persona_names, weights=persona_weights, k=num_episodes
        )

        overall = AggregateMetrics()
        per_persona: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
        episodes: list[EpisodeResult] = []

        t0 = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(matchup_name, total=num_episodes)
            for i, (scenario, persona) in enumerate(
                zip(sampled_scenarios, sampled_personas)
            ):
                ep = run_episode(
                    env=env,
                    learner_gen=learner_gen,
                    opponent_gen=opponent_gen,
                    scenario=scenario,
                    agent_ids=scenario["agent_ids"],
                    persona=persona,
                    episode_id=i,
                    learner_label=learner_cfg["label"],
                    opponent_label=opponent_cfg["label"],
                    max_turns=max_turns,
                    temperature=temperature,
                    top_p=top_p,
                )
                episodes.append(ep)
                overall.add(ep)
                per_persona[ep.persona].add(ep)
                progress.advance(task)

        elapsed = time.time() - t0
        logger.info(f"Finished {matchup_name} in {elapsed:.1f}s")

        per_persona_summaries = {p: m.summary() for p, m in sorted(per_persona.items())}
        matchup_summary = {
            "matchup": matchup_name,
            "dataset": dataset,
            "learner": learner_cfg["label"],
            "opponent": opponent_cfg["label"],
            "overall": overall.summary(),
            "per_persona": per_persona_summaries,
            "elapsed_seconds": round(elapsed, 1),
        }
        all_summaries[matchup_name] = matchup_summary
        all_results[matchup_name] = episodes

        _save_matchup(output_dir, matchup_name, matchup_summary, episodes)

        if isinstance(learner_gen, LocalGenerator):
            del learner_gen.model
            if not same_local_model and isinstance(opponent_gen, LocalGenerator):
                del opponent_gen.model
            if torch is not None:
                torch.cuda.empty_cache()

        print_matchup_report(matchup_summary)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print_comparison_table(all_summaries)
    return all_summaries


def _save_matchup(
    output_dir: Path,
    matchup_name: str,
    matchup_summary: dict,
    episodes: list[EpisodeResult],
) -> None:
    matchup_dir = output_dir / matchup_name
    matchup_dir.mkdir(parents=True, exist_ok=True)

    with open(matchup_dir / "metrics.json", "w") as f:
        json.dump(matchup_summary, f, indent=2)

    serializable_episodes = [
        {
            "episode_id": ep.episode_id,
            "persona": ep.persona,
            "outcome": ep.outcome,
            "learner_points": ep.learner_points,
            "opponent_points": ep.opponent_points,
            "num_turns": ep.num_turns,
            "who_terminated": ep.who_terminated,
            "learner_agent_id": ep.learner_agent_id,
            "opponent_agent_id": ep.opponent_agent_id,
            "learner_format_ok": ep.learner_format_ok,
            "learner_total_turns": ep.learner_total_turns,
            "learner_malformed_deals": ep.learner_malformed_deals,
            "opponent_format_ok": ep.opponent_format_ok,
            "opponent_total_turns": ep.opponent_total_turns,
            "opponent_malformed_deals": ep.opponent_malformed_deals,
            "learner_messages": ep.learner_messages,
            "opponent_messages": ep.opponent_messages,
        }
        for ep in episodes
    ]
    with open(matchup_dir / "episodes.json", "w") as f:
        json.dump(serializable_episodes, f, indent=2)

    outcomes_seen: dict[str, list[EpisodeResult]] = defaultdict(list)
    for ep in episodes:
        outcomes_seen[ep.outcome].append(ep)

    for outcome, eps in outcomes_seen.items():
        with open(matchup_dir / f"dialogues_{outcome}.txt", "w") as f:
            for ep in eps:
                f.write(format_dialogue(ep))


def score_negotiate_logs(log_dir: str) -> dict:
    """Re-score saved negotiation episodes and print reports.

    Accepts a directory containing matchup subdirs with episodes.json,
    or a parent directory that will be searched recursively.
    """
    root = Path(log_dir)
    if not root.is_dir():
        logger.error(f"Directory not found: {root}")
        return {}

    episode_files = sorted(root.rglob("episodes.json"))
    if not episode_files:
        logger.warning(f"No episodes.json files found under {root}")
        return {}

    all_summaries: dict[str, dict] = {}

    for ep_path in episode_files:
        matchup_name = ep_path.parent.name
        with open(ep_path) as f:
            raw_episodes = json.load(f)

        dataset = None
        metrics_path = ep_path.parent / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                dataset = json.load(f).get("dataset")

        overall = AggregateMetrics()
        per_persona: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)

        for raw in raw_episodes:
            ep = EpisodeResult(
                episode_id=raw["episode_id"],
                learner_label=raw.get(
                    "learner_label",
                    matchup_name.split("_vs_")[0] if "_vs_" in matchup_name else "",
                ),
                opponent_label=raw.get(
                    "opponent_label",
                    matchup_name.split("_vs_")[-1] if "_vs_" in matchup_name else "",
                ),
                persona=raw["persona"],
                outcome=raw["outcome"],
                learner_points=raw.get("learner_points"),
                opponent_points=raw.get("opponent_points"),
                num_turns=raw["num_turns"],
                learner_messages=raw.get("learner_messages", []),
                opponent_messages=raw.get("opponent_messages", []),
                learner_agent_id=raw.get("learner_agent_id", ""),
                opponent_agent_id=raw.get("opponent_agent_id", ""),
                who_terminated=raw.get("who_terminated", ""),
                learner_total_turns=raw.get("learner_total_turns", 0),
                learner_format_ok=raw.get("learner_format_ok", 0),
                learner_malformed_deals=raw.get("learner_malformed_deals", 0),
                opponent_total_turns=raw.get("opponent_total_turns", 0),
                opponent_format_ok=raw.get("opponent_format_ok", 0),
                opponent_malformed_deals=raw.get("opponent_malformed_deals", 0),
            )
            overall.add(ep)
            per_persona[ep.persona].add(ep)

        per_persona_summaries = {p: m.summary() for p, m in sorted(per_persona.items())}
        matchup_summary = {
            "matchup": matchup_name,
            "dataset": dataset,
            "overall": overall.summary(),
            "per_persona": per_persona_summaries,
        }
        all_summaries[matchup_name] = matchup_summary
        print_matchup_report(matchup_summary)

    print_comparison_table(all_summaries)
    return all_summaries


def main(
    config_path: str = "eval/configs/negotiate.yaml", evaluate_only: str | None = None
):
    if evaluate_only:
        score_negotiate_logs(evaluate_only)
        return
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_evaluation(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Negotiation evaluation")
    parser.add_argument("--config", default="eval/configs/negotiate.yaml")
    parser.add_argument(
        "--evaluate-only",
        type=str,
        default=None,
        metavar="LOG_DIR",
        help="Score existing negotiate logs in LOG_DIR instead of running episodes",
    )
    args = parser.parse_args()
    main(args.config, args.evaluate_only)
