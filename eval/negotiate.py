from __future__ import annotations

import ast
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.prompts import build_system_prompt
from rl.utils import (
    extract_tag,
    flip_deal_perspective,
    last_opponent_action_is_submit,
    parse_submit_deal,
    strip_thought,
)

POINTS = {"High": 5, "Medium": 4, "Low": 3}
MAX_POINTS = 36

PERSONAS = {
    "uncompromising": (
        "You are a tough negotiator. You insist on getting your top-priority "
        "items and rarely make concessions. Only accept a deal if you receive "
        "at least 2 units of your highest-value item."
    ),
    "selfish": (
        "You are a self-interested negotiator. Always claim 3 units of your "
        "top item in every offer. Make small concessions only if the "
        "negotiation is about to collapse."
    ),
    "anchoring": (
        "You are a strategic negotiator. Open with an extreme offer claiming "
        "all 3 units of your top two items. Move slowly and never concede "
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
    who_terminated: str = ""  # learner | opponent
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

        joint_scores = [
            lp + op for lp, op in zip(deals, opp_deals)
        ]
        avg_joint = (
            round(sum(joint_scores) / len(joint_scores), 2) if joint_scores else None
        )

        score_ratios = [
            lp / (lp + op) if (lp + op) > 0 else 0.5
            for lp, op in zip(deals, opp_deals)
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
            "opponent_malformed_deal_rate": round(self.opponent_malformed_deals / ot, 3),
            "learner_total_turns": self.learner_total_turns,
            "opponent_total_turns": self.opponent_total_turns,
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_path: str,
    base_model: str | None = None,
    dtype: str = "bfloat16",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    adapter_config = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.isdir(model_path) and os.path.exists(adapter_config)

    if is_lora:
        if not base_model:
            raise ValueError(
                f"LoRA adapter at {model_path} requires base_model to be set"
            )
        logger.info(f"Loading base model {base_model} + LoRA from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def generate_turn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
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


def run_episode(
    learner_model: AutoModelForCausalLM,
    learner_tokenizer: AutoTokenizer,
    opponent_model: AutoModelForCausalLM,
    opponent_tokenizer: AutoTokenizer,
    participant_info: dict,
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

    learner_system = build_system_prompt(participant_info, learner_id)
    opponent_system = (
        PERSONAS[persona] + "\n\n" + build_system_prompt(participant_info, opponent_id)
    )

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
            response = generate_turn(
                learner_model,
                learner_tokenizer,
                learner_msgs,
                temperature,
                top_p,
            )
            learner_msgs.append({"role": "assistant", "content": response})
            opponent_msgs.append(
                {
                    "role": "user",
                    "content": flip_deal_perspective(strip_thought(response)),
                }
            )
        else:
            if len(opponent_msgs) == 1:
                opponent_msgs.append(
                    {"role": "user", "content": "Begin the negotiation."}
                )
            response = generate_turn(
                opponent_model,
                opponent_tokenizer,
                opponent_msgs,
                temperature,
                top_p,
            )
            opponent_msgs.append({"role": "assistant", "content": response})
            learner_msgs.append(
                {
                    "role": "user",
                    "content": flip_deal_perspective(strip_thought(response)),
                }
            )

        thought = extract_tag(response, "thought")
        talk = extract_tag(response, "talk")
        action = extract_tag(response, "action")

        format_ok = False
        malformed_deal = False
        if thought is not None and talk is not None and action is not None:
            try:
                order_ok = (
                    response.index("<thought>")
                    < response.index("<talk>")
                    < response.index("<action>")
                )
            except ValueError:
                order_ok = False

            if order_ok:
                if re.search(r"\[SUBMIT_DEAL\]", action, re.IGNORECASE):
                    deal = parse_submit_deal(action)
                    format_ok = deal is not None and all(
                        0 <= v <= 3 for v in deal.values()
                    )
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
            parsed_deal = parse_submit_deal(action)
            if parsed_deal is not None:
                last_submit_deal = parsed_deal

        if action and "[ACCEPT_DEAL]" in action:
            result.outcome = "deal"
            result.who_terminated = "learner" if is_learner_turn else "opponent"
            if last_submit_deal is not None:
                learner_alloc = (
                    {item: 3 - qty for item, qty in last_submit_deal.items()}
                    if is_learner_turn
                    else last_submit_deal
                )
                opponent_alloc = (
                    last_submit_deal
                    if is_learner_turn
                    else {item: 3 - qty for item, qty in last_submit_deal.items()}
                )
                lp_map = {
                    participant_info[learner_id]["value2issue"][lv].lower(): pts
                    for lv, pts in POINTS.items()
                }
                op_map = {
                    participant_info[opponent_id]["value2issue"][lv].lower(): pts
                    for lv, pts in POINTS.items()
                }
                result.learner_points = sum(
                    qty * lp_map.get(item.lower(), 0)
                    for item, qty in learner_alloc.items()
                )
                result.opponent_points = sum(
                    qty * op_map.get(item.lower(), 0)
                    for item, qty in opponent_alloc.items()
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
            a = extract_tag(msg["content"], "action")
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


def load_scenarios(csv_path: str) -> list[dict]:
    scenarios = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pi = ast.literal_eval(row["participant_info"])
            scenarios.append({"participant_info": pi, "agent_ids": list(pi.keys())})
    return scenarios


def run_evaluation(cfg: dict) -> dict:
    output_dir = Path(cfg.get("output_dir", "logs/negotiate"))
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(cfg["csv_path"])
    logger.info(f"Loaded {len(scenarios)} scenarios from {cfg['csv_path']}")

    num_episodes = cfg.get("num_episodes", 200)
    max_turns = cfg.get("max_turns", 18)
    temperature = cfg.get("temperature", 0.7)
    top_p = cfg.get("top_p", 0.9)
    seed = cfg.get("seed", 42)
    random.seed(seed)

    persona_names = list(cfg.get("persona_weights", {p: 0.25 for p in PERSONAS}).keys())
    persona_weights = [
        cfg.get("persona_weights", {p: 0.25 for p in PERSONAS})[p]
        for p in persona_names
    ]

    matchups = cfg.get("matchups", [])
    all_results: dict[str, list[EpisodeResult]] = {}
    all_summaries: dict[str, dict] = {}

    for matchup in matchups:
        learner_cfg = matchup["learner"]
        opponent_cfg = matchup["opponent"]
        matchup_name = f"{learner_cfg['label']}_vs_{opponent_cfg['label']}"
        logger.info(f"\n{'#' * 60}\nMatchup: {matchup_name}\n{'#' * 60}")

        learner_model, learner_tok = load_model_and_tokenizer(
            learner_cfg["model_path"],
            learner_cfg.get("base_model"),
            learner_cfg.get("dtype", "bfloat16"),
        )
        if opponent_cfg["model_path"] == learner_cfg["model_path"]:
            opponent_model, opponent_tok = learner_model, learner_tok
            logger.info("Opponent is same model as learner — sharing weights")
        else:
            opponent_model, opponent_tok = load_model_and_tokenizer(
                opponent_cfg["model_path"],
                opponent_cfg.get("base_model"),
                opponent_cfg.get("dtype", "bfloat16"),
            )

        sampled_scenarios = random.choices(scenarios, k=num_episodes)
        sampled_personas = random.choices(
            persona_names, weights=persona_weights, k=num_episodes
        )

        overall = AggregateMetrics()
        per_persona: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
        episodes: list[EpisodeResult] = []

        t0 = time.time()
        for i, (scenario, persona) in enumerate(
            tqdm(
                zip(sampled_scenarios, sampled_personas),
                total=num_episodes,
                desc=matchup_name,
            )
        ):
            ep = run_episode(
                learner_model=learner_model,
                learner_tokenizer=learner_tok,
                opponent_model=opponent_model,
                opponent_tokenizer=opponent_tok,
                participant_info=scenario["participant_info"],
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

        elapsed = time.time() - t0
        logger.info(f"Finished {matchup_name} in {elapsed:.1f}s")

        per_persona_summaries = {
            p: m.summary() for p, m in sorted(per_persona.items())
        }
        matchup_summary = {
            "matchup": matchup_name,
            "learner": learner_cfg["label"],
            "opponent": opponent_cfg["label"],
            "overall": overall.summary(),
            "per_persona": per_persona_summaries,
            "elapsed_seconds": round(elapsed, 1),
        }
        all_summaries[matchup_name] = matchup_summary
        all_results[matchup_name] = episodes

        # Save per-matchup results
        matchup_dir = output_dir / matchup_name
        matchup_dir.mkdir(parents=True, exist_ok=True)

        with open(matchup_dir / "metrics.json", "w") as f:
            json.dump(matchup_summary, f, indent=2)

        serializable_episodes = []
        for ep in episodes:
            serializable_episodes.append(
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
            )
        with open(matchup_dir / "episodes.json", "w") as f:
            json.dump(serializable_episodes, f, indent=2)

        # Save readable dialogue samples per outcome category
        outcomes_seen: dict[str, list[EpisodeResult]] = defaultdict(list)
        for ep in episodes:
            outcomes_seen[ep.outcome].append(ep)

        for outcome, eps in outcomes_seen.items():
            sample = eps[:20]
            with open(matchup_dir / f"dialogues_{outcome}.txt", "w") as f:
                for ep in sample:
                    f.write(format_dialogue(ep))

        # Free opponent if it was separately loaded
        if opponent_cfg["model_path"] != learner_cfg["model_path"]:
            del opponent_model
            torch.cuda.empty_cache()

        del learner_model
        torch.cuda.empty_cache()

        _print_matchup_report(matchup_summary)

    # Save combined summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    _print_comparison_table(all_summaries)
    return all_summaries


def _print_matchup_report(summary: dict):
    o = summary["overall"]
    print(f"\n{'=' * 60}")
    print(f"  {summary['matchup']}")
    print(f"{'=' * 60}")
    print(f"  Episodes:          {o['total_episodes']}")
    print(f"  Deal rate:         {o['deal_rate']:.1%}  ({o['deal_count']})")
    print(f"  Walk-away rate:    {o['walk_away_rate']:.1%}  ({o['walk_away_count']})")
    print(
        f"  Reject-loop rate:  {o['reject_loop_rate']:.1%}  ({o['reject_loop_count']})"
    )
    print(f"  Max-turns rate:    {o['max_turns_rate']:.1%}  ({o['max_turns_count']})")
    print(f"  Avg learner pts:   {o['avg_learner_points']}  (std {o['std_learner_points']})")
    print(f"  Avg opponent pts:  {o['avg_opponent_points']}")
    print(f"  Avg joint score:   {o['avg_joint_score']}")
    print(f"  Avg score ratio:   {o['avg_score_ratio']}")
    print(f"  Avg turns (deal):  {o['avg_turns_to_deal']}")
    print(f"  Avg turns (all):   {o['avg_turns_all']}")
    print(f"  Points/turn:       {o['points_per_turn']}")
    print()
    print(f"  Learner format:    {o['learner_format_rate']:.1%}  ({o['learner_total_turns']} turns)")
    print(f"  Learner bad deals: {o['learner_malformed_deal_rate']:.1%}")
    print(f"  Opponent format:   {o['opponent_format_rate']:.1%}  ({o['opponent_total_turns']} turns)")
    print(f"  Opponent bad deals:{o['opponent_malformed_deal_rate']:.1%}")
    print()

    for persona, pm in summary["per_persona"].items():
        sr = pm['avg_score_ratio']
        sr_str = f"{sr:.2f}" if sr is not None else "—"
        js = pm['avg_joint_score']
        js_str = f"{js:.1f}" if js is not None else "—"
        print(
            f"  [{persona}]  deal={pm['deal_rate']:.0%}  "
            f"learner_pts={pm['avg_learner_points']}  "
            f"opp_pts={pm['avg_opponent_points']}  "
            f"ratio={sr_str}  joint={js_str}  "
            f"turns={pm['avg_turns_to_deal']}  "
            f"fmt={pm['learner_format_rate']:.0%}  "
            f"(n={pm['total_episodes']})"
        )
    print()


def _print_comparison_table(all_summaries: dict):
    if len(all_summaries) < 2:
        return

    header_fields = [
        "deal_rate",
        "avg_learner_points",
        "avg_opponent_points",
        "avg_joint_score",
        "avg_score_ratio",
        "avg_turns_to_deal",
        "points_per_turn",
        "learner_format_rate",
    ]
    col_w = 16

    print(f"\n{'=' * 60}")
    print("  A/B COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Matchup':<40} " + " ".join(f"{h:>{col_w}}" for h in header_fields))
    print(f"  {'-' * 40} " + " ".join("-" * col_w for _ in header_fields))

    for name, s in all_summaries.items():
        o = s["overall"]
        vals = []
        for h in header_fields:
            v = o.get(h)
            if v is None:
                vals.append(f"{'—':>{col_w}}")
            elif "rate" in h:
                vals.append(f"{v:>{col_w}.1%}")
            else:
                vals.append(f"{v:>{col_w}.2f}")
        print(f"  {name:<40} " + " ".join(vals))

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Negotiation evaluation")
    parser.add_argument("--config", default="eval/negotiate.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_evaluation(cfg)


if __name__ == "__main__":
    main()
