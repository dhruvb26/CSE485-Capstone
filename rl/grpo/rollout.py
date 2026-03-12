from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.sft.data import POINTS, build_system_prompt, parse_tag_content

log = logging.getLogger(__name__)

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
        "all 3 units of your top two items. Move slowly — never concede more "
        "than 1 unit per turn."
    ),
    "cooperative": (
        "You are a friendly negotiator who values reaching a deal. You're "
        "willing to split items fairly and respond positively to reasonable "
        "offers."
    ),
}

TERMINAL_ACTIONS = {"[ACCEPT_DEAL]", "[WALK_AWAY]", "[REJECT_DEAL]"}
_SUBMIT_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)", re.IGNORECASE
)


@dataclass
class Episode:
    learner_agent_id: str
    opponent_agent_id: str
    persona: str
    learner_messages: list[dict] = field(default_factory=list)
    opponent_messages: list[dict] = field(default_factory=list)
    learner_turns: list[int] = field(default_factory=list)
    outcome: str = "max_turns"
    learner_points: int | None = None
    opponent_points: int | None = None
    participant_info: dict = field(default_factory=dict)


def _extract_action(text: str) -> str | None:
    """Return the raw action string from inside <action>...</action>."""
    return parse_tag_content(text, "<action>", "</action>")


def _parse_submit_deal(action_str: str) -> dict[str, int] | None:
    """Parse ``[SUBMIT_DEAL] food:F water:W firewood:FW`` into a dict."""
    m = _SUBMIT_RE.search(action_str)
    if m is None:
        return None
    return {"food": int(m.group(1)), "water": int(m.group(2)), "firewood": int(m.group(3))}


def _compute_points(alloc: dict[str, int], participant_info: dict, agent_id: str) -> int:
    """Compute the total points for *agent_id* given an item allocation dict."""
    v2i = participant_info[agent_id]["value2issue"]
    issue2pts = {}
    for level in ("High", "Medium", "Low"):
        issue2pts[v2i[level].lower()] = POINTS[level]
    total = 0
    for item, qty in alloc.items():
        total += qty * issue2pts.get(item.lower(), 0)
    return total


def _strip_to_user_view(text: str) -> str:
    """Strip the <thought> tag from an assistant reply so the opponent only
    sees <talk> and <action> (matching the SFT data convention)."""
    thought = parse_tag_content(text, "<thought>", "</thought>")
    cleaned = text
    if thought is not None:
        cleaned = re.sub(r"<thought>.*?</thought>\s*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _sample_persona(weights: dict[str, float]) -> str:
    names = list(weights.keys())
    probs = [weights[n] for n in names]
    return random.choices(names, weights=probs, k=1)[0]


@torch.no_grad()
def _generate_turn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
) -> str:
    """Generate a single assistant turn given a message history."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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


def _is_repeat_loop(messages: list[dict], window: int = 3) -> bool:
    """Return True if the last *window* assistant turns contain identical
    SUBMIT_DEAL actions (a degenerate negotiation loop)."""
    deals: list[str] = []
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        action = _extract_action(msg["content"])
        if action and "[SUBMIT_DEAL]" in action:
            deals.append(action.strip())
        if len(deals) >= window:
            break
    if len(deals) < window:
        return False
    return len(set(deals)) == 1


def run_episode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    participant_info: dict,
    agent_ids: list[str],
    persona: str,
    max_turns: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Episode:
    """Play out a single negotiation episode via self-play.

    The learner is randomly assigned one of the two agent roles; the opponent
    gets the other role with a persona prefix on its system prompt.
    """
    learner_id, opponent_id = agent_ids[0], agent_ids[1]
    if random.random() < 0.5:
        learner_id, opponent_id = opponent_id, learner_id

    learner_sys = build_system_prompt(participant_info, learner_id)
    opponent_sys = PERSONAS[persona] + "\n\n" + build_system_prompt(participant_info, opponent_id)

    learner_msgs: list[dict] = [{"role": "system", "content": learner_sys}]
    opponent_msgs: list[dict] = [{"role": "system", "content": opponent_sys}]

    ep = Episode(
        learner_agent_id=learner_id,
        opponent_agent_id=opponent_id,
        persona=persona,
        participant_info=participant_info,
    )

    learner_goes_first = random.random() < 0.5
    last_submit_deal: dict[str, int] | None = None

    for turn_idx in range(max_turns):
        is_learner_turn = (turn_idx % 2 == 0) == learner_goes_first

        if is_learner_turn:
            if len(learner_msgs) == 1:
                learner_msgs.append({"role": "user", "content": "Begin the negotiation."})

            response = _generate_turn(
                model, tokenizer, learner_msgs, temperature, top_p
            )
            learner_msgs.append({"role": "assistant", "content": response})
            ep.learner_turns.append(len(learner_msgs) - 1)

            user_view = _strip_to_user_view(response)
            opponent_msgs.append({"role": "user", "content": user_view})

            action = _extract_action(response)
            if action:
                deal = _parse_submit_deal(action)
                if deal is not None:
                    last_submit_deal = deal
        else:
            if len(opponent_msgs) == 1:
                opponent_msgs.append({"role": "user", "content": "Begin the negotiation."})

            response = _generate_turn(
                model, tokenizer, opponent_msgs, temperature, top_p
            )
            opponent_msgs.append({"role": "assistant", "content": response})

            user_view = _strip_to_user_view(response)
            learner_msgs.append({"role": "user", "content": user_view})

            action = _extract_action(response)
            if action:
                deal = _parse_submit_deal(action)
                if deal is not None:
                    last_submit_deal = deal

        if action and "[ACCEPT_DEAL]" in action:
            ep.outcome = "deal"
            if last_submit_deal is not None:
                acceptor = "learner" if is_learner_turn else "opponent"
                proposer = "opponent" if acceptor == "learner" else "learner"
                proposer_alloc = last_submit_deal
                acceptor_alloc = {k: 3 - v for k, v in proposer_alloc.items()}

                proposer_id = ep.learner_agent_id if proposer == "learner" else ep.opponent_agent_id
                acceptor_id = ep.learner_agent_id if acceptor == "learner" else ep.opponent_agent_id

                proposer_pts = _compute_points(proposer_alloc, participant_info, proposer_id)
                acceptor_pts = _compute_points(acceptor_alloc, participant_info, acceptor_id)

                if proposer == "learner":
                    ep.learner_points = proposer_pts
                    ep.opponent_points = acceptor_pts
                else:
                    ep.learner_points = acceptor_pts
                    ep.opponent_points = proposer_pts
            break

        if action and "[WALK_AWAY]" in action:
            ep.outcome = "walk_away"
            break

        if action and "[REJECT_DEAL]" in action:
            pass

        if _is_repeat_loop(learner_msgs):
            ep.outcome = "reject_loop"
            break

    ep.learner_messages = learner_msgs
    ep.opponent_messages = opponent_msgs
    return ep


def run_self_play(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    scenarios: list[dict],
    rollout_config,
) -> list[Episode]:
    """Generate negotiation episodes via self-play over a list of scenarios.

    Args:
        model: The language model (with LoRA adapter loaded).
        tokenizer: Corresponding tokenizer.
        scenarios: List of dicts with ``participant_info`` and ``agent_ids``.
        rollout_config: A ``RolloutConfig`` with max_turns, temperature,
            top_p, and persona_weights.

    Returns:
        List of completed ``Episode`` objects.
    """
    model.eval()
    episodes: list[Episode] = []
    for scenario in tqdm(scenarios, desc="Self-play rollout", unit="episode"):
        persona = _sample_persona(rollout_config.persona_weights)
        ep = run_episode(
            model=model,
            tokenizer=tokenizer,
            participant_info=scenario["participant_info"],
            agent_ids=scenario["agent_ids"],
            persona=persona,
            max_turns=rollout_config.max_turns,
            temperature=rollout_config.temperature,
            top_p=rollout_config.top_p,
        )
        episodes.append(ep)

    outcomes = {}
    for ep in episodes:
        outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1
    log.warning(
        "Self-play complete: %d episodes — %s",
        len(episodes),
        ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items())),
    )
    return episodes
