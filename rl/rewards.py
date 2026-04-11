from __future__ import annotations

import json
import re

from loguru import logger
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rl.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT
from rl.utils import (
    extract_tag,
    last_opponent_action_is_submit,
    parse_submit_deal,
)

_judge_client: OpenAI | None = None
_judge_model: str = ""


class EmptyResponseError(APIError):
    """OpenRouter returned a response with no choices."""

    def __init__(self) -> None:
        # APIError requires message, body, and request; fake a minimal one.
        super().__init__(
            message="OpenRouter returned choices=None",
            request=None,  # type: ignore[arg-type]
            body=None,
        )


def configure_judge(
    *,
    model: str,
    base_url: str = "",
    api_key_env: str = "OPENROUTER_API_KEY",
) -> None:
    """Initialise the shared judge client. Must be called before training.

    The API key is read from the env variable named by *api_key_env*.
    For OpenRouter BYOK, configure your provider key in the OpenRouter
    dashboard — the API call itself always authenticates with the
    OpenRouter key.
    """
    import os

    global _judge_client, _judge_model
    if not model:
        raise ValueError("judge.model is required but was empty")

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise EnvironmentError(f"${api_key_env} is not set in the environment")

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    _judge_client = OpenAI(**kwargs)
    _judge_model = model
    logger.info(
        "Judge configured: model={} base_url={}", model, base_url or "(default)"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(
        (APIError, APITimeoutError, RateLimitError, EmptyResponseError)
    ),
    reraise=True,
)
def _judge_call(client: OpenAI, *, system_prompt: str, user_content: str) -> str:
    response = client.chat.completions.create(
        model=_judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    if not response.choices:
        raise EmptyResponseError()
    content = response.choices[0].message.content
    if content is None:
        raise EmptyResponseError()
    return content.strip()


def length_reward(completions, **kwargs) -> list[float]:
    """
    Reward based on the total character length of the completion.

    Encourages moderately long responses. The reward scales linearly up to
    750 characters (mapped to 1.0), penalises completions shorter than 50
    characters, and decays gently for completions longer than 750 characters
    (divided by 1000 so detailed thoughts with arithmetic aren't penalised).

    Args:
        completions: List of completions from the model. Each completion is
            a list containing a single message dict with a ``"content"`` key.

    Returns:
        A list of float rewards in the range [-1.0, 1.0], one per completion.
    """
    rewards: list[float] = []
    for completion in completions:
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        n = len(text)
        if n < 50:
            rewards.append(-1.0)
        elif n <= 750:
            rewards.append(min(1.0, n / 750.0))
        else:
            rewards.append(max(-1.0, 1.0 - (n - 750) / 1000.0))
    return rewards


def _format_conversation_context(prompt, max_turns: int = 3) -> str:
    """Extract the last few conversation turns from a prompt for the judge."""
    if not prompt:
        return "(no conversation history yet)"
    messages = prompt if isinstance(prompt, list) else []
    recent = [m for m in messages if m.get("role") in ("user", "assistant")][-max_turns * 2 :]
    if not recent:
        return "(no conversation history yet)"
    lines = []
    for m in recent:
        role = "Agent" if m["role"] == "assistant" else "Neighbor"
        content = m["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def judge_reward(completions, **kwargs) -> list[float]:
    """LLM-as-a-judge reward scoring thought quality and talk–action alignment.

    Sends each completion's ``<thought>``, ``<talk>``, and ``<action>`` along
    with the agent's negotiation context and recent conversation history to a
    judge model. The judge returns a 0-10 score (5 pts for thought quality,
    5 pts for talk–action alignment) which is normalised to [0.0, 1.0].

    Requires the ``system_prompt`` and ``prompts`` columns in the dataset.

    Returns:
        A list of float rewards in [0.0, 1.0], one per completion. Returns
        0.0 for completions missing ``<thought>`` or ``<talk>`` tags, or when
        the API call fails.
    """
    if _judge_client is None:
        raise RuntimeError(
            "Judge not configured. Call configure_judge() before training."
        )

    system_prompts: list[str] = kwargs.get("system_prompt", [])
    prompts = kwargs.get("prompts", [])
    rewards: list[float] = []

    for i, completion in enumerate(completions):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        thought = extract_tag(text, "thought")
        talk = extract_tag(text, "talk")
        if thought is None or talk is None:
            rewards.append(0.0)
            continue

        action = extract_tag(text, "action") or "[TALK]"
        ctx = system_prompts[i] if i < len(system_prompts) else ""
        prompt = prompts[i] if i < len(prompts) else []
        conversation_context = _format_conversation_context(prompt)

        try:
            raw = _judge_call(
                _judge_client,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_content=JUDGE_USER_PROMPT.format(
                    system_prompt=ctx,
                    conversation_context=conversation_context,
                    thought=thought,
                    talk=talk,
                    action=action,
                ),
            )
            m = re.search(r"\{[^}]*\}", raw)
            score = float(json.loads(m.group())["score"]) if m else 0.0
            rewards.append(max(0.0, min(10.0, score)) / 10.0)
        except Exception as exc:
            logger.exception("Judge call failed for completion {} ({})", i, exc)
            rewards.append(0.0)

    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """
    Binary reward for strict Thought-Talk-Action format compliance.

    Returns 1.0 only when **all** of the following hold:

    1. All three tags — ``<thought>``, ``<talk>``, ``<action>`` — are present.
    2. They appear in the correct order (thought → talk → action).
    3. The action content is valid:

       * ``[SUBMIT_DEAL]`` must include a parseable ``food:N water:N
         firewood:N`` string.
       * ``[TALK]``, ``[ACCEPT_DEAL]``, ``[REJECT_DEAL]``, ``[WALK_AWAY]``
         must appear alone with no trailing parameters.
       * ``[ACCEPT_DEAL]`` is only valid when the opponent's most recent
         action (last user message) was ``[SUBMIT_DEAL]``.

    Any violation yields 0.0 — no partial credit.

    Args:
        completions: List of completions from the model.
        **kwargs: Additional keyword arguments forwarded by GRPOTrainer.
            ``prompts`` (list) is used for the ``[ACCEPT_DEAL]`` context check.

    Returns:
        A list of float rewards (0.0 or 1.0), one per completion.
    """
    prompts = kwargs.get("prompts", [])
    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )

        thought = extract_tag(text, "thought")
        talk = extract_tag(text, "talk")
        action = extract_tag(text, "action")

        if thought is None or talk is None or action is None:
            rewards.append(0.0)
            continue

        if not (
            text.index("<thought>") < text.index("<talk>") < text.index("<action>")
        ):
            rewards.append(0.0)
            continue

        action_valid = False
        if re.search(r"\[SUBMIT_DEAL\]", action, re.IGNORECASE):
            deal = parse_submit_deal(action)
            action_valid = (
                deal is not None and all(0 <= v <= 3 for v in deal.values())
            )
        elif re.compile(
            r"^\s*\[(TALK|ACCEPT_DEAL|REJECT_DEAL|WALK_AWAY)\]\s*$", re.IGNORECASE
        ).match(action):
            if re.search(r"\[ACCEPT_DEAL\]", action, re.IGNORECASE):
                prompt = prompts[i] if i < len(prompts) else ""
                action_valid = last_opponent_action_is_submit(prompt)
            else:
                action_valid = True

        rewards.append(1.0 if action_valid else 0.0)
    return rewards


def _compute_points(alloc: dict, food_pts: int, water_pts: int, firewood_pts: int) -> int:
    """Dot-product of an item allocation with the learner's per-item point values."""
    return (
        alloc["food"] * food_pts
        + alloc["water"] * water_pts
        + alloc["firewood"] * firewood_pts
    )


def points_reward(completions, **kwargs) -> list[float]:
    """Reward that incentivises the learner to maximise its own points.

    For deal actions (``[SUBMIT_DEAL]``, ``[ACCEPT_DEAL]``), the reward is
    the learner's normalised score: ``points / max_points``, scaled to
    ``[-0.5, 1.5]`` so that bad deals are penalised and good deals are
    strongly rewarded.  ``[WALK_AWAY]`` yields ``-1.0`` (no points at all).
    All other actions return ``0.0``.

    Args:
        completions: List of completions from the model.
        **kwargs: Must contain ``food_points``, ``water_points``,
            ``firewood_points``, ``max_points``, and ``last_opponent_offer``
            lists forwarded from the dataset columns.

    Returns:
        A list of float rewards, one per completion.
    """
    food_pts: list = kwargs.get("food_points", [])
    water_pts: list = kwargs.get("water_points", [])
    fire_pts: list = kwargs.get("firewood_points", [])
    max_pts: list = kwargs.get("max_points", [])
    opponent_offers: list = kwargs.get("last_opponent_offer", [])

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        action = extract_tag(text, "action")
        if action is None:
            rewards.append(0.0)
            continue

        action_upper = action.upper()
        fp = food_pts[i] if i < len(food_pts) else 3
        wp = water_pts[i] if i < len(water_pts) else 3
        fwp = fire_pts[i] if i < len(fire_pts) else 3
        mp = max_pts[i] if i < len(max_pts) else 36

        if "[SUBMIT_DEAL]" in action_upper:
            deal = parse_submit_deal(action)
            if deal is None:
                rewards.append(0.0)
                continue
            points = _compute_points(deal, fp, wp, fwp)
            rewards.append((points / mp) * 2.0 - 0.5)

        elif "[ACCEPT_DEAL]" in action_upper:
            raw = opponent_offers[i] if i < len(opponent_offers) else None
            alloc = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(alloc, dict):
                rewards.append(0.0)
                continue
            points = _compute_points(alloc, fp, wp, fwp)
            rewards.append((points / mp) * 2.0 - 0.5)

        elif "[WALK_AWAY]" in action_upper:
            rewards.append(-1.0)

        else:
            rewards.append(0.0)

    return rewards
