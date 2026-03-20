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

from rl.prompts import THOUGHT_JUDGE_SYSTEM_PROMPT, THOUGHT_JUDGE_USER_PROMPT
from rl.utils import (
    extract_discussed_deal,
    extract_tag,
    last_opponent_action_is_submit,
    parse_submit_deal,
    prompt_to_text,
)

_judge_client: OpenAI | None = None
_judge_model: str = ""


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
    logger.info("Judge configured: model={} base_url={}", model, base_url or "(default)")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
    reraise=True,
)
def _judge_call(client: OpenAI, *, system_prompt: str, user_content: str) -> str:
    response = client.chat.completions.create(
        model=_judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=1.0,
    )
    return response.choices[0].message.content.strip()


def length_reward(completions, **kwargs) -> list[float]:
    """
    Reward based on the total character length of the completion.

    Encourages moderately long responses. The reward scales linearly up to
    500 characters (mapped to 1.0), penalises completions shorter than 50
    characters, and decays for completions longer than 500 characters.

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
        elif n <= 500:
            rewards.append(min(1.0, n / 500.0))
        else:
            rewards.append(max(-1.0, 1.0 - (n - 500) / 500.0))
    return rewards


def thought_judge_reward(completions, **kwargs) -> list[float]:
    """
    LLM-as-a-judge reward that evaluates the quality of the ``<thought>`` tag
    using an OpenAI-compatible API (OpenAI, OpenRouter, etc.).

    Sends each completion's thought (and action, when present) along with the
    agent's negotiation context to a judge model. The judge returns a 0-10
    score which is normalised to [0.0, 1.0]. For ``[SUBMIT_DEAL]`` actions the
    full action string is included so the judge can verify arithmetic and
    strategic consistency.

    Requires the ``system_prompt`` column in the dataset so it appears in
    *kwargs* as a list of strings carrying each agent's priorities.

    Args:
        completions: List of completions from the model. Each completion is
            a list containing a single message dict with a ``"content"`` key.
        **kwargs: Must contain ``"system_prompt"`` — a list of strings (one
            per completion) with the agent's negotiation system prompt that
            includes item priorities and point values.

    Returns:
        A list of float rewards in [0.0, 1.0], one per completion. Returns
        0.0 for completions missing a ``<thought>`` tag or when the API call
        fails.
    """
    if _judge_client is None:
        raise RuntimeError(
            "Judge not configured. Call configure_judge() before training."
        )

    system_prompts: list[str] = kwargs.get("system_prompt", [])
    rewards: list[float] = []

    for i, completion in enumerate(completions):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        thought = extract_tag(text, "thought")
        if thought is None:
            rewards.append(0.0)
            continue

        action = extract_tag(text, "action") or "[TALK]"
        ctx = system_prompts[i] if i < len(system_prompts) else ""

        try:
            raw = _judge_call(
                _judge_client,
                system_prompt=THOUGHT_JUDGE_SYSTEM_PROMPT,
                user_content=THOUGHT_JUDGE_USER_PROMPT.format(
                    system_prompt=ctx,
                    thought=thought,
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
            action_valid = parse_submit_deal(action) is not None
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


def outcome_reward(completions, **kwargs) -> list[float]:
    """
    Completion-level reward based on the agent's selected action.

    This function inspects each completion's ``<action>...</action>`` tag
    directly (instead of using episode-level outcome metadata from
    ``prepare_dataset``).

    Rewards:
      - ``[ACCEPT_DEAL]``: ``1.0``
      - ``[SUBMIT_DEAL]`` but not ``[ACCEPT_DEAL]``: ``0.3``
      - ``[WALK_AWAY]``: ``-0.5``
      - ``[TALK]`` or any other action (or missing ``<action>`` tag): ``0.0``

    Args:
        completions: List of completions from the model.
        **kwargs: Unused (kept for GRPOTrainer compatibility).

    Returns:
        A list of float rewards, one per completion.
    """
    rewards: list[float] = []
    for completion in completions:
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        action = extract_tag(text, "action")
        if action is None:
            rewards.append(0.0)
            continue

        if re.search(r"\[ACCEPT_DEAL\]", action, re.IGNORECASE):
            rewards.append(1.0)
        elif re.search(r"\[SUBMIT_DEAL\]", action, re.IGNORECASE):
            rewards.append(0.3)
        elif re.search(r"\[WALK_AWAY\]", action, re.IGNORECASE):
            rewards.append(-0.5)
        else:
            rewards.append(0.0)

    return rewards


def arithmetic_reward(completions, **kwargs) -> list[float]:
    """
    Reward that validates the arithmetic of ``[SUBMIT_DEAL]`` actions.

    Performs two checks:

    1. **Range check** — every item value must be in [0, 3].
    2. **Consistency check** — the submitted deal must match the most recently
       discussed deal in the conversation.  The discussed deal is extracted
       from (a) the last ``<thought>`` tag with explicit ``N item x M pts``
       arithmetic for all three items, or (b) the last ``[SUBMIT_DEAL]`` in
       the prompt.  If no prior deal can be determined the check is skipped.

    Completions without a ``[SUBMIT_DEAL]`` action receive a neutral 0.5.
    Malformed or inconsistent deals receive 0.0.

    Args:
        completions: List of completions from the model. Each completion is
            a list containing a single message dict with a ``"content"`` key.
        **kwargs: Additional keyword arguments forwarded by GRPOTrainer.
            ``prompts`` (list) is used for the conversation consistency check.

    Returns:
        A list of float rewards, one per completion. 1.0 for valid and
        consistent deals, 0.0 for invalid/inconsistent deals, 0.5 for
        non-deal actions.
    """
    prompts = kwargs.get("prompts", [])
    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        action = extract_tag(text, "action")

        if action is None or not re.search(r"\[SUBMIT_DEAL\]", action, re.IGNORECASE):
            rewards.append(0.5)
            continue

        deal = parse_submit_deal(action)
        if deal is None:
            rewards.append(0.0)
            continue

        if not all(0 <= v <= 3 for v in deal.values()):
            rewards.append(0.0)
            continue

        prompt = prompts[i] if i < len(prompts) else ""
        discussed = extract_discussed_deal(prompt_to_text(prompt))
        if discussed is not None and discussed != deal:
            rewards.append(0.0)
            continue

        rewards.append(1.0)

    return rewards
