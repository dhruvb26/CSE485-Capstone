import asyncio
import logging
import os

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rl.config import GenerateConfig
from rl.sft.data import (
    STRUCTURAL,
    build_annotation_context,
    merge_annotations,
    parse_tag_content,
    render_turn,
)

log = logging.getLogger(__name__)


def create_openai_client(config: GenerateConfig) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from the generate config.

    Reads the API key from the environment variable named in
    config.api_key_env. If config.base_url is set (e.g. for
    OpenRouter or a local vLLM server), it is forwarded to the client.
    When no API key is found (typical for local vLLM), a placeholder
    is used so the client still initialises.
    """
    api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
    kwargs: dict = {"api_key": api_key or "no-key-required"}
    if config.base_url:
        kwargs["base_url"] = config.base_url

    return AsyncOpenAI(**kwargs)


def _make_retry_decorator(config: GenerateConfig):
    """Build a tenacity retry decorator from config values."""
    return retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=config.retry_min_wait,
            max=config.retry_max_wait,
        ),
        retry=retry_if_exception_type((APIError, RateLimitError, APIConnectionError)),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )


async def call_completion(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
    config: GenerateConfig,
) -> str:
    """Call the OpenAI chat completions API for a single request.

    Retries on transient API errors using exponential backoff as
    configured in the generate config.
    """
    retry_decorator = _make_retry_decorator(config)

    @retry_decorator
    async def _inner():
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    return await _inner()


async def run_batch_completions(
    client: AsyncOpenAI,
    requests: list[dict],
    config: GenerateConfig,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[int, str]:
    """Run multiple completion requests concurrently with bounded parallelism.

    Returns a dict mapping each request's turn_idx to the raw response text.
    When *semaphore* is provided (e.g. a global one shared across conversations),
    it is used instead of creating a per-batch semaphore.
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _limited(req: dict) -> tuple[int, str | None]:
        async with semaphore:
            try:
                text = await call_completion(
                    client, req["messages"], config.model, config.temperature, config
                )
                return req["turn_idx"], text
            except Exception:
                log.warning(
                    "Completion failed for turn %d after retries", req["turn_idx"]
                )
                return req["turn_idx"], None

    results = await asyncio.gather(*[_limited(r) for r in requests])
    return {turn_idx: text for turn_idx, text in results}


async def annotate_agent(
    client: AsyncOpenAI,
    chat_logs: list[dict],
    participant_info: dict,
    agent_id: str,
    config: GenerateConfig,
    sample_id: str | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> list[dict]:
    """Run GPT annotation for a single agent sequentially so that each turn's
    prompt includes the agent's previously generated thoughts.

    Callers should limit concurrency at the agent level (e.g. via a
    semaphore around this call) so that at most ``max_concurrent`` agents
    run in parallel — each one making sequential API calls.

    If *shutdown_event* is set, stops after the current in-flight call and
    returns partial results.
    """
    tag = sample_id or agent_id
    gpt_system, priorities_str = build_annotation_context(participant_info, agent_id)

    history_lines: list[str] = []
    annotations: dict[int, dict] = {}

    for idx, turn in enumerate(chat_logs):
        is_me = turn["id"] == agent_id
        text = turn["text"]
        _, _, rendered = render_turn(turn, agent_id)

        if not is_me:
            history_lines.append(f"Them: {rendered}")
            continue

        if shutdown_event is not None and shutdown_event.is_set():
            log.info("Shutdown requested — returning partial annotations for %s", tag)
            break

        is_structural = text in STRUCTURAL
        history_block = (
            "\n".join(history_lines)
            if history_lines
            else "(opening turn - no prior messages)"
        )

        user_prompt_parts = [
            f"Agent priorities:\n{priorities_str}\n",
            f"Conversation so far:\n{history_block}\n",
            f"Action taken this turn: {rendered}",
        ]
        if not is_structural:
            user_prompt_parts.append(f'Talk text this turn: "{text}"')

        if is_structural:
            user_prompt_parts.append(
                "\nThis is a structural action with no human text. "
                "Generate both <thought>...</thought> and <talk>...</talk>."
            )
        else:
            user_prompt_parts.append(
                "\nGenerate only <thought>...</thought> for this turn."
            )

        messages = [
            {"role": "system", "content": gpt_system},
            {"role": "user", "content": "\n".join(user_prompt_parts)},
        ]

        max_talk_retries = 3
        thought = None
        talk = None

        for attempt in range(1, max_talk_retries + 1):
            try:
                resp_text = await call_completion(
                    client, messages, config.model, config.temperature, config
                )
            except Exception:
                log.warning(
                    "Completion failed for turn %d after retries (sample %s)", idx, tag
                )
                resp_text = None

            if resp_text is None:
                break

            thought = parse_tag_content(resp_text, "<thought>", "</thought>")
            talk = (
                parse_tag_content(resp_text, "<talk>", "</talk>")
                if is_structural
                else None
            )

            if not is_structural or talk is not None:
                break

            if attempt < max_talk_retries:
                log.warning(
                    "GPT returned no <talk> for structural turn %d (%s), "
                    "sample %s — retrying (%d/%d)",
                    idx, text, tag, attempt, max_talk_retries,
                )

        if resp_text is None:
            log.warning("No response for turn %d, sample %s", idx, tag)
            history_lines.append(f"You: {rendered}")
            continue

        if is_structural and talk is None:
            log.warning(
                "GPT returned no <talk> for structural turn %d (%s), "
                "sample %s — exhausted %d retries",
                idx, text, tag, max_talk_retries,
            )

        annotations[idx] = {"thought": thought, "talk": talk}

        thought_prefix = f"<thought>{thought}</thought> " if thought else ""
        history_lines.append(f"You: {thought_prefix}{rendered}")

    return merge_annotations(chat_logs, participant_info, agent_id, annotations, sample_id=sample_id)
