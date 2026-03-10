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
    build_annotation_requests,
    merge_annotations,
    parse_tag_content,
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
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[list[dict], list[list[dict]]]:
    """Run GPT annotation for a single agent and return merged SFT messages
    together with the OpenAI prompts that were used for annotation.

    Returns (sft_messages, external_prompts) where external_prompts is a list
    of message lists -- one per annotation request sent to the API.
    """
    reqs = build_annotation_requests(chat_logs, participant_info, agent_id)

    raw_responses = await run_batch_completions(client, reqs, config, semaphore)

    annotations: dict[int, dict] = {}
    for req in reqs:
        turn_idx = req["turn_idx"]
        text = raw_responses.get(turn_idx)
        if text is None:
            log.warning("No response for turn %d, agent %s", turn_idx, agent_id)
            continue

        thought = parse_tag_content(text, "<thought>", "</thought>")
        talk = (
            parse_tag_content(text, "<talk>", "</talk>") if req["needs_talk"] else None
        )

        if req["needs_talk"] and talk is None:
            log.warning(
                "GPT returned no <talk> for structural turn %d, agent %s",
                turn_idx,
                agent_id,
            )

        annotations[turn_idx] = {"thought": thought, "talk": talk}

    sft_messages = merge_annotations(chat_logs, participant_info, agent_id, annotations)
    external_prompts = [req["messages"] for req in reqs]
    return sft_messages, external_prompts
