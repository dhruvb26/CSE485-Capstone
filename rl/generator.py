import os
from dataclasses import dataclass

from loguru import logger
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rl.prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    ANNOTATION_USER_ACTION,
    ANNOTATION_USER_TALK,
    ANNOTATION_USER_TALK_WITH_ACTION,
    SYNTHETIC_OPENER_PROMPT,
    build_system_prompt,
)
from rl.utils import extract_tag


@dataclass
class SyntheticDataGeneratorConfig:
    model: str
    temperature: float
    base_url: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"


class ParseError(ValueError):
    """Raised when a required XML tag is missing from the completion."""


class SyntheticDataGenerator:
    RETRYABLE_EXCEPTIONS = (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        ParseError,
    )

    def __init__(self, config: SyntheticDataGeneratorConfig):
        self.config = config
        api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
        base_url = config.base_url if config.base_url else None
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _append_message(self, messages: list[dict], role: str, content: str):
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})

    def build_requests(
        self,
        chat_logs: list[dict],
        participant_info: dict,
        agent_id: str,
        row_id: str,
    ):
        """Build the sequence of annotation requests for one agent.

        Args:
            chat_logs: Raw conversation turns from the source dataset.
            participant_info: Scenario metadata for both negotiation agents.
            agent_id: Agent being annotated.
            row_id: Stable identifier for logging and retries.

        Returns:
            A generator that yields request payloads and ultimately returns the
            final GRPO/SFT-compatible messages list via ``StopIteration.value``.

        Raises:
            KeyError: If required scenario metadata is missing.
            StopIteration: When the caller has supplied all parsed responses.
        """
        v2i = participant_info[agent_id]["value2issue"]
        v2r = participant_info[agent_id]["value2reason"]
        priority_points = {"High": 5, "Medium": 4, "Low": 3}

        priorities_context = "\n".join(
            f"- {v2i[p]} ({priority_points[p]} points) - {v2r[p]}"
            for p in ("High", "Medium", "Low")
        )

        # Pre-process: flatten consecutive same-agent turns.
        # Reject-Deal + Submit-Deal -> keep only the Submit-Deal
        # Reject-Deal + talk        -> single turn with talk + reject_deal action
        turns: list[dict] = []
        for turn in chat_logs:
            prev = turns[-1] if turns else None
            if prev and prev["id"] == turn["id"] and prev["text"] == "Reject-Deal":
                if turn["text"] == "Submit-Deal":
                    turns[-1] = turn
                else:
                    turns[-1] = {
                        "id": turn["id"],
                        "text": turn["text"],
                        "task_data": {
                            "data": "reject_deal",
                            "issue2youget": prev["task_data"]["issue2youget"],
                            "issue2theyget": prev["task_data"]["issue2theyget"],
                        },
                    }
            else:
                turns.append(turn)

        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(participant_info, agent_id)}
        ]
        history_lines: list[str] = []

        # If the first turn belongs to the active agent, we need a synthetic
        # user message so the output follows system -> user -> assistant
        if turns and turns[0]["id"] == agent_id:
            opponent_id = next(k for k in participant_info if k != agent_id)
            opp_v2i = participant_info[opponent_id]["value2issue"]
            opp_v2r = participant_info[opponent_id]["value2reason"]
            opp_priorities_block = "\n".join(
                f"  {opp_v2i[p]} ({priority_points[p]} points) - {opp_v2r[p]}"
                for p in ("High", "Medium", "Low")
            )
            result = yield {
                "messages": [
                    {"role": "system", "content": SYNTHETIC_OPENER_PROMPT.format(
                        priorities_block=opp_priorities_block,
                    )},
                    {"role": "user", "content": "Generate an opening message."},
                ],
                "metadata": {
                    "type": "synthetic_opener",
                    "row_id": row_id,
                    "agent_id": agent_id,
                },
            }
            opener_text = result["text"]
            self._append_message(
                messages, "user",
                f"<talk>{opener_text}</talk> <action>[TALK]</action>",
            )
            history_lines.append(
                f"Neighbor: <talk>{opener_text}</talk> <action>[TALK]</action>"
            )

        for turn in turns:
            is_me = turn["id"] == agent_id
            text = turn["text"]
            task_data = turn["task_data"]

            is_reject_with_talk = (
                task_data.get("data") == "reject_deal"
                and text not in ("Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away")
            )
            is_action = (
                not is_reject_with_talk
                and (
                    task_data["data"] != ""
                    or text in ("Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away")
                )
            )

            if not is_me:
                if is_reject_with_talk:
                    action_str = self._to_action_str(text, task_data, is_me=False)
                    user_content = f"<talk>{text}</talk> <action>{action_str}</action>"
                    history_lines.append(f"Neighbor: {user_content}")
                elif is_action:
                    action_str = self._to_action_str(text, task_data, is_me=False)
                    user_content = f"<action>{action_str}</action>"
                    history_lines.append(f"Neighbor: {user_content}")
                else:
                    user_content = f"<talk>{text}</talk> <action>[TALK]</action>"
                    history_lines.append(f"Neighbor: {user_content}")

                self._append_message(messages, "user", user_content)
                continue

            history_str = (
                "\n".join(history_lines) if history_lines else "No prior conversation."
            )

            if is_reject_with_talk:
                action_str = self._to_action_str(text, task_data, is_me=True)
                user_prompt = ANNOTATION_USER_TALK_WITH_ACTION.format(
                    priorities_context=priorities_context,
                    history_str=history_str,
                    text=text,
                    action_str=action_str,
                )
                result = yield {
                    "messages": [
                        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "metadata": {
                        "row_id": row_id,
                        "turn_index": len(history_lines),
                        "agent_id": agent_id,
                        "text": text,
                        "action_type": "talk",
                        "has_talk": True,
                    },
                }

                assistant_content = (
                    f"<thought>{result['thought']}</thought>\n"
                    f"<talk>{text}</talk>\n"
                    f"<action>{action_str}</action>"
                )
                history_lines.append(
                    f"You: <thought>{result['thought']}</thought> "
                    f"<talk>{text}</talk> <action>{action_str}</action>"
                )

            elif not is_action:
                user_prompt = ANNOTATION_USER_TALK.format(
                    priorities_context=priorities_context,
                    history_str=history_str,
                    text=text,
                )
                result = yield {
                    "messages": [
                        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "metadata": {
                        "row_id": row_id,
                        "turn_index": len(history_lines),
                        "agent_id": agent_id,
                        "text": text,
                        "action_type": "talk",
                    },
                }

                assistant_content = (
                    f"<thought>{result['thought']}</thought>\n"
                    f"<talk>{text}</talk>\n"
                    f"<action>[TALK]</action>"
                )
                history_lines.append(
                    f"You: <thought>{result['thought']}</thought> "
                    f"<talk>{text}</talk> <action>[TALK]</action>"
                )

            else:
                action_str = self._to_action_str(text, task_data, is_me=True)
                user_prompt = ANNOTATION_USER_ACTION.format(
                    priorities_context=priorities_context,
                    history_str=history_str,
                    action_str=action_str,
                )
                result = yield {
                    "messages": [
                        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "metadata": {
                        "row_id": row_id,
                        "turn_index": len(history_lines),
                        "agent_id": agent_id,
                        "text": text,
                        "action_type": text,
                        "action_str": action_str,
                    },
                }

                assistant_content = (
                    f"<thought>{result['thought']}</thought>\n"
                    f"<talk>{result['talk']}</talk>\n"
                    f"<action>{action_str}</action>"
                )
                history_lines.append(
                    f"You: <thought>{result['thought']}</thought> "
                    f"<talk>{result['talk']}</talk> <action>{action_str}</action>"
                )

            self._append_message(messages, "assistant", assistant_content)

        return messages

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    async def _make_req(
        self, messages: list[dict], required_tags: list[str], row_id: str
    ) -> dict[str, str]:
        """Request one model completion and extract the required XML tags.

        Args:
            messages: Chat payload sent to the model.
            required_tags: XML tags that must be present in the completion.
            row_id: Stable identifier for logging and retries.

        Returns:
            A mapping from tag name to extracted text.

        Raises:
            ParseError: If a required XML tag is missing.
            Exception: If the underlying API call fails.
        """
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )

        completion = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else "N/A"
        logger.debug("Received completion | row_id={} tokens={}", row_id, tokens)

        parsed: dict[str, str] = {}
        for tag in required_tags:
            value = extract_tag(completion, tag)
            if value is None:
                logger.warning(
                    "Missing <{}> tag, will retry | row_id={} completion={!r}",
                    tag,
                    row_id,
                    completion,
                )
                raise ParseError(f"Missing <{tag}> in completion: {completion!r}")
            parsed[tag] = value

        return parsed

    async def make_request(
        self,
        chat_logs: list[dict],
        participant_info: dict,
        agent_id: str,
        row_id: str,
        dataset: str = "casino",
        row_idx: int = 0,
    ) -> dict:
        """Generate one fully annotated conversation for a single agent.

        Args:
            chat_logs: Raw conversation turns from the source dataset.
            participant_info: Scenario metadata for both negotiation agents.
            agent_id: Agent being annotated.
            row_id: Stable identifier for logging and retries.
            dataset: Dataset name stored in the output payload.
            row_idx: Row index stored in the output payload.

        Returns:
            A conversation record with ``id``, ``dataset``, ``row_idx``,
            ``agent_id``, and annotated ``messages``.

        Raises:
            Exception: If request generation or API calls fail.
        """
        gen = self.build_requests(chat_logs, participant_info, agent_id, row_id)
        logger.info("Starting annotation generation | row_id={}", row_id)

        messages = []
        try:
            request = next(gen)

            while True:
                meta = request["metadata"]

                if meta.get("type") == "synthetic_opener":
                    required_tags = ["text"]
                elif meta.get("has_talk") or meta.get("action_type") == "talk":
                    required_tags = ["thought"]
                else:
                    required_tags = ["thought", "talk"]

                logger.debug(
                    "Requesting completion | row_id={} turn={} type={}",
                    row_id,
                    meta.get("turn_index", "opener"),
                    meta.get("type", meta.get("action_type")),
                )

                parsed = await self._make_req(
                    request["messages"], required_tags, row_id
                )

                request = gen.send(parsed)

        except StopIteration as e:
            messages = e.value
        except Exception:
            logger.exception("Annotation generation failed | row_id={}", row_id)
            raise

        logger.info("Finished annotation generation | row_id={}", row_id)

        return {
            "id": f"{dataset}_{row_idx}_{agent_id}",
            "dataset": dataset,
            "row_idx": row_idx,
            "agent_id": agent_id,
            "messages": messages,
        }

    @staticmethod
    def _to_action_str(text: str, task_data: dict, is_me: bool) -> str:
        if task_data.get("data") == "reject_deal":
            return "[REJECT_DEAL]"
        if text == "Submit-Deal":
            alloc = task_data["issue2youget"] if is_me else task_data["issue2theyget"]
            return (
                f"[SUBMIT_DEAL] food:{alloc.get('Food', 0)} "
                f"water:{alloc.get('Water', 0)} firewood:{alloc.get('Firewood', 0)}"
            )
        if text == "Accept-Deal":
            return "[ACCEPT_DEAL]"
        if text == "Reject-Deal":
            return "[REJECT_DEAL]"
        if text == "Walk-Away":
            return "[WALK_AWAY]"
        return f"[{text.upper().replace('-', '_')}]"
