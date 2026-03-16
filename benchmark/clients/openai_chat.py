import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIChat:
    """Chat client that delegates to the OpenAI Chat Completions API.

    Reads ``OPENAI_API_KEY`` and ``OPENAI_MODEL`` from environment variables
    (defaults to ``gpt-4o`` if ``OPENAI_MODEL`` is unset). All requests use
    ``temperature=0.0`` for deterministic output.
    """

    def __init__(self) -> None:
        self.model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, instructions: str, messages: list[dict]) -> str:
        """Send a conversation to OpenAI and return the assistant's reply.

        Args:
            instructions: System-level prompt prepended to the conversation.
            messages: Conversation history as a list of ``{"role", "content"}``
                dicts.

        Returns:
            The assistant's response text, or an empty string on failure.
        """
        try:
            completion = self.client.chat.completions.create(
                temperature=0.0,
                model=self.model,
                messages=[
                    {"role": "system", "content": instructions},
                    *[
                        {"role": message["role"], "content": message["content"]}
                        for message in messages
                    ],
                ],
            )
            if (
                completion.choices
                and completion.choices[0].message
                and completion.choices[0].message.content
            ):
                content = completion.choices[0].message.content
                return content or ""
        except Exception:
            logger.info("Error sending query to OpenAI. Returning empty string.")
            return ""
        return ""
