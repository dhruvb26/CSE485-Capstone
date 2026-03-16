import logging

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LocalChat:
    """Chat client that talks to a local OpenAI-compatible server (e.g. vLLM).

    Posts to ``{base_url}/chat/completions`` using the standard OpenAI
    request format with ``temperature=0.0`` and ``max_tokens=1024``.
    """

    def __init__(self, model: str, base_url: str) -> None:
        self.model: str = model
        self.base_url: str = base_url

    def chat(self, instructions: str, messages: list[dict]) -> str:
        """Send a conversation to the local server and return the assistant's reply.

        Args:
            instructions: System-level prompt prepended to the conversation.
            messages: Conversation history as a list of ``{"role", "content"}``
                dicts.

        Returns:
            The assistant's response text, or an empty string on failure.
        """
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": instructions},
                        *[
                            {"role": message["role"], "content": message["content"]}
                            for message in messages
                        ],
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.0,
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            logger.info("Error sending query to LocalChat. Returning empty string.")
            return ""
