from __future__ import annotations

import logging
import os
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _ThoughtResponse(BaseModel):
    reasoning: str = Field(
        description=(
            "A concise step-by-step chain of reasoning (1-3 sentences) that explains how to "
            "arrive at the correct answer. Plain prose only — no JSON, no XML tags, no answer repetition."
        )
    )


_SYSTEM_PROMPT = (
    "You are a reasoning assistant for negotiation tasks. "
    "You will be given a task prompt and the correct answer. "
)

_USER_TMPL = """\
Task prompt:

{prompt}

Correct answer (JSON): {answer_json}

Explain the reasoning that leads to this answer."""


class LLMThoughtGenerator:
    def __init__(self, model: str, api_key_env: str):
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Environment variable '{api_key_env}' is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self, prompt: str, answer_json: str, det_thought: str
    ) -> tuple[str, str]:
        user_msg = _USER_TMPL.format(prompt=prompt, answer_json=answer_json)
        try:
            resp = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format=_ThoughtResponse,
                temperature=0.7,
                max_tokens=300,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Model refused to respond")
            reasoning = re.sub(r"[\uD800-\uDFFF]", "", parsed.reasoning).strip()
            return reasoning, "llm"
        except Exception as exc:
            logger.warning(
                "LLM thought generation failed (%s); using deterministic.", exc
            )
            return det_thought, "deterministic"
