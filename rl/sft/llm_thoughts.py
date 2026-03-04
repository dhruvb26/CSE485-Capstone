from __future__ import annotations

import asyncio
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


class _TurnThoughtResponse(BaseModel):
    thought: str = Field(
        description=(
            "A concise internal thought (2-4 sentences) explaining the negotiation reasoning "
            "for this turn. Include point/price arithmetic and partner priority estimate."
        )
    )

_EVAL_SYSTEM = (
    "You are a reasoning assistant for negotiation tasks. "
    "You will be given a task prompt and the correct answer. "
)

_EVAL_USER = """\
Task prompt:

{prompt}

Correct answer (JSON): {answer_json}

Explain the reasoning that leads to this answer."""

_TURN_SYSTEM = (
    "You are generating internal reasoning for a negotiation agent. "
    "Given the negotiation context, the agent's utterance, and the action taken, "
    "write a concise internal thought that explains the reasoning behind this turn."
)

_TURN_USER = """\
Negotiation context (the agent's prompt):

{prompt}

The agent's next turn:
- Utterance: "{talk}"
- Action: {action_json}
{strategy_line}
Write a concise internal thought (2-4 sentences) explaining the reasoning \
leading to this action. Include:
1. Point/price arithmetic (calculate values of current/proposed offers)
2. Partner priority estimate based on dialogue
3. Justification for the chosen action type"""


def _sanitize(text: str) -> str:
    return re.sub(r"[\uD800-\uDFFF]", "", text).strip()


# ===================================================================
# Synchronous generator (kept for backward compat)
# ===================================================================


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
        user_msg = _EVAL_USER.format(prompt=prompt, answer_json=answer_json)
        try:
            resp = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": _EVAL_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_format=_ThoughtResponse,
                temperature=0.7,
                max_tokens=500,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Model refused to respond")
            return _sanitize(parsed.reasoning), "llm"
        except Exception as exc:
            logger.warning(
                "LLM thought generation failed (%s); using deterministic.", exc
            )
            return det_thought, "deterministic"

class AsyncLLMThoughtGenerator:
    """Generates thoughts for many rows in parallel using AsyncOpenAI."""

    def __init__(
        self,
        model: str,
        api_key_env: str,
        max_concurrency: int = 30,
    ):
        from openai import AsyncOpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Environment variable '{api_key_env}' is not set.")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.sem = asyncio.Semaphore(max_concurrency)

    async def _generate_one(self, row: dict) -> tuple[str, str]:
        """Generate a single thought, returning (thought_text, source)."""
        is_turn = "talk" in row

        if is_turn:
            import json
            strat = row.get("strategy_label")
            strategy_line = f"\nStrategy annotation: {strat}" if strat else ""
            user_msg = _TURN_USER.format(
                prompt=row["prompt"],
                talk=row["talk"],
                action_json=json.dumps(row["action"]),
                strategy_line=strategy_line,
            )
            system_msg = _TURN_SYSTEM
            response_fmt = _TurnThoughtResponse
        else:
            user_msg = _EVAL_USER.format(
                prompt=row["prompt"], answer_json=row["answer_json"],
            )
            system_msg = _EVAL_SYSTEM
            response_fmt = _ThoughtResponse

        async with self.sem:
            try:
                resp = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format=response_fmt,
                    temperature=0.7,
                    max_tokens=300,
                )
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Model refused to respond")
                text = parsed.thought if is_turn else parsed.reasoning
                return _sanitize(text), "llm"
            except Exception as exc:
                logger.warning("LLM thought failed (%s); using deterministic.", exc)
                return row["det_thought"], "deterministic"

    async def generate_batch(
        self, rows: list[dict],
    ) -> list[tuple[str, str]]:
        """Process all rows in parallel, returning (thought, source) per row."""
        tasks = [self._generate_one(row) for row in rows]
        return await asyncio.gather(*tasks)
