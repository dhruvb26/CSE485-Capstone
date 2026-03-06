"""Async LLM thought generation for negotiation turn SFT data.

Given a turn row (prompt, talk, action, strategy_label), calls an external
LLM to produce a concise internal reasoning thought that explains the
negotiation logic behind the turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from tqdm.auto import tqdm

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _TurnThoughtResponse(BaseModel):
    thought: str = Field(
        description=(
            "A concise internal thought (2-4 sentences) explaining the negotiation reasoning "
            "for this turn. Include explicit point arithmetic with calculations "
            "(e.g., '2 food x 4pts = 8pts') and partner priority estimate."
        )
    )


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
1. Explicit point arithmetic with numerical calculations shown (e.g., '2 food x 4pts = 8pts, total = 8 + 5 = 13pts')
2. Partner priority estimate based on dialogue
3. Justification for the chosen action type"""


def _sanitize(text: str) -> str:
    return re.sub(r"[\uD800-\uDFFF]", "", text).strip()


class AsyncLLMThoughtGenerator:
    """Generates turn-level thoughts in parallel using AsyncOpenAI."""

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
        """Generate a single turn thought, returning (thought_text, source)."""
        strat = row.get("strategy_label")
        strategy_line = f"\nStrategy annotation: {strat}" if strat else ""
        user_msg = _TURN_USER.format(
            prompt=row["prompt"],
            talk=row["talk"],
            action_json=json.dumps(row["action"]),
            strategy_line=strategy_line,
        )

        async with self.sem:
            try:
                resp = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _TURN_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format=_TurnThoughtResponse,
                    temperature=0.7,
                    max_completion_tokens=2200,
                )
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Model refused to respond")
                return _sanitize(parsed.thought), "llm"
            except Exception as exc:
                logger.warning("LLM thought failed (%s); skipping row.", exc)
                return "", "failed"

    async def generate_batch(
        self,
        rows: list[dict],
    ) -> list[tuple[str, str]]:
        """Process all rows in parallel, returning (thought, source) per row."""
        pbar = tqdm(total=len(rows), desc="LLM thoughts", unit="row")

        async def _wrapped(row: dict) -> tuple[str, str]:
            result = await self._generate_one(row)
            pbar.update(1)
            return result

        tasks = [_wrapped(row) for row in rows]
        results = await asyncio.gather(*tasks)
        pbar.close()
        return results
