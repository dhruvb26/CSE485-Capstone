"""Async hindsight thought generation for negotiation turn SFT data.

Given a turn row with full hindsight context (both agents' point values,
complete dialogue, and final outcome), calls an LLM to produce the thought
that a perfect strategist would have written at that turn knowing the full
picture.  This is the STaR "rationalization" step applied to negotiation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class _HindsightThoughtResponse(BaseModel):
    thought: str = Field(
        description=(
            "A concise internal thought (4-6 sentences) that a perfect negotiation "
            "strategist would write at this turn. Must follow all four steps in order: "
            "(1) point arithmetic for the chosen AND one alternative allocation, "
            "(2) partner priority estimate inferred from dialogue cues only, "
            "(3) best concession identification, "
            "(4) grounded justification referencing step 1 or 2."
        )
    )


_HINDSIGHT_SYSTEM = (
    "You are a perfect negotiation strategist reviewing a completed negotiation "
    "in hindsight. You have access to the partner's true point values and the "
    "final outcome, but you must NEVER state them as known facts. Use them only "
    "to confirm which dialogue signals were most informative, not to replace "
    "inference with fact. The thought you write must read as if the agent "
    "deduced everything from the conversation alone."
)

_HINDSIGHT_USER = """\
COMPLETE NEGOTIATION CONTEXT (hindsight):

Agent's point values: {agent_values}
Partner's TRUE point values (for your reference only — do NOT state these \
as known facts in the thought): {partner_values}

Items available: 3 Food, 3 Water, 3 Firewood

Full dialogue:
{full_dialogue}

Final outcome: {outcome}

---

You are reviewing TURN {turn_index} of this dialogue.

The agent's prompt at this turn:
{prompt}

The agent's action at this turn:
- Utterance: "{talk}"
- Action: {action_json}
{strategy_line}
Write the <thought> that a perfect strategist would have written at this \
moment. Follow these four steps IN ORDER:

{steps}"""

_STEPS_EARLY = """\
Step 1 — Partner priority estimate (inference only): Estimate the partner's \
priorities based ONLY on what they have said or offered so far in the \
dialogue. You know their true values — use them only to confirm which \
signals were most informative, not to replace inference with fact. Never \
write "the partner values X at Npts" as a stated fact.

Step 2 — Point arithmetic (chosen + alternative): Compute the exact points \
for the proposed allocation (e.g., '2 food x 4pts = 8pts, total = 11pts'). \
Then compute at least one alternative allocation the agent could have \
proposed instead, showing why the chosen one is better.

Step 3 — Best concession: Identify which item is the best concession — the \
one that costs the agent the least points but gains the most goodwill given \
the partner's inferred priorities from step 1.

Step 4 — Grounded justification: Justify the specific action taken. This \
justification MUST reference something from step 1 (the partner inference) \
or step 2 (the arithmetic comparison). Do not introduce new reasoning \
disconnected from the prior steps."""

_STEPS_LATE = """\
Step 1 — Point arithmetic (chosen + alternative): Compute the exact points \
for the proposed allocation (e.g., '2 food x 4pts = 8pts, total = 11pts'). \
Then compute at least one alternative allocation the agent could have \
proposed instead, showing why the chosen one is better.

Step 2 — Partner priority estimate (inference only): Estimate the partner's \
priorities based ONLY on what they have said or offered so far in the \
dialogue. You know their true values — use them only to confirm which \
signals were most informative, not to replace inference with fact. Never \
write "the partner values X at Npts" as a stated fact.

Step 3 — Best concession: Identify which item is the best concession — the \
one that costs the agent the least points but gains the most goodwill given \
the partner's inferred priorities from step 2.

Step 4 — Grounded justification: Justify the specific action taken. This \
justification MUST reference something from step 1 (the arithmetic \
comparison) or step 2 (the partner inference). Do not introduce new \
reasoning disconnected from the prior steps."""

_EARLY_TURN_THRESHOLD = 4


def _sanitize(text: str) -> str:
    return re.sub(r"[\uD800-\uDFFF]", "", text).strip()


def _fmt_values(vals: dict[str, int]) -> str:
    return ", ".join(f"{k.title()}={v}pts" for k, v in vals.items())


def _fmt_outcome(outcome: dict) -> str:
    if not outcome.get("deal_reached"):
        return "No deal reached (walk-away)."
    agent_alloc = outcome.get("agent_allocation", {})
    partner_alloc = outcome.get("partner_allocation", {})
    agent_pts = outcome.get("agent_points", "?")
    partner_pts = outcome.get("partner_points", "?")
    parts = [
        f"{k}: agent={agent_alloc.get(k, '?')}/partner={partner_alloc.get(k, '?')}"
        for k in ("food", "water", "firewood")
    ]
    return (
        f"Deal reached. Allocation: {', '.join(parts)}. "
        f"Agent scored {agent_pts}pts, partner scored {partner_pts}pts."
    )


class AsyncHindsightThoughtGenerator:
    """Generates hindsight-rationalized thoughts in parallel using AsyncOpenAI.

    Drop-in replacement for ``AsyncLLMThoughtGenerator`` — exposes the same
    ``generate_batch(rows)`` interface so the rest of the SFT pipeline is
    unchanged.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str,
        max_concurrency: int = 30,
        base_url: str | None = None,
    ):
        from openai import AsyncOpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Environment variable '{api_key_env}' is not set.")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers={"X-Provider-Key": os.environ.get("ANTHROPIC_API_KEY")})
        self.model = model
        self.sem = asyncio.Semaphore(max_concurrency)

    def _build_prompt(self, row: dict) -> tuple[str, str]:
        """Build the system and user messages for a single row.

        Step ordering is conditional on turn index: early turns (< 4) lead
        with partner inference since the agent needs a partner model before
        it can reason about offers; late turns lead with arithmetic since the
        strategic question is largely resolved.
        """
        strat = row.get("strategy_label")
        strategy_line = f"\nStrategy annotation: {strat}" if strat else ""

        turn_idx = row.get("turn_index", 0)
        try:
            turn_idx = int(turn_idx)
        except (TypeError, ValueError):
            turn_idx = 0
        steps = _STEPS_EARLY if turn_idx < _EARLY_TURN_THRESHOLD else _STEPS_LATE

        user_msg = _HINDSIGHT_USER.format(
            agent_values=_fmt_values(row["agent_values"]),
            partner_values=_fmt_values(row["partner_values"]),
            full_dialogue=row["full_dialogue"],
            outcome=_fmt_outcome(row["dialogue_outcome"]),
            turn_index=row.get("turn_index", "?"),
            prompt=row["prompt"],
            talk=row["talk"],
            action_json=json.dumps(row["action"]),
            strategy_line=strategy_line,
            steps=steps,
        )
        return _HINDSIGHT_SYSTEM, user_msg

    async def _generate_one(self, row: dict) -> tuple[str, str, str]:
        """Return (thought_text, source, raw_prompt_to_4o)."""
        system_msg, user_msg = self._build_prompt(row)
        raw_prompt = f"[SYSTEM]\n{system_msg}\n\n[USER]\n{user_msg}"

        async with self.sem:
            try:
                resp = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format=_HindsightThoughtResponse,
                    temperature=0.7,
                    max_completion_tokens=2200,
                )
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Model refused to respond")
                return _sanitize(parsed.thought), "hindsight", raw_prompt
            except Exception as exc:
                logger.warning("Hindsight thought failed (%s); skipping row.", exc)
                return "", "failed", raw_prompt

    async def generate_batch(
        self,
        rows: list[dict],
    ) -> list[tuple[str, str, str]]:
        """Process all rows in parallel, returning (thought, source, raw_prompt) per row."""
        pbar = tqdm(total=len(rows), desc="Hindsight thoughts", unit="row")

        async def _wrapped(row: dict) -> tuple[str, str, str]:
            result = await self._generate_one(row)
            pbar.update(1)
            return result

        tasks = [_wrapped(row) for row in rows]
        results = await asyncio.gather(*tasks)
        pbar.close()
        return results
