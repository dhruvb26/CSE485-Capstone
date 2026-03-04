"""Clone opponent personas for self-play diversity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    """A behavioural persona injected into the clone opponent's system prompt."""
    name: str
    system_prompt_suffix: str
    description: str


PERSONAS: dict[str, Persona] = {
    "uncompromising": Persona(
        name="uncompromising",
        system_prompt_suffix=(
            "You are an uncompromising negotiator. You insist on receiving your "
            "highest-priority items and rarely make concessions. Only agree to a "
            "deal if you receive at least 70% of your maximum possible points."
        ),
        description="Insists on highest-priority items, rarely concedes.",
    ),
    "selfish": Persona(
        name="selfish",
        system_prompt_suffix=(
            "You are a selfish negotiator. You always claim 2-3 units of your "
            "top-valued item in every offer. You anchor high and make minimal "
            "concessions."
        ),
        description="Claims 2-3 units of top-valued item in every offer.",
    ),
    "anchoring": Persona(
        name="anchoring",
        system_prompt_suffix=(
            "You are an anchoring negotiator. You open with an extreme position "
            "requesting nearly all items. You move slowly toward the middle, "
            "making very small concessions each turn."
        ),
        description="Opens extreme, moves slowly.",
    ),
    "cooperative": Persona(
        name="cooperative",
        system_prompt_suffix=(
            "You are a cooperative negotiator. You try to find mutually "
            "beneficial outcomes and are willing to make fair concessions. "
            "You aim for deals where both parties get reasonable value."
        ),
        description="Seeks mutually beneficial outcomes, fair concessions.",
    ),
}
