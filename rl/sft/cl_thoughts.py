"""Deterministic thought templates for CraigslistBargain SFT tasks."""

from __future__ import annotations


def action_inference(action: str, last_text: str) -> str:
    return (
        f'Analysing the last utterance: "{last_text[:120]}..." '
        f"This corresponds to a {action} action."
    )


def price_reasoning(
    price: float, listing_price: float, role: str, last_text: str
) -> str:
    pct = (price / listing_price * 100) if listing_price > 0 else 0
    direction = "below" if price < listing_price else "at or above"
    return (
        f"The listing price is ${listing_price:.2f}. "
        f"The {role}'s last mentioned price is ${price:.2f} ({pct:.0f}% of listing, {direction} asking). "
        f'From utterance: "{last_text[:120]}..."'
    )
