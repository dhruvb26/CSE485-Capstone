"""Turn-based negotiation environment for GRPO self-play."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from rl.env.actions import Action, ActionType
from rl.env.personas import Persona
from rl.env.scenario import ITEMS, Scenario
from rl.verifiers.format import parse_xml_turn

_MAX_TALK_LEN = 200
_TALK_RE = re.compile(r"<talk>(.*?)</talk>", re.DOTALL | re.IGNORECASE)


@dataclass
class Turn:
    """Record of a single turn in the negotiation."""
    agent: str
    raw_output: str
    thought: str
    talk: str
    action: Action | None
    valid: bool


@dataclass
class NegotiationEnv:
    """Stateful negotiation environment.

    Manages alternating turns between a learner and a clone opponent,
    validates outputs, and tracks the dialogue history.
    """
    scenario: Scenario
    persona: Persona
    max_turns: int
    history: list[Turn] = field(default_factory=list)
    _done: bool = False
    _deal_reached: bool = False
    _agent_deal_action: Action | None = None
    _partner_deal_action: Action | None = None

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def deal_reached(self) -> bool:
        return self._deal_reached

    @property
    def current_turn(self) -> int:
        return len(self.history)

    @property
    def is_learner_turn(self) -> bool:
        return len(self.history) % 2 == 0

    def reset(self, scenario: Scenario, persona: Persona, max_turns: int) -> None:
        self.scenario = scenario
        self.persona = persona
        self.max_turns = max_turns
        self.history = []
        self._done = False
        self._deal_reached = False
        self._agent_deal_action = None
        self._partner_deal_action = None

    def step(self, raw_output: str) -> Turn:
        """Process one agent turn. Returns the recorded Turn.

        Call alternately for learner (even turns) and clone (odd turns).
        """
        if self._done:
            raise RuntimeError("Episode is already done.")

        agent = "learner" if self.is_learner_turn else "clone"

        thought, talk, action, valid = _parse_agent_output(raw_output, self.scenario)

        turn = Turn(
            agent=agent,
            raw_output=raw_output,
            thought=thought,
            talk=talk,
            action=action,
            valid=valid,
        )
        self.history.append(turn)

        if not valid:
            if self.current_turn >= self.max_turns:
                self._done = True
            return turn

        if action is not None and action.type == ActionType.ACCEPT:
            if len(self.history) < 2:
                turn.valid = False
                if self.current_turn >= self.max_turns:
                    self._done = True
                return turn
            prev = self.history[-2]
            if prev.action is None or prev.action.type not in (
                ActionType.OFFER, ActionType.COUNTER
            ):
                turn.valid = False
                if self.current_turn >= self.max_turns:
                    self._done = True
                return turn
            self._done = True
            self._deal_reached = True
            self._agent_deal_action = prev.action
            self._partner_deal_action = prev.action
            return turn

        if action is not None and action.type == ActionType.REJECT:
            self._done = True
            return turn

        if self.current_turn >= self.max_turns:
            self._done = True

        return turn

    def agent_points(self) -> int:
        """Points the learner gets if the current deal is accepted."""
        if not self._deal_reached or self._agent_deal_action is None:
            return 0
        return sum(
            self._agent_deal_action.allocations.get(item, 0) * self.scenario.agent_values[item]
            for item in ITEMS
        )

    def partner_points(self) -> int:
        """Points the clone gets from the deal."""
        if not self._deal_reached or self._agent_deal_action is None:
            return 0
        return sum(
            (self.scenario.items[item] - self._agent_deal_action.allocations.get(item, 0))
            * self.scenario.partner_values[item]
            for item in ITEMS
        )

    def build_learner_prompt(self) -> str:
        """Build the prompt for the learner's next turn.

        Format instructions appear both at the top (survives truncation)
        and at the bottom (matches SFT training layout).
        """
        vals = self.scenario.agent_values
        max_pts = self.scenario.agent_max_points

        format_spec = (
            "Produce your next negotiation turn using the following XML format:\n"
            "<thought>your internal reasoning (point arithmetic, partner priority "
            "estimate, justification)</thought>\n"
            "<talk>your natural language response to your partner</talk>\n"
            '<action>{"type": "offer"|"counter"|"accept"|"reject", '
            "...item allocations for yourself}</action>"
        )

        lines = [
            "You are negotiating with your campsite neighbor over extra supply "
            "of food, water, and firewood for your camping trip.",
            "You MUST respond with exactly one <thought>, one <talk>, and one <action> XML block.",
            "",
            "Items available: 3 Food, 3 Water, 3 Firewood",
            (
                f"Your point values: "
                f"Food={vals['food']}pts, "
                f"Water={vals['water']}pts, "
                f"Firewood={vals['firewood']}pts"
            ),
            f"Max possible points: {max_pts}pts",
            "",
        ]

        if self.history:
            lines.append("Dialogue so far:")
            for turn in self.history:
                speaker = "You" if turn.agent == "learner" else "Partner"
                talk = turn.talk[:_MAX_TALK_LEN]
                lines.append(f"{speaker}: {talk}")
            lines.append("")

        lines.append(format_spec)
        return "\n".join(lines)

    def build_clone_prompt(self) -> str:
        """Build the prompt for the clone opponent's next turn.

        Same structure as the learner prompt but from the partner's
        perspective, with the clone persona injected.
        """
        vals = self.scenario.partner_values
        max_pts = self.scenario.partner_max_points

        format_spec = (
            "Produce your next negotiation turn using the following XML format:\n"
            "<thought>your internal reasoning (point arithmetic, partner priority "
            "estimate, justification)</thought>\n"
            "<talk>your natural language response to your partner</talk>\n"
            '<action>{"type": "offer"|"counter"|"accept"|"reject", '
            "...item allocations for yourself}</action>"
        )

        lines = [
            "You are negotiating with your campsite neighbor over extra supply "
            "of food, water, and firewood for your camping trip.",
            self.persona.system_prompt_suffix,
            "You MUST respond with exactly one <thought>, one <talk>, and one <action> XML block.",
            "",
            "Items available: 3 Food, 3 Water, 3 Firewood",
            (
                f"Your point values: "
                f"Food={vals['food']}pts, "
                f"Water={vals['water']}pts, "
                f"Firewood={vals['firewood']}pts"
            ),
            f"Max possible points: {max_pts}pts",
            "",
        ]

        if self.history:
            lines.append("Dialogue so far:")
            for turn in self.history:
                speaker = "Partner" if turn.agent == "learner" else "You"
                talk = turn.talk[:_MAX_TALK_LEN]
                lines.append(f"{speaker}: {talk}")
            lines.append("")

        lines.append(format_spec)
        return "\n".join(lines)


def _parse_agent_output(
    raw: str, scenario: Scenario
) -> tuple[str, str, Action | None, bool]:
    """Parse raw model output into (thought, talk, action, valid).

    When full XML parsing fails, we salvage just the ``<talk>`` content
    (if present) or fall back to a short placeholder.  We **never** dump
    the entire raw output into ``talk`` because that would leak private
    ``<thought>`` content into the dialogue history and corrupt
    subsequent prompts.
    """
    parsed = parse_xml_turn(raw)
    if parsed is None:
        talk_m = _TALK_RE.search(raw)
        if talk_m:
            talk = talk_m.group(1).strip()[:_MAX_TALK_LEN]
        else:
            talk = "[no response]"
        return "", talk, None, False

    thought = parsed["thought"]
    talk = parsed["talk"][:_MAX_TALK_LEN]

    try:
        action_dict = json.loads(parsed["action_raw"])
    except json.JSONDecodeError:
        return thought, talk, None, False

    if not isinstance(action_dict, dict) or "type" not in action_dict:
        return thought, talk, None, False

    try:
        action = Action.from_dict(action_dict)
    except (ValueError, KeyError):
        return thought, talk, None, False

    if action.type in (ActionType.OFFER, ActionType.COUNTER):
        ok, _ = action.validate(scenario.items)
        if not ok:
            return thought, talk, action, False

    return thought, talk, action, True
