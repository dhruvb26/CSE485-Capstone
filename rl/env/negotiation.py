"""Turn-based negotiation environment for GRPO self-play."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from rl.env.actions import Action, ActionType
from rl.env.personas import Persona
from rl.env.scenario import ITEMS, Scenario
from rl.verifiers.format import check_format


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
            self._done = True
            return turn

        if action is not None and action.type == ActionType.ACCEPT:
            self._done = True
            if len(self.history) >= 2:
                prev = self.history[-2]
                if prev.action is not None and prev.action.type in (
                    ActionType.OFFER, ActionType.COUNTER
                ):
                    self._deal_reached = True
                    if agent == "learner":
                        self._agent_deal_action = prev.action
                        self._partner_deal_action = prev.action
                    else:
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
        """Build the prompt for the learner's next turn."""
        lines = [
            "You are negotiating over food, water, and firewood (3 of each).",
            "",
            "Your point values:",
        ]
        for item in ITEMS:
            lines.append(f"  {item}: {self.scenario.agent_values[item]} points each")
        lines.append("")

        if self.history:
            lines.append("Dialogue so far:")
            for turn in self.history:
                speaker = "You" if turn.agent == "learner" else "Partner"
                lines.append(f"  {speaker}: {turn.talk}")
            lines.append("")

        lines.append(
            "Respond with a JSON object containing: thought, talk, action."
        )
        return "\n".join(lines)

    def build_clone_prompt(self) -> str:
        """Build the prompt for the clone opponent's next turn."""
        lines = [
            "You are negotiating over food, water, and firewood (3 of each).",
            self.persona.system_prompt_suffix,
            "",
            "Your point values:",
        ]
        for item in ITEMS:
            lines.append(f"  {item}: {self.scenario.partner_values[item]} points each")
        lines.append("")

        if self.history:
            lines.append("Dialogue so far:")
            for turn in self.history:
                speaker = "Partner" if turn.agent == "learner" else "You"
                lines.append(f"  {speaker}: {turn.talk}")
            lines.append("")

        lines.append(
            "Respond with a JSON object containing: thought, talk, action."
        )
        return "\n".join(lines)


def _parse_agent_output(
    raw: str, scenario: Scenario
) -> tuple[str, str, Action | None, bool]:
    """Parse raw model output into (thought, talk, action, valid)."""
    fmt_ok, _ = check_format(raw)
    if not fmt_ok:
        return "", raw.strip(), None, False

    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return "", raw.strip(), None, False
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return "", raw.strip(), None, False

    thought = str(parsed.get("thought", ""))
    talk = str(parsed.get("talk", ""))

    action_raw = parsed.get("action")
    if not isinstance(action_raw, dict) or "type" not in action_raw:
        return thought, talk, None, False

    try:
        action = Action.from_dict(action_raw)
    except (ValueError, KeyError):
        return thought, talk, None, False

    if action.type in (ActionType.OFFER, ActionType.COUNTER):
        ok, _ = action.validate(scenario.items)
        if not ok:
            return thought, talk, action, False

    return thought, talk, action, True
