import ast
import asyncio
import csv
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

import chz
import tinker
import trackio
from dotenv import load_dotenv
from tinker import ModelInput
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.completers import (
    MessageCompleter,
    StopCondition,
    TinkerMessageCompleter,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer, get_text_content
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

load_dotenv()

POINTS = {"High": 5, "Medium": 4, "Low": 3}
MAX_POINTS = 3 * 5 + 3 * 4 + 3 * 3

SYSTEM_PROMPT = """You are negotiating with your campsite neighbor over 3 packages each of food, water, and firewood. Each item allocation must be 0-3 and both sides must sum to 3 per item.

Your priorities:

{items_block}

Your reply must include these 3 parts in order:

Thought: your private strategic reasoning (not shown to the neighbor).
Talk: what you say to the neighbor. Be concise.
Action: exactly one of the following:

1. [SUBMIT_DEAL] food:F water:W firewood:FW if you want to propose a deal. F, W, FW are YOUR allocations (0-3 each). Your neighbor receives 3 minus each value. When your neighbor proposes a [SUBMIT_DEAL], the values shown are what YOU would receive.
2. [ACCEPT_DEAL] if you agree to the neighbor's most recent [SUBMIT_DEAL]. This ends the negotiation and closes the deal.
3. [REJECT_DEAL] if you want to reject the neighbor's most recent offer and await a new offer.
4. [WALK_AWAY] if you believe a good deal cannot be reached. This ends the negotiation with no deal."""


@dataclass(frozen=True)
class Scenario:
    participant_info: dict
    agent_ids: tuple[str, ...]


def load_scenarios(csv_path: str) -> list[Scenario]:
    scenarios = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pi = ast.literal_eval(row["participant_info"])
            scenarios.append(Scenario(participant_info=pi, agent_ids=tuple(pi.keys())))
    return scenarios


def build_system_prompt(participant_info: dict, agent_id: str) -> str:
    v2i = participant_info[agent_id]["value2issue"]
    v2r = participant_info[agent_id]["value2reason"]
    items_block = "\n  ".join(
        f"{v2i[p]} ({POINTS[p]} points) - {v2r[p]}" for p in POINTS
    )
    return SYSTEM_PROMPT.format(items_block=items_block)


def compute_points(deal: dict[str, int], value2issue: dict[str, str]) -> int:
    issue2points = {v.lower(): POINTS[k] for k, v in value2issue.items()}
    return sum(units * issue2points.get(item, 0) for item, units in deal.items())


def parse_submit_deal(text: str) -> dict[str, int] | None:
    m = re.search(
        r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if m is None:
        return None
    return {
        "food": int(m.group(1)),
        "water": int(m.group(2)),
        "firewood": int(m.group(3)),
    }


def flip_deal_perspective(text: str) -> str:
    def _flip(m: re.Match) -> str:
        f, w, fw = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"[SUBMIT_DEAL] food:{3 - f} water:{3 - w} firewood:{3 - fw}"

    return re.sub(
        r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)",
        _flip,
        text,
        flags=re.IGNORECASE,
    )


def extract_section(text: str, label: str) -> str | None:
    m = re.search(
        rf"^{label}:\s*(.+?)(?=^(?:Thought|Talk|Action):|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def strip_thought(text: str) -> str:
    return re.sub(
        r"^Thought:.*?(?=^(?:Talk|Action):|\Z)",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    ).strip()


OUTCOMES = ("deal", "walk_away", "max_turns", "format_violation")


def _outcome_metrics(outcome: str, **extra: float) -> dict[str, float]:
    return {f"outcome_{o}": float(o == outcome) for o in OUTCOMES} | extra


class NegotiationEnv(Env):
    """One negotiation episode. The learner (policy) negotiates against a
    frozen opponent (base model via MessageCompleter).

    Tinker calls initial_observation() to get the first prompt, then alternates:
      1. Learner generates a response (Tinker handles this)
      2. We receive it in step(), play the opponent's turn, return next observation

    Reward is only given at episode end:
      deal  -> learner_points / MAX_POINTS
      no deal / walk away -> 0.0
      format violation -> -1.0
    """

    def __init__(
        self,
        opponent: MessageCompleter,
        scenario: Scenario,
        renderer: Renderer,
        max_rounds: int = 6,
    ):
        self.opponent = opponent
        self.scenario = scenario
        self.renderer = renderer
        self.max_rounds = max_rounds

        ids = scenario.agent_ids
        self.learner_id, self.opponent_id = (
            (ids[0], ids[1]) if random.random() < 0.5 else (ids[1], ids[0])
        )

        self.learner_sys: Message = {
            "role": "system",
            "content": build_system_prompt(scenario.participant_info, self.learner_id),
        }
        self.opponent_sys: Message = {
            "role": "system",
            "content": build_system_prompt(scenario.participant_info, self.opponent_id),
        }

        # Conversation from each side's perspective
        self.learner_turns: list[Message] = []
        self.opponent_turns: list[Message] = []
        self.round = 0
        self.last_submit_deal: dict[str, int] | None = None
        self.last_submit_by: str | None = None

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _learner_convo(self) -> list[Message]:
        if not self.learner_turns:
            return [
                self.learner_sys,
                {"role": "user", "content": "Begin the negotiation."},
            ]
        return [self.learner_sys] + self.learner_turns

    def _opponent_convo(self) -> list[Message]:
        return [self.opponent_sys] + self.opponent_turns

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return (
            self.renderer.build_generation_prompt(self._learner_convo()),
            self.stop_condition,
        )

    async def step(
        self, action: Action, *, extra: ActionExtra | None = None
    ) -> StepResult:
        """Process one round: learner spoke, now opponent responds."""
        # Parse the learner's response
        (learner_msg, _) = self.renderer.parse_response(action)
        learner_text = get_text_content(learner_msg)

        # Check format
        if not (
            extract_section(learner_text, "Thought")
            and extract_section(learner_text, "Talk")
            and extract_section(learner_text, "Action")
        ):
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=-1.0,
                metrics=_outcome_metrics("format_violation"),
            )

        # Add learner's message to both conversation histories
        self.learner_turns.append({"role": "assistant", "content": learner_text})
        visible = flip_deal_perspective(strip_thought(learner_text))
        self.opponent_turns.append({"role": "user", "content": visible})

        # Check learner's action for terminal conditions
        learner_action = extract_section(learner_text, "Action") or ""
        learner_action_upper = learner_action.strip().upper()

        parsed = parse_submit_deal(learner_action)
        if parsed is not None:
            self.last_submit_deal = parsed
            self.last_submit_by = "learner"

        # Learner accepts opponent's deal
        if (
            "[ACCEPT_DEAL]" in learner_action_upper
            and self.last_submit_deal is not None
            and self.last_submit_by == "opponent"
        ):
            learner_alloc = {
                item: 3 - qty for item, qty in self.last_submit_deal.items()
            }
            points = compute_points(
                learner_alloc,
                self.scenario.participant_info[self.learner_id]["value2issue"],
            )
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=points / MAX_POINTS,
                metrics=_outcome_metrics("deal", learner_points=points),
            )

        if "[WALK_AWAY]" in learner_action_upper:
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
                metrics=_outcome_metrics("walk_away"),
            )

        self.round += 1
        if self.round >= self.max_rounds:
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
                metrics=_outcome_metrics("max_turns"),
            )

        # Opponent's turn
        opponent_msg = await self.opponent(self._opponent_convo())
        opponent_text = get_text_content(opponent_msg)

        self.opponent_turns.append({"role": "assistant", "content": opponent_text})
        visible_opp = flip_deal_perspective(strip_thought(opponent_text))
        self.learner_turns.append({"role": "user", "content": visible_opp})

        # Check opponent's action
        opp_action = extract_section(opponent_text, "Action") or ""
        opp_action_upper = opp_action.strip().upper()

        opp_parsed = parse_submit_deal(opp_action)
        if opp_parsed is not None:
            self.last_submit_deal = opp_parsed
            self.last_submit_by = "opponent"

        # Opponent accepts learner's deal
        if (
            "[ACCEPT_DEAL]" in opp_action_upper
            and self.last_submit_deal is not None
            and self.last_submit_by == "learner"
        ):
            points = compute_points(
                self.last_submit_deal,
                self.scenario.participant_info[self.learner_id]["value2issue"],
            )
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=points / MAX_POINTS,
                metrics=_outcome_metrics("deal", learner_points=points),
            )

        if "[WALK_AWAY]" in opp_action_upper:
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
                metrics=_outcome_metrics("walk_away"),
            )

        # Continue negotiation
        return StepResult(
            next_observation=self.renderer.build_generation_prompt(
                self._learner_convo()
            ),
            next_stop_condition=self.stop_condition,
            episode_done=False,
            reward=0.0,
        )


@dataclass(frozen=True)
class NegotiationGroupBuilder(EnvGroupBuilder):
    """Builds G copies of the same scenario for GRPO advantage computation."""

    opponent: MessageCompleter
    scenario: Scenario
    renderer: Renderer
    num_envs: int
    max_rounds: int = 6

    async def make_envs(self) -> Sequence[Env]:
        return [
            NegotiationEnv(self.opponent, self.scenario, self.renderer, self.max_rounds)
            for _ in range(self.num_envs)
        ]


@dataclass(frozen=True)
class NegotiationDataset(RLDataset):
    """Produces batches of scenario groups for the training loop."""

    opponent: MessageCompleter
    scenarios: Sequence[Scenario]
    renderer: Renderer
    batch_size: int
    group_size: int
    max_rounds: int = 6

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self.scenarios)
        batch_scenarios = [
            self.scenarios[(start + i) % len(self.scenarios)]
            for i in range(self.batch_size)
        ]
        return [
            NegotiationGroupBuilder(
                opponent=self.opponent,
                scenario=s,
                renderer=self.renderer,
                num_envs=self.group_size,
                max_rounds=self.max_rounds,
            )
            for s in batch_scenarios
        ]

    def __len__(self) -> int:
        return len(self.scenarios) // self.batch_size


@chz.chz
class NegotiationDatasetBuilder(RLDatasetBuilder):
    csv_path: str = "data/casino/ca.train.csv"
    eval_csv_path: str | None = None
    batch_size: int = 2
    group_size: int = 2
    max_rounds: int = 6
    model_name_for_tokenizer: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    num_epochs: int = 1
    opponent_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    opponent_temperature: float = 0.7
    eval_csv_path: str | None = None
    eval_group_size: int = 4

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        service_client = tinker.ServiceClient()

        # Opponent is the frozen base model
        opp_renderer_name = model_info.get_recommended_renderer_name(
            self.opponent_model
        )
        opp_tokenizer = get_tokenizer(self.opponent_model)
        opp_renderer = get_renderer(opp_renderer_name, opp_tokenizer)
        opp_client = service_client.create_sampling_client(
            base_model=self.opponent_model
        )
        opponent = TinkerMessageCompleter(
            sampling_client=opp_client,
            renderer=opp_renderer,
            max_tokens=300,
            temperature=self.opponent_temperature,
        )

        # Learner renderer
        learner_renderer_name = model_info.get_recommended_renderer_name(
            self.model_name_for_tokenizer
        )
        learner_tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        learner_renderer = get_renderer(learner_renderer_name, learner_tokenizer)

        scenarios = load_scenarios(self.csv_path)
        random.shuffle(scenarios)
        scenarios = scenarios * self.num_epochs

        dataset = NegotiationDataset(
            opponent=opponent,
            scenarios=scenarios,
            renderer=learner_renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            max_rounds=self.max_rounds,
        )
        eval_dataset = None
        if self.eval_csv_path:
            eval_scenarios = load_scenarios(self.eval_csv_path)
            eval_dataset = NegotiationDataset(
                opponent=opponent,
                scenarios=eval_scenarios,
                renderer=learner_renderer,
                batch_size=len(eval_scenarios),
                group_size=self.eval_group_size,
                max_rounds=self.max_rounds,
            )
        return dataset, eval_dataset


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    csv_path: str = "data/casino/ca.train.csv"
    eval_csv_path: str | None = None
    group_size: int = 2
    batch_size: int = 2
    num_epochs: int = 2
    learning_rate: float = 3e-5
    lora_rank: int = 32  # Tinker default; LoRA-only platform
    max_tokens: int = 300
    max_rounds: int = 6

    opponent_temperature: float = 0.7
    kl_penalty_coef: float = 0.0  # paper Table 5
    save_every: int = 10
    eval_every: int = 0
    wandb_project: str | None = "agent-rlvr"
    trackio_project: str = "agent-rlvr"
    trackio_space_id: str = "dhruvb26/agent-rlvr"
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(cli: CLIConfig) -> tuple[train.Config, str]:
    from datetime import datetime

    renderer_name = model_info.get_recommended_renderer_name(cli.model_name)
    run_name = f"tinker-rlvr-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    dataset_builder = NegotiationDatasetBuilder(
        csv_path=cli.csv_path,
        batch_size=cli.batch_size,
        group_size=cli.group_size,
        max_rounds=cli.max_rounds,
        model_name_for_tokenizer=cli.model_name,
        num_epochs=cli.num_epochs,
        opponent_model=cli.model_name,
        opponent_temperature=cli.opponent_temperature,
        eval_csv_path=cli.eval_csv_path,
    )

    log_path = f"logs/tinker-negotiation/{run_name}"

    config = train.Config(
        model_name=cli.model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        kl_penalty_coef=cli.kl_penalty_coef,
        save_every=cli.save_every,
        eval_every=cli.eval_every,
        wandb_project=cli.wandb_project,
        wandb_name=run_name if cli.wandb_project else None,
    )

    return config, run_name


def sync_to_trackio(
    log_path: str, project: str, name: str, space_id: str, cli: CLIConfig
) -> None:
    """Read Tinker's metrics.jsonl after training and push to trackio with config."""
    import json
    from pathlib import Path

    metrics_file = Path(log_path) / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"No metrics file at {metrics_file}")
        return

    trackio.init(
        project=project,
        name=name,
        space_id=space_id,
        config={
            "model_name": cli.model_name,
            "learning_rate": cli.learning_rate,
            "lora_rank": cli.lora_rank,
            "opponent_temperature": cli.opponent_temperature,
            "kl_penalty_coef": cli.kl_penalty_coef,
            "group_size": cli.group_size,
            "batch_size": cli.batch_size,
            "num_epochs": cli.num_epochs,
            "max_tokens": cli.max_tokens,
            "max_rounds": cli.max_rounds,
            "csv_path": cli.csv_path,
        },
    )
    with open(metrics_file) as f:
        for line in f:
            m = json.loads(line)
            trackio.log(m, step=int(m.get("progress/batch", 0)))
    trackio.finish()
    print(f"Synced metrics to trackio: {project}/{name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="tinkering/config.yaml", help="YAML config file"
    )
    args, remaining = parser.parse_known_args()

    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cli = CLIConfig(**{k: v for k, v in cfg.items() if hasattr(CLIConfig, k)})
    else:
        cli = chz.entrypoint(CLIConfig)

    config, run_name = build_config(cli)
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli.behavior_if_log_dir_exists
    )
    asyncio.run(train.main(config))
    sync_to_trackio(
        config.log_path, cli.trackio_project, run_name, cli.trackio_space_id, cli
    )
