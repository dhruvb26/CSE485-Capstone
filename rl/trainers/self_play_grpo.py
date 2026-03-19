from __future__ import annotations

import ast
import csv
import json
import os
import random
from dataclasses import dataclass, field

import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from trl import GRPOConfig, GRPOTrainer

from rl.config import ModelConfig, SelfPlayConfig
from rl.prompts import build_system_prompt
from rl.rewards import (
    arithmetic_reward,
    configure_judge,
    format_reward,
    length_reward,
    outcome_reward,
    thought_judge_reward,
)
from rl.trainers.base import BaseTrainer
from rl.utils import extract_tag, parse_submit_deal, strip_thought


@dataclass
class Episode:
    learner_agent_id: str
    opponent_agent_id: str
    persona: str
    learner_messages: list[dict] = field(default_factory=list)
    opponent_messages: list[dict] = field(default_factory=list)
    learner_turns: list[int] = field(default_factory=list)
    outcome: str = "max_turns"
    learner_points: int | None = None
    opponent_points: int | None = None
    participant_info: dict = field(default_factory=dict)


class SelfPlayGRPOTrainer(BaseTrainer):
    """GRPO trainer that first generates negotiation episodes via self-play."""

    points = {"High": 5, "Medium": 4, "Low": 3}
    personas = {
        "uncompromising": (
            "You are a tough negotiator. You insist on getting your top-priority "
            "items and rarely make concessions. Only accept a deal if you receive "
            "at least 2 units of your highest-value item."
        ),
        "selfish": (
            "You are a self-interested negotiator. Always claim 3 units of your "
            "top item in every offer. Make small concessions only if the "
            "negotiation is about to collapse."
        ),
        "anchoring": (
            "You are a strategic negotiator. Open with an extreme offer claiming "
            "all 3 units of your top two items. Move slowly and never concede "
            "more than 1 unit per turn."
        ),
        "cooperative": (
            "You are a friendly negotiator who values reaching a deal. You are "
            "willing to split items fairly and respond positively to reasonable "
            "offers."
        ),
    }

    def __init__(self, model_config: ModelConfig, config: SelfPlayConfig):
        super().__init__(model_config)
        self.self_play_config = config

    def load_model(self):
        super().load_model()

        sft_checkpoint = self.self_play_config.sft_checkpoint
        if sft_checkpoint and os.path.isdir(sft_checkpoint):
            self.load_checkpoint(sft_checkpoint)

        return self.model, self.tokenizer

    @torch.no_grad()
    def _generate_turn(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _run_episode(
        self,
        participant_info: dict,
        agent_ids: list[str],
        persona: str,
    ) -> Episode:
        """Play a single negotiation episode with the same model on both sides.

        Args:
            participant_info: Scenario metadata with each agent's priorities.
            agent_ids: The two agent ids participating in the episode.
            persona: Opponent persona name to prepend to the opponent prompt.

        Returns:
            A completed self-play episode.

        Raises:
            Exception: If generation fails for either side.
        """
        cfg = self.self_play_config
        learner_id, opponent_id = agent_ids[0], agent_ids[1]
        if random.random() < 0.5:
            learner_id, opponent_id = opponent_id, learner_id

        learner_system_prompt = build_system_prompt(participant_info, learner_id)
        opponent_system_prompt = (
            self.personas[persona]
            + "\n\n"
            + build_system_prompt(participant_info, opponent_id)
        )

        learner_messages: list[dict] = [
            {"role": "system", "content": learner_system_prompt}
        ]
        opponent_messages: list[dict] = [
            {"role": "system", "content": opponent_system_prompt}
        ]

        episode = Episode(
            learner_agent_id=learner_id,
            opponent_agent_id=opponent_id,
            persona=persona,
            participant_info=participant_info,
        )
        learner_goes_first = random.random() < 0.5
        last_submit_deal: dict[str, int] | None = None

        for turn_index in range(cfg.max_turns):
            is_learner_turn = (turn_index % 2 == 0) == learner_goes_first

            if is_learner_turn:
                if len(learner_messages) == 1:
                    learner_messages.append(
                        {"role": "user", "content": "Begin the negotiation."}
                    )
                response = self._generate_turn(
                    learner_messages,
                    cfg.temperature,
                    cfg.top_p,
                )
                learner_messages.append({"role": "assistant", "content": response})
                opponent_messages.append(
                    {"role": "user", "content": strip_thought(response)}
                )
                episode.learner_turns.append(len(learner_messages) - 1)
            else:
                if len(opponent_messages) == 1:
                    opponent_messages.append(
                        {"role": "user", "content": "Begin the negotiation."}
                    )
                response = self._generate_turn(
                    opponent_messages,
                    cfg.temperature,
                    cfg.top_p,
                )
                opponent_messages.append({"role": "assistant", "content": response})
                learner_messages.append(
                    {"role": "user", "content": strip_thought(response)}
                )

            action = extract_tag(response, "action")
            if action:
                parsed_deal = parse_submit_deal(action)
                if parsed_deal is not None:
                    last_submit_deal = parsed_deal

            if action and "[ACCEPT_DEAL]" in action:
                episode.outcome = "deal"
                if last_submit_deal is not None:
                    learner_allocation = (
                        {item: 3 - qty for item, qty in last_submit_deal.items()}
                        if is_learner_turn
                        else last_submit_deal
                    )
                    opponent_allocation = (
                        last_submit_deal
                        if is_learner_turn
                        else {item: 3 - qty for item, qty in last_submit_deal.items()}
                    )
                    learner_points = {
                        participant_info[learner_id]["value2issue"][
                            level
                        ].lower(): points
                        for level, points in self.points.items()
                    }
                    opponent_points = {
                        participant_info[opponent_id]["value2issue"][
                            level
                        ].lower(): points
                        for level, points in self.points.items()
                    }
                    episode.learner_points = sum(
                        qty * learner_points.get(item.lower(), 0)
                        for item, qty in learner_allocation.items()
                    )
                    episode.opponent_points = sum(
                        qty * opponent_points.get(item.lower(), 0)
                        for item, qty in opponent_allocation.items()
                    )
                break

            if action and "[WALK_AWAY]" in action:
                episode.outcome = "walk_away"
                break

            recent_deals: list[str] = []
            for message in reversed(learner_messages):
                if message["role"] != "assistant":
                    continue
                recent_action = extract_tag(message["content"], "action")
                if recent_action and "[SUBMIT_DEAL]" in recent_action:
                    recent_deals.append(recent_action)
                if len(recent_deals) >= 3:
                    break
            if len(recent_deals) >= 3 and len(set(recent_deals)) == 1:
                episode.outcome = "reject_loop"
                break

        episode.learner_messages = learner_messages
        episode.opponent_messages = opponent_messages
        return episode

    def prepare_dataset(self) -> Dataset:
        """Generate self-play episodes, then slice them into GRPO prompt examples.

        Args:
            None.

        Returns:
            A dataset containing prompt prefixes and reward metadata.

        Raises:
            ValueError: If no training samples can be produced.
            Exception: If the scenario CSV cannot be read.
        """
        cfg = self.self_play_config
        scenarios: list[dict] = []

        try:
            with open(cfg.csv_path, encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    participant_info = ast.literal_eval(row["participant_info"])
                    scenarios.append(
                        {
                            "participant_info": participant_info,
                            "agent_ids": list(participant_info.keys()),
                        }
                    )
        except Exception:
            logger.exception(f"Failed to load self-play scenarios from {cfg.csv_path}")
            raise

        if cfg.num_episodes and len(scenarios) > cfg.num_episodes:
            scenarios = random.sample(scenarios, cfg.num_episodes)

        logger.info(f"Loaded {len(scenarios)} scenarios from {cfg.csv_path}")

        self.model.eval()
        episodes: list[Episode] = []
        persona_names = list(cfg.persona_weights.keys())
        persona_weights = [cfg.persona_weights[name] for name in persona_names]

        for scenario in tqdm(scenarios, desc="Self-play rollout", unit="episode"):
            persona = random.choices(persona_names, weights=persona_weights, k=1)[0]
            episodes.append(
                self._run_episode(
                    participant_info=scenario["participant_info"],
                    agent_ids=scenario["agent_ids"],
                    persona=persona,
                )
            )

        outcomes: dict[str, int] = {}
        for episode in episodes:
            outcomes[episode.outcome] = outcomes.get(episode.outcome, 0) + 1

        logger.info(
            f"Self-play complete: {len(episodes)} episodes - "
            + ", ".join(f"{name}: {count}" for name, count in sorted(outcomes.items()))
        )

        rows: list[dict] = []
        for episode in episodes:
            learner_value_to_issue = episode.participant_info[episode.learner_agent_id][
                "value2issue"
            ]
            point_map = {
                learner_value_to_issue[level].lower(): self.points[level]
                for level in ("High", "Medium", "Low")
            }
            split_at = max(1, int(len(episode.learner_turns) * cfg.prompt_split))

            for learner_turn_index in episode.learner_turns[split_at:]:
                prompt = episode.learner_messages[:learner_turn_index]
                if not prompt or prompt[-1]["role"] != "user":
                    continue

                last_opponent_offer = None
                for message in reversed(prompt):
                    if message["role"] != "user":
                        continue
                    last_opponent_offer = parse_submit_deal(message["content"])
                    if last_opponent_offer is not None:
                        break

                rows.append(
                    {
                        "prompt": prompt,
                        "system_prompt": episode.learner_messages[0]["content"],
                        "food_points": point_map.get("food", 3),
                        "water_points": point_map.get("water", 3),
                        "firewood_points": point_map.get("firewood", 3),
                        "max_points": 36,
                        "last_opponent_offer": (
                            json.dumps(last_opponent_offer)
                            if last_opponent_offer is not None
                            else "null"
                        ),
                        "episode_outcome": episode.outcome,
                        "episode_learner_points": (
                            episode.learner_points
                            if episode.learner_points is not None
                            else -1
                        ),
                        "opponent_persona": episode.persona,
                    }
                )

        if not rows:
            raise ValueError("No training samples produced from self-play episodes")

        logger.info(
            f"Built GRPO dataset: {len(rows)} per-turn samples from {len(episodes)} episodes"
        )
        return Dataset.from_list(rows)

    def build_trainer(self, train_dataset: Dataset) -> GRPOTrainer:
        cfg = self.self_play_config

        training_args = GRPOConfig(
            output_dir=cfg.output_dir,
            num_generations=cfg.num_generations,
            beta=cfg.beta,
            num_train_epochs=1,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            gradient_checkpointing=cfg.gradient_checkpointing,
            bf16=cfg.bf16,
            report_to=cfg.report_to,
            **cfg.extra_kwargs,
        )
        peft_config = self._build_peft_config(cfg.lora)

        os.makedirs(training_args.output_dir, exist_ok=True)

        return GRPOTrainer(
            model=self.model,
            args=training_args,
            reward_funcs=[
                length_reward,
                thought_judge_reward,
                format_reward,
                arithmetic_reward,
                outcome_reward,
            ],
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

    def train(self, resume_from: str | None = None) -> None:
        if self.model is None or self.tokenizer is None:
            self.load_model()

        cfg = self.self_play_config
        self._patch_chat_template()

        for iteration in range(cfg.num_online_iterations):
            logger.info(f"Online iteration {iteration + 1}/{cfg.num_online_iterations}")

            dataset = self.prepare_dataset()
            trainer = self.build_trainer(dataset)

            if iteration == 0 and resume_from:
                ckpt = True if resume_from == "latest" else resume_from
            else:
                ckpt = None

            trainer.train(resume_from_checkpoint=ckpt)
            trainer.save_model()

            self.model = trainer.model.merge_and_unload()
            logger.info(f"Merged LoRA adapters after iteration {iteration + 1}")

        merged_path = os.path.join(cfg.output_dir, "merged-final")
        os.makedirs(merged_path, exist_ok=True)
        self.model.save_pretrained(merged_path)
        self.tokenizer.save_pretrained(merged_path)
        logger.info(f"Training complete. Merged model saved to {merged_path}")

    @classmethod
    def run(
        cls,
        model_config: ModelConfig,
        config: SelfPlayConfig,
        resume_from: str | None = None,
    ) -> None:
        configure_judge(
            model=config.judge.model,
            base_url=config.judge.base_url,
            api_key_env=config.judge.api_key_env,
        )
        trainer = cls(model_config, config)
        trainer.train(resume_from=resume_from)
