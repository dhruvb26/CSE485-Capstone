from __future__ import annotations

import trackio
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TrackioLoggingCallback(TrainerCallback):
    """Bridges TRL's training metrics to an existing trackio run.

    Use with ``report_to="none"`` so TRL doesn't create its own trackio
    session. Call ``trackio.init(...)`` before training starts, then add
    this callback to the trainer to forward every ``on_log`` payload.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs:
            trackio.log(logs, step=state.global_step)
