"""Evaluation orchestrator that runs SysEval CaSiNo tasks against a GRPO model."""

from __future__ import annotations

import logging
import os
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from rl.config import EvalConfig

log = logging.getLogger(__name__)


def _build_syseval_args(config: EvalConfig) -> Namespace:
    """Build a mock argparse.Namespace that SysEval handlers expect."""
    return Namespace(
        storage_dir=config.storage_dir,
        dataset_name="casino",
        model_name="grpo_model",
        task_name=config.tasks,
        num_instances=config.num_instances,
        max_num_instances=config.num_instances,
        use_cot=False,
        num_multishot=0,
        num_prior_utts=0,
        num_utts_partial_dial=-1,
        hf_model_str="grpo_model",
        openai_model_str="",
    )


def run_evaluation(config: EvalConfig):
    """Run SysEval CaSiNo evaluation with the GRPO-trained model.

    1. Creates a timestamped run directory under ``storage_dir``
    2. Adds SysEval repo to ``sys.path``
    3. Loads the CaSiNo dataset handler (from SysEval)
    4. Loads the GRPO model via our custom handler
    5. Runs each requested task
    6. Saves raw completions to ``completions.jsonl``
    7. Computes metrics on the saved log files
    """
    syseval_path = os.path.abspath(config.syseval_path)
    if not os.path.isdir(syseval_path):
        raise FileNotFoundError(
            f"SysEval-NegoLLMs repo not found at {syseval_path}"
        )

    if syseval_path not in sys.path:
        sys.path.insert(0, syseval_path)

    original_cwd = os.getcwd()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(original_cwd, config.storage_dir, f"eval_{stamp}")
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    log.warning("Eval run directory: %s", run_dir)

    completions_path = os.path.join(run_dir, "completions.jsonl")

    os.chdir(syseval_path)

    try:
        from nego_datasets.casino import CasinoHandler
        import utils as syseval_utils

        args = _build_syseval_args(config)
        args.storage_dir = run_dir

        log.warning("Loading CaSiNo test dataset from SysEval")
        dataset_handler = CasinoHandler("casino", args)

        log.warning("Loading GRPO model for evaluation")
        from rl.eval.model_handler import GRPOModelHandler

        model_handler = GRPOModelHandler(
            "grpo_model", args, config,
            completions_path=completions_path,
        )

        task_names = [t.strip() for t in config.tasks.split(",") if t.strip()]
        log.warning("Running %d evaluation tasks: %s", len(task_names), task_names)

        for i, task_name in enumerate(task_names):
            log.warning("[%d/%d] Evaluating task: %s", i + 1, len(task_names), task_name)
            model_handler.set_current_task(task_name)
            task_handler = syseval_utils.get_task_handler(task_name, args)
            task_handler.evaluate(dataset_handler, model_handler)

        model_handler.save_completions()

        log.warning("All tasks complete. Logs saved to %s", run_dir)

        _compute_metrics(run_dir)

    finally:
        os.chdir(original_cwd)


def _compute_metrics(storage_dir: str):
    """Run SysEval's evaluate_logs to compute per-task metrics."""
    try:
        from scripts.evaluate_logs import run as run_eval_logs

        log_dir = Path(storage_dir) / "logs"
        log.warning("Computing metrics from %s", log_dir)
        run_eval_logs(log_dir=log_dir, verbose=True, output_path=None)
    except Exception:
        log.exception("Failed to compute metrics — logs are still saved")
