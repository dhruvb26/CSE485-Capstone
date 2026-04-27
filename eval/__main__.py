"""Unified CLI for the eval module.

Usage:
    python -m eval tasks  [--config ...] [--evaluate-only] [--list-tasks]
    python -m eval negotiate [--config ...] [--evaluate-only LOG_DIR]
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation framework for negotiation LLMs",
    )
    sub = parser.add_subparsers(dest="command")

    # --- tasks subcommand (SysEval benchmark tasks) ---
    tasks_p = sub.add_parser("tasks", help="Run SysEval benchmark tasks")
    tasks_p.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "tasks.yaml"),
        help="Path to YAML config file",
    )
    tasks_p.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only score existing logs (skip model inference)",
    )
    tasks_p.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available evaluation tasks and exit",
    )

    # --- negotiate subcommand (self-play dialogue evaluation) ---
    neg_p = sub.add_parser("negotiate", help="Run self-play negotiation evaluation")
    neg_p.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "negotiate.yaml"),
        help="Path to negotiate YAML config file",
    )
    neg_p.add_argument(
        "--evaluate-only",
        type=str,
        default=None,
        metavar="LOG_DIR",
        help="Score existing negotiate logs in LOG_DIR instead of running episodes",
    )

    args = parser.parse_args()

    if args.command == "tasks":
        from eval.main import (
            list_available_tasks,
            load_config,
            run_evaluation,
            score_logs,
        )

        if args.list_tasks:
            list_available_tasks()
            return

        config = load_config(args.config)

        if args.evaluate_only or config.get("evaluate_only", False):
            storage_dir = config.get("storage_dir", "./logs/eval")
            score_logs(storage_dir)
        else:
            run_evaluation(config)

    elif args.command == "negotiate":
        from eval.negotiate import main as negotiate_main

        negotiate_main(args.config, args.evaluate_only)

    else:
        parser.print_help()


main()
