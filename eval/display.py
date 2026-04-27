from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _fmt(val, fmt: str = ".2f", suffix: str = "") -> str:
    if val is None:
        return "\u2014"
    if "%" in fmt:
        return f"{val:{fmt}}"
    return f"{val:{fmt}}{suffix}"


def print_matchup_report(summary: dict) -> None:
    o = summary["overall"]

    lines = Text()
    lines.append(f"Episodes:          {o['total_episodes']}\n")
    lines.append(
        f"Deal rate:         {_fmt(o['deal_rate'], '.1%')}  ({o['deal_count']})\n"
    )
    lines.append(
        f"Walk-away rate:    {_fmt(o['walk_away_rate'], '.1%')}  ({o['walk_away_count']})\n"
    )
    lines.append(
        f"Reject-loop rate:  {_fmt(o['reject_loop_rate'], '.1%')}  ({o['reject_loop_count']})\n"
    )
    lines.append(
        f"Max-turns rate:    {_fmt(o['max_turns_rate'], '.1%')}  ({o['max_turns_count']})\n"
    )
    lines.append("\n")
    lines.append(
        f"Avg learner pts:   {_fmt(o['avg_learner_points'])}  (std {_fmt(o['std_learner_points'])})\n"
    )
    lines.append(f"Avg opponent pts:  {_fmt(o['avg_opponent_points'])}\n")
    lines.append(f"Avg joint score:   {_fmt(o['avg_joint_score'])}\n")
    lines.append(f"Avg score ratio:   {_fmt(o['avg_score_ratio'], '.3f')}\n")
    lines.append(f"Avg turns (deal):  {_fmt(o['avg_turns_to_deal'])}\n")
    lines.append(f"Avg turns (all):   {_fmt(o['avg_turns_all'])}\n")
    lines.append(f"Points/turn:       {_fmt(o['points_per_turn'], '.3f')}\n")
    lines.append("\n")
    lines.append(
        f"Learner format:    {_fmt(o['learner_format_rate'], '.1%')}  ({o['learner_total_turns']} turns)\n"
    )
    lines.append(
        f"Learner bad deals: {_fmt(o['learner_malformed_deal_rate'], '.1%')}\n"
    )
    lines.append(
        f"Opponent format:   {_fmt(o['opponent_format_rate'], '.1%')}  ({o['opponent_total_turns']} turns)\n"
    )
    lines.append(
        f"Opponent bad deals:{_fmt(o['opponent_malformed_deal_rate'], '.1%')}\n"
    )

    title = summary["matchup"]
    if summary.get("dataset"):
        title = f"{title}  [{summary['dataset']}]"
    panel = Panel(lines, title=title, border_style="blue", padding=(1, 2))
    console.print(panel)

    if summary.get("per_persona"):
        table = Table(title="Per-Persona Breakdown", show_lines=False)
        table.add_column("Persona", style="cyan")
        table.add_column("Deal%", justify="right")
        table.add_column("Learner Pts", justify="right")
        table.add_column("Opp Pts", justify="right")
        table.add_column("Ratio", justify="right")
        table.add_column("Joint", justify="right")
        table.add_column("Turns", justify="right")
        table.add_column("Format%", justify="right")
        table.add_column("N", justify="right")

        for persona, pm in summary["per_persona"].items():
            table.add_row(
                persona,
                _fmt(pm["deal_rate"], ".0%"),
                _fmt(pm["avg_learner_points"]),
                _fmt(pm["avg_opponent_points"]),
                _fmt(pm["avg_score_ratio"], ".3f"),
                _fmt(pm["avg_joint_score"], ".1f"),
                _fmt(pm["avg_turns_to_deal"]),
                _fmt(pm["learner_format_rate"], ".0%"),
                str(pm["total_episodes"]),
            )
        console.print(table)
    console.print()


def print_comparison_table(all_summaries: dict) -> None:
    if len(all_summaries) < 2:
        return

    table = Table(title="A/B Comparison", show_lines=True, border_style="green")
    table.add_column("Matchup", style="bold")
    table.add_column("Dataset", style="dim")
    table.add_column("Deal%", justify="right")
    table.add_column("Learner Pts", justify="right")
    table.add_column("Opp Pts", justify="right")
    table.add_column("Joint", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Pts/Turn", justify="right")
    table.add_column("Format%", justify="right")

    for name, s in all_summaries.items():
        o = s["overall"]
        table.add_row(
            name,
            s.get("dataset") or "—",
            _fmt(o["deal_rate"], ".1%"),
            _fmt(o["avg_learner_points"]),
            _fmt(o["avg_opponent_points"]),
            _fmt(o["avg_joint_score"]),
            _fmt(o["avg_score_ratio"], ".3f"),
            _fmt(o["avg_turns_to_deal"]),
            _fmt(o["points_per_turn"], ".3f"),
            _fmt(o["learner_format_rate"], ".1%"),
        )

    console.print(table)
    console.print()


def print_task_scores(results: list[dict]) -> None:
    if not results:
        console.print("[yellow]No evaluable logs found.[/yellow]")
        return

    by_model: dict[tuple, list[dict]] = {}
    for r in results:
        key = (r["dataset"], r["model"])
        by_model.setdefault(key, []).append(r)

    for (dataset, model), rows in sorted(by_model.items()):
        table = Table(
            title=f"{dataset} | {model}",
            show_lines=False,
            border_style="blue",
        )
        table.add_column("Task", style="cyan")
        table.add_column("Metric")
        table.add_column("Score", justify="right", style="bold")
        table.add_column("N", justify="right", style="dim")

        for r in sorted(rows, key=lambda x: x["task"]):
            score_str = (
                f"{r['score']:.4f}"
                if isinstance(r["score"], (int, float))
                else str(r["score"])
            )
            table.add_row(r["task"], r["metric"], score_str, str(r["n"]))

        console.print(table)
        console.print()
