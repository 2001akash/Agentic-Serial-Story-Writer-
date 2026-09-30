from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .engine import StoryEngine
from .provider import OpenAICompatibleProvider
from .state import RunState


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="serial-writer", description="Plan and write a resumable serial story.")
    commands = root.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a new story run")
    create.add_argument("--id", required=True, help="short run identifier")
    create.add_argument("--premise", required=True, help="one-line story premise")
    create.add_argument("--runs-dir", type=Path, default=Path(".story-runs"))
    create.add_argument("--episodes", type=int, default=200, choices=range(1, 201))
    create.add_argument("--budget-usd", type=float, default=3.0)
    create.add_argument("--episode-cap-usd", type=float, default=0.15)

    for name, help_text in (
        ("plan", "generate the complete arc plan for human review"),
        ("approve-plan", "approve the edited plan.json"),
        ("status", "show progress and cost"),
        ("resume", "reopen a run at its next unapproved episode"),
        ("draft", "write or revise the next episode draft"),
        ("revise", "revise the pending episode using new feedback"),
        ("approve", "approve the pending draft, optionally with edits and guidance"),
        ("events", "show trace and retry events"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("run_id")
        command.add_argument("--runs-dir", type=Path, default=Path(".story-runs"))
        if name in ("draft", "revise"):
            command.add_argument("--feedback", default="")
        if name == "plan":
            command.add_argument(
                "--auto-approve",
                action="store_true",
                help="approve the plan after all episode outlines pass validation",
            )
        if name == "approve":
            command.add_argument("--feedback", default="", help="guidance for future episodes")
            command.add_argument("--text-file", type=Path, help="approve this edited episode text")
        if name == "events":
            command.add_argument("--tail", type=int, default=30)

    rewrite = commands.add_parser("rewrite", help="edit an approved episode and stale later episodes")
    rewrite.add_argument("run_id")
    rewrite.add_argument("--episode", type=int, required=True)
    rewrite.add_argument("--text-file", type=Path, required=True)
    rewrite.add_argument("--feedback", default="", help="carry this correction into future episodes")
    rewrite.add_argument("--runs-dir", type=Path, default=Path(".story-runs"))

    estimate = commands.add_parser("estimate", help="estimate model cost and generation time")
    estimate.add_argument("--episodes", type=int, default=200)
    estimate.add_argument("--revision-rate", type=float, default=0.25)
    estimate.add_argument("--seconds-per-call", type=float, default=18.0)
    return root


def _run_path(runs_dir: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run id must be a simple folder name")
    path = runs_dir / run_id
    if not (path / "run.json").exists():
        raise FileNotFoundError(f"run not found: {path}")
    return path


def _engine(runs_dir: Path, run_id: str) -> StoryEngine:
    path = _run_path(runs_dir, run_id)
    return StoryEngine(RunState.load(path / "run.json"), path, OpenAICompatibleProvider())


def _print_status(engine: StoryEngine) -> None:
    state = engine.state
    approved = sum(item.get("status") == "approved" for item in state.episodes)
    draft = next((item for item in state.episodes if item.get("status") == "draft"), None)
    stale = sum(item.get("status") == "stale" for item in state.episodes)
    print(f"Run: {state.run_id}")
    print(f"Plan: {'approved' if state.plan_approved else 'needs review'}")
    print(f"Episodes: {approved}/{state.total_episodes} approved; {stale} stale")
    print(f"Next: {state.next_episode_number() or 'complete'}")
    if draft:
        print(f"Pending draft: episode {draft['number']} ({draft.get('word_count', 0)} words)")
    print(f"Cost: ${state.usage['cost_usd']:.4f} / ${state.budget_usd:.2f}")
    print(f"Tokens: {int(state.usage['input_tokens'])} input, {int(state.usage['output_tokens'])} output")


def _estimate(episodes: int, revision_rate: float, seconds_per_call: float) -> None:
    if episodes < 1 or not 0 <= revision_rate <= 1 or seconds_per_call <= 0:
        raise ValueError("episodes and seconds must be positive; revision rate must be between 0 and 1")
    provider = OpenAICompatibleProvider()
    input_tokens = 4500 * episodes * (1 + revision_rate) + 3000
    output_tokens = 1250 * episodes * (1 + revision_rate) + 9000
    cost = (input_tokens * provider.input_price + output_tokens * provider.output_price) / 1_000_000
    planning_calls = 1 + (episodes + 4) // 5
    calls = planning_calls + episodes * (1 + revision_rate)
    seconds = calls * seconds_per_call
    print(f"Estimated model cost: ${cost:.2f} at {provider.model} pricing")
    print(f"Estimated generation time: {seconds / 3600:.1f} hours at {seconds_per_call:g}s per call")
    print(
        f"Assumptions: {input_tokens:,.0f} input tokens, {output_tokens:,.0f} output tokens, "
        f"{planning_calls} planning calls, "
        f"{revision_rate:.0%} episodes need one revision; excludes human review and edited-episode reconciliation."
    )
    print(
        "Actual usage is recorded per call; configure STORY_INPUT_USD_PER_MILLION and "
        "STORY_OUTPUT_USD_PER_MILLION for your provider/model rates."
    )


def _read_episode_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("# Episode "):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "create":
            if args.budget_usd <= 0 or args.episode_cap_usd <= 0:
                raise ValueError("cost limits must be positive")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.id):
                raise ValueError("id may contain letters, numbers, underscores, and hyphens")
            run_dir = args.runs_dir / args.id
            run_dir.mkdir(parents=True, exist_ok=False)
            state = RunState.create(args.id, args.premise, args.episodes, args.budget_usd, args.episode_cap_usd)
            state.log("run_created", total_episodes=args.episodes, budget_usd=args.budget_usd)
            state.save(run_dir / "run.json")
            print(f"Created {run_dir}. Next: serial-writer plan {args.id}")
            return 0

        if args.command == "estimate":
            _estimate(args.episodes, args.revision_rate, args.seconds_per_call)
            return 0

        engine = _engine(args.runs_dir, args.run_id)
        if args.command == "plan":
            plan = engine.generate_plan()
            print(f"Generated {len(plan['episodes'])} outlines: {engine.run_dir / 'plan.json'}")
            if args.auto_approve:
                engine.approve_plan()
                print("Plan validated and approved. Next: serial-writer draft " + args.run_id)
            else:
                print("Review/edit plan.json, then run: serial-writer approve-plan " + args.run_id)
        elif args.command == "approve-plan":
            engine.approve_plan()
            print("Plan approved. Next: serial-writer draft " + args.run_id)
        elif args.command in ("status", "resume"):
            _print_status(engine)
            if args.command == "resume" and engine.state.plan_approved:
                print(f"Resume with: serial-writer draft {args.run_id}")
        elif args.command in ("draft", "revise"):
            draft = engine.draft_episode(args.feedback)
            print(f"Drafted episode {draft['number']}: {draft['word_count']} words")
            review_path = engine.run_dir / "episodes" / f"episode-{draft['number']:03d}-draft.md"
            print(f"Review: {review_path}")
            print("Approve with optional feedback: serial-writer approve " + args.run_id + " --feedback \"...\"")
        elif args.command == "approve":
            number = engine.state.next_episode_number()
            if number is None:
                raise ValueError("the run is already complete")
            edited_text = _read_episode_text(args.text_file) if args.text_file else None
            episode = engine.approve_episode(number, args.feedback, edited_text)
            print(f"Approved episode {number}. Next episode: {engine.state.next_episode_number() or 'complete'}")
            if args.feedback:
                print("Feedback was added to persistent guidance for later episodes.")
        elif args.command == "rewrite":
            text = _read_episode_text(args.text_file)
            engine.rewrite_episode(args.episode, text, args.feedback)
            print(f"Rewrote episode {args.episode}; later approved episodes are stale and will be regenerated.")
        elif args.command == "events":
            events = engine.state.events[-args.tail:] if args.tail > 0 else []
            for event in events:
                print(json.dumps(event, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())