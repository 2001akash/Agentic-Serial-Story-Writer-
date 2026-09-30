from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunState:
    run_id: str
    premise: str
    total_episodes: int = 200
    budget_usd: float = 3.0
    episode_cost_cap_usd: float = 0.15
    plan: dict[str, Any] | None = None
    plan_progress: dict[str, Any] | None = None
    plan_approved: bool = False
    episodes: list[dict[str, Any]] = field(default_factory=list)
    canon: dict[str, Any] = field(default_factory=lambda: {
        "characters": {},
        "facts": [],
        "open_threads": [],
        "rolling_summary": "",
    })
    editor_guidance: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    })
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        run_id: str,
        premise: str,
        total_episodes: int = 200,
        budget_usd: float = 3.0,
        episode_cost_cap_usd: float = 0.15,
    ) -> RunState:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        if not premise.strip():
            raise ValueError("premise must not be empty")
        if not 1 <= total_episodes <= 200:
            raise ValueError("total_episodes must be between 1 and 200")
        if budget_usd <= 0 or episode_cost_cap_usd <= 0:
            raise ValueError("cost limits must be positive")
        return cls(
            run_id=run_id,
            premise=premise.strip(),
            total_episodes=total_episodes,
            budget_usd=budget_usd,
            episode_cost_cap_usd=episode_cost_cap_usd,
        )

    @classmethod
    def load(cls, path: Path) -> RunState:
        with path.open("r", encoding="utf-8") as file:
            return cls(**json.load(file))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self), ensure_ascii=False, indent=2)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def log(self, name: str, **details: Any) -> None:
        self.events.append({"at": utc_now(), "event": name, **details})

    def next_episode_number(self) -> int | None:
        approved = {
            episode["number"]
            for episode in self.episodes
            if episode.get("status") == "approved"
        }
        return next(
            (number for number in range(1, self.total_episodes + 1) if number not in approved),
            None,
        )

    def episode(self, number: int) -> dict[str, Any] | None:
        return next(
            (episode for episode in self.episodes if episode.get("number") == number),
            None,
        )

    def save_draft(self, episode: dict[str, Any]) -> None:
        number = episode["number"]
        if number != self.next_episode_number():
            raise ValueError("draft must target the first unapproved episode")
        existing = self.episode(number)
        if existing:
            episode["revision"] = existing.get("revision", 0) + 1
            self.episodes[self.episodes.index(existing)] = episode
        else:
            episode["revision"] = 1
            self.episodes.append(episode)
        episode["status"] = "draft"
        self.log("episode_drafted", number=number, revision=episode["revision"])

    def approve_draft(self, number: int, feedback: str = "") -> dict[str, Any]:
        episode = self.episode(number)
        if not episode or episode.get("status") != "draft":
            raise ValueError(f"episode {number} has no pending draft")
        if number != self.next_episode_number():
            raise ValueError("episodes must be approved in order")
        episode["status"] = "approved"
        episode["approved_at"] = utc_now()
        self._apply_continuity(episode.get("continuity", {}))
        if feedback.strip():
            self.editor_guidance.append({"episode": number, "text": feedback.strip()})
        self.log("episode_approved", number=number, feedback=feedback.strip())
        return episode

    def replace_approved_episode(
        self, number: int, text: str, continuity: dict[str, Any], feedback: str = ""
    ) -> None:
        episode = self.episode(number)
        if not episode or episode.get("status") != "approved":
            raise ValueError(f"episode {number} is not approved")
        episode.setdefault("revisions", []).append({
            "text": episode["text"],
            "continuity": episode.get("continuity", {}),
            "replaced_at": utc_now(),
        })
        episode.update({"text": text, "continuity": continuity, "edited_at": utc_now()})
        for later in self.episodes:
            if later["number"] > number and later.get("status") in {"approved", "draft"}:
                later["status"] = "stale"
        if feedback.strip():
            self.editor_guidance.append({"episode": number, "text": feedback.strip()})
        self._rebuild_canon(number)
        self.log("episode_rewritten", number=number, invalidated_after=number)

    def add_usage(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens
        self.usage["cost_usd"] += cost_usd

    def _apply_continuity(self, delta: dict[str, Any]) -> None:
        characters = self.canon.setdefault("characters", {})
        characters.update(delta.get("characters", {}))
        facts = self.canon.setdefault("facts", [])
        for fact in delta.get("retracted_facts", []):
            facts[:] = [item for item in facts if item != fact]
        for fact in delta.get("facts", []):
            if fact not in facts:
                facts.append(fact)
        threads = self.canon.setdefault("open_threads", [])
        for thread in delta.get("resolved_threads", []):
            threads[:] = [item for item in threads if item != thread]
        for thread in delta.get("open_threads", []):
            if thread not in threads:
                threads.append(thread)
        if delta.get("summary"):
            self.canon["rolling_summary"] = delta["summary"]

    def _rebuild_canon(self, through: int) -> None:
        self.canon = {
            "characters": {},
            "facts": [],
            "open_threads": [],
            "rolling_summary": "",
        }
        for episode in sorted(self.episodes, key=lambda item: item["number"]):
            if episode["number"] <= through and episode.get("status") == "approved":
                self._apply_continuity(episode.get("continuity", {}))