from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .provider import Completion, InvalidModelOutput, OpenAICompatibleProvider
from .state import RunState


class StoryEngine:
    #: How many times a malformed-but-valid-JSON outline batch is corrected before giving up.
    _outline_shape_attempts = 3

    def __init__(
        self,
        state: RunState,
        run_dir: Path,
        provider: OpenAICompatibleProvider | None = None,
    ) -> None:
        self.state = state
        self.run_dir = run_dir
        self.provider = provider or OpenAICompatibleProvider()

    @property
    def state_path(self) -> Path:
        return self.run_dir / "run.json"

    def save(self) -> None:
        self.state.save(self.state_path)

    def generate_plan(self) -> dict[str, Any]:
        if self.state.plan_approved:
            raise ValueError("the plan is already approved")
        progress = self.state.plan_progress
        if progress is None:
            blueprint = self._generate_blueprint()
            progress = {"blueprint": blueprint, "episodes": []}
            self.state.plan_progress = progress
            self.state.log("plan_blueprint_saved")
            self.save()

        blueprint = progress.get("blueprint")
        self.validate_blueprint(blueprint, self.state.total_episodes)
        saved_episodes = progress.get("episodes")
        if not isinstance(saved_episodes, list):
            raise ValueError("saved plan progress is invalid; inspect run.json before continuing")
        outlines = {item.get("number"): item for item in saved_episodes if isinstance(item, dict)}
        batch_size = 5
        for start in range(1, self.state.total_episodes + 1, batch_size):
            end = min(start + batch_size - 1, self.state.total_episodes)
            missing = [number for number in range(start, end + 1) if number not in outlines]
            for attempt in range(1, 6):
                if not missing:
                    break
                try:
                    batch = self._generate_outline_batch(blueprint, missing)
                except InvalidModelOutput as error:
                    if len(missing) == 1:
                        raise
                    self.state.log(
                        "plan_batch_degraded_to_single_outlines",
                        requested=missing,
                        reason=str(error),
                    )
                    self.save()
                    batch = []
                    for number in missing:
                        batch.extend(self._generate_outline_batch(blueprint, [number]))
                outlines.update({item["number"]: item for item in batch})
                self.state.plan_progress["episodes"] = [outlines[number] for number in sorted(outlines)]
                missing = [number for number in range(start, end + 1) if number not in outlines]
                self.state.log(
                    "plan_outline_batch_saved",
                    start_episode=start,
                    end_episode=end,
                    returned_count=len(batch),
                    missing=missing,
                    attempt=attempt,
                )
                self.save()
            if missing:
                raise ValueError(
                    f"plan batch {start}-{end} is still missing episodes {missing} after 5 requests; "
                    "rerun plan to continue from the saved checkpoint"
                )

        plan = {**blueprint, "episodes": [outlines[number] for number in range(1, self.state.total_episodes + 1)]}
        self.validate_plan(plan, self.state.total_episodes)
        self.state.plan = plan
        self.state.plan_progress = None
        self.state.plan_approved = False
        self.state.log("plan_generated", episode_count=len(plan["episodes"]))
        self._write_plan()
        self.save()
        return plan

    def _generate_blueprint(self) -> dict[str, Any]:
        prompt = f"""Design the series-level blueprint for a {self.state.total_episodes}-episode serial.
Premise: {self.state.premise}

Return JSON only, with series_logline, promise, acts, characters, long_threads, and guardrails.
Create 8-10 acts with contiguous start_episode/end_episode ranges covering 1..{self.state.total_episodes}.
Each character needs a role, defining_facts, relationships, and beginning-to-end arc. Each long thread
needs id, question, introduced_episode, and planned_resolution_episode. Do NOT include episode outlines.
Make the escalation and character changes distinct across the run; keep this blueprint concise."""
        completion = self._call_json("plan_blueprint", [
            {"role": "system", "content": "You are a serial-story architect. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ], max_tokens=5000)
        self.validate_blueprint(completion.data, self.state.total_episodes)
        return completion.data

    def _generate_outline_batch(
        self, blueprint: dict[str, Any], requested_numbers: list[int]
    ) -> list[dict[str, Any]]:
        prompt = f"""Write one compact outline for each requested episode number: {requested_numbers}.
Premise: {self.state.premise}
Series blueprint: {json.dumps(blueprint, ensure_ascii=False)}

Return JSON with an episodes array containing exactly {len(requested_numbers)} objects. Each object must have
number, act, title, dramatic_beat, and ending_hook. Keep each field concise (one sentence maximum).
Use only these episode numbers, in this order: {requested_numbers}.
Make every beat distinct, advance the act/character arcs, and make each ending_hook a specific question
or imminent consequence. Respect planned long-thread introductions and resolutions."""
        messages = [
            {"role": "system", "content": "You create tightly structured serial episode outlines. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        last_error = ""
        for attempt in range(1, self._outline_shape_attempts + 1):
            completion = self._call_json("plan_outline_batch", messages, max_tokens=1800)
            try:
                episodes = self._extract_outline_episodes(completion.data)
                numbers = [item.get("number") for item in episodes]
                if not numbers:
                    raise ValueError("the episodes array was empty")
                if (
                    not all(isinstance(number, int) and not isinstance(number, bool) for number in numbers)
                    or len(set(numbers)) != len(numbers)
                    or any(number not in requested_numbers for number in numbers)
                ):
                    raise ValueError(
                        f"episode numbers {numbers} are invalid; expected a subset of {requested_numbers} with no repeats"
                    )
                if any(
                    not all(item.get(key) for key in ("title", "dramatic_beat", "ending_hook"))
                    for item in episodes
                ):
                    raise ValueError("every episode needs a non-empty title, dramatic_beat, and ending_hook")
            except ValueError as shape_error:
                last_error = str(shape_error)
                self.state.log(
                    "plan_outline_shape_invalid",
                    requested=requested_numbers,
                    attempt=attempt,
                    error=last_error,
                    returned_keys=sorted(str(key) for key in completion.data),
                )
                self.save()
                if attempt == self._outline_shape_attempts:
                    raise ValueError(
                        f"plan request for episodes {requested_numbers} did not return usable outlines: {last_error}"
                    ) from shape_error
                messages = messages + [
                    {"role": "assistant", "content": json.dumps(completion.data, ensure_ascii=False)[:4000]},
                    {
                        "role": "user",
                        "content": (
                            f"That response was unusable: {last_error}. Return the same {len(requested_numbers)} "
                            f"outlines again, wrapped in a single JSON object with an \"episodes\" array of exactly "
                            f"{len(requested_numbers)} objects using only the numbers {requested_numbers}. Each object "
                            "needs number, act, title, dramatic_beat, and ending_hook. JSON only, no commentary."
                        ),
                    },
                ]
                continue
            return sorted(episodes, key=lambda item: item["number"])
        raise ValueError(
            f"plan request for episodes {requested_numbers} did not return usable outlines: {last_error}"
        )

    @staticmethod
    def _extract_outline_episodes(data: Any) -> list[dict[str, Any]]:
        """Pull the episode outlines out of a model response, tolerating minor shape drift."""
        if not isinstance(data, dict):
            raise ValueError("the response is not a JSON object")
        for key in ("episodes", "episode_outlines", "outlines", "items", "results"):
            if key in data:
                candidate = data[key]
                break
        else:
            raise ValueError(
                f"the response has no episodes array; top-level keys were {sorted(str(key) for key in data)}"
            )
        if not isinstance(candidate, list) or not all(isinstance(item, dict) for item in candidate):
            raise ValueError("the episodes value is not an array of objects")
        normalized: list[dict[str, Any]] = []
        for item in candidate:
            episode = dict(item)
            number = episode.get("number", episode.get("episode_number", episode.get("episode")))
            if isinstance(number, str) and number.strip().isdigit():
                number = int(number.strip())
            episode["number"] = number
            if episode.get("act") is None and "act" not in episode:
                episode["act"] = episode.get("act_number")
            normalized.append(episode)
        return normalized

    def approve_plan(self) -> None:
        plan_path = self.run_dir / "plan.json"
        if not plan_path.exists():
            raise ValueError("no plan.json found; run the plan command first")
        with plan_path.open("r", encoding="utf-8") as file:
            plan = json.load(file)
        self.validate_plan(plan, self.state.total_episodes)
        self.state.plan = plan
        self.state.plan_approved = True
        self.state.log("plan_approved", source="plan.json")
        self.save()

    @staticmethod
    def validate_blueprint(plan: dict[str, Any], total_episodes: int) -> None:
        if not isinstance(plan, dict):
            raise ValueError("plan blueprint must be a JSON object")
        if not isinstance(plan.get("acts"), list) or not plan["acts"]:
            raise ValueError("plan must include the act structure")
        next_episode = 1
        for act in plan["acts"]:
            if not isinstance(act, dict):
                raise ValueError("each act must be a JSON object")
            start = act.get("start_episode")
            end = act.get("end_episode")
            if not isinstance(start, int) or not isinstance(end, int) or start != next_episode or end < start:
                raise ValueError("acts must cover episode ranges continuously from episode 1")
            next_episode = end + 1
        if next_episode != total_episodes + 1:
            raise ValueError("act ranges must cover the full serial")
        if not isinstance(plan.get("characters"), list):
            raise ValueError("plan must include the character arcs")
        if not isinstance(plan.get("long_threads"), list):
            raise ValueError("plan must include long_threads, even when empty")

    @staticmethod
    def validate_plan(plan: dict[str, Any], total_episodes: int) -> None:
        StoryEngine.validate_blueprint(plan, total_episodes)
        episodes = plan.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != total_episodes:
            raise ValueError(f"plan must contain exactly {total_episodes} episode outlines")
        if not all(isinstance(item, dict) for item in episodes):
            raise ValueError("each episode outline must be a JSON object")
        numbers = [item.get("number") for item in episodes]
        if not all(isinstance(number, int) for number in numbers) or sorted(numbers) != list(range(1, total_episodes + 1)):
            raise ValueError("plan episode numbers must cover 1..N exactly once")
        for item in episodes:
            if not all(item.get(key) for key in ("title", "dramatic_beat", "ending_hook")):
                raise ValueError("each episode needs a title, dramatic_beat, and ending_hook")

    def draft_episode(self, feedback: str = "") -> dict[str, Any]:
        if not self.state.plan_approved or not self.state.plan:
            raise ValueError("approve the plan before drafting episodes")
        number = self.state.next_episode_number()
        if number is None:
            raise ValueError("all episodes are already approved")
        existing = self.state.episode(number)
        prior_text = existing.get("text", "") if existing else ""
        context = self._episode_context(number)
        first = self._write_episode(number, context, feedback, prior_text)
        for revision_number in range(1, 5):
            word_count = self._word_count(first["text"])
            review = first.get("self_review", {})
            if not review.get("revision_needed") and 400 <= word_count <= 700:
                break
            reasons = "; ".join(review.get("notes", [])) or f"word count is {word_count}"
            if word_count > 700:
                correction = (
                    f"This is trim pass {revision_number} of 4. Rewrite the episode body to 450-500 words; "
                    f"the hard maximum is 700 words. The previous version had {word_count} words. "
                    "Cut secondary descriptions, repeated explanations, and extra scenes first. Preserve the "
                    "planned beat, canon, and one distinct ending hook. Stop writing immediately after the hook; "
                    "do not append an episode recap or meta commentary."
                )
            elif word_count < 400:
                correction = (
                    f"This is revision pass {revision_number} of 4. Expand the existing episode to 450-500 words "
                    f"and keep it within 400-700 words. The previous version had {word_count} words. "
                    "Add only story-relevant action or dialogue; preserve canon and the ending hook."
                )
            else:
                correction = f"This is revision pass {revision_number} of 4. Fix the self-review issues: {reasons}."
            revised = self._write_episode(
                number,
                context,
                f"{correction}\nAdditional editor feedback: {feedback}",
                first["text"],
            )
            first = revised
        word_count = self._word_count(first["text"])
        if not 400 <= word_count <= 700:
            self.state.log("draft_rejected", number=number, word_count=word_count)
            self.save()
            raise ValueError(f"draft has {word_count} words after four revisions; required range is 400-700")
        draft = {
            "number": number,
            "title": self._outline(number).get("title", f"Episode {number}"),
            "text": first["text"].strip(),
            "word_count": word_count,
            "continuity": first.get("continuity", {}),
            "self_review": first.get("self_review", {}),
            "status": "draft",
        }
        self.state.save_draft(draft)
        self._write_episode_file(draft, "draft")
        self.save()
        return draft

    def approve_episode(self, number: int, feedback: str = "", edited_text: str | None = None) -> dict[str, Any]:
        episode = self.state.episode(number)
        if not episode or episode.get("status") != "draft":
            raise ValueError(f"episode {number} has no pending draft")
        if edited_text is not None and edited_text.strip() != episode["text"].strip():
            if not 400 <= self._word_count(edited_text) <= 700:
                raise ValueError("edited episode must contain 400-700 words")
            continuity = self.reconcile_episode(number, edited_text)
            episode["text"] = edited_text.strip()
            episode["continuity"] = continuity
            episode["word_count"] = self._word_count(edited_text)
        approved = self.state.approve_draft(number, feedback)
        self._write_episode_file(approved, "approved")
        self.save()
        return approved

    def rewrite_episode(
        self, number: int, edited_text: str, feedback: str = ""
    ) -> None:
        if not 400 <= self._word_count(edited_text) <= 700:
            raise ValueError("edited episode must contain 400-700 words")
        continuity = self.reconcile_episode(number, edited_text)
        self.state.replace_approved_episode(number, edited_text.strip(), continuity, feedback)
        self._write_episode_file(self.state.episode(number), "approved")
        for episode in self.state.episodes:
            if episode.get("status") == "stale":
                self._write_episode_file(episode, "stale")
        self.save()

    def reconcile_episode(self, number: int, text: str) -> dict[str, Any]:
        completion = self._call_json("continuity_reconciliation", [
            {"role": "system", "content": "Extract only facts supported by the supplied story text. Return valid JSON."},
            {"role": "user", "content": f"""For episode {number}, return JSON with:
summary (concise cumulative story-so-far summary), characters (object mapping names to current state),
facts (new durable facts), open_threads (newly opened questions), resolved_threads (exact thread labels closed).
retracted_facts (exact prior canon statements this edited text explicitly disproves).
Existing canon for reference: {json.dumps(self.state.canon, ensure_ascii=False)}
Episode text:\n{text}"""},
        ], max_tokens=1600, episode_number=number)
        return completion.data

    def _write_episode(
        self, number: int, context: dict[str, Any], feedback: str, previous_draft: str
    ) -> dict[str, Any]:
        prompt = f"""Write episode {number} of {self.state.total_episodes} in about 500 words (400-700 inclusive).
    The episode_text value itself must contain no more than 700 words; do not include the JSON fields in this count.
Use the attached outline as a binding beat, not a checklist. End on a concrete, earned hook.
Maintain established facts, relationships, timeline, and open threads. Do not recap at length.

Return JSON with episode_text, continuity (summary, characters object, facts array, retracted_facts array,
open_threads array, resolved_threads array), and self_review (revision_needed boolean, notes array).
The summary should update the rolling summary across all approved episodes so far. Keep canon deltas
specific and only record facts established by this episode. Do not invent resolutions to planned threads.

STORY STATE:\n{json.dumps(context, ensure_ascii=False)}
EDITOR FEEDBACK:\n{feedback or "None"}
PREVIOUS DRAFT TO REVISE (if any):\n{previous_draft or "None"}"""
        completion = self._call_json("episode_draft", [
            {"role": "system", "content": "Write vivid, specific serial fiction. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ], max_tokens=2400, episode_number=number)
        data = completion.data
        text = data.get("episode_text", data.get("text", ""))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("model returned an empty episode")
        data["text"] = text
        return data

    def _call_json(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        episode_number: int | None = None,
    ) -> Completion:
        episode_spend = sum(
            float(event.get("cost_usd", 0))
            for event in self.state.events
            if event.get("episode") == episode_number and event.get("event") == "model_call_completed"
        ) if episode_number is not None else 0.0
        estimate_input = max(1, sum(len(item["content"]) for item in messages) // 4)
        estimated_cost = (
            estimate_input * self.provider.input_price + max_tokens * self.provider.output_price
        ) / 1_000_000
        if self.state.usage["cost_usd"] + estimated_cost > self.state.budget_usd:
            raise ValueError("run cost budget would be exceeded; no model call made")
        if episode_number is not None and episode_spend + estimated_cost > self.state.episode_cost_cap_usd:
            raise ValueError("episode cost cap would be exceeded; no model call made")
        last_error: Exception | None = None
        for attempt in (1, 2):
            self.state.log("model_call_started", stage=stage, episode=episode_number, attempt=attempt)
            self.save()
            try:
                completion = self.provider.complete_json(messages, max_tokens)
                if completion.fallback_used:
                    self.state.log("json_mode_fallback_used", stage=stage, episode=episode_number)
                self.state.add_usage(completion.input_tokens, completion.output_tokens, completion.cost_usd)
                self.state.log(
                    "model_call_completed",
                    stage=stage,
                    episode=episode_number,
                    attempt=attempt,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    latency_seconds=round(completion.latency_seconds, 3),
                    cost_usd=completion.cost_usd,
                )
                self.save()
                run_over_budget = self.state.usage["cost_usd"] > self.state.budget_usd
                episode_over_budget = (
                    episode_number is not None
                    and episode_spend + completion.cost_usd > self.state.episode_cost_cap_usd
                )
                if run_over_budget or episode_over_budget:
                    self.state.log(
                        "budget_limit_reached",
                        stage=stage,
                        episode=episode_number,
                        run_over_budget=run_over_budget,
                        episode_over_budget=episode_over_budget,
                    )
                    self.save()
                    raise ValueError("model response exceeded a cost limit and was not accepted")
                return completion
            except Exception as error:
                last_error = error
                if isinstance(error, InvalidModelOutput):
                    self.state.add_usage(
                        error.input_tokens,
                        error.output_tokens,
                        error.cost_usd,
                    )
                    self.state.log(
                        "model_output_invalid",
                        stage=stage,
                        episode=episode_number,
                        input_tokens=error.input_tokens,
                        output_tokens=error.output_tokens,
                        latency_seconds=round(error.latency_seconds, 3),
                        cost_usd=error.cost_usd,
                    )
                self.state.log("model_call_failed", stage=stage, episode=episode_number, attempt=attempt, error=str(error))
                self.save()
                if attempt == 2 or not self._is_transient(error):
                    raise
                self.state.log("model_retry", stage=stage, episode=episode_number, next_attempt=2)
                self.save()
        raise RuntimeError("model call failed") from last_error

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        message = str(error).lower()
        return any(value in message for value in ("http 429", "http 500", "http 502", "http 503", "http 504", "timed out", "connection failed"))

    def _episode_context(self, number: int) -> dict[str, Any]:
        approved = sorted(
            (episode for episode in self.state.episodes
             if episode.get("status") == "approved" and episode["number"] < number),
            key=lambda episode: episode["number"],
        )
        outline = self._outline(number)
        act = next((item for item in self.state.plan["acts"]
                    if item.get("start_episode", 0) <= number <= item.get("end_episode", 0)), None)
        next_milestones = [
            {key: item.get(key) for key in ("number", "title", "dramatic_beat", "ending_hook")}
            for item in self.state.plan["episodes"][number:number + 5]
        ]
        return {
            "premise": self.state.premise,
            "episode_number": number,
            "episode_outline": outline,
            "current_act": act,
            "next_milestones": next_milestones,
            "arc_logline": self.state.plan.get("series_logline"),
            "rolling_summary": self.state.canon.get("rolling_summary", ""),
            "canon_characters": self.state.canon.get("characters", {}),
            "canon_facts": self.state.canon.get("facts", []),
            "open_threads": self.state.canon.get("open_threads", []),
            "long_threads": self.state.plan.get("long_threads", []),
            "recent_episode_summaries": [
                {"number": item["number"], "summary": item.get("continuity", {}).get("summary", "")}
                for item in approved[-5:]
            ],
            "editor_guidance": self.state.editor_guidance,
            "guardrails": self.state.plan.get("guardrails", []),
        }

    def _outline(self, number: int) -> dict[str, Any]:
        return self.state.plan["episodes"][number - 1]

    def _write_plan(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "plan.json"
        path.write_text(json.dumps(self.state.plan, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_episode_file(self, episode: dict[str, Any], status: str) -> None:
        directory = self.run_dir / "episodes"
        directory.mkdir(parents=True, exist_ok=True)
        number = episode["number"]
        path = directory / f"episode-{number:03d}-{status}.md"
        path.write_text(f"# Episode {number}: {episode.get('title', '')}\n\n{episode['text'].strip()}\n", encoding="utf-8")

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))