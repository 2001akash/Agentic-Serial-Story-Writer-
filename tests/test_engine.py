import tempfile
import unittest
from pathlib import Path

from serial_writer.engine import StoryEngine
from serial_writer.provider import Completion, InvalidModelOutput
from serial_writer.state import RunState


def outline(number):
    return {
        "number": number,
        "act": 1,
        "title": f"Turn {number}",
        "dramatic_beat": f"The rider discovers clue {number}.",
        "ending_hook": f"Who left clue {number}?",
    }


def episode_result(number):
    return {
        "episode_text": " ".join(f"word{index}" for index in range(420)),
        "continuity": {
            "summary": f"The rider reaches stop {number}.",
            "characters": {"Nia": f"At stop {number}."},
            "facts": [f"Stop {number} is part of the route."],
            "open_threads": [f"Who marked stop {number}?"],
            "resolved_threads": [],
        },
        "self_review": {"revision_needed": False, "notes": []},
    }


def episode_result_with_words(number, count):
    result = episode_result(number)
    result["episode_text"] = " ".join(f"word{index}" for index in range(count))
    return result


class FakeProvider:
    input_price = 0.15
    output_price = 0.60

    def __init__(self):
        self.responses = [
            {
                "series_logline": "A rider follows a route of impossible addresses.",
                "acts": [{"start_episode": 1, "end_episode": 2, "title": "The route", "summary": "The mystery opens."}],
                "characters": [{"name": "Nia", "role": "rider", "arc": "From wary to brave."}],
                "long_threads": [{"id": "route", "question": "Who changed the route?"}],
                "guardrails": ["Nia rides a blue bicycle."],
            },
            {"episodes": [outline(1), outline(2)]},
            episode_result(1),
            episode_result(2),
            {
                "summary": "Nia rewrites the route and discovers stop one is missing.",
                "characters": {"Nia": "Knows the route can be altered."},
                "facts": ["Stop one was removed from the route."],
                "retracted_facts": ["Stop 1 is part of the route."],
                "open_threads": [],
                "resolved_threads": [],
            },
        ]
        self.requests = []

    def complete_json(self, messages, max_tokens):
        self.requests.append(messages)
        return Completion(self.responses.pop(0), 100, 100, 0.01, 0.0001)


class InterruptedPlanProvider:
    input_price = 0.15
    output_price = 0.60

    def __init__(self, responses, fail_after=None):
        self.responses = list(responses)
        self.fail_after = fail_after
        self.calls = 0
        self.requests = []

    def complete_json(self, messages, max_tokens):
        if self.fail_after is not None and self.calls == self.fail_after:
            raise RuntimeError("simulated interruption")
        self.requests.append(messages)
        self.calls += 1
        return Completion(self.responses.pop(0), 50, 50, 0.01, 0.0001)


class DegradingPlanProvider:
    input_price = 0.15
    output_price = 0.60

    def __init__(self, blueprint):
        self.responses = [
            blueprint,
            {"episodes": [outline(1)]},
            {"episodes": [outline(2)]},
        ]
        self.requests = []

    def complete_json(self, messages, max_tokens):
        self.requests.append(messages)
        if len(self.requests) == 2:
            raise InvalidModelOutput("malformed response", 10, 11, 0.1, 0.0001)
        return Completion(self.responses.pop(0), 20, 20, 0.01, 0.0001)


def plan_blueprint(total):
    return {
        "series_logline": "A rider follows impossible addresses.",
        "acts": [{"start_episode": 1, "end_episode": total, "title": "The route"}],
        "characters": [{"name": "Nia", "arc": "From wary to brave."}],
        "long_threads": [],
        "guardrails": [],
    }


class ShapeDriftingPlanProvider:
    input_price = 0.15
    output_price = 0.60

    def __init__(self, total, bad_shapes):
        self.total = total
        self.responses = [plan_blueprint(total)] + list(bad_shapes)
        self.requests = []
        self.calls = 0

    def complete_json(self, messages, max_tokens):
        self.requests.append(messages)
        self.calls += 1
        if self.responses:
            return Completion(self.responses.pop(0), 50, 50, 0.01, 0.0001)
        return Completion(
            {"episodes": [outline(number) for number in range(1, self.total + 1)]},
            50,
            50,
            0.01,
            0.0001,
        )


class StoryEngineTests(unittest.TestCase):
    def test_episode_draft_gets_second_trim_revision_when_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            state = RunState.create("sample", "A rider follows impossible addresses.", 1)
            state.plan_approved = True
            state.plan = {
                **plan_blueprint(1),
                "episodes": [outline(1)],
            }
            provider = FakeProvider()
            provider.responses = [
                episode_result_with_words(1, 900),
                episode_result_with_words(1, 720),
                episode_result_with_words(1, 500),
            ]
            engine = StoryEngine(state, Path(directory) / "sample", provider)

            draft = engine.draft_episode()

            self.assertEqual(draft["word_count"], 500)
            self.assertEqual(len(provider.requests), 3)
            self.assertIn("hard maximum is 700 words", provider.requests[1][-1]["content"])
            self.assertIn("about 500 words", provider.requests[2][-1]["content"])

    def test_episode_draft_continues_trimming_until_within_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            state = RunState.create("sample", "A rider follows impossible addresses.", 1)
            state.plan_approved = True
            state.plan = {**plan_blueprint(1), "episodes": [outline(1)]}
            provider = FakeProvider()
            provider.responses = [
                episode_result_with_words(1, 880),
                episode_result_with_words(1, 810),
                episode_result_with_words(1, 740),
                episode_result_with_words(1, 650),
            ]
            engine = StoryEngine(state, Path(directory) / "sample", provider)

            draft = engine.draft_episode()

            self.assertEqual(draft["word_count"], 650)
            self.assertEqual(len(provider.requests), 4)
            self.assertIn("trim pass 3 of 4", provider.requests[3][-1]["content"])

    def test_feedback_propagates_and_rewrite_stales_later_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sample"
            state = RunState.create("sample", "A rider follows impossible addresses.", 2)
            provider = FakeProvider()
            engine = StoryEngine(state, run_dir, provider)

            engine.generate_plan()
            engine.approve_plan()
            engine.draft_episode()
            engine.approve_episode(1, feedback="Make Nia more cautious around strangers.")
            engine.draft_episode()
            self.assertIn("Make Nia more cautious around strangers", provider.requests[3][-1]["content"])
            engine.approve_episode(2)

            engine.rewrite_episode(
                1,
                " ".join(f"rewrite{index}" for index in range(420)),
                feedback="The route was changed before the first delivery.",
            )

            self.assertEqual(state.episode(2)["status"], "stale")
            self.assertEqual(state.next_episode_number(), 2)
            self.assertIn("Stop one was removed from the route.", state.canon["facts"])
            self.assertNotIn("Stop 1 is part of the route.", state.canon["facts"])
            self.assertEqual(state.usage["input_tokens"], 500)

    def test_plan_generation_resumes_after_saved_outline_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sample"
            state = RunState.create("sample", "A rider follows impossible addresses.", 12)
            blueprint = {
                "series_logline": "A rider follows impossible addresses.",
                "acts": [{"start_episode": 1, "end_episode": 12, "title": "The route"}],
                "characters": [{"name": "Nia", "arc": "From wary to brave."}],
                "long_threads": [],
                "guardrails": [],
            }
            first_batch = {"episodes": [outline(number) for number in range(1, 6)]}
            first_provider = InterruptedPlanProvider([blueprint, first_batch], fail_after=2)
            first_engine = StoryEngine(state, run_dir, first_provider)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                first_engine.generate_plan()

            resumed_state = RunState.load(run_dir / "run.json")
            resumed_provider = InterruptedPlanProvider([
                {"episodes": [outline(number) for number in range(6, 11)]},
                {"episodes": [outline(11), outline(12)]},
            ])
            resumed_engine = StoryEngine(resumed_state, run_dir, resumed_provider)
            plan = resumed_engine.generate_plan()

            self.assertEqual(len(plan["episodes"]), 12)
            self.assertEqual(resumed_provider.calls, 2)
            self.assertIsNone(resumed_state.plan_progress)

    def test_plan_requests_only_outline_numbers_missing_from_partial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sample"
            state = RunState.create("sample", "A rider follows impossible addresses.", 10)
            blueprint = {
                "series_logline": "A rider follows impossible addresses.",
                "acts": [{"start_episode": 1, "end_episode": 10, "title": "The route"}],
                "characters": [{"name": "Nia", "arc": "From wary to brave."}],
                "long_threads": [],
                "guardrails": [],
            }
            provider = InterruptedPlanProvider([
                blueprint,
                {"episodes": [outline(number) for number in range(1, 4)]},
                {"episodes": [outline(4), outline(5)]},
                {"episodes": [outline(number) for number in range(6, 11)]},
            ])
            engine = StoryEngine(state, run_dir, provider)

            plan = engine.generate_plan()

            self.assertEqual([item["number"] for item in plan["episodes"]], list(range(1, 11)))
            self.assertEqual(provider.calls, 4)
            self.assertIn("[4, 5]", provider.requests[2][-1]["content"])

    def test_malformed_multi_outline_response_degrades_to_individual_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sample"
            state = RunState.create("sample", "A rider follows impossible addresses.", 2)
            blueprint = {
                "series_logline": "A rider follows impossible addresses.",
                "acts": [{"start_episode": 1, "end_episode": 2, "title": "The route"}],
                "characters": [{"name": "Nia", "arc": "From wary to brave."}],
                "long_threads": [],
                "guardrails": [],
            }
            provider = DegradingPlanProvider(blueprint)
            engine = StoryEngine(state, run_dir, provider)

            plan = engine.generate_plan()

            self.assertEqual(len(plan["episodes"]), 2)
            self.assertEqual(len(provider.requests), 4)
            self.assertIn("[1]", provider.requests[2][-1]["content"])
            self.assertIn("[2]", provider.requests[3][-1]["content"])
            self.assertEqual(state.usage["input_tokens"], 70)
            self.assertTrue(any(event["event"] == "plan_batch_degraded_to_single_outlines" for event in state.events))

    def test_outline_batch_with_wrong_key_is_repaired_on_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sample"
            state = RunState.create("sample", "A rider follows impossible addresses.", 2)
            provider = ShapeDriftingPlanProvider(2, [
                {"episode_list": [outline(1), outline(2)]},
                {"episodes": [{"episode_number": "1", "title": "T1", "dramatic_beat": "B1", "ending_hook": "H1"},
                              {"episode_number": "2", "title": "T2", "dramatic_beat": "B2", "ending_hook": "H2"}]},
            ])
            engine = StoryEngine(state, run_dir, provider)

            plan = engine.generate_plan()

            self.assertEqual([item["number"] for item in plan["episodes"]], [1, 2])
            self.assertTrue(any(event["event"] == "plan_outline_shape_invalid" for event in state.events))

    def test_repeatedly_malformed_outline_shape_fails_with_detail(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sample"
            state = RunState.create("sample", "A rider follows impossible addresses.", 2)
            provider = ShapeDriftingPlanProvider(2, [{"note": "no outlines here"}] * 5)
            engine = StoryEngine(state, run_dir, provider)

            with self.assertRaisesRegex(ValueError, "no episodes array"):
                engine.generate_plan()


if __name__ == "__main__":
    unittest.main()