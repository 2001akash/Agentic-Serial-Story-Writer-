import tempfile
import unittest
from pathlib import Path

from serial_writer.state import RunState


class RunStateTests(unittest.TestCase):
    def test_state_resumes_at_first_unapproved_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            state = RunState.create("sample", "A rider finds impossible addresses.", 3)
            state.plan = {"acts": []}
            state.plan_approved = True
            state.save_draft({
                "number": 1,
                "text": "Episode one.",
                "continuity": {"facts": ["The route starts at dawn."], "summary": "The route begins."},
            })
            state.approve_draft(1)
            state.save(path)

            resumed = RunState.load(path)

            self.assertEqual(resumed.next_episode_number(), 2)
            self.assertEqual(resumed.canon["facts"], ["The route starts at dawn."])

    def test_rewriting_episode_invalidates_later_episodes(self):
        state = RunState.create("sample", "A rider finds impossible addresses.", 3)
        state.plan_approved = True
        for number in (1, 2):
            state.save_draft({"number": number, "text": f"Episode {number}.", "continuity": {}})
            state.approve_draft(number)

        state.replace_approved_episode(1, "Rewritten.", {"facts": ["New truth."]})

        self.assertEqual(state.episode(2)["status"], "stale")
        self.assertEqual(state.next_episode_number(), 2)
        self.assertEqual(state.canon["facts"], ["New truth."])


if __name__ == "__main__":
    unittest.main()