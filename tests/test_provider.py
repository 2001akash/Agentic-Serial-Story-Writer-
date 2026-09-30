import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from serial_writer.provider import OpenAICompatibleProvider


class ProviderConfigTests(unittest.TestCase):
    def test_groq_defaults_and_credentials(self):
        with patch.dict(os.environ, {
            "STORY_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
        }, clear=True):
            provider = OpenAICompatibleProvider()

        self.assertEqual(provider.api_key, "test-key")
        self.assertEqual(provider.base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(provider.model, "openai/gpt-oss-20b")

    def test_groq_model_and_url_can_be_overridden(self):
        with patch.dict(os.environ, {
            "STORY_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "STORY_BASE_URL": "https://proxy.example/v1/",
            "STORY_MODEL": "custom-model",
        }, clear=True):
            provider = OpenAICompatibleProvider()

        self.assertEqual(provider.base_url, "https://proxy.example/v1")
        self.assertEqual(provider.model, "custom-model")

    def test_unknown_provider_is_rejected(self):
        with patch.dict(os.environ, {"STORY_PROVIDER": "unknown"}, clear=True):
            with self.assertRaisesRegex(ValueError, "STORY_PROVIDER"):
                OpenAICompatibleProvider()

    def test_env_file_loads_provider_and_preserves_shell_override(self):
        from dotenv import load_dotenv as load_dotenv_file

        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory, ".env")
            env_path.write_text(
                "STORY_PROVIDER=groq\nGROQ_API_KEY=dotenv-key\nSTORY_MODEL=dotenv-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"STORY_MODEL": "shell-model"}, clear=True):
                with patch(
                    "serial_writer.provider.load_dotenv",
                    side_effect=lambda override=False: load_dotenv_file(env_path, override=override),
                ):
                    provider = OpenAICompatibleProvider()

        self.assertEqual(provider.provider, "groq")
        self.assertEqual(provider.api_key, "dotenv-key")
        self.assertEqual(provider.model, "shell-model")

    def test_groq_retries_json_validation_failure_without_json_mode(self):
        calls = []

        class JsonValidationError(Exception):
            body = {"error": {"code": "json_validate_failed"}}

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if "response_format" in kwargs:
                    raise JsonValidationError("JSON validation failed")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"episodes": []}'))],
                    usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
                )

        fake_groq = SimpleNamespace(
            Groq=lambda api_key: SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            )
        )
        with patch.dict(os.environ, {
            "STORY_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
        }, clear=True), patch.dict(sys.modules, {"groq": fake_groq}):
            completion = OpenAICompatibleProvider().complete_json(
                [{"role": "user", "content": "Return JSON."}],
                max_tokens=100,
            )

        self.assertEqual(completion.data, {"episodes": []})
        self.assertTrue(completion.fallback_used)
        self.assertEqual(len(calls), 2)
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertEqual(calls[1]["reasoning_effort"], "low")

    def test_groq_repairs_malformed_json_and_counts_both_calls(self):
        calls = []

        def response(content, prompt_tokens, completion_tokens):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
            )

        class FakeCompletions:
            responses = [
                response('{"episodes":[{"number":1,"title":"unfinished', 12, 20),
                response('{"episodes":[{"number":1,"title":"fixed"}]}', 15, 10),
            ]

            def create(self, **kwargs):
                calls.append(kwargs)
                return self.responses.pop(0)

        fake_groq = SimpleNamespace(
            Groq=lambda api_key: SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            )
        )
        with patch.dict(os.environ, {
            "STORY_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
        }, clear=True), patch.dict(sys.modules, {"groq": fake_groq}):
            completion = OpenAICompatibleProvider().complete_json(
                [{"role": "user", "content": "Return an outline."}],
                max_tokens=100,
            )

        self.assertEqual(completion.data, {"episodes": [{"number": 1, "title": "fixed"}]})
        self.assertEqual(completion.input_tokens, 27)
        self.assertEqual(completion.output_tokens, 30)
        self.assertTrue(completion.fallback_used)
        self.assertEqual(len(calls), 2)
        self.assertIn("unfinished", calls[1]["messages"][1]["content"])
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertEqual(calls[1]["reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()