from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class Completion:
    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float
    fallback_used: bool = False


class InvalidModelOutput(ValueError):
    def __init__(
        self,
        message: str,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float,
        cost_usd: float,
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_seconds = latency_seconds
        self.cost_usd = cost_usd


class OpenAICompatibleProvider:
    """Client for OpenAI-compatible chat completion APIs."""

    def __init__(self) -> None:
        load_dotenv(override=False)
        self.provider = os.environ.get("STORY_PROVIDER", "openai").strip().lower()

        if self.provider not in {"openai", "groq"}:
            raise ValueError("STORY_PROVIDER must be 'openai' or 'groq'")

        if self.provider == "groq":
            self.api_key = os.environ.get("GROQ_API_KEY", "")
            self.model = os.environ.get(
                "STORY_MODEL",
                "openai/gpt-oss-20b",
            )
            self.base_url = os.environ.get(
                "STORY_BASE_URL",
                "https://api.groq.com/openai/v1",
            ).rstrip("/")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
            self.model = os.environ.get(
                "STORY_MODEL",
                "gpt-4o-mini",
            )
            self.base_url = os.environ.get(
                "STORY_BASE_URL",
                "https://api.openai.com/v1",
            ).rstrip("/")

        self.input_price = float(
            os.environ.get("STORY_INPUT_USD_PER_MILLION", "0.15")
        )
        self.output_price = float(
            os.environ.get("STORY_OUTPUT_USD_PER_MILLION", "0.60")
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> Completion:

        if not self.api_key:
            key_name = (
                "GROQ_API_KEY"
                if self.provider == "groq"
                else "OPENAI_API_KEY"
            )
            raise RuntimeError(
                f"{key_name} is not set; configure it to call the model"
            )

        # Use Groq SDK for Groq because urllib requests are being rejected.
        if self.provider == "groq":
            from groq import Groq

            client = Groq(api_key=self.api_key)
            reasoning_options = (
                {"reasoning_effort": "low"}
                if self.model.startswith("openai/gpt-oss-")
                else {}
            )

            started = time.perf_counter()
            fallback_used = False
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    **reasoning_options,
                )
            except Exception as error:
                if not self._is_json_validation_error(error):
                    raise
                fallback_used = True
                retry_messages = list(messages)
                retry_messages[-1] = {
                    **retry_messages[-1],
                    "content": retry_messages[-1]["content"]
                    + "\n\nReturn exactly one valid JSON object. Do not use markdown fences or add commentary.",
                }
                response = client.chat.completions.create(
                    model=self.model,
                    messages=retry_messages,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    **reasoning_options,
                )

            content = response.choices[0].message.content

            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                )

            usage = response.usage
            input_tokens = int(
                usage.prompt_tokens if usage else 0
            )
            output_tokens = int(
                usage.completion_tokens if usage else 0
            )

            try:
                result = self._parse_json_object(content)
            except (json.JSONDecodeError, ValueError) as parse_error:
                fallback_used = True
                repair_messages = [
                    {
                        "role": "system",
                        "content": "Repair malformed JSON. Return exactly one valid JSON object, without markdown. Preserve the original data and finish any truncated value conservatively.",
                    },
                    {
                        "role": "user",
                        "content": f"Repair this model response into valid JSON:\n\n{content}",
                    },
                ]
                repaired = client.chat.completions.create(
                    model=self.model,
                    messages=repair_messages,
                    temperature=0,
                    max_tokens=max(max_tokens, 3000),
                    **reasoning_options,
                )
                repaired_content = repaired.choices[0].message.content
                if isinstance(repaired_content, list):
                    repaired_content = "".join(
                        part.get("text", "")
                        for part in repaired_content
                    )
                repair_usage = repaired.usage
                if repair_usage:
                    input_tokens += int(repair_usage.prompt_tokens)
                    output_tokens += int(repair_usage.completion_tokens)
                try:
                    result = self._parse_json_object(repaired_content)
                except (json.JSONDecodeError, ValueError) as repair_error:
                    latency = time.perf_counter() - started
                    cost = (
                        input_tokens * self.input_price
                        + output_tokens * self.output_price
                    ) / 1_000_000
                    raise InvalidModelOutput(
                        f"model returned malformed JSON and one repair attempt failed: {repair_error}",
                        input_tokens,
                        output_tokens,
                        latency,
                        cost,
                    ) from parse_error

            latency = time.perf_counter() - started

            cost = (
                input_tokens * self.input_price
                + output_tokens * self.output_price
            ) / 1_000_000

            return Completion(
                data=result,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=latency,
                cost_usd=cost,
                fallback_used=fallback_used,
            )

        # Keep the existing urllib implementation for OpenAI.
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")

        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.perf_counter()

        try:
            with urlopen(request, timeout=150) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as error:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
            )[:1000]

            raise RuntimeError(
                f"model HTTP {error.code}: {detail}"
            ) from error

        except URLError as error:
            raise RuntimeError(
                f"model connection failed: {error.reason}"
            ) from error

        latency = time.perf_counter() - started

        content = response_data["choices"][0]["message"]["content"]

        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
            )

        result = json.loads(content)

        usage = response_data.get("usage", {})

        input_tokens = int(
            usage.get(
                "prompt_tokens",
                max(1, len(json.dumps(messages)) // 4),
            )
        )

        output_tokens = int(
            usage.get(
                "completion_tokens",
                max(1, len(content) // 4),
            )
        )

        cost = (
            input_tokens * self.input_price
            + output_tokens * self.output_price
        ) / 1_000_000

        return Completion(
            result,
            input_tokens,
            output_tokens,
            latency,
            cost,
        )

    @staticmethod
    def _is_json_validation_error(error: Exception) -> bool:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            details = body.get("error", {})
            if isinstance(details, dict) and details.get("code") == "json_validate_failed":
                return True
        return "json_validate_failed" in str(error).lower()

    @staticmethod
    def _parse_json_object(content: str | None) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model returned empty content instead of a JSON object")
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            if start < 0:
                raise ValueError("model response did not contain a JSON object")
            result, _ = json.JSONDecoder().raw_decode(content[start:])
        if not isinstance(result, dict):
            raise ValueError("model response must be a JSON object")
        return result