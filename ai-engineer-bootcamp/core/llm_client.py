"""Provider-agnostic LLM client abstraction (Gemini & Groq)."""

from __future__ import annotations

import os
from dataclasses import dataclass
import time
from typing import Any

from google import genai

from core.config import get_settings
from core.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class UsageMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float


class LLMClient:
    """Simple LLM client with `chat` and usage logging."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.llm_model
        self.temperature = (
            settings.llm_temperature if temperature is None else float(temperature)
        )
        self.input_cost_per_1m_tokens = settings.input_cost_per_1m_tokens
        self.output_cost_per_1m_tokens = settings.output_cost_per_1m_tokens

        if self.provider == "gemini":
            # Prefer GEMINI_MODEL env var over the generic LLM_MODEL.
            self.model = model or os.environ.get("GEMINI_MODEL", settings.llm_model)
            # The SDK reads GEMINI_API_KEY from environment variables.
            self.client = genai.Client()
        elif self.provider == "groq":
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("Missing GROQ_API_KEY environment variable.")
            self.model = model or os.environ.get(
                "GROQ_MODEL", "llama-3.3-70b-versatile"
            )
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. Supported: 'gemini', 'groq'."
            )

    def chat(self, messages: list[dict[str, str]] | str, **kwargs: Any) -> dict[str, Any]:
        """Send messages to the configured model and return response + metadata."""
        if self.provider == "groq":
            return self._chat_groq(messages, **kwargs)

        prompt = self._messages_to_prompt(messages)
        if not prompt:
            raise ValueError("messages cannot be empty.")

        started = time.perf_counter()
        try:
            config = {"temperature": self.temperature}
            extra_config = kwargs.pop("config", None)
            if isinstance(extra_config, dict):
                config.update(extra_config)
            config.update(kwargs)
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            logger.exception(
                "LLM call failed | provider=%s | model=%s",
                self.provider,
                self.model,
            )
            raise RuntimeError("Failed to call the LLM provider.") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)
        estimated_cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)

        metrics = UsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
        )
        self.log_usage(metrics)

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("The LLM returned an empty response.")

        return {
            "response": text,
            "metadata": {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "usage": {
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens,
                },
                "latency_ms": round(metrics.latency_ms, 2),
                "estimated_cost_usd": round(metrics.estimated_cost_usd, 8),
            },
        }

    def log_usage(self, metrics: UsageMetrics) -> None:
        """Log usage metrics for every LLM call."""
        logger.info(
            (
                "llm_call | provider=%s | model=%s | prompt_tokens=%d "
                "| completion_tokens=%d | total_tokens=%d | latency_ms=%.2f "
                "| estimated_cost_usd=%.8f"
            ),
            self.provider,
            self.model,
            metrics.prompt_tokens,
            metrics.completion_tokens,
            metrics.total_tokens,
            metrics.latency_ms,
            metrics.estimated_cost_usd,
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_1m_tokens
        output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_1m_tokens
        return input_cost + output_cost

    def _extract_usage(self, response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            usage = getattr(response, "usage_metadata", None)

        prompt_tokens = self._read_usage_value(
            usage,
            "prompt_tokens",
            "prompt_token_count",
            "input_tokens",
            "input_token_count",
        )
        completion_tokens = self._read_usage_value(
            usage,
            "completion_tokens",
            "candidates_token_count",
            "output_tokens",
            "output_token_count",
        )
        total_tokens = self._read_usage_value(
            usage,
            "total_tokens",
            "total_token_count",
        )

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        return prompt_tokens, completion_tokens, total_tokens

    def _read_usage_value(self, usage: Any, *fields: str) -> int:
        if usage is None:
            return 0

        for field_name in fields:
            value = getattr(usage, field_name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(field_name)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    def _chat_groq(self, messages: list[dict[str, str]] | str, **kwargs: Any) -> dict[str, Any]:
        """Handle chat via the Groq API (OpenAI-compatible)."""
        if isinstance(messages, str):
            groq_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            groq_messages = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ]
        else:
            raise TypeError("messages must be either a string or a list of dicts.")

        config = kwargs.pop("config", None) or {}
        if isinstance(config, dict):
            config = dict(config)
        else:
            config = {}
        config.update(kwargs)

        max_tokens = config.pop("max_output_tokens", config.pop("max_tokens", 1024))
        temperature = config.pop("temperature", self.temperature)

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            logger.exception(
                "LLM call failed | provider=%s | model=%s",
                self.provider,
                self.model,
            )
            raise RuntimeError("Failed to call the LLM provider.") from exc

        latency_ms = (time.perf_counter() - started) * 1000

        text = (response.choices[0].message.content or "").strip()

        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)

        metrics = UsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
        )
        self.log_usage(metrics)

        if not text:
            raise RuntimeError("The LLM returned an empty response.")

        return {
            "response": text,
            "metadata": {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "usage": {
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens,
                },
                "latency_ms": round(metrics.latency_ms, 2),
                "estimated_cost_usd": round(metrics.estimated_cost_usd, 8),
            },
        }

    def _messages_to_prompt(self, messages: list[dict[str, str]] | str) -> str:
        if isinstance(messages, str):
            return messages.strip()

        if not isinstance(messages, list):
            raise TypeError("messages must be either a string or a list of dictionaries.")

        lines: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                raise TypeError("Each message must be a dictionary with role/content.")
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")

        return "\n".join(lines).strip()
