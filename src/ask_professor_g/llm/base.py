from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .parser import extract_json


class LLMClient(ABC):
    @abstractmethod
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        image_path: str | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        image_path: str | None = None,
    ) -> dict[str, Any] | list[Any]:
        return extract_json(self.generate_text(system_prompt, user_prompt, image_path=image_path))


def build_llm_client(settings: dict[str, Any]) -> LLMClient:
    provider = (settings.get("provider") or "gemini").lower()
    if provider == "gemini":
        from .gemini import GeminiClient

        return GeminiClient.from_settings(settings)
    if provider in {"openai-compatible", "openai", "custom"}:
        from .openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient.from_settings(settings)
    raise ValueError(f"Unsupported LLM provider: {provider}")

