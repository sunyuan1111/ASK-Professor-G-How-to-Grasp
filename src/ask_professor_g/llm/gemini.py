from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .base import LLMClient


@dataclass
class GeminiClient(LLMClient):
    model: str
    api_key: str | None = None
    timeout: int = 180
    max_retries: int = 3

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "GeminiClient":
        return cls(
            model=settings.get("model") or os.environ.get("GRASP_LLM_MODEL", "gemini-3-pro-preview"),
            api_key=os.environ.get("GEMINI_API_KEY"),
            timeout=int(settings.get("timeout") or os.environ.get("GRASP_LLM_TIMEOUT", 180)),
            max_retries=int(settings.get("max_retries") or os.environ.get("GRASP_LLM_MAX_RETRIES", 3)),
        )

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        image_path: str | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for provider 'gemini'.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install google-genai to use provider 'gemini'.") from exc

        client = genai.Client(api_key=self.api_key)
        parts: list[Any] = [user_prompt]
        if image_path:
            with open(image_path, "rb") as handle:
                image_bytes = handle.read()
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        request_options={"timeout": self.timeout},
                    ),
                )
                return response.text or ""
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Gemini request failed: {last_error}")

