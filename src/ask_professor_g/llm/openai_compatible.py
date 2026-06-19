from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import LLMClient


@dataclass
class OpenAICompatibleClient(LLMClient):
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 180
    max_retries: int = 3

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "OpenAICompatibleClient":
        return cls(
            model=settings.get("model") or os.environ.get("GRASP_LLM_MODEL", "gpt-4o"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=settings.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
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
            raise RuntimeError("OPENAI_API_KEY is required for provider 'openai-compatible'.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use provider 'openai-compatible'.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image_path:
            mime = "image/png" if Path(image_path).suffix.lower() != ".jpg" else "image/jpeg"
            with open(image_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI-compatible request failed: {last_error}")

