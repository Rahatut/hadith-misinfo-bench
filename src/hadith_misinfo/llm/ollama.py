"""Ollama local LLM adapter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from hadith_misinfo.llm.base import CompleteFn

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b"


class OllamaAdapter:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = httpx.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def as_complete_fn(self) -> CompleteFn:
        return self.complete
