"""OpenAI LLM adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hadith_misinfo.llm.base import CompleteFn

_DEFAULT_MODEL = "gpt-4o-mini"


import os

class OpenAIAdapter:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from hadith_misinfo.config import settings

        self.model = model or _DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or settings.openai_base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.base_url and self.api_key and self.api_key.startswith("sk-or-v1-"):
            self.base_url = "https://openrouter.ai/api/v1"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package not installed.\n"
                    "Run: pip install 'hadith-misinfo-bench[openai]'"
                ) from e
            kwargs = {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def as_complete_fn(self) -> CompleteFn:
        return self.complete
