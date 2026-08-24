"""Anthropic Claude adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hadith_misinfo.llm.base import CompleteFn

_DEFAULT_MODEL = "claude-3-haiku-20240307"


class AnthropicAdapter:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.model = model or _DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package not installed.\n"
                    "Run: pip install 'hadith-misinfo-bench[anthropic]'"
                ) from e
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text if message.content else ""

    def as_complete_fn(self) -> CompleteFn:
        return self.complete
