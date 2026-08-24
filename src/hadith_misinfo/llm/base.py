"""LLM base protocol, retry wrapper, and factory."""

from __future__ import annotations

import time
from typing import Callable, Protocol

# The canonical function signature used throughout the codebase
CompleteFn = Callable[[str], str]

# Thread-local current complete function (used by paraphrase module as fallback)
_current_complete_fn: CompleteFn | None = None


class LLMAdapter(Protocol):
    """Protocol for all LLM provider adapters."""

    def complete(self, prompt: str) -> str:
        """Send a prompt and return the text response."""
        ...

    def as_complete_fn(self) -> CompleteFn:
        """Return a bare callable suitable for use as CompleteFn."""
        ...


# ── Retry wrapper ─────────────────────────────────────────────────────────────

_RETRYABLE_CODES = {402, 429, 503, 529, 500, 502}


def retry_complete(
    fn: CompleteFn,
    max_attempts: int = 8,
    base_delay: float = 3.0,
    backoff: float = 1.5,
) -> CompleteFn:
    """Wrap a CompleteFn with exponential back-off retry."""

    def _wrapped(prompt: str) -> str:
        delay = base_delay
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(prompt)
            except Exception as exc:
                last_exc = exc
                code = _extract_status_code(exc)
                if code is not None and code not in _RETRYABLE_CODES:
                    raise
                if attempt < max_attempts:
                    exc_str = str(exc)
                    if "Retry-After" in exc_str:
                        sleep_time = 122.0
                    elif code in (402, 429):
                        sleep_time = max(delay, 30.0)
                    else:
                        sleep_time = delay

                    print(
                        f"[retry] Attempt {attempt}/{max_attempts} failed "
                        f"({exc}). Waiting {sleep_time:.1f}s for in-flight settlement..."
                    )
                    time.sleep(sleep_time)
                    delay *= backoff
        raise RuntimeError(f"All {max_attempts} attempts failed.") from last_exc

    return _wrapped


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)
    return None


# ── Factory ───────────────────────────────────────────────────────────────────

def make_complete_fn(
    provider: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retry: bool = True,
    retry_attempts: int = 8,
    retry_base_delay: float = 3.0,
) -> CompleteFn:
    """Factory that returns a CompleteFn for the given provider."""
    if provider == "openai":
        from hadith_misinfo.llm.openai import OpenAIAdapter
        adapter = OpenAIAdapter(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "anthropic":
        from hadith_misinfo.llm.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "ollama":
        from hadith_misinfo.llm.ollama import OllamaAdapter
        adapter = OllamaAdapter(model=model, temperature=temperature)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            "Supported: 'openai', 'anthropic', 'ollama'."
        )

    fn = adapter.as_complete_fn()
    if retry:
        fn = retry_complete(fn, max_attempts=retry_attempts, base_delay=retry_base_delay)
    return fn
