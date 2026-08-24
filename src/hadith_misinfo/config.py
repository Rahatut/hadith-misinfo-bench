"""Centralised configuration for HadithMisinfoBench.

Configuration is resolved in the following priority order (highest wins):
  1. Environment variables (prefixed HMB_)
  2. .env file in the project root
  3. configs/default.yaml
  4. Built-in defaults defined here

Usage
-----
>>> from hadith_misinfo.config import settings
>>> print(settings.llm_model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/hadith_misinfo/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes")


# ── Path resolution ────────────────────────────────────────────────────────────

def _resolve(rel: str) -> Path:
    """Resolve a path relative to the project root."""
    return _PROJECT_ROOT / rel


@dataclass
class Settings:
    """All tunable settings for the pipeline.

    Attributes are set from env vars at import time.  Mutate at runtime
    only in tests or script-level overrides.
    """

    # ── Paths ─────────────────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    data_dir: Path = field(
        default_factory=lambda: _resolve(_env("HMB_DATA_DIR", "data"))
    )
    results_dir: Path = field(
        default_factory=lambda: _resolve(_env("HMB_RESULTS_DIR", "results"))
    )
    indices_dir: Path = field(
        default_factory=lambda: _resolve(_env("HMB_INDICES_DIR", "data/indices"))
    )

    # ── Raw data subdirectories ───────────────────────────────────────────────
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def mahaddat_raw_dir(self) -> Path:
        return self.raw_dir / "mahaddat"

    @property
    def hadith_json_raw_dir(self) -> Path:
        return self.raw_dir / "hadith-json"

    @property
    def al_zaman_raw_dir(self) -> Path:
        return self.raw_dir / "al-zaman"

    @property
    def alzaman_raw_dir(self) -> Path:
        return self.al_zaman_raw_dir

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    # ── Benchmark files ───────────────────────────────────────────────────────
    @property
    def benchmark_path(self) -> Path:
        return self.processed_dir / "benchmark_dataset_a.jsonl"

    @property
    def evidence_path(self) -> Path:
        return self.processed_dir / "evidence.jsonl"

    @property
    def dataset_c_path(self) -> Path:
        return self.processed_dir / "social_media.jsonl"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    dense_model: str = field(
        default_factory=lambda: _env("HMB_DENSE_MODEL", "BAAI/bge-m3")
    )
    default_k: int = field(
        default_factory=lambda: _env_int("HMB_DEFAULT_K", 5)
    )
    retrieval_text_mode: str = field(
        default_factory=lambda: _env("HMB_RETRIEVAL_TEXT_MODE", "arabic_plus_english")
    )

    # ── Index subdirectories ──────────────────────────────────────────────────
    @property
    def bm25_index_dir(self) -> Path:
        return self.indices_dir / "bm25"

    @property
    def dense_index_dir(self) -> Path:
        return self.indices_dir / "dense"

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: str = field(
        default_factory=lambda: _env("HMB_LLM_PROVIDER", "openai")
    )
    llm_model: str = field(
        default_factory=lambda: _env("HMB_LLM_MODEL", "gpt-4o-mini")
    )
    llm_temperature: float = field(
        default_factory=lambda: _env_float("HMB_LLM_TEMPERATURE", 0.0)
    )
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int("HMB_LLM_MAX_TOKENS", 512)
    )
    llm_retry_attempts: int = field(
        default_factory=lambda: _env_int("HMB_LLM_RETRY_ATTEMPTS", 10)
    )
    llm_retry_base_delay: float = field(
        default_factory=lambda: _env_float("HMB_LLM_RETRY_BASE_DELAY", 5.0)
    )

    # ── OpenAI / OpenRouter ───────────────────────────────────────────────────
    @property
    def openai_api_key(self) -> str:
        return _env("OPENAI_API_KEY", "")

    @property
    def openai_base_url(self) -> str:
        return _env("OPENAI_BASE_URL", "")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b")
    )

    # ── Benchmark sampling ────────────────────────────────────────────────────
    benchmark_n_authentic: int = field(
        default_factory=lambda: _env_int("HMB_N_AUTHENTIC", 250)
    )
    benchmark_n_fabricated: int = field(
        default_factory=lambda: _env_int("HMB_N_FABRICATED", 250)
    )
    benchmark_seed: int = field(
        default_factory=lambda: _env_int("HMB_SEED", 42)
    )

    # ── Dataset C ─────────────────────────────────────────────────────────────
    dataset_c_sample_size: int = field(
        default_factory=lambda: _env_int("HMB_DATASET_C_SAMPLE", 100)
    )

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [
            self.processed_dir,
            self.results_dir,
            self.bm25_index_dir,
            self.dense_index_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere
settings = Settings()
