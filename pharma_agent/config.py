"""Runtime configuration, read once from the environment (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty env var among `names`."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip("'\"")
    return default


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str | None
    chat_model: str
    embedding_model: str
    corpus_dir: Path
    cache_dir: Path
    max_tool_rounds: int
    max_revisions: int


@lru_cache
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example).")
    return Settings(
        api_key=api_key,
        # The provided .env ships `OPENAI_BASE_UR` (typo) — accept both spellings.
        base_url=_env("OPENAI_BASE_URL", "OPENAI_BASE_UR"),
        chat_model=_env("OPENAI_MODEL", default="gpt-4o-mini"),
        embedding_model=_env("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small"),
        corpus_dir=Path(_env("CORPUS_DIR", default=str(PROJECT_ROOT / "corpus"))),
        cache_dir=Path(_env("CACHE_DIR", default=str(PROJECT_ROOT / ".cache"))),
        max_tool_rounds=int(_env("AGENT_MAX_TOOL_ROUNDS", default="6")),
        max_revisions=int(_env("AGENT_MAX_REVISIONS", default="1")),
    )
