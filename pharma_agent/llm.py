"""Thin wrapper around the OpenAI-compatible client (LiteLLM proxy)."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from openai import OpenAI

from .config import get_settings

log = logging.getLogger(__name__)


@lru_cache
def client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.api_key, base_url=s.base_url, max_retries=3, timeout=90)


def chat(messages: list[dict[str, Any]], tools: list[dict] | None = None):
    """Single chat completion; returns the assistant message object."""
    kwargs: dict[str, Any] = {"model": get_settings().chat_model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    return client().chat.completions.create(**kwargs).choices[0].message


def chat_json(system: str, user: str) -> dict[str, Any]:
    """Chat completion constrained to a JSON object."""
    resp = client().chat.completions.create(
        model=get_settings().chat_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        log.warning("Model returned invalid JSON: %s", content[:200])
        return {}


def embed(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    model = get_settings().embedding_model
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = client().embeddings.create(model=model, input=texts[i : i + batch_size])
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out
