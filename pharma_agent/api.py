"""FastAPI app: GET /health, POST /ask, POST /ask/stream (SSE) and the web UI."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import Agent, AgentResources
from .config import get_settings
from .corpus import load_corpus
from .index import HybridIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pharma_agent")
STATIC = Path(__file__).parent / "static"

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    docs, chunks = load_corpus(s.corpus_dir)
    index = HybridIndex(chunks, s.cache_dir, s.embedding_model)
    state["agent"] = Agent(AgentResources(docs, index))
    state["stats"] = {"documents": len(docs), "chunks": len(chunks), "model": s.chat_model}
    log.info("Index ready: %s", state["stats"])
    yield
    state.clear()


app = FastAPI(title="Pharma Research Agent", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["What are the main contraindications for warfarin?"])


def _agent() -> Agent:
    agent = state.get("agent")
    if agent is None:
        raise HTTPException(503, "Agent not ready")
    return agent


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", **state.get("stats", {})}


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    try:
        return _agent().ask(req.question.strip())
    except HTTPException:
        raise
    except Exception as exc:  # surface upstream LLM errors cleanly
        log.exception("ask failed")
        raise HTTPException(502, f"Agent error: {exc}") from exc


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    agent = _agent()

    def events() -> Iterator[str]:
        try:
            for ev in agent.stream(req.question.strip()):
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            log.exception("stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
