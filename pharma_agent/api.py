"""FastAPI app: GET /health, POST /ask and the web UI."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import agent


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not (agent.INDEX_DIR / "chunks.json").exists():  # the index ships with the repo; rebuild only if missing
        from .ingest import build
        build()
    agent.index()  # load the in-memory index once at startup
    yield


app = FastAPI(title="Pharma Research Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(agent.index().chunks)}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return agent.ask(request.question.strip())
    except Exception as exc:  # surface upstream LLM errors cleanly
        raise HTTPException(502, f"Agent error: {exc}") from exc


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
