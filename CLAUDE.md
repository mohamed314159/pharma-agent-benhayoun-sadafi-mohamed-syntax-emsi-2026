# CLAUDE.md — project instructions for Claude Code

Agentic RAG assistant over a pharmacology corpus (Syntax GenAI assignment), reusing the architecture
from a previous internship.

## Commands
- Install: `uv sync`
- Rebuild the index (semantic chunking + LLM titles): `uv run python -m pharma_agent.ingest`
- Run API + UI: `uv run uvicorn pharma_agent.api:app` → http://localhost:8000
- Eval (4 public questions): `uv run python -m tests.eval_questions --verbose`
- Docker: `docker build -t agent . && docker run -p 8000:8000 --env-file .env agent`

## Conventions
- Keep the code minimal: 3 Python files (`ingest.py`, `agent.py`, `api.py`).
- Pipeline (LangGraph `StateGraph`): planner (sub-questions + tool choice: search_corpus always, read_label_section,
  glossary_lookup) → one `Send` branch per sub-question (tools → comparator/critic) → synthesizer.
  search_corpus = hybrid 75/25, 50 → rerank → 5. LLM calls use the OpenAI SDK directly. Prompts are constants in `agent.py`.
- Chunking parameters (500 target, 600 max, 300 min, 10 % overlap, titles ≤ 10 words) come from my spec: don't change them silently.
- Config only via env vars; never commit `.env`. Answers must cite `[file]` and end with the disclaimer.
- Keep dependencies pinned (`uv.lock` + exported `requirements.txt`).
- Working method: restate my request and wait for confirmation before big changes.
