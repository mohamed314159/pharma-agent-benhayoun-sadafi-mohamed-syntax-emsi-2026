# CLAUDE.md — project instructions for Claude Code

Agentic RAG assistant over a pharmacology corpus (Syntax GenAI assignment).

## Commands
- Install: `uv sync`
- Run API + UI: `uv run uvicorn pharma_agent.api:app --reload` → http://localhost:8000
- Eval (4 public questions): `uv run python -m tests.eval_questions --verbose`
- Docker: `docker build -t agent . && docker run -p 8000:8000 --env-file .env agent`

## Conventions
- Hand-written agent loop in `pharma_agent/agent.py` (no LangChain): plan → tools → draft → critic → answer.
- Every passage shown to the writer must go through `EvidenceStore` so it gets an `[E#]` id.
- Prompts live only in `pharma_agent/prompts.py`.
- Config only via env vars (`pharma_agent/config.py`); never commit `.env`.
- Answers must cite sources and end with the not-medical-advice disclaimer.
- Keep dependencies pinned (`uv.lock` + exported `requirements.txt`).
