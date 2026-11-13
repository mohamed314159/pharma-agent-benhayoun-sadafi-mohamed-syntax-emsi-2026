# 💊 Pharma Research Agent

Agentic research assistant that answers compound pharmacology questions over a local corpus of
FDA labels (DailyMed) and MedlinePlus monographs, with **planning, agent-driven tool use, a
grounding critic and cited answers**.

> Syntax GenAI Bootcamp — coding assignment · Mohamed BENHAYOUN SADAFI (EMSI) · 2026
> ⚠️ Technical exercise — **not medical advice**.

## Quick start

```bash
cp .env.example .env          # fill OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
docker build -t agent . && docker run -p 8000:8000 --env-file .env agent
```

Open <http://localhost:8000> for the UI, or call the API:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What are the main contraindications for warfarin?"}'
```

Local dev without Docker: `uv sync && uv run uvicorn pharma_agent.api:app --reload`

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `200 {"status":"ok","documents":21,"chunks":546,...}` |
| `POST` | `/ask` | `{"question": "..."}` → `{answer, sources[], plan, tool_calls[], review, disclaimer, latency_s}` |
| `POST` | `/ask/stream` | Same pipeline as Server-Sent Events (plan, each tool call, drafts, critic rounds, final) — used by the UI |
| `GET` | `/` | Web UI |

## Architecture

```
question
   │
   ▼
┌────────┐  JSON: language, question type, drugs, sub-questions, in_scope
│ 1 PLAN │──────────────────────────────────────────────────────────────┐
└────────┘                                                              │
   ▼                                                                    │
┌────────────────────────────┐   OpenAI function calling, the LLM       │
│ 2-3 RESEARCH (tool loop)   │   chooses tools & args (≤ 6 rounds,      │
│  search_corpus   (hybrid)  │   parallel calls allowed)                │
│  get_section     (labels)  │──► every passage → EvidenceStore [E1..En]│
│  list_sections             │                                          │
│  lookup_glossary           │                                          │
└────────────────────────────┘                                          │
   ▼                                                                    │
┌────────┐  answer from evidence only, every sentence cites [E#]        │
│ DRAFT  │◄─────────────────────────────────────────────┐               │
└────────┘                                              │ issues +      │
   ▼                                                    │ follow-up     │
┌──────────────┐  LLM judge: unsupported / ambiguous /  │ searches      │
│ 4 CRITIC     │  missing / citation  + deterministic   │ (≤1 revision) │
│              │  check of cited ids  ──── revise ──────┘               │
└──────────────┘                                                        │
   ▼ pass                                                               │
┌──────────┐  [E#] → numbered sources (file › section), disclaimer       │
│ 5 ANSWER │◄────────────────────────────────────────────────────────────┘
└──────────┘
```

| Module | Role |
|---|---|
| `pharma_agent/corpus.py` | Parses markdown into **heading-aware chunks** (a chunk never spans two sections and carries its heading path: `4 CONTRAINDICATIONS`, `7.3 Drugs that Increase Bleeding Risk`) |
| `pharma_agent/index.py` | **In-memory hybrid retrieval**: numpy cosine similarity on embeddings + BM25, fused with Reciprocal Rank Fusion; drug/source filters; embeddings cached on disk |
| `pharma_agent/tools.py` | The 4 local tools + JSON schemas + `EvidenceStore` |
| `pharma_agent/agent.py` | The hand-written agent loop (generator of events) |
| `pharma_agent/prompts.py` | All prompts in one place |
| `pharma_agent/api.py` | FastAPI app (REST + SSE + static UI) |
| `pharma_agent/static/index.html` | Single-file UI: live agent trace, clickable citations, critic report |

### Design choices

- **Own loop instead of LangChain/LangGraph** — the workflow is 5 explicit steps; a ~200-line loop
  over the OpenAI SDK is easier to read, debug and explain, and works with any OpenAI-compatible
  endpoint (the LiteLLM proxy) with zero adapters.
- **Section-aware chunking** — drug labels are strongly structured; questions like "main
  contraindications" need the *whole* section, so the agent has `get_section` for exhaustive lists
  and `search_corpus` for targeted facts. The model picks, not the code.
- **Hybrid retrieval** — embeddings handle paraphrases, BM25 handles exact tokens (drug names,
  `eGFR`, `CYP2C9`, numbers). RRF avoids score calibration.
- **Evidence ids end-to-end** — tools register every passage they return (`[E7]`); the writer must
  cite ids; the critic checks claims against those passages, plus a deterministic check that every
  cited id exists. Ids are finally renumbered into user-facing `[1] dailymed-warfarin.md › 4 CONTRAINDICATIONS`.
- **Critic with feedback loop** — when it finds issues it proposes follow-up queries, the agent
  retrieves more evidence and rewrites (bounded by `AGENT_MAX_REVISIONS`).
- **Robust config** — the provided `.env` names the proxy URL `OPENAI_BASE_UR` (missing `L`); both
  spellings are accepted.

## Evaluation

```bash
uv run python -m tests.eval_questions --verbose
```

Runs the 4 public questions and checks that the expected source files are cited:

| Question | Expected | Result |
|---|---|---|
| Main contraindications of warfarin | dailymed-warfarin | ✅ cited, critic pass |
| Warfarin vs apixaban contraindications | dailymed-warfarin, dailymed-apixaban | ✅ cited, critic pass |
| Warfarin + sertraline bleeding & monitoring | dailymed-warfarin, dailymed-sertraline | ✅ cited, critic pass |
| Metformin at eGFR 25 | dailymed-metformin | ✅ cited (threshold < 30 mL/min/1.73 m²), critic pass |

## Configuration

| Variable | Default | |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_BASE_URL` / `OPENAI_BASE_UR` | OpenAI | LiteLLM proxy URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `AGENT_MAX_TOOL_ROUNDS` | `6` | research loop budget |
| `AGENT_MAX_REVISIONS` | `1` | critic → rewrite rounds |

## AI assistance

Built with Claude Code (as encouraged by the assignment). The AI coding configuration is versioned
in [`.claude/`](.claude/) and [`CLAUDE.md`](CLAUDE.md). Every file was reviewed and I can walk
through any design decision.
