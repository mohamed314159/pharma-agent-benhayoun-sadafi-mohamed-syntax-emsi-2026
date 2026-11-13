"""Orchestration: decompose -> route (JSON route + confidence) -> RAG agents -> comparator -> synthesizer.

Same architecture as the multi-source RAG assistant I designed during a previous internship,
adapted to this corpus: the two agents are the DailyMed agent (FDA labels) and the MedlinePlus
agent (consumer monographs), plus a glossary agent.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
load_dotenv(ROOT / ".env")


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)  # `docker --env-file` keeps quotes, so strip them
    return value.strip().strip("'\"") if value else default


# The provided .env names the proxy URL OPENAI_BASE_UR (missing L): accept both spellings.
client = OpenAI(api_key=env("OPENAI_API_KEY"), base_url=env("OPENAI_BASE_URL") or env("OPENAI_BASE_UR"), max_retries=3)
CHAT_MODEL = env("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

ALPHA, CANDIDATES, TOP_K, MIN_CONFIDENCE = 0.75, 50, 5, 0.5  # 75% semantic / 25% keyword, 50 -> rerank -> 5
DISCLAIMER = ("⚠️ Generated from a limited set of FDA labels and MedlinePlus monographs for information only. "
              "This is not medical advice: consult a qualified healthcare professional.")
CATALOG = {  # summary of each route, the only thing the router sees
    "dailymed": "FDA drug labels (authoritative): boxed warning, indications, dosage and renal/hepatic adjustments, "
                "contraindications, warnings, drug interactions, specific populations, pharmacology, numeric thresholds.",
    "medlineplus": "MedlinePlus consumer monographs: why it is prescribed, how to take it, precautions, "
                   "what to tell the doctor, side effects, storage, overdose — in plain language.",
    "glossary": "Definitions of pharmacology terms and abbreviations (INR, CYP, SSRI, NSAID, DOAC, half-life...).",
}

# ----------------------------------------------------------------------------- LLM helpers
def chat(system: str, user: str, as_json: bool = False):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **({"response_format": {"type": "json_object"}} if as_json else {}),
    )
    content = response.choices[0].message.content or ""
    if not as_json:
        return content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), 256):
        vectors += [d.embedding for d in client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 256]).data]
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ----------------------------------------------------------------------------- in-memory hybrid index
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Index:
    def __init__(self):
        data = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = data["chunks"]
        texts = [f"{c['title']}\n{c['text']}" for c in self.chunks]
        same_model = data["embedding_model"] == EMBED_MODEL
        self.vectors = np.load(INDEX_DIR / "embeddings.npy").astype(np.float32) if same_model else embed(texts)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.drugs = sorted({c["file"].split("-", 1)[1][:-3] for c in self.chunks if "-" in c["file"]})

    def hybrid_search(self, query: str, collection: str) -> list[dict]:
        """Weaviate-style hybrid search: min-max normalised scores, 0.75 x semantic + 0.25 x BM25."""
        ids = np.array([i for i, c in enumerate(self.chunks) if c["collection"] == collection])
        semantic = self.vectors[ids] @ embed([query])[0]
        keyword = np.array(self.bm25.get_batch_scores(tokenize(query), ids))
        norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        scores = ALPHA * norm(semantic) + (1 - ALPHA) * norm(keyword)
        return [self.chunks[ids[j]] for j in np.argsort(-scores)[:CANDIDATES]]


@cache
def index() -> Index:
    return Index()


# ----------------------------------------------------------------------------- prompts
PLANNER = """You orchestrate a pharmacology research assistant. Corpus drugs: {drugs}.
Routes (catalog summary):
{catalog}

1. Decompose the user question into self-contained sub-questions (just one if it is not decomposable).
   A question about several drugs (comparison, interaction) gets at least one sub-question PER DRUG.
2. Classify each sub-question: give every useful route with a confidence score between 0 and 1.
Return JSON: {{"language": "<language of the question>", "in_scope": true, "out_of_scope_reply": null,
 "sub_questions": [{{"question": "...", "routes": [{{"route": "dailymed", "confidence": 0.9}}]}}]}}
Set in_scope=false only if the question is not about medicines (then write a short polite out_of_scope_reply)."""

RERANKER = """Rate how useful each numbered excerpt is to answer the question, from 0 (irrelevant) to 10 (answers it directly).
Return JSON {"scores": [...]} with exactly one integer per excerpt, in the same order."""

AGENT = """You are the {route} agent. Answer the question using ONLY the excerpts below.
Cite the source file in brackets after each fact, e.g. [dailymed-warfarin.md]. Keep numbers and units exact.
If the excerpts do not contain the answer, reply exactly: NOT FOUND."""

COMPARATOR = """You compare the answers given by several agents to the same sub-question, each with the excerpts it used
and their reranker scores (0-10, same scale for every agent).
- Check whether each answer actually answers the sub-question.
- Check every claim against the excerpts: flag claims that are not supported, and clinically ambiguous points
  (a threshold given without its conditions, a generalisation, a missing caveat).
- If both answers are supported and consistent, keep both. If they contradict each other, keep the most reliable one
  (the FDA label / the better reranker scores).
Return JSON ("kept" uses the exact route names: dailymed, medlineplus, glossary): {"kept": ["dailymed"], "contradiction": false,
 "issues": [{"claim": "...", "type": "unsupported|ambiguous|contradiction", "detail": "..."}],
 "verified_answer": "<answer to the sub-question with only the supported claims, keeping the [file] citations>"}"""

SYNTHESIZER = """Answer the user's question from the verified sub-answers only. Do not add outside knowledge.
Keep the [file] citations after each fact. Start with a direct 1-2 sentence answer, then details with markdown
(a table for comparisons). If issues were flagged, add a short "Points to verify" section.
Do not write a medical disclaimer (it is added automatically). Write in {language}."""


# ----------------------------------------------------------------------------- agents
def rag_agent(route: str, question: str) -> dict:
    candidates = index().hybrid_search(question, route)
    excerpts = "\n\n".join(f"[{i}] {c['file']} › {c['section']} — {c['title']}\n{c['text']}" for i, c in enumerate(candidates))
    scores = chat(RERANKER, f"QUESTION: {question}\n\nEXCERPTS:\n{excerpts}", as_json=True).get("scores", [])
    if len(scores) != len(candidates):  # malformed reranker output: keep the hybrid order
        scores = list(range(len(candidates), 0, -1))
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])[:TOP_K]
    top = [candidates[i] | {"rerank_score": scores[i]} for i in order]
    context = "\n\n".join(f"{c['file']} › {c['section']} — {c['title']}\n{c['text']}" for c in top)
    answer = chat(AGENT.format(route=route), f"QUESTION: {question}\n\nEXCERPTS:\n{context}")
    return {"answer": answer, "chunks": top}


def answer_sub_question(sub: dict) -> dict:
    routes = [r for r in sub.get("routes", []) if r.get("route") in CATALOG]
    chosen = [r["route"] for r in routes if r.get("confidence", 0) >= MIN_CONFIDENCE]
    chosen = chosen or [max(routes, key=lambda r: r.get("confidence", 0))["route"] if routes else "dailymed"]
    with ThreadPoolExecutor() as pool:
        agents = dict(zip(chosen, pool.map(lambda r: rag_agent(r, sub["question"]), chosen)))
    report = "\n\n".join(
        f"=== {route} agent ===\nANSWER: {a['answer']}\nEXCERPTS:\n"
        + "\n".join(f"(reranker {c['rerank_score']}) {c['file']} › {c['section']}: {c['text']}" for c in a["chunks"])
        for route, a in agents.items()
    )
    review = chat(COMPARATOR, f"SUB-QUESTION: {sub['question']}\n\n{report}", as_json=True)
    return {"question": sub["question"], "routes": routes, "agents": agents, "review": review}


def ask(question: str) -> dict:
    start = time.time()
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in CATALOG.items())
    plan = chat(PLANNER.format(drugs=", ".join(index().drugs), catalog=catalog), question, as_json=True)
    subs = plan.get("sub_questions") or []
    if not plan.get("in_scope", True) or not subs:
        answer, details = plan.get("out_of_scope_reply") or "I can only answer questions about the drugs in my corpus.", []
    else:
        with ThreadPoolExecutor() as pool:
            details = list(pool.map(answer_sub_question, subs))
        verified = "\n\n".join(
            f"SUB-QUESTION: {d['question']}\nVERIFIED ANSWER: {d['review'].get('verified_answer', '')}\n"
            f"ISSUES: {json.dumps(d['review'].get('issues', []), ensure_ascii=False)}" for d in details
        )
        answer = chat(SYNTHESIZER.format(language=plan.get("language", "English")), f"QUESTION: {question}\n\n{verified}")
    cited = set(re.findall(r"\[([\w.-]+\.md)\]", answer))
    sources = {(c["file"], c["section"]): c for d in details for a in d["agents"].values() for c in a["chunks"] if c["file"] in cited}
    return {
        "question": question,
        "answer": f"{answer}\n\n---\n{DISCLAIMER}",
        "sources": [{k: c[k] for k in ("file", "section", "title", "rerank_score")} for c in sources.values()],
        "details": details,
        "disclaimer": DISCLAIMER,
        "latency_s": round(time.time() - start, 1),
    }
