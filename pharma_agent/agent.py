"""LangGraph orchestration: planner -> one parallel branch per sub-question (tools -> comparator) -> synthesizer.

Same architecture as the multi-source RAG assistant I designed during a previous internship,
adapted to this corpus. Three local tools, chosen by the planner for each sub-question:
  1. search_corpus      hybrid search + reranking in DailyMed and/or MedlinePlus (always called)
  2. read_label_section exact lookup of a whole section of an FDA label (precise drug + section)
  3. glossary_lookup    exact definitions from GLOSSARY.md (when a term must be defined)
"""

import json
import operator
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Annotated, Any, TypedDict

import numpy as np
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import OpenAI
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
load_dotenv(ROOT / ".env")


def env(name: str, default: str = "") -> str:
    value = os.getenv(name)  # `docker --env-file` keeps quotes, so strip them
    return value.strip().strip("'\"") if value else default


# The provided .env names the proxy URL OPENAI_BASE_UR (missing L): accept both spellings.
client = OpenAI(api_key=env("OPENAI_API_KEY") or None, base_url=env("OPENAI_BASE_URL") or env("OPENAI_BASE_UR") or None,
                max_retries=3)
CHAT_MODEL = env("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

ALPHA, CANDIDATES, TOP_K, MIN_CONFIDENCE = 0.75, 50, 5, 0.5  # 75% semantic / 25% keyword, 50 -> rerank -> 5
MAX_SECTION_CHARS = 8000  # read_label_section output cap
DISCLAIMER = ("⚠️ Generated from a limited set of FDA labels and MedlinePlus monographs for information only. "
              "This is not medical advice: consult a qualified healthcare professional.")
CATALOG = {  # summary of each search_corpus source, the only thing the planner sees
    "dailymed": "FDA drug labels (authoritative): boxed warning, indications, dosage and renal/hepatic adjustments, "
                "contraindications, warnings, drug interactions, specific populations, pharmacology, numeric thresholds.",
    "medlineplus": "MedlinePlus consumer monographs: why it is prescribed, how to take it, precautions, "
                   "what to tell the doctor, side effects, storage, overdose — in plain language.",
}

# ----------------------------------------------------------------------------- LLM helpers
def chat(system: str, user: str, as_json: bool = False) -> Any:
    """Plain text, or a dict parsed from the model's JSON (`{}` if malformed) when as_json."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"} if as_json else {"type": "text"},
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
        corpus = ROOT / "corpus"
        self.sections = {  # (drug, heading) -> text of each `##` section of the FDA labels
            (path.stem.split("-", 1)[1], heading.strip()): body.strip()
            for path in sorted(corpus.glob("dailymed-*.md"))
            for heading, body in re.findall(r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", path.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL)
        }
        self.section_names = sorted({re.sub(r"^[\d.]+ ", "", h).upper() for _, h in self.sections if not h.startswith("WARNING:")})
        self.section_names.append("BOXED WARNING")
        glossary = re.findall(r"^- \*\*(.+?)\*\*: (.+(?:\n {2}.+)*)", (corpus / "GLOSSARY.md").read_text(encoding="utf-8"), re.MULTILINE)
        self.glossary = {term: re.sub(r"\s+", " ", definition) for term, definition in glossary}

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
Tools:
1. search_corpus — ALWAYS called: hybrid search + reranking in one or more sources:
{catalog}
2. read_label_section(drug, section) — returns a whole section of a drug's FDA label, word for word.
   Call it ONLY when the sub-question targets a precise drug AND a precise section. Sections: {sections}.
3. glossary_lookup(terms) — exact definitions from the glossary. Call it ONLY when a term must be defined
   to answer (e.g. INR, SSRI, NSAID, DOAC, CYP, half-life).

1. Decompose the user question into self-contained sub-questions (just one if it is not decomposable).
   A question about several drugs (comparison, interaction) gets at least one sub-question PER DRUG.
2. For each sub-question: give every useful search_corpus source with a confidence score between 0 and 1,
   the label sections to read (often empty) and the glossary terms (often empty).
Return JSON: {{"language": "<language of the question>", "in_scope": true, "out_of_scope_reply": null,
 "sub_questions": [{{"question": "...", "routes": [{{"route": "dailymed", "confidence": 0.9}}],
  "sections": [{{"drug": "warfarin", "section": "CONTRAINDICATIONS"}}], "terms": []}}]}}
Set in_scope=false only if the question is not about medicines (then write a short polite out_of_scope_reply)."""

RERANKER = """Rate how useful each numbered excerpt is to answer the question, from 0 (irrelevant) to 10 (answers it directly).
Return JSON {"scores": [...]} with exactly one integer per excerpt, in the same order."""

AGENT = """You are the {route} agent. Answer the question using ONLY the excerpts below.
Cite the source file in brackets after each fact, e.g. [dailymed-warfarin.md]. Keep numbers and units exact.
If the excerpts do not contain the answer, reply exactly: NOT FOUND."""

COMPARATOR = """You compare the answers given by several search agents to the same sub-question, each with the excerpts it used
and their reranker scores (0-10, same scale for every agent). You may also get EXACT EXCERPTS returned by
read_label_section (whole FDA label sections) and glossary_lookup (definitions): they are verbatim and authoritative.
- Check whether each answer actually answers the sub-question; complete it with facts from the exact excerpts if needed.
- Check every claim against the excerpts: flag claims that are not supported, and clinically ambiguous points
  (a threshold given without its conditions, a generalisation, a missing caveat).
- If both answers are supported and consistent, keep both. If they contradict each other, keep the most reliable one
  (the FDA label / the better reranker scores).
Return JSON ("kept" uses the exact route names: dailymed, medlineplus): {"kept": ["dailymed"], "contradiction": false,
 "issues": [{"claim": "...", "type": "unsupported|ambiguous|contradiction", "detail": "..."}],
 "verified_answer": "<answer to the sub-question with only the supported claims, keeping the [file] citations>"}"""

SYNTHESIZER = """Answer the user's question from the verified sub-answers only. Do not add outside knowledge.
Keep the [file] citations after each fact. Start with a direct 1-2 sentence answer, then details with markdown
(a table for comparisons). If issues were flagged, add a short "Points to verify" section.
Do not write a medical disclaimer (it is added automatically). Write in {language}."""


# ----------------------------------------------------------------------------- tools
def search_corpus(route: str, question: str) -> dict:
    """Tool 1: hybrid search (50 candidates) -> LLM reranker -> top 5 -> cited answer."""
    candidates = index().hybrid_search(question, route)
    excerpts = "\n\n".join(f"[{i}] {c['file']} › {c['section']} — {c['title']}\n{c['text']}" for i, c in enumerate(candidates))
    scores = chat(RERANKER, f"QUESTION: {question}\n\nEXCERPTS:\n{excerpts}", as_json=True).get("scores", [])
    if len(scores) != len(candidates):  # malformed reranker output: keep the hybrid order
        scores = list(range(len(candidates), 0, -1))
    retrieved = [c | {"rerank_score": s, "kept": False} for c, s in zip(candidates, scores)]  # hybrid order
    top = sorted(retrieved, key=lambda c: -c["rerank_score"])[:TOP_K]
    for c in top:
        c["kept"] = True  # same dicts as in `retrieved`: flags them there too
    context = "\n\n".join(f"{c['file']} › {c['section']} — {c['title']}\n{c['text']}" for c in top)
    answer = chat(AGENT.format(route=route), f"QUESTION: {question}\n\nEXCERPTS:\n{context}")
    return {"answer": answer, "chunks": top, "retrieved": retrieved}


def read_label_section(drug: str, section: str) -> dict:
    """Tool 2: exact lookup of a whole section of an FDA label (no similarity search)."""
    wanted = "WARNING:" if section.upper() == "BOXED WARNING" else section.upper()
    hits = [f"## {h}\n{t}" for (d, h), t in index().sections.items() if d == drug.lower() and wanted in h.upper()]
    return {"drug": drug, "section": section, "file": f"dailymed-{drug.lower()}.md",
            "text": "\n\n".join(hits)[:MAX_SECTION_CHARS] or "NOT FOUND"}


def glossary_lookup(terms: list[str]) -> list[dict]:
    """Tool 3: exact definitions from GLOSSARY.md (case-insensitive match on the defined term)."""
    return [{"term": k, "definition": v} for k, v in index().glossary.items() if any(t and t.lower() in k.lower() for t in terms)]


# ----------------------------------------------------------------------------- agents
def answer_sub_question(sub: dict) -> dict:
    routes = [r for r in sub.get("routes", []) if r.get("route") in CATALOG]
    chosen = [r["route"] for r in routes if r.get("confidence", 0) >= MIN_CONFIDENCE]
    chosen = chosen or [max(routes, key=lambda r: r.get("confidence", 0))["route"] if routes else "dailymed"]
    with ThreadPoolExecutor() as pool:  # tool 1 is always called, on every chosen source
        agents = dict(zip(chosen, pool.map(lambda r: search_corpus(r, sub["question"]), chosen)))
    sections = [read_label_section(str(s.get("drug", "")), str(s.get("section", ""))) for s in sub.get("sections") or []]
    glossary = glossary_lookup([str(t) for t in sub.get("terms") or []])
    report = "\n\n".join(
        f"=== search_corpus({route}) agent ===\nANSWER: {a['answer']}\nEXCERPTS:\n"
        + "\n".join(f"(reranker {c['rerank_score']}) {c['file']} › {c['section']}: {c['text']}" for c in a["chunks"])
        for route, a in agents.items()
    )
    report += "".join(f"\n\n=== read_label_section({s['drug']}, {s['section']}) [{s['file']}] ===\n{s['text']}" for s in sections)
    if glossary:
        report += "\n\n=== glossary_lookup [GLOSSARY.md] ===\n" + "\n".join(f"{g['term']}: {g['definition']}" for g in glossary)
    review = chat(COMPARATOR, f"SUB-QUESTION: {sub['question']}\n\n{report}", as_json=True)
    return {"question": sub["question"], "routes": routes, "agents": agents, "sections": sections, "glossary": glossary,
            "review": review}


# ----------------------------------------------------------------------------- LangGraph orchestration
class State(TypedDict):
    question: str
    plan: dict
    details: Annotated[list[dict], operator.add]  # one entry per sub-question, appended by the parallel branches
    answer: str


class Branch(TypedDict):  # input of one parallel sub_question branch
    sub: dict


def planner(state: State) -> dict:
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in CATALOG.items())
    prompt = PLANNER.format(drugs=", ".join(index().drugs), catalog=catalog, sections=", ".join(index().section_names))
    return {"plan": chat(prompt, state["question"], as_json=True)}


def fan_out(state: State) -> list[Send] | str:
    """One parallel branch per sub-question, or straight to the synthesizer when out of scope."""
    subs = state["plan"].get("sub_questions") or []
    if not state["plan"].get("in_scope", True) or not subs:
        return "synthesizer"
    return [Send("sub_question", Branch(sub=sub)) for sub in subs]


def sub_question(state: Branch) -> dict:
    return {"details": [answer_sub_question(state["sub"])]}  # tools -> comparator


def synthesizer(state: State) -> dict:
    if not state["details"]:
        return {"answer": state["plan"].get("out_of_scope_reply") or "I can only answer questions about the drugs in my corpus."}
    verified = "\n\n".join(
        f"SUB-QUESTION: {d['question']}\nVERIFIED ANSWER: {d['review'].get('verified_answer', '')}\n"
        f"ISSUES: {json.dumps(d['review'].get('issues', []), ensure_ascii=False)}" for d in state["details"]
    )
    prompt = SYNTHESIZER.format(language=state["plan"].get("language", "English"))
    return {"answer": chat(prompt, f"QUESTION: {state['question']}\n\n{verified}")}


builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("sub_question", sub_question)
builder.add_node("synthesizer", synthesizer)
builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", fan_out, ["sub_question", "synthesizer"])
builder.add_edge("sub_question", "synthesizer")
builder.add_edge("synthesizer", END)
graph = builder.compile()  # graph.get_graph().draw_mermaid() prints the diagram


def ask(question: str) -> dict:
    start = time.time()
    result = graph.invoke(State(question=question, plan={}, details=[], answer=""))
    answer, details = result["answer"], result["details"]
    cited = set(re.findall(r"\[([\w.-]+\.md)\]", answer))
    sources = {(c["file"], c["section"]): c for d in details for a in d["agents"].values() for c in a["chunks"] if c["file"] in cited}
    for d in details:  # exact excerpts of tools 2 and 3, when the answer cites them
        for s in d["sections"]:
            if s["file"] in cited and s["text"] != "NOT FOUND":
                sources[(s["file"], s["section"])] = {"file": s["file"], "section": s["section"], "title": "whole section (read_label_section)"}
        if "GLOSSARY.md" in cited and d["glossary"]:
            sources[("GLOSSARY.md", d["question"])] = {"file": "GLOSSARY.md", "section": ", ".join(g["term"] for g in d["glossary"]),
                                                      "title": "definitions (glossary_lookup)"}
    return {
        "question": question,
        "answer": f"{answer}\n\n---\n{DISCLAIMER}",
        "sources": [{k: c.get(k) for k in ("file", "section", "title", "rerank_score")} for c in sources.values()],
        "details": details,
        "disclaimer": DISCLAIMER,
        "latency_s": round(time.time() - start, 1),
    }
