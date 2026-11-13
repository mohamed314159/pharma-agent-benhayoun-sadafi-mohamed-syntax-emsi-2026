"""Local tools exposed to the agent through OpenAI function calling.

Every passage a tool returns is registered in an `EvidenceStore` and tagged with
a stable id (`[E3]`). The writer must cite those ids and the critic checks
claims against them, so grounding is enforced end-to-end.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .corpus import Document, Section
from .index import HybridIndex, tokenize

MAX_SECTION_CHARS = 9000


@dataclass
class Evidence:
    id: str
    file: str
    section: str
    text: str

    def render(self) -> str:
        return f"[{self.id}] {self.file} › {self.section}\n{self.text}"


@dataclass
class EvidenceStore:
    items: dict[str, Evidence] = field(default_factory=dict)
    _seen: dict[tuple[str, str, int], str] = field(default_factory=dict)

    def add(self, file: str, section: str, text: str) -> Evidence:
        key = (file, section, hash(text))
        if key in self._seen:
            return self.items[self._seen[key]]
        ev = Evidence(f"E{len(self.items) + 1}", file, section, text)
        self.items[ev.id] = ev
        self._seen[key] = ev.id
        return ev


def _stem(token: str) -> str:
    return token[:6]


class Toolbox:
    """Tools bound to one corpus/index and one request's evidence store."""

    def __init__(self, docs: list[Document], index: HybridIndex, store: EvidenceStore):
        self.docs = {(d.source, d.drug): d for d in docs}
        self.drugs = sorted({d.drug for d in docs if d.source != "glossary"})
        self.index = index
        self.store = store
        self.calls: list[dict[str, Any]] = []
        self._impl: dict[str, Callable[..., str]] = {
            "search_corpus": self.search_corpus,
            "get_section": self.get_section,
            "list_sections": self.list_sections,
            "lookup_glossary": self.lookup_glossary,
        }

    # ------------------------------------------------------------------ schemas
    def schemas(self) -> list[dict[str, Any]]:
        drug_enum = {"type": "string", "enum": self.drugs}
        source_enum = {"type": "string", "enum": ["dailymed", "medlineplus"]}

        def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }

        return [
            fn(
                "search_corpus",
                "Hybrid semantic+keyword search over all drug documents. Best for open questions, "
                "cross-drug facts, or when you don't know which section holds the answer.",
                {
                    "query": {"type": "string", "description": "Focused search query."},
                    "drugs": {"type": "array", "items": drug_enum, "description": "Restrict to these drugs."},
                    "source": source_enum | {"description": "dailymed = FDA label (authoritative); medlineplus = consumer info."},
                    "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                ["query"],
            ),
            fn(
                "get_section",
                "Return the FULL text of a named section of a drug document (with its sub-sections), "
                "e.g. 'contraindications', 'drug interactions', 'renal impairment', 'boxed warning'. "
                "Prefer this for exhaustive lists like contraindications.",
                {
                    "drug": drug_enum,
                    "section": {"type": "string", "description": "Section name or keywords."},
                    "source": source_enum | {"default": "dailymed"},
                },
                ["drug", "section"],
            ),
            fn(
                "list_sections",
                "List the section headings (table of contents) of a drug document.",
                {"drug": drug_enum, "source": source_enum | {"default": "dailymed"}},
                ["drug"],
            ),
            fn(
                "lookup_glossary",
                "Define a pharmacology term or abbreviation (INR, CYP, SSRI, eGFR...) from GLOSSARY.md.",
                {"term": {"type": "string"}},
                ["term"],
            ),
        ]

    # ---------------------------------------------------------------- dispatch
    def run(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return f"ERROR: arguments are not valid JSON: {raw_args!r}"
        fn = self._impl.get(name)
        if fn is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            result = fn(**args)
        except TypeError as exc:
            result = f"ERROR: bad arguments for {name}: {exc}"
        self.calls.append({"tool": name, "args": args, "preview": result[:300]})
        return result

    # ------------------------------------------------------------------- tools
    def search_corpus(self, query: str, drugs: list[str] | None = None, source: str | None = None, k: int = 5) -> str:
        hits = self.index.search(query, k=max(1, min(int(k), 10)), drugs=drugs, sources=[source] if source else None)
        if not hits:
            return "No passages found. Try broader terms or remove filters."
        return "\n\n".join(self.store.add(h.chunk.file, h.chunk.section, h.chunk.text).render() for h in hits)

    def _doc(self, drug: str, source: str) -> Document | None:
        return self.docs.get((source.lower(), drug.lower()))

    def get_section(self, drug: str, section: str, source: str = "dailymed") -> str:
        doc = self._doc(drug, source)
        if doc is None:
            return f"ERROR: no {source} document for {drug!r}. Known drugs: {', '.join(self.drugs)}"
        q = {_stem(t) for t in tokenize(section)}

        def score(sec: Section) -> float:
            head = sec.path[-1] if sec.path else ""
            h = {_stem(t) for t in tokenize(head)}
            full = {_stem(t) for t in tokenize(sec.title)}
            # leaf-heading overlap matters most; shallower headings win ties
            return 2 * len(q & h) + len(q & full) - 0.1 * len(sec.path)

        best = max(doc.sections, key=score)
        if score(best) <= 0:
            return f"No section matching {section!r}. Use list_sections to see headings."
        prefix = best.path
        matched = [s for s in doc.sections if s.path[: len(prefix)] == prefix]
        out, total = [], 0
        for s in matched:
            if total > MAX_SECTION_CHARS:
                out.append("[... section truncated, use search_corpus for the rest ...]")
                break
            out.append(self.store.add(doc.file, s.title, s.text).render())
            total += len(s.text)
        return "\n\n".join(out)

    def list_sections(self, drug: str, source: str = "dailymed") -> str:
        doc = self._doc(drug, source)
        if doc is None:
            return f"ERROR: no {source} document for {drug!r}."
        seen: list[str] = []
        for s in doc.sections:
            if s.title not in seen:
                seen.append(s.title)
        return f"{doc.file}:\n" + "\n".join(f"- {t}" for t in seen)

    def lookup_glossary(self, term: str) -> str:
        doc = self._doc("glossary", "glossary")
        if doc is None:
            return "Glossary not available."
        pattern = re.compile(re.escape(term.strip()), re.IGNORECASE)
        lines = [
            line.strip()
            for s in doc.sections
            for line in s.text.splitlines()
            if pattern.search(line)
        ]
        if not lines:
            return f"No glossary entry for {term!r}."
        ev = self.store.add(doc.file, f"Glossary: {term}", "\n".join(lines))
        return ev.render()
