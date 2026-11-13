"""Load the markdown corpus and split it into section-aware chunks.

Drug labels are highly structured (`## 4 CONTRAINDICATIONS`, `### 7.3 Drugs that
Increase Bleeding Risk`...), so we chunk along headings rather than by fixed
size: a chunk never mixes two sections, and every chunk carries its heading
path, which makes citations precise and retrieval much sharper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$")
MAX_CHUNK_CHARS = 1800


@dataclass
class Section:
    file: str
    drug: str
    source: str  # "dailymed" | "medlineplus" | "glossary"
    path: list[str]  # heading path, e.g. ["4 CONTRAINDICATIONS"]
    text: str

    @property
    def title(self) -> str:
        return " > ".join(self.path) if self.path else "(intro)"


@dataclass
class Chunk:
    id: int
    file: str
    drug: str
    source: str
    section: str
    text: str
    embed_text: str = field(repr=False, default="")


@dataclass
class Document:
    file: str
    drug: str
    source: str
    title: str
    sections: list[Section]


def _parse_name(path: Path) -> tuple[str, str]:
    stem = path.stem.lower()
    if stem == "glossary":
        return "glossary", "glossary"
    source, _, drug = stem.partition("-")
    return source, drug


def parse_document(path: Path) -> Document:
    source, drug = _parse_name(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    title = path.stem
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(path.name, drug, source, [h for _, h in stack], text))
        buf.clear()

    for line in lines:
        m = HEADING_RE.match(line)
        if not m:
            buf.append(line)
            continue
        level, heading = len(m.group(1)), m.group(2)
        if level == 1:
            title = heading
            continue
        flush()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
    flush()
    return Document(path.name, drug, source, title, sections)


def _split(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split long text on paragraph (then line) boundaries."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|\n(?=[-*|] )", text):
        if cur and len(cur) + len(para) > limit:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def load_corpus(corpus_dir: Path) -> tuple[list[Document], list[Chunk]]:
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"No .md/.txt files found in {corpus_dir}")
    docs = [parse_document(p) for p in files]
    chunks: list[Chunk] = []
    for doc in docs:
        for sec in doc.sections:
            for piece in _split(sec.text):
                header = f"{doc.title} — {sec.title}"
                chunks.append(
                    Chunk(len(chunks), doc.file, doc.drug, doc.source, sec.title, piece, f"{header}\n{piece}")
                )
    return docs, chunks
