"""Ingestion: semantic chunking, section by section (the method I specified during a previous internship).

For each `##` section of each document:
 1. split into sentences with spaCy
 2. embed every sentence
 3-4. walk the sentences: same chunk as the previous sentence if cosine > threshold, else new chunk;
      the threshold is found by binary search so that chunks average 500 characters
 5. chunks > 600 chars are cut into ceil(size/500) pieces with 10% overlap
 6. chunks < 300 chars are merged with the previous chunk, else the next one, if the result stays <= 600
 7. the LLM writes a title/context of at most 10 words for each chunk; title + chunk are embedded

Run once: `python -m pharma_agent.ingest` -> index/chunks.json + index/embeddings.npy
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import spacy

from .agent import EMBED_MODEL, INDEX_DIR, ROOT, chat, embed

TARGET, MAX, MIN = 500, 600, 300
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


def sections(path):
    """Yield (section title, lines) for each `##` section of a markdown file."""
    current, lines = "Introduction", []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if lines:
                yield current, lines
            current, lines = line[3:].strip(), []
        elif line.strip() and not line.startswith("# "):
            lines.append(line.lstrip("#").strip())
    if lines:
        yield current, lines


def group(sentences, cosines, threshold):
    chunks = [sentences[0]]
    for sentence, cos in zip(sentences[1:], cosines):
        if cos > threshold:
            chunks[-1] += " " + sentence
        else:
            chunks.append(sentence)
    return chunks


def semantic_chunks(sentences, vectors):
    if sum(map(len, sentences)) <= MAX:
        return [" ".join(sentences)]
    cosines = (vectors[1:] * vectors[:-1]).sum(axis=1)
    low, high = cosines.min() - 1e-6, cosines.max() + 1e-6
    for _ in range(25):  # binary search: higher threshold -> more splits -> smaller chunks
        threshold = (low + high) / 2
        if np.mean([len(c) for c in group(sentences, cosines, threshold)]) > TARGET:
            low = threshold
        else:
            high = threshold
    return group(sentences, cosines, (low + high) / 2)


def split_big(chunks):
    out = []
    for chunk in chunks:
        if len(chunk) <= MAX:
            out.append(chunk)
            continue
        pieces = math.ceil(len(chunk) / TARGET)
        step = len(chunk) / pieces
        overlap = int(step * 0.10)
        out += [chunk[max(0, int(i * step) - overlap): int((i + 1) * step)] for i in range(pieces)]
    return out


def merge_small(chunks):
    merged = True
    while merged:
        merged = False
        for i, chunk in enumerate(chunks):
            if len(chunk) >= MIN:
                continue
            if i > 0 and len(chunks[i - 1]) + len(chunk) + 1 <= MAX:
                chunks[i - 1] += " " + chunk
            elif i + 1 < len(chunks) and len(chunk) + len(chunks[i + 1]) + 1 <= MAX:
                chunks[i + 1] = chunk + " " + chunks[i + 1]
            else:
                continue
            del chunks[i]
            merged = True
            break
    return chunks


def build():
    sections_ = []  # (file, section, sentences)
    for path in sorted((ROOT / "corpus").glob("*.md")):
        for title, lines in sections(path):
            sentences = [s.text.strip() for doc in nlp.pipe(lines) for s in doc.sents if s.text.strip()]
            sections_.append((path.name, title, sentences))

    vectors = embed([s for _, _, sentences in sections_ for s in sentences])  # all sentences in one go
    chunks, offset = [], 0
    for file, section, sentences in sections_:
        section_vectors = vectors[offset: offset + len(sentences)]
        offset += len(sentences)
        collection = file.split("-")[0].lower() if "-" in file else "glossary"
        for text in merge_small(split_big(semantic_chunks(sentences, section_vectors))):
            chunks.append({"file": file, "collection": collection, "section": section, "text": text})

    def title(chunk):
        prompt = f"Write a title/context of at most 10 words for this excerpt of {chunk['file']} (section {chunk['section']}). Reply with the title only."
        try:
            return chat(prompt, chunk["text"]).strip('"')[:150]
        except Exception:  # a refused/failed call must not break the ingestion
            return f"{chunk['file'][:-3]} — {chunk['section']}"

    with ThreadPoolExecutor(8) as pool:
        for chunk, t in zip(chunks, pool.map(title, chunks)):
            chunk["title"] = t

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embed([f"{c['title']}\n{c['text']}" for c in chunks]).astype(np.float16))
    payload = {"embedding_model": EMBED_MODEL, "chunks": chunks}
    (INDEX_DIR / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = [len(c["text"]) for c in chunks]
    print(f"{len(sections_)} sections -> {len(chunks)} chunks, mean {np.mean(sizes):.0f} chars, min {min(sizes)}, max {max(sizes)}")


if __name__ == "__main__":
    build()
