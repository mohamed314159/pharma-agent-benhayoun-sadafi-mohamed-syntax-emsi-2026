"""In-memory hybrid retriever: dense cosine similarity + BM25, fused with RRF.

Dense embeddings capture paraphrases ("blood thinner" ~ "anticoagulant") while
BM25 nails exact tokens that embeddings blur (drug names, "eGFR", "CYP2C9",
numeric thresholds). Reciprocal Rank Fusion merges both rankings without
having to calibrate their scores.

Embeddings are cached on disk (keyed by content hash + model) so restarts are
instant; the index itself lives purely in memory (numpy).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import llm
from .corpus import Chunk

log = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = set(
    "a an and are as at be by for from has have if in into is it its of on or such that the their then "
    "there these they this to was were will with what which who how does do can should not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            out[i] = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf
            )
        return out


class HybridIndex:
    def __init__(self, chunks: list[Chunk], cache_dir: Path, model: str):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.embed_text) for c in chunks])
        self.vectors = self._load_embeddings(cache_dir, model)

    def _load_embeddings(self, cache_dir: Path, model: str) -> np.ndarray:
        keys = [hashlib.sha1(f"{model}\n{c.embed_text}".encode()).hexdigest() for c in self.chunks]
        cache_file = cache_dir / "embeddings.npz"
        cache: dict[str, np.ndarray] = {}
        if cache_file.exists():
            data = np.load(cache_file)
            cache = dict(zip(data["keys"].tolist(), data["vecs"]))
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            log.info("Embedding %d/%d chunks...", len(missing), len(keys))
            vecs = llm.embed([self.chunks[i].embed_text for i in missing])
            for i, v in zip(missing, vecs):
                cache[keys[i]] = np.asarray(v, dtype=np.float32)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, keys=np.array(keys), vecs=np.stack([cache[k] for k in keys]))
        mat = np.stack([cache[k] for k in keys]).astype(np.float32)
        return mat / np.linalg.norm(mat, axis=1, keepdims=True)

    def search(
        self,
        query: str,
        k: int = 6,
        drugs: list[str] | None = None,
        sources: list[str] | None = None,
        rrf_k: int = 60,
    ) -> list[Hit]:
        mask = np.ones(len(self.chunks), dtype=bool)
        if drugs:
            wanted = {d.lower() for d in drugs}
            mask &= np.array([c.drug in wanted for c in self.chunks])
        if sources:
            wanted = {s.lower() for s in sources}
            mask &= np.array([c.source in wanted for c in self.chunks])
        if not mask.any():
            return []

        q = np.asarray(llm.embed([query])[0], dtype=np.float32)
        dense = self.vectors @ (q / np.linalg.norm(q))
        sparse = self.bm25.scores(tokenize(query))

        idx = np.flatnonzero(mask)
        fused = np.zeros(len(self.chunks))
        for scores in (dense, sparse):
            order = idx[np.argsort(-scores[idx])]
            for rank, i in enumerate(order):
                fused[i] += 1.0 / (rrf_k + rank + 1)
        best = idx[np.argsort(-fused[idx])][:k]
        return [Hit(self.chunks[i], float(fused[i])) for i in best]
