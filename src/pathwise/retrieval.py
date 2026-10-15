"""Hybrid retrieval over the catalog chunks.

Keyword search (BM25) catches exact terms and course codes; dense search (local
bge-small embeddings) catches meaning ("courses about search engines" finds
Information Retrieval). Their rankings are merged with reciprocal rank fusion
(the default). An optional cross-encoder can re-rank the fused results, but on
this catalog it did not beat plain fusion (see evals/results.md). Everything runs locally,
so retrieval costs nothing per query.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from .catalog import Chunk

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CACHE = Path(os.environ.get("PATHWISE_CACHE", Path.home() / ".pathwise" / "cache"))
MODES = ("bm25", "dense", "hybrid", "rerank")
CODE = re.compile(r"\b([A-Za-z]{2})[\s-]?(\d{3})\b")
STOP = set("a an the of to in on for and or is are do does i my me can what which how when who with be it this that you your".split())


class Hit(BaseModel):
    chunk: Chunk
    score: float
    rank: int


def tokenize(text: str) -> list[str]:
    text = CODE.sub(lambda m: f" {m.group(1).lower()}{m.group(2)} ", text)  # "CS 345" -> one token "cs345"
    return [t for t in re.findall(r"[a-z0-9+]+", text.lower()) if t not in STOP]


def mentioned_codes(text: str) -> list[str]:
    return list(dict.fromkeys(f"{a.upper()} {n}" for a, n in CODE.findall(text)))


class LocalEmbedder:
    """bge-small via fastembed (ONNX, CPU). Downloads about 70 MB once."""
    def __init__(self, model: str = EMBED_MODEL):
        from fastembed import TextEmbedding
        self.model_name = model
        self.model = TextEmbedding(model)

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.passage_embed(texts)))

    def query(self, text: str) -> np.ndarray:
        return np.array(list(self.model.query_embed(text)))[0]


class LocalReranker:
    def __init__(self, model: str = RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, texts)]


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder=None, reranker=None, cache: Path | None = CACHE, mentions=mentioned_codes):
        self.chunks = chunks
        self.mentions = mentions  # finds courses a query names; the catalog's version also knows titles and abbreviations
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self._embedder, self._reranker, self._vectors = embedder, reranker, None
        self.cache = cache

    # Models load lazily so keyword-only use (and tests) never touch them.
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = LocalEmbedder()
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = LocalReranker()
        return self._reranker

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            texts = [c.text for c in self.chunks]
            name = getattr(self.embedder, "model_name", type(self.embedder).__name__)
            key = hashlib.sha256(json.dumps([name, texts]).encode()).hexdigest()[:16]
            path = self.cache / f"vectors-{key}.npy" if self.cache else None
            if path and path.exists():
                self._vectors = np.load(path)
            else:
                self._vectors = _normalize(self.embedder.passages(texts))
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, self._vectors)
        return self._vectors

    def _bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [i for i in np.argsort(-scores) if scores[i] > 0]

    def _dense_rank(self, query: str) -> list[int]:
        q = _normalize(self.embedder.query(query))
        return list(np.argsort(-(self.vectors @ q)))

    def search(self, query: str, k: int = 6, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "bm25":
            order = self._bm25_rank(query)
        elif mode == "dense":
            order = self._dense_rank(query)
        else:
            # Reciprocal rank fusion of the two rankings (k = 60, the usual constant).
            fused: dict[int, float] = {}
            for ranking in (self._bm25_rank(query)[:20], self._dense_rank(query)[:20]):
                for r, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r + 1)
            order = sorted(fused, key=lambda i: -fused[i])
            if mode == "rerank":
                pool = order[:15]
                scores = self.reranker.scores(query, [self.chunks[i].text for i in pool])
                order = [i for _, i in sorted(zip(scores, pool), key=lambda x: -x[0])]

        # A course the query names is always in the results: students often ask about one directly.
        named = [i for code in self.mentions(query) for i, c in enumerate(self.chunks) if c.id == f"course:{code}"]
        order = list(dict.fromkeys(named + list(order)))
        return [Hit(chunk=self.chunks[i], score=round(1.0 / (r + 1), 4), rank=r + 1) for r, i in enumerate(order[:k])]
