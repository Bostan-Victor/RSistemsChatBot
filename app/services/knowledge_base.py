from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


_DIACRITICS_TABLE = str.maketrans(
    {
        "ă": "a",
        "â": "a",
        "î": "i",
        "ș": "s",
        "ş": "s",
        "ț": "t",
        "ţ": "t",
        "Ă": "a",
        "Â": "a",
        "Î": "i",
        "Ș": "s",
        "Ş": "s",
        "Ț": "t",
        "Ţ": "t",
    }
)


def _normalize_text(text: str) -> str:
    return text.translate(_DIACRITICS_TABLE).lower()


# Minimal, pragmatic synonyms/normalization for HoReCa/RSistems domain.
# Canonical tokens are chosen to match what exists in the KB markdown.
_TOKEN_SYNONYMS: dict[str, str] = {
    # inventory / stock
    "inventar": "stocuri",
    "marfa": "stocuri",
    "marfuri": "stocuri",
    "stoc": "stocuri",
    "depozit": "stocuri",
    "pierderi": "pierderilor",
    # POS / fiscal
    "pos": "pos",
    "fiscalizare": "fiscale",
    # reports
    "raport": "rapoarte",
    "rapoarte": "rapoarte",
    "analiza": "analize",
    "analize": "analize",
    "statistica": "rapoarte",
    "statistici": "rapoarte",
    # staff
    "chelner": "ospatar",
    "chelneri": "ospatar",
    "ospatar": "ospatar",
    "ospatari": "ospatar",
    "angajati": "personal",
    # kitchen / delivery
    "bucatarie": "kds",
    "kds": "kds",
    "livrare": "delivery",
    "livrari": "delivery",
    "curier": "delivery",
    "curieri": "delivery",

    # pricing
    "pret": "preturi",
    "pretul": "preturi",
    "pretului": "preturi",
    "preturi": "preturi",
    "preturile": "preturi",
    "cost": "preturi",
    "costa": "preturi",
    "costuri": "preturi",
    "costurile": "preturi",
    "tarif": "preturi",
    "tarife": "preturi",
    "abonament": "preturi",
    "abonamente": "preturi",
    "oferta": "preturi",
    "ofertare": "preturi",
}

_PHRASE_EXPANSIONS: dict[str, set[str]] = {
    "bon fiscal": {"bonuri", "fiscale"},
    "bonuri fiscale": {"bonuri", "fiscale"},
    "gestiune stocuri": {"stocuri"},
    "gestiune marfa": {"stocuri"},
    "comenzi online": {"online", "delivery"},
    "ordine online": {"online", "delivery"},

    # pricing
    "cat costa": {"preturi"},
    "cat costa?": {"preturi"},
    "as vrea preturi": {"preturi"},
    "vreau preturi": {"preturi"},
    "am nevoie de preturi": {"preturi"},
}


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_text(text)
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    tokens = {t for t in tokens if len(t) >= 3}

    # Expand multi-word phrases to canonical tokens
    for phrase, extra_tokens in _PHRASE_EXPANSIONS.items():
        if phrase in normalized:
            tokens |= extra_tokens

    # Expand/normalize single tokens
    expanded: set[str] = set(tokens)
    for t in tokens:
        canon = _TOKEN_SYNONYMS.get(t)
        if canon:
            expanded.add(canon)

    return expanded


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    # Protect against zero vectors
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


@dataclass(frozen=True)
class KBChunk:
    source: str
    text: str
    tokens: set[str]


@dataclass(frozen=True)
class KBEmbeddingChunk:
    id: str
    source: str
    text: str
    embedding: list[float]


_INDEX_CACHE: dict[str, tuple[float, str, list[KBEmbeddingChunk]]] = {}


def _load_index_cached(index_path: str) -> tuple[str, list[KBEmbeddingChunk]]:
    """Load kb_index.json with a simple mtime cache.

    Returns (embedding_model, chunks).
    """
    mtime = os.path.getmtime(index_path)
    cached = _INDEX_CACHE.get(index_path)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    with open(index_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    embedding_model = str(data.get("embedding_model") or "")
    chunks_raw = data.get("chunks") or []
    chunks: list[KBEmbeddingChunk] = []
    for item in chunks_raw:
        try:
            chunks.append(
                KBEmbeddingChunk(
                    id=str(item.get("id")),
                    source=str(item.get("source")),
                    text=str(item.get("text")),
                    embedding=list(item.get("embedding")),
                )
            )
        except Exception:
            continue

    _INDEX_CACHE[index_path] = (mtime, embedding_model, chunks)
    return embedding_model, chunks


class KnowledgeBase:
    """Simple file-based knowledge base (MVP).

    Reads Markdown files from a directory and retrieves relevant chunks by token overlap.
    """

    def __init__(
        self,
        *,
        kb_dir: str,
        index_path: str | None = None,
        openai_api_key: str | None = None,
        embedding_model: str | None = None,
        ) -> None:
        self.kb_dir = kb_dir
        self.index_path = index_path
        self.openai_api_key = openai_api_key or ""
        self.embedding_model_override = embedding_model

        self._chunks: list[KBChunk] = []
        self._embedding_chunks: list[KBEmbeddingChunk] = []
        self._embedding_model: str | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        # Preferred path: embeddings index (kb_index.json)
        if self.index_path and os.path.isfile(self.index_path):
            model_from_index, chunks = _load_index_cached(self.index_path)
            self._embedding_model = model_from_index or self.embedding_model_override
            self._embedding_chunks = chunks
            self._loaded = True
            return

        chunks: list[KBChunk] = []
        if not os.path.isdir(self.kb_dir):
            self._chunks = []
            self._loaded = True
            return

        for name in sorted(os.listdir(self.kb_dir)):
            if not name.lower().endswith(".md"):
                continue
            path = os.path.join(self.kb_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                continue

            # Chunk by blank lines (simple & robust)
            raw_parts = re.split(r"\n\s*\n", content)
            for part in raw_parts:
                part = part.strip()
                if not part:
                    continue
                tokens = _tokenize(part)
                if not tokens:
                    continue
                chunks.append(KBChunk(source=name, text=part, tokens=tokens))

        self._chunks = chunks
        self._loaded = True

    def search(self, query: str, *, k: int = 4, min_score: float = 0.25) -> list[str]:
        self.load()

        # Embeddings-based retrieval (preferred)
        if self._embedding_chunks:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            model = self.embedding_model_override or self._embedding_model or "text-embedding-3-small"
            client = OpenAI(api_key=self.openai_api_key)
            resp = client.embeddings.create(model=model, input=query)
            query_vec = resp.data[0].embedding

            scored: list[tuple[float, KBEmbeddingChunk]] = []
            for c in self._embedding_chunks:
                s = _cosine_similarity(query_vec, c.embedding)
                scored.append((s, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [c for s, c in scored[:k] if s >= min_score]
            return [f"[{c.source}]\n{c.text}" for c in top]

        # Fallback: token overlap (if no embeddings index available)
        q = _tokenize(query)
        if not q or not self._chunks:
            return []

        scored2: list[tuple[float, KBChunk]] = []
        for chunk in self._chunks:
            overlap = len(q & chunk.tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(q), 1)
            scored2.append((score, chunk))

        scored2.sort(key=lambda x: x[0], reverse=True)
        top2 = [c for s, c in scored2[:k] if s >= 0.12]
        return [f"[{c.source}]\n{c.text}" for c in top2]
