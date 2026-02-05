"""
Journal search backend — read-only, pre-indexed embedding search.

Queries an existing journal.db that has been indexed with
build-embedding-index.py (entry_embeddings table with nomic-embed-text vectors).

No new deps. Uses ollama HTTP API + stdlib cosine similarity.
"""

import json
import sqlite3
import struct
import urllib.request
from typing import List, Optional, Dict, Any

from .base import SearchBackend, SearchResult


class JournalSearchBackend(SearchBackend):
    """
    Read-only search backend over a pre-indexed journal DB.

    Expects a SQLite DB with:
        - entries (id, date, text, ...)          — the raw journal entries
        - entry_embeddings (date, text_chunk, embedding BLOB)  — pre-computed vectors

    Usage:
        backend = JournalSearchBackend(
            db_path="/path/to/journal.db",
            embedder_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
            label="DayOne Journal",
        )
        results = backend.search("running with Guan Lingfeng", limit=3)
    """

    def __init__(
        self,
        db_path: str,
        embedder_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        label: str = "journal",
        year_range: Optional[tuple[int, int]] = None,
    ):
        """
        Args:
            db_path: Path to the journal SQLite DB (with entry_embeddings).
            embedder_model: Ollama model name for query embedding.
            ollama_url: Ollama base URL.
            label: Label for results (shows up in recall output).
            year_range: Optional (min_year, max_year) filter. None = all years.
        """
        self.db_path = db_path
        self.embedder_model = embedder_model
        self.ollama_url = ollama_url.rstrip("/")
        self.label = label
        self.year_range = year_range
        self._conn: Optional[sqlite3.Connection] = None

    # ── connection ────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    # ── embedding ─────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Embed text via Ollama HTTP API."""
        req = urllib.request.Request(
            f"{self.ollama_url}/api/embeddings",
            data=json.dumps({"model": self.embedder_model, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return resp["embedding"]

    # ── cosine ────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _blob_to_vec(blob: bytes) -> list[float]:
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    # ── SearchBackend interface ───────────────────────────────

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Embed query → cosine similarity against all stored vectors → top-N."""
        if not query.strip():
            return []

        query_vec = self._embed(query)
        conn = self._get_conn()
        cur = conn.cursor()

        # Optionally join with entries to filter by year
        if self.year_range:
            min_y, max_y = self.year_range
            cur.execute(
                "SELECT ee.date, ee.text_chunk, ee.embedding "
                "FROM entry_embeddings ee "
                "WHERE CAST(SUBSTR(ee.date, 1, 4) AS INTEGER) BETWEEN ? AND ?",
                (min_y, max_y),
            )
        else:
            cur.execute("SELECT date, text_chunk, embedding FROM entry_embeddings")

        scored: list[tuple[float, str, str]] = []
        for date, text_chunk, blob in cur.fetchall():
            vec = self._blob_to_vec(blob)
            sim = self._cosine(query_vec, vec)
            scored.append((sim, date, text_chunk))

        scored.sort(reverse=True)

        results = []
        for sim, date, text_chunk in scored[:limit]:
            results.append(
                SearchResult(
                    text=text_chunk,
                    score=sim,
                    source=f"{self.label}:{date}",
                    section=None,
                    date=date,
                    metadata={"cosine_sim": sim},
                )
            )
        return results

    # ── read-only stubs ───────────────────────────────────────

    def index(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """No-op. Journal is pre-indexed externally."""
        pass

    def delete(self, doc_id: str) -> None:
        """No-op."""
        pass

    def clear(self) -> None:
        """No-op."""
        pass

    # ── info ──────────────────────────────────────────────────

    def count(self) -> int:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entry_embeddings")
        return cur.fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
