"""
Generic pre-indexed vector search backend.

Queries any SQLite DB that has a table with pre-computed embedding blobs.
Read-only — the embeddings are built externally (e.g. build-embedding-index.py).
Table structure and column names are fully configurable.

No pip deps. Uses ollama HTTP API (stdlib urllib) + cosine similarity.

Example config (memfas.yaml):
    external_sources:
      - type: preindexed_vec
        db_path: /path/to/some.db
        label: "My Indexed Data"
        table: entry_embeddings
        key_col: date          # surfaced as .date in results
        text_col: text_chunk   # the text snippet
        embedding_col: embedding  # BLOB of packed float32
        embedder_model: nomic-embed-text
        ollama_url: http://localhost:11434
        max_results: 3
"""

import json
import sqlite3
import struct
import urllib.request
from typing import List, Optional, Dict, Any

from .base import SearchBackend, SearchResult


class PreIndexedVecBackend(SearchBackend):
    """
    Read-only vector search over a pre-indexed SQLite table.

    Args:
        db_path: Path to SQLite DB.
        table: Table name containing the embeddings.
        key_col: Column used as the result identifier (e.g. "date").
        text_col: Column containing the text.
        embedding_col: Column containing the embedding BLOB (packed float32).
        embedder_model: Ollama model name for query embedding.
        ollama_url: Ollama base URL.
        label: Display label for results in recall output.
        key_filter: Optional SQL WHERE clause added to queries (e.g. year filtering).
    """

    def __init__(
        self,
        db_path: str,
        table: str = "entry_embeddings",
        key_col: str = "date",
        text_col: str = "text_chunk",
        embedding_col: str = "embedding",
        embedder_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        label: str = "external",
        key_filter: Optional[str] = None,
    ):
        self.db_path = db_path
        self.table = table
        self.key_col = key_col
        self.text_col = text_col
        self.embedding_col = embedding_col
        self.embedder_model = embedder_model
        self.ollama_url = ollama_url.rstrip("/")
        self.label = label
        self.key_filter = key_filter  # raw SQL WHERE clause, user is responsible
        self._conn: Optional[sqlite3.Connection] = None

    # ── connection ────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    # ── embedding ─────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        req = urllib.request.Request(
            f"{self.ollama_url}/api/embeddings",
            data=json.dumps({"model": self.embedder_model, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return resp["embedding"]

    # ── vector helpers ────────────────────────────────────────

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
        """Embed query → cosine sim against all rows → top-N."""
        if not query.strip():
            return []

        query_vec = self._embed(query)
        conn = self._get_conn()
        cur = conn.cursor()

        # Build query safely — table/col names are set at init, not user input
        sql = (
            f"SELECT {self.key_col}, {self.text_col}, {self.embedding_col} "
            f"FROM {self.table}"
        )
        if self.key_filter:
            sql += f" WHERE {self.key_filter}"

        cur.execute(sql)

        scored: list[tuple[float, str, str]] = []
        for key, text, blob in cur.fetchall():
            vec = self._blob_to_vec(blob)
            sim = self._cosine(query_vec, vec)
            scored.append((sim, str(key), text))

        scored.sort(reverse=True)

        return [
            SearchResult(
                text=text,
                score=sim,
                source=f"{self.label}:{key}",
                section=None,
                date=key,  # key_col value surfaced as date
                metadata={"cosine_sim": sim, "key": key},
            )
            for sim, key, text in scored[:limit]
        ]

    # ── read-only stubs ───────────────────────────────────────

    def index(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """No-op. This backend is pre-indexed externally."""
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
        cur.execute(f"SELECT COUNT(*) FROM {self.table}")
        return cur.fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
