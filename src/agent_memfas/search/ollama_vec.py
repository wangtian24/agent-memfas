"""
Ollama-backed vector store — read-write embedding search.

Stores embeddings as raw float blobs in plain SQLite (no sqlite-vec).
Cosine similarity in Python. Zero pip deps beyond ollama running locally.

Designed to run in parallel with FTS5Backend on the same documents —
FTS5 catches keywords, this catches meaning. Memory merges + dedupes.
"""

import json
import sqlite3
import struct
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import SearchBackend, SearchResult


class OllamaVecBackend(SearchBackend):
    """
    Vector search backend using Ollama embeddings + plain SQLite blobs.

    Tables (in the same DB as FTS5):
        vec_store_docs   — doc_id, text, source, section, date, embedding BLOB
    
    Usage:
        backend = OllamaVecBackend("./memfas.db")
        backend.index("doc1", "Hello world", {"source": "MEMORY.md"})
        results = backend.search("greeting")
    """

    def __init__(
        self,
        db_path: str,
        embedder_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
    ):
        self.db_path = str(db_path)
        self.embedder_model = embedder_model
        self.ollama_url = ollama_url.rstrip("/")
        self._conn: Optional[sqlite3.Connection] = None
        self._setup_tables()

    # ── connection ────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    # ── table setup ───────────────────────────────────────────

    def _setup_tables(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_store_docs (
                doc_id TEXT PRIMARY KEY,
                text TEXT,
                source TEXT,
                section TEXT,
                date TEXT,
                embedding BLOB,
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    # ── ollama ────────────────────────────────────────────────

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
    def _vec_to_blob(vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _blob_to_vec(blob: bytes) -> list[float]:
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    # ── SearchBackend interface ───────────────────────────────

    def index(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Embed and store a document."""
        metadata = metadata or {}
        embedding = self._embed(text[:3000])  # truncate for embedding (same as journal)
        blob = self._vec_to_blob(embedding)

        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO vec_store_docs
                (doc_id, text, source, section, date, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            text[:10000],
            metadata.get("source", ""),
            metadata.get("section", ""),
            metadata.get("date", datetime.now().isoformat()),
            blob,
        ))
        conn.commit()

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Embed query → cosine sim against all stored docs → top-N."""
        if not query.strip():
            return []

        query_vec = self._embed(query)
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT doc_id, text, source, section, date, embedding FROM vec_store_docs")

        scored: list[tuple[float, str, str, str, str, str]] = []
        for doc_id, text, source, section, date, blob in cur.fetchall():
            vec = self._blob_to_vec(blob)
            sim = self._cosine(query_vec, vec)
            scored.append((sim, doc_id, text, source, section, date))

        scored.sort(reverse=True)

        return [
            SearchResult(
                text=text,
                score=sim,
                source=source,
                section=section,
                date=date,
                metadata={"doc_id": doc_id, "cosine_sim": sim},
            )
            for sim, doc_id, text, source, section, date in scored[:limit]
        ]

    def delete(self, doc_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM vec_store_docs WHERE doc_id = ?", (doc_id,))
        conn.commit()

    def clear(self) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM vec_store_docs")
        conn.commit()

    def count(self) -> int:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM vec_store_docs")
        return cur.fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
