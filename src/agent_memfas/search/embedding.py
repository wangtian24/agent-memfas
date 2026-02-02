"""Embedding-based vector search backend using sqlite-vec."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .base import SearchBackend, SearchResult

if TYPE_CHECKING:
    from ..embedders.base import Embedder


class EmbeddingBackend(SearchBackend):
    """
    Vector similarity search backend using embeddings + sqlite-vec.
    
    Uses sqlite-vec for KNN search. Requires:
    - An Embedder (FastEmbed, Ollama, or custom)
    - sqlite-vec extension
    
    Install: pip install sqlite-vec fastembed
    
    Usage:
        from agent_memfas.embedders.fastembed import FastEmbedEmbedder
        
        embedder = FastEmbedEmbedder()
        backend = EmbeddingBackend("./memfas.db", embedder)
        backend.index("doc1", "Hello world")
        results = backend.search("greeting")
    """
    
    def __init__(
        self,
        db_path: str,
        embedder: "Embedder",
        table_prefix: str = "vec"
    ):
        """
        Initialize embedding backend.
        
        Args:
            db_path: Path to SQLite database
            embedder: Embedder instance for generating vectors
            table_prefix: Prefix for table names (default: "vec")
        """
        self.db_path = str(db_path)
        self.embedder = embedder
        self.table_prefix = table_prefix
        self._conn: Optional[sqlite3.Connection] = None
        self._setup_tables()
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._load_extension()
        return self._conn
    
    def _load_extension(self):
        """Load sqlite-vec extension."""
        try:
            import sqlite_vec
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
        except ImportError:
            raise ImportError(
                "sqlite-vec not installed. Install with: pip install sqlite-vec"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sqlite-vec: {e}")
    
    def _setup_tables(self):
        """Create tables for vector storage."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = self._get_conn()
        cur = conn.cursor()
        
        dims = self.embedder.dimensions
        
        # Document metadata table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_prefix}_docs (
                doc_id TEXT PRIMARY KEY,
                text TEXT,
                source TEXT,
                section TEXT,
                date TEXT,
                metadata TEXT,
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Vector table using sqlite-vec
        cur.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_prefix}_vectors 
            USING vec0(embedding float[{dims}])
        """)
        
        # Mapping table (rowid in vec table -> doc_id)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_prefix}_map (
                vec_rowid INTEGER PRIMARY KEY,
                doc_id TEXT UNIQUE,
                FOREIGN KEY (doc_id) REFERENCES {self.table_prefix}_docs(doc_id)
            )
        """)
        
        conn.commit()
    
    def index(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Index a document with its embedding."""
        metadata = metadata or {}
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Generate embedding
        embedding = self.embedder.embed(text)
        
        # Remove old entry if exists
        self.delete(doc_id)
        
        # Insert document metadata
        cur.execute(f"""
            INSERT INTO {self.table_prefix}_docs (doc_id, text, source, section, date, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            text[:10000],
            metadata.get("source", ""),
            metadata.get("section", ""),
            metadata.get("date", datetime.now().isoformat()),
            json.dumps(metadata)
        ))
        
        # Insert vector
        cur.execute(f"""
            INSERT INTO {self.table_prefix}_vectors (embedding) VALUES (?)
        """, (self._serialize_vector(embedding),))
        
        vec_rowid = cur.lastrowid
        
        # Insert mapping
        cur.execute(f"""
            INSERT INTO {self.table_prefix}_map (vec_rowid, doc_id) VALUES (?, ?)
        """, (vec_rowid, doc_id))
        
        conn.commit()
    
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search using vector similarity (KNN)."""
        if not query.strip():
            return []
        
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Generate query embedding
        query_vec = self.embedder.embed(query)
        
        # KNN search via sqlite-vec
        # Note: vec0 requires k=? constraint, not LIMIT
        cur.execute(f"""
            SELECT 
                m.doc_id,
                d.text,
                d.source,
                d.section,
                d.date,
                v.distance
            FROM {self.table_prefix}_vectors v
            JOIN {self.table_prefix}_map m ON v.rowid = m.vec_rowid
            JOIN {self.table_prefix}_docs d ON m.doc_id = d.doc_id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
        """, (self._serialize_vector(query_vec), limit))
        
        results = []
        for row in cur.fetchall():
            doc_id, text, source, section, date, distance = row
            # Convert distance to similarity score (lower distance = higher score)
            score = 1.0 / (1.0 + distance)
            
            results.append(SearchResult(
                text=text,
                score=score,
                source=source,
                section=section,
                date=date,
                metadata={"doc_id": doc_id, "distance": distance}
            ))
        
        return results
    
    def delete(self, doc_id: str) -> None:
        """Remove a document and its vector."""
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Get vec_rowid
        cur.execute(f"""
            SELECT vec_rowid FROM {self.table_prefix}_map WHERE doc_id = ?
        """, (doc_id,))
        row = cur.fetchone()
        
        if row:
            vec_rowid = row[0]
            cur.execute(f"DELETE FROM {self.table_prefix}_vectors WHERE rowid = ?", (vec_rowid,))
            cur.execute(f"DELETE FROM {self.table_prefix}_map WHERE doc_id = ?", (doc_id,))
        
        cur.execute(f"DELETE FROM {self.table_prefix}_docs WHERE doc_id = ?", (doc_id,))
        conn.commit()
    
    def clear(self) -> None:
        """Clear all documents and vectors."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {self.table_prefix}_docs")
        cur.execute(f"DELETE FROM {self.table_prefix}_map")
        # Recreate vector table (no DELETE support in vec0)
        cur.execute(f"DROP TABLE IF EXISTS {self.table_prefix}_vectors")
        cur.execute(f"""
            CREATE VIRTUAL TABLE {self.table_prefix}_vectors 
            USING vec0(embedding float[{self.embedder.dimensions}])
        """)
        conn.commit()
    
    def count(self) -> int:
        """Return document count."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self.table_prefix}_docs")
        return cur.fetchone()[0]
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def _serialize_vector(self, vector: List[float]) -> bytes:
        """Serialize vector to bytes for sqlite-vec."""
        import struct
        return struct.pack(f"{len(vector)}f", *vector)
