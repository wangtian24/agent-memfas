"""SQLite FTS5 full-text search backend. Zero dependencies."""

import sqlite3
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from .base import SearchBackend, SearchResult


class FTS5Backend(SearchBackend):
    """
    SQLite FTS5 full-text search backend.
    
    This is the default backend. Uses BM25 ranking with optional
    recency weighting. Zero external dependencies.
    
    Usage:
        backend = FTS5Backend("./memfas.db")
        backend.index("doc1", "Hello world", {"source": "test.md"})
        results = backend.search("hello")
    """
    
    def __init__(
        self,
        db_path: str,
        recency_weight: float = 0.1,
        min_score: float = 0.0
    ):
        """
        Initialize FTS5 backend.
        
        Args:
            db_path: Path to SQLite database
            recency_weight: Weight for recency scoring (0 = disabled)
            min_score: Minimum score threshold for results
        """
        self.db_path = str(db_path)
        self.recency_weight = recency_weight
        self.min_score = min_score
        self._conn: Optional[sqlite3.Connection] = None
        self._setup_tables()
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn
    
    def _setup_tables(self):
        """Create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Main documents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS search_docs (
                doc_id TEXT PRIMARY KEY,
                text TEXT,
                source TEXT,
                section TEXT,
                date TEXT,
                metadata TEXT,
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # FTS5 virtual table
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                doc_id,
                text,
                source,
                section,
                content='search_docs',
                content_rowid='rowid'
            )
        """)
        
        # Sync triggers
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS search_ai AFTER INSERT ON search_docs BEGIN
                INSERT INTO search_fts(rowid, doc_id, text, source, section)
                VALUES (new.rowid, new.doc_id, new.text, new.source, new.section);
            END
        """)
        
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS search_ad AFTER DELETE ON search_docs BEGIN
                INSERT INTO search_fts(search_fts, rowid, doc_id, text, source, section)
                VALUES ('delete', old.rowid, old.doc_id, old.text, old.source, old.section);
            END
        """)
        
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS search_au AFTER UPDATE ON search_docs BEGIN
                INSERT INTO search_fts(search_fts, rowid, doc_id, text, source, section)
                VALUES ('delete', old.rowid, old.doc_id, old.text, old.source, old.section);
                INSERT INTO search_fts(rowid, doc_id, text, source, section)
                VALUES (new.rowid, new.doc_id, new.text, new.source, new.section);
            END
        """)
        
        conn.commit()
    
    def index(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Index a document."""
        import json
        
        metadata = metadata or {}
        conn = self._get_conn()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT OR REPLACE INTO search_docs (doc_id, text, source, section, date, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            text[:10000],  # Limit text size
            metadata.get("source", ""),
            metadata.get("section", ""),
            metadata.get("date", datetime.now().isoformat()),
            json.dumps(metadata)
        ))
        
        conn.commit()
    
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search using FTS5 with BM25 ranking."""
        # Sanitize query for FTS5
        safe_query = re.sub(r'["\'\(\)\*\:\^\?\+\-\~\{\}\[\]\|\\\/]', ' ', query)
        safe_query = ' '.join(safe_query.split())
        
        if not safe_query.strip():
            return []
        
        conn = self._get_conn()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT d.doc_id, d.text, d.source, d.section, d.date, bm25(search_fts) as score
                FROM search_docs d
                JOIN search_fts fts ON d.rowid = fts.rowid
                WHERE search_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (safe_query, limit * 2))  # Get extra for recency re-ranking
            
            results = []
            now = datetime.now()
            
            for row in cur.fetchall():
                doc_id, text, source, section, date, bm25_score = row
                
                # Apply recency weighting
                recency_score = 1.0
                if date and self.recency_weight > 0:
                    try:
                        mem_date = datetime.fromisoformat(date)
                        days_old = (now - mem_date).days
                        recency_score = 1.0 / (1.0 + days_old * self.recency_weight * 0.01)
                    except:
                        pass
                
                final_score = abs(bm25_score) * recency_score
                
                if final_score >= self.min_score:
                    results.append(SearchResult(
                        text=text,
                        score=final_score,
                        source=source,
                        section=section,
                        date=date,
                        metadata={"doc_id": doc_id}
                    ))
            
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except sqlite3.OperationalError:
            return []
    
    def delete(self, doc_id: str) -> None:
        """Remove a document."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM search_docs WHERE doc_id = ?", (doc_id,))
        conn.commit()
    
    def clear(self) -> None:
        """Clear all documents."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM search_docs")
        cur.execute("DELETE FROM search_fts")
        conn.commit()
    
    def count(self) -> int:
        """Return document count."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM search_docs")
        return cur.fetchone()[0]
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
