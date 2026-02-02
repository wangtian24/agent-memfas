"""
Core Memory class for agent-memfas.

Implements dual-store memory:
- Type 1 (Fast): Keyword triggers for instant recall
- Type 2 (Slow): Pluggable search backend (FTS5 default, embeddings optional)
"""

import sqlite3
import re
import hashlib
from pathlib import Path
from glob import glob
from datetime import datetime
from typing import Optional, Union, List, TYPE_CHECKING
from dataclasses import dataclass

from .config import Config, TriggerConfig
from .search.base import SearchBackend, SearchResult
from .search.fts5 import FTS5Backend

if TYPE_CHECKING:
    from .embedders.base import Embedder


@dataclass
class MemoryResult:
    """A single memory retrieval result."""
    source: str
    text: str
    score: float
    trigger: Optional[str] = None
    hint: Optional[str] = None
    date: Optional[str] = None
    
    def __str__(self) -> str:
        prefix = f"[{self.trigger}] {self.hint}" if self.trigger else f"[{self.source}]"
        snippet = self.text[:200] + "..." if len(self.text) > 200 else self.text
        return f"{prefix}\n  > {snippet}"


def _create_embedder_from_config(config: Config) -> Optional["Embedder"]:
    """Create an embedder from config settings."""
    embedder_type = config.search.embedder_type
    embedder_model = config.search.embedder_model
    
    if not embedder_type:
        return None
    
    if embedder_type == "fastembed":
        from .embedders.fastembed import FastEmbedEmbedder
        if embedder_model:
            return FastEmbedEmbedder(model=embedder_model)
        return FastEmbedEmbedder()
    elif embedder_type == "ollama":
        from .embedders.ollama import OllamaEmbedder
        if embedder_model:
            return OllamaEmbedder(model=embedder_model)
        return OllamaEmbedder()
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


def _create_backend(
    config: Config,
    backend_type: Optional[str] = None,
    embedder: Optional["Embedder"] = None
) -> SearchBackend:
    """
    Create a search backend based on config.
    
    Args:
        config: Config object
        backend_type: Override backend type ("fts5" or "embedding")
        embedder: Embedder instance (auto-created from config if not provided)
    
    Returns:
        SearchBackend instance
    """
    backend_type = backend_type or config.search.backend
    
    if backend_type == "embedding":
        # Try to create embedder from config if not provided
        if embedder is None:
            embedder = _create_embedder_from_config(config)
        if embedder is None:
            raise ValueError(
                "Embedder required for embedding backend. Either pass embedder= "
                "or set search.embedder_type in config."
            )
        from .search.embedding import EmbeddingBackend
        return EmbeddingBackend(config.db_path, embedder)
    else:
        return FTS5Backend(
            config.db_path,
            recency_weight=config.search.recency_weight,
            min_score=config.search.min_score
        )


class Memory:
    """
    Dual-store memory system for AI agents.
    
    Type 1 (Fast): Keyword triggers - instant pattern matching
    Type 2 (Slow): Pluggable search backend (FTS5 or embeddings)
    
    Usage:
        # Default (FTS5, zero deps)
        mem = Memory("./memfas.yaml")
        context = mem.recall("How's the family?")
        
        # With embeddings (optional deps)
        from agent_memfas.embedders.fastembed import FastEmbedEmbedder
        mem = Memory("./memfas.yaml", embedder=FastEmbedEmbedder())
    """
    
    def __init__(
        self,
        config: Union[str, Config, None] = None,
        search_backend: Optional[str] = None,
        embedder: Optional["Embedder"] = None
    ):
        """
        Initialize memory system.
        
        Args:
            config: Path to config file, Config object, or None for auto-detect
            search_backend: Override search backend type ("fts5" or "embedding")
            embedder: Embedder for embedding backend (optional)
        """
        if config is None:
            self.config = Config.default(".")
        elif isinstance(config, str):
            if Path(config).exists():
                self.config = Config.load(config)
            else:
                self.config = Config.default(config)
        else:
            self.config = config
        
        # Initialize search backend
        self._backend = _create_backend(self.config, search_backend, embedder)
        self._embedder = embedder
        
        # Triggers DB (separate from search backend)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_triggers_db()
    
    def _ensure_triggers_db(self):
        """Ensure triggers table exists."""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        
        # Keyword triggers table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS keyword_triggers (
                id INTEGER PRIMARY KEY,
                keyword TEXT UNIQUE,
                hint TEXT,
                memory_ids TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Load triggers from config
        self._load_config_triggers()
    
    def _load_config_triggers(self):
        """Load triggers from config into database."""
        if not self.config.triggers:
            return
        
        conn = self._get_conn()
        cur = conn.cursor()
        
        for t in self.config.triggers:
            cur.execute("""
                INSERT OR IGNORE INTO keyword_triggers (keyword, hint, memory_ids)
                VALUES (?, ?, ?)
            """, (t.keyword.lower(), t.hint, ",".join(map(str, t.memory_ids))))
        
        conn.commit()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.config.db_path)
        return self._conn
    
    def close(self):
        """Close database connections."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._backend.close()
    
    # ============ Type 1: Fast (Keyword Triggers) ============
    
    def add_trigger(self, keyword: str, hint: str, memory_ids: list[int] = None):
        """
        Add a keyword trigger for fast recall.
        
        Args:
            keyword: Word or phrase to trigger on
            hint: Brief description of what this triggers
            memory_ids: Optional list of specific memory IDs to return
        """
        conn = self._get_conn()
        cur = conn.cursor()
        
        ids_str = ",".join(map(str, memory_ids or []))
        
        cur.execute("""
            INSERT INTO keyword_triggers (keyword, hint, memory_ids)
            VALUES (?, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET
                hint = excluded.hint,
                memory_ids = excluded.memory_ids
        """, (keyword.lower(), hint, ids_str))
        
        conn.commit()
    
    def remove_trigger(self, keyword: str):
        """Remove a keyword trigger."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM keyword_triggers WHERE keyword = ?", (keyword.lower(),))
        conn.commit()
    
    def list_triggers(self) -> list[dict]:
        """List all keyword triggers."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT keyword, hint, memory_ids FROM keyword_triggers ORDER BY keyword")
        return [
            {"keyword": row[0], "hint": row[1], "memory_ids": row[2]}
            for row in cur.fetchall()
        ]
    
    def _check_triggers(self, text: str) -> list[MemoryResult]:
        """Check text for keyword triggers (Type 1 - fast path)."""
        conn = self._get_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT keyword, hint, memory_ids FROM keyword_triggers")
        triggers = cur.fetchall()
        
        text_lower = text.lower()
        results = []
        
        for keyword, hint, memory_ids in triggers:
            if keyword in text_lower:
                if memory_ids:
                    # Search for associated memories by ID
                    ids = [int(x) for x in memory_ids.split(",") if x]
                    for doc_id in ids:
                        search_results = self._backend.search(str(doc_id), limit=1)
                        if search_results:
                            r = search_results[0]
                            results.append(MemoryResult(
                                source=r.source,
                                text=r.text,
                                score=1.0,
                                trigger=keyword,
                                hint=hint,
                                date=r.date
                            ))
                else:
                    results.append(MemoryResult(
                        source="trigger",
                        text=hint,
                        score=1.0,
                        trigger=keyword,
                        hint=hint
                    ))
        
        return results
    
    # ============ Type 2: Slow (Search Backend) ============
    
    def search(self, query: str, limit: int = None) -> list[MemoryResult]:
        """
        Search memories using the configured backend.
        
        Args:
            query: Search query
            limit: Max results (default from config)
        
        Returns:
            List of matching MemoryResult objects
        """
        if limit is None:
            limit = self.config.search.max_results
        
        search_results = self._backend.search(query, limit)
        
        return [
            MemoryResult(
                source=r.source,
                text=r.text,
                score=r.score,
                date=r.date
            )
            for r in search_results
        ]
    
    # ============ Combined Recall ============
    
    def recall(self, context: str, include_always_load: bool = True) -> str:
        """
        Recall relevant memories for the given context.
        
        This is the main entry point. It:
        1. Checks Type 1 triggers first (fast)
        2. Falls back to Type 2 search if needed (slow)
        3. Formats results for LLM context injection
        
        Args:
            context: Natural language context/query
            include_always_load: Whether to include always-load sources
        
        Returns:
            Formatted memory context string
        """
        results = []
        
        # Type 1: Check triggers (fast path)
        trigger_results = self._check_triggers(context)
        if trigger_results:
            results.extend(trigger_results)
        
        # Type 2: Search if no triggers or need more context
        if len(results) < self.config.search.max_results:
            search_results = self.search(
                context, 
                limit=self.config.search.max_results - len(results)
            )
            results.extend(search_results)
        
        # Format output
        if not results:
            return ""
        
        output_parts = ["📚 Memory Context:\n"]
        
        # Group by trigger vs search
        triggered = [r for r in results if r.trigger]
        searched = [r for r in results if not r.trigger]
        
        if triggered:
            output_parts.append("**Triggered Memories:**")
            for r in triggered:
                output_parts.append(str(r))
        
        if searched:
            output_parts.append("\n**Related Memories:**")
            for r in searched:
                output_parts.append(str(r))
        
        return "\n".join(output_parts)
    
    # ============ Indexing ============
    
    def _doc_id(self, source: str, section: str, text: str) -> str:
        """Generate stable document ID."""
        content = f"{source}:{section}:{text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def index_file(self, path: str, file_type: str = "markdown"):
        """
        Index a single file into memory.
        
        Args:
            path: Path to file
            file_type: Type of file (markdown, json, text)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        content = path.read_text()
        
        if file_type == "markdown":
            self._index_markdown(str(path), content)
        elif file_type == "json":
            self._index_json(str(path), content)
        else:
            self._index_text(str(path), content)
    
    def _index_markdown(self, source: str, content: str):
        """Index markdown content, splitting by headers."""
        sections = re.split(r'^(#{1,6}\s+.+)$', content, flags=re.MULTILINE)
        
        current_section = "intro"
        chunks = []
        
        for i, part in enumerate(sections):
            if re.match(r'^#{1,6}\s+', part):
                current_section = part.strip()
            elif part.strip():
                chunks.append((current_section, part.strip()))
        
        if not chunks:
            chunks = [("content", content)]
        
        for section, text in chunks:
            if len(text) > 100:
                doc_id = self._doc_id(source, section, text)
                self._backend.index(doc_id, text[:10000], {
                    "source": source,
                    "section": section,
                    "date": datetime.now().isoformat()
                })
    
    def _index_json(self, source: str, content: str):
        """Index JSON content."""
        import json as json_lib
        
        try:
            data = json_lib.loads(content)
            text = json_lib.dumps(data, indent=2)[:10000]
            doc_id = self._doc_id(source, "root", text)
            self._backend.index(doc_id, text, {
                "source": source,
                "section": "root",
                "date": datetime.now().isoformat()
            })
        except json_lib.JSONDecodeError:
            pass
    
    def _index_text(self, source: str, content: str):
        """Index plain text content."""
        chunk_size = 1000
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                section = f"chunk_{i // chunk_size}"
                doc_id = self._doc_id(source, section, chunk)
                self._backend.index(doc_id, chunk, {
                    "source": source,
                    "section": section,
                    "date": datetime.now().isoformat()
                })
    
    def index_sources(self):
        """Index all sources from config."""
        for source in self.config.sources:
            paths = glob(source.path)
            for path in paths:
                try:
                    self.index_file(path, source.type)
                except Exception as e:
                    print(f"Warning: Failed to index {path}: {e}")
    
    def clear(self):
        """Clear all indexed memories (keeps triggers)."""
        self._backend.clear()
    
    def stats(self) -> dict:
        """Get memory system statistics."""
        conn = self._get_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM keyword_triggers")
        trigger_count = cur.fetchone()[0]
        
        return {
            "memories": self._backend.count(),
            "triggers": trigger_count,
            "backend": type(self._backend).__name__,
            "db_path": self.config.db_path
        }
    
    def suggest_triggers(self, min_occurrences: int = 3, limit: int = 20) -> list[dict]:
        """
        Suggest potential triggers based on frequently occurring terms.
        
        Args:
            min_occurrences: Minimum times a term must appear
            limit: Maximum suggestions to return
        
        Returns:
            List of {"term": str, "count": int, "type": str} suggestions
        """
        from collections import Counter
        
        # Get all text directly from the backend's underlying storage
        # This is backend-specific, but both FTS5 and Embedding use similar schemas
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Try the FTS5 backend table first
        try:
            cur.execute("SELECT text FROM search_docs")
            rows = cur.fetchall()
        except:
            # Fallback: try vec_docs for embedding backend
            try:
                cur.execute("SELECT text FROM vec_docs")
                rows = cur.fetchall()
            except:
                rows = []
        
        all_text = " ".join(row[0] for row in rows if row[0])
        
        if not all_text:
            return []
        
        existing = {t["keyword"].lower() for t in self.list_triggers()}
        suggestions = []
        
        # Capitalized words (entities)
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', all_text)
        cap_counts = Counter(w.lower() for w in capitalized)
        
        for term, count in cap_counts.most_common(limit * 2):
            if count >= min_occurrences and term not in existing:
                suggestions.append({"term": term, "count": count, "type": "entity"})
        
        # Frequent terms
        words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
        word_counts = Counter(words)
        stopwords = {"this", "that", "with", "from", "have", "been", "were", "they", 
                     "their", "would", "could", "should", "about", "into", "more",
                     "some", "than", "then", "when", "what", "which", "there"}
        
        for term, count in word_counts.most_common(limit * 3):
            if count >= min_occurrences and term not in existing and term not in stopwords:
                suggestions.append({"term": term, "count": count, "type": "term"})
        
        seen = set()
        unique = []
        for s in sorted(suggestions, key=lambda x: x["count"], reverse=True):
            if s["term"] not in seen:
                seen.add(s["term"])
                unique.append(s)
        
        return unique[:limit]
    
    # ============ Backend Access ============
    
    @property
    def backend(self) -> SearchBackend:
        """Access the underlying search backend."""
        return self._backend
    
    def reindex(self, new_backend: Optional[str] = None, embedder: Optional["Embedder"] = None):
        """
        Re-index all documents with a new backend.
        
        Args:
            new_backend: New backend type ("fts5" or "embedding")
            embedder: New embedder (required for embedding backend)
        """
        # Get all current documents directly from the database
        conn = self._get_conn()
        cur = conn.cursor()
        
        documents = []
        
        # Try FTS5 backend table
        try:
            cur.execute("SELECT doc_id, text, source, section, date FROM search_docs")
            documents = cur.fetchall()
        except:
            pass
        
        # If empty, try embedding backend table
        if not documents:
            try:
                cur.execute("SELECT doc_id, text, source, section, date FROM vec_docs")
                documents = cur.fetchall()
            except:
                pass
        
        if not documents:
            return
        
        # Create new backend
        new = _create_backend(self.config, new_backend, embedder)
        
        # Re-index each document
        for doc_id, text, source, section, date in documents:
            new.index(doc_id, text, {
                "source": source or "",
                "section": section or "",
                "date": date or ""
            })
        
        # Swap backends
        self._backend.close()
        self._backend = new
        self._embedder = embedder
