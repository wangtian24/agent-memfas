"""
Core Memory class for agent-memfas.

Implements dual-store memory:
- Type 1 (Fast): Keyword triggers for instant recall
- Type 2 (Slow): FTS5 search for deliberate lookup
"""

import sqlite3
import re
from pathlib import Path
from glob import glob
from datetime import datetime
from typing import Optional, Union
from dataclasses import dataclass

from .config import Config, TriggerConfig


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


class Memory:
    """
    Dual-store memory system for AI agents.
    
    Type 1 (Fast): Keyword triggers - instant pattern matching
    Type 2 (Slow): FTS5 search - deliberate semantic lookup
    
    Usage:
        mem = Memory("./memfas.yaml")
        context = mem.recall("How's the family?")
        mem.add_trigger("tahoe", "Family ski trips")
    """
    
    def __init__(self, config: Union[str, Config, None] = None):
        """
        Initialize memory system.
        
        Args:
            config: Path to config file, Config object, or None for auto-detect
        """
        if config is None:
            self.config = Config.default(".")
        elif isinstance(config, str):
            if Path(config).exists():
                self.config = Config.load(config)
            else:
                # Treat as directory path
                self.config = Config.default(config)
        else:
            self.config = config
        
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()
    
    def _ensure_db(self):
        """Ensure database exists with correct schema."""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        
        # Main memories table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                source TEXT,
                section TEXT,
                text TEXT,
                date TEXT,
                type TEXT DEFAULT 'markdown',
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # FTS5 virtual table for full-text search
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                source,
                section,
                text,
                content='memories',
                content_rowid='id'
            )
        """)
        
        # Triggers for keeping FTS in sync
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, source, section, text)
                VALUES (new.id, new.source, new.section, new.text);
            END
        """)
        
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, source, section, text)
                VALUES ('delete', old.id, old.source, old.section, old.text);
            END
        """)
        
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
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
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
                # Get associated memories if specified
                if memory_ids:
                    ids = [int(x) for x in memory_ids.split(",") if x]
                    cur.execute(f"""
                        SELECT source, text, date FROM memories
                        WHERE id IN ({','.join('?' * len(ids))})
                    """, ids)
                    for row in cur.fetchall():
                        results.append(MemoryResult(
                            source=row[0],
                            text=row[1],
                            score=1.0,
                            trigger=keyword,
                            hint=hint,
                            date=row[2]
                        ))
                else:
                    # Return hint as placeholder
                    results.append(MemoryResult(
                        source="trigger",
                        text=hint,
                        score=1.0,
                        trigger=keyword,
                        hint=hint
                    ))
        
        return results
    
    # ============ Type 2: Slow (FTS5 Search) ============
    
    def search(self, query: str, limit: int = None) -> list[MemoryResult]:
        """
        Search memories using FTS5 (Type 2 - slow path).
        
        Args:
            query: Search query
            limit: Max results (default from config)
        
        Returns:
            List of matching MemoryResult objects
        """
        if limit is None:
            limit = self.config.search.max_results
        
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Sanitize query for FTS5
        safe_query = re.sub(r'["\'\(\)\*\:\^\?\+\-\~\{\}\[\]\|\\\/]', ' ', query)
        safe_query = ' '.join(safe_query.split())  # Normalize whitespace
        
        if not safe_query.strip():
            return []
        
        try:
            cur.execute("""
                SELECT m.source, m.text, m.date, bm25(memories_fts) as score
                FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (safe_query, limit * 2))  # Get extra for recency re-ranking
            
            results = []
            now = datetime.now()
            
            for row in cur.fetchall():
                source, text, date, bm25_score = row
                
                # Apply recency weighting
                recency_score = 1.0
                if date and self.config.search.recency_weight > 0:
                    try:
                        mem_date = datetime.fromisoformat(date)
                        days_old = (now - mem_date).days
                        recency_score = 1.0 / (1.0 + days_old * self.config.search.recency_weight * 0.01)
                    except:
                        pass
                
                final_score = abs(bm25_score) * recency_score
                
                if final_score >= self.config.search.min_score:
                    results.append(MemoryResult(
                        source=source,
                        text=text,
                        score=final_score,
                        date=date
                    ))
            
            # Sort by final score and limit
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except sqlite3.OperationalError:
            # FTS query failed, return empty
            return []
    
    # ============ Combined Recall ============
    
    def recall(self, context: str, include_always_load: bool = True) -> str:
        """
        Recall relevant memories for the given context.
        
        This is the main entry point. It:
        1. Checks Type 1 triggers first (fast)
        2. Falls back to Type 2 search if needed (slow)
        3. Optionally includes always-load memories
        4. Formats results for LLM context injection
        
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
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Split by headers
        sections = re.split(r'^(#{1,6}\s+.+)$', content, flags=re.MULTILINE)
        
        current_section = "intro"
        chunks = []
        
        for i, part in enumerate(sections):
            if re.match(r'^#{1,6}\s+', part):
                current_section = part.strip()
            elif part.strip():
                chunks.append((current_section, part.strip()))
        
        # If no headers found, index as single chunk
        if not chunks:
            chunks = [("content", content)]
        
        # Insert chunks
        for section, text in chunks:
            if len(text) > 100:  # Skip tiny chunks
                cur.execute("""
                    INSERT INTO memories (source, section, text, date, type)
                    VALUES (?, ?, ?, ?, 'markdown')
                """, (source, section, text[:10000], datetime.now().isoformat()))
        
        conn.commit()
    
    def _index_json(self, source: str, content: str):
        """Index JSON content."""
        import json
        conn = self._get_conn()
        cur = conn.cursor()
        
        try:
            data = json.loads(content)
            text = json.dumps(data, indent=2)[:10000]
            cur.execute("""
                INSERT INTO memories (source, section, text, date, type)
                VALUES (?, ?, ?, ?, 'json')
            """, (source, "root", text, datetime.now().isoformat()))
            conn.commit()
        except json.JSONDecodeError:
            pass
    
    def _index_text(self, source: str, content: str):
        """Index plain text content."""
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Split into ~1000 char chunks
        chunk_size = 1000
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                cur.execute("""
                    INSERT INTO memories (source, section, text, date, type)
                    VALUES (?, ?, ?, ?, 'text')
                """, (source, f"chunk_{i // chunk_size}", chunk, datetime.now().isoformat()))
        
        conn.commit()
    
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
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM memories")
        cur.execute("DELETE FROM memories_fts")
        conn.commit()
    
    def stats(self) -> dict:
        """Get memory system statistics."""
        conn = self._get_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM memories")
        memory_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM keyword_triggers")
        trigger_count = cur.fetchone()[0]
        
        cur.execute("SELECT SUM(LENGTH(text)) FROM memories")
        total_chars = cur.fetchone()[0] or 0
        
        return {
            "memories": memory_count,
            "triggers": trigger_count,
            "total_chars": total_chars,
            "db_path": self.config.db_path
        }
    
    def suggest_triggers(self, min_occurrences: int = 3, limit: int = 20) -> list[dict]:
        """
        Suggest potential triggers based on frequently occurring terms.
        
        Analyzes indexed content to find:
        1. Capitalized words (likely proper nouns/entities)
        2. Frequently occurring terms
        
        Args:
            min_occurrences: Minimum times a term must appear
            limit: Maximum suggestions to return
        
        Returns:
            List of {"term": str, "count": int, "type": str} suggestions
        """
        from collections import Counter
        
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Get all indexed text
        cur.execute("SELECT text FROM memories")
        all_text = " ".join(row[0] for row in cur.fetchall())
        
        if not all_text:
            return []
        
        # Get existing triggers to exclude
        existing = {t["keyword"].lower() for t in self.list_triggers()}
        
        suggestions = []
        
        # Method 1: Capitalized words (proper nouns, entities)
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', all_text)
        cap_counts = Counter(w.lower() for w in capitalized)
        
        for term, count in cap_counts.most_common(limit * 2):
            if count >= min_occurrences and term not in existing:
                suggestions.append({
                    "term": term,
                    "count": count,
                    "type": "entity"
                })
        
        # Method 2: Frequent multi-word phrases (bigrams)
        words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        
        for phrase, count in bigram_counts.most_common(limit):
            if count >= min_occurrences and phrase not in existing:
                # Skip common phrases
                if phrase not in ("the the", "of the", "in the", "to the", "and the"):
                    suggestions.append({
                        "term": phrase,
                        "count": count,
                        "type": "phrase"
                    })
        
        # Method 3: Standalone frequent terms (4+ chars to avoid noise)
        word_counts = Counter(words)
        stopwords = {"this", "that", "with", "from", "have", "been", "were", "they", 
                     "their", "would", "could", "should", "about", "into", "more",
                     "some", "than", "then", "when", "what", "which", "there"}
        
        for term, count in word_counts.most_common(limit * 3):
            if (count >= min_occurrences and 
                term not in existing and 
                term not in stopwords and
                len(term) >= 4):
                suggestions.append({
                    "term": term,
                    "count": count,
                    "type": "term"
                })
        
        # Deduplicate and sort by count
        seen = set()
        unique = []
        for s in sorted(suggestions, key=lambda x: x["count"], reverse=True):
            if s["term"] not in seen:
                seen.add(s["term"])
                unique.append(s)
        
        return unique[:limit]
