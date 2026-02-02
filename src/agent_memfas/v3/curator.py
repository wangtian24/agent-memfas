"""Main context curator for v3."""

import time
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

from .types import MemoryChunk, ContextResponse, Topic
from .topic import TopicDetector
from .scorer import RelevanceScorer
from .budget import TokenBudget
from .session import SessionState
from .telemetry import TelemetryLogger, TurnMetrics
from .levels import resolve_level, get_preset, LevelPreset, DEFAULT_LEVEL

if TYPE_CHECKING:
    from ..memory import Memory
    from ..embedders.base import Embedder


class ContextCurator:
    """
    Main entry point for v3 dynamic context curation.
    
    Proactively curates memory context each turn:
    1. Detects current topic
    2. Scores all memories by relevance
    3. Fills token budget with top memories
    4. Returns focused, curated context
    
    Usage:
        curator = ContextCurator("./memfas.yaml")
        
        result = curator.get_context(
            query="what's the project status?",
            session_id="main",
            baseline_tokens=85000
        )
        
        print(result.context)  # Inject into prompt
        print(result.tokens_saved)  # The value we provide
    """
    
    def __init__(
        self,
        config_path: str,
        level: Union[int, str, None] = None,
        token_budget: Optional[int] = None,
        telemetry_path: Optional[str] = None,
        embedder: Optional["Embedder"] = None
    ):
        """
        Initialize context curator.
        
        Args:
            config_path: Path to memfas config file
            level: Curation level 1-5, name ("minimal"/"balanced"/"full"), 
                   or "auto" (defaults to 3). Controls aggressiveness.
            token_budget: Override token budget (otherwise uses level preset)
            telemetry_path: Path for telemetry logs (default: ./memfas-telemetry.jsonl)
            embedder: Optional embedder override
        """
        from ..memory import Memory
        
        # Load memory system
        self.memory = Memory(config_path, embedder=embedder)
        
        # Get embedder (from memory or create one)
        if embedder:
            self._embedder = embedder
        elif hasattr(self.memory, '_embedder') and self.memory._embedder:
            self._embedder = self.memory._embedder
        else:
            # Create default embedder
            from ..embedders.fastembed import FastEmbedEmbedder
            self._embedder = FastEmbedEmbedder()
        
        # Resolve level and preset
        self._level = resolve_level(level)
        self._preset = get_preset(self._level)
        
        # Initialize components with preset values
        self.topic_detector = TopicDetector(self._embedder)
        self.scorer = RelevanceScorer(
            self._embedder,
            recency_weight=self._preset.recency_weight
        )
        
        # Use explicit budget if provided, otherwise use preset
        effective_budget = token_budget if token_budget is not None else self._preset.token_budget
        self.budget = TokenBudget(total_budget=effective_budget)
        
        # Session management
        self.sessions: Dict[str, SessionState] = {}
        
        # Telemetry
        telem_path = telemetry_path or "./memfas-telemetry.jsonl"
        self.telemetry = TelemetryLogger(telem_path)
    
    def get_context(
        self,
        query: str,
        session_id: str = "default",
        baseline_tokens: int = 0,
        include_triggers: bool = True,
        recent_history: Optional[List[str]] = None,
        level: Union[int, str, None] = None
    ) -> ContextResponse:
        """
        Get curated context for the current query.
        
        This is the main entry point. Call this before each Claude inference
        to get focused, relevant context.
        
        Args:
            query: Current user query
            session_id: Session identifier
            baseline_tokens: Current context size (for measuring savings)
            include_triggers: Whether to include Type 1 triggers
            recent_history: Recent messages for topic detection
            level: Override curation level for this query (1-5 or name)
        
        Returns:
            ContextResponse with curated context and metrics
        """
        start_time = time.perf_counter()
        
        # Resolve effective level (per-query override or default)
        effective_level = self._level
        effective_preset = self._preset
        if level is not None:
            effective_level = resolve_level(level)
            effective_preset = get_preset(effective_level)
            # Temporarily adjust budget for this query
            self.budget.set_budget(effective_preset.token_budget)
        
        # Get or create session
        session = self.sessions.setdefault(
            session_id,
            SessionState(session_id)
        )
        
        # 1. Detect topic
        topic = self.topic_detector.detect(query, recent_history)
        topic_shifted = self.topic_detector.detect_shift()
        
        # 2. Get all memory chunks
        memories = self._get_all_memories()
        
        # 3. Score by relevance
        boost_ids = session.get_boost_ids()
        scored = self.scorer.score(
            query,
            memories,
            current_topic=topic,
            boost_ids=boost_ids
        )
        
        # 4. Filter by min_score threshold (based on level)
        min_score = effective_preset.min_score
        max_results = effective_preset.max_results
        filtered_scored = [
            sm for sm in scored 
            if sm.score >= min_score
        ][:max_results]
        
        # Track how many were filtered
        filtered_count = len(scored) - len(filtered_scored)
        
        # 5. Get triggers (Type 1 fast path)
        triggers = []
        trigger_tokens = 0
        if include_triggers:
            trigger_results = self.memory._check_triggers(query)
            triggers = trigger_results
            # Estimate trigger tokens
            for t in triggers:
                trigger_tokens += self.budget.count_tokens(
                    f"[{t.trigger}] {t.hint}"
                )
        
        # 6. Allocate budget from filtered+scored memories
        curated = self.budget.allocate(filtered_scored, reserved=trigger_tokens)
        
        # 7. Format context
        context = self._format_context(triggers, curated.memories)
        total_tokens = curated.total_tokens + trigger_tokens
        
        # 8. Calculate metrics
        latency = (time.perf_counter() - start_time) * 1000
        tokens_saved = max(0, baseline_tokens - total_tokens)
        compression = total_tokens / max(baseline_tokens, 1)
        
        # 9. Update session state
        session.on_turn(
            query,
            topic,
            [m.id for m in curated.memories]
        )
        
        # 10. Restore budget if we used a per-query override
        if level is not None:
            self.budget.set_budget(self._preset.token_budget)
        
        # 11. Log telemetry
        self.telemetry.log_turn(TurnMetrics(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            turn_number=session.turn_count,
            query=query[:200],  # Truncate for log
            query_tokens=self.budget.count_tokens(query),
            detected_topic=topic.name,
            topic_shifted=topic_shifted,
            baseline_context_tokens=baseline_tokens,
            curated_context_tokens=total_tokens,
            memories_scored=len(scored),
            memories_included=len(curated.memories),
            memories_dropped=len(curated.dropped),
            memories_filtered=filtered_count,
            triggers_matched=len(triggers),
            tokens_saved=tokens_saved,
            compression_ratio=compression,
            latency_ms=latency,
            curation_level=effective_level,
            curation_level_name=effective_preset.name,
            min_score_threshold=min_score,
            top_memories=[
                {
                    "id": sm.memory.id,
                    "score": round(sm.score, 4),
                    "source": sm.memory.source,
                    "snippet": sm.memory.text[:100]
                }
                for sm in scored[:5]
            ],
            score_weights=self.scorer.get_weights()
        ))
        
        # 12. Build response
        return ContextResponse(
            context=context,
            tokens_used=total_tokens,
            budget=effective_preset.token_budget,
            memories_included=len(curated.memories),
            memories_dropped=len(curated.dropped),
            memories_filtered=filtered_count,
            triggers_matched=len(triggers),
            topic=topic.name,
            topic_shifted=topic_shifted,
            baseline_tokens=baseline_tokens,
            tokens_saved=tokens_saved,
            compression_ratio=compression,
            latency_ms=latency,
            curation_level=effective_level,
            curation_level_name=effective_preset.name,
            top_memories=[
                {
                    "id": sm.memory.id,
                    "score": round(sm.score, 4),
                    "source": sm.memory.source
                }
                for sm in scored[:5]
            ]
        )
    
    def end_session(self, session_id: str):
        """
        End a session and log summary.
        
        Args:
            session_id: Session to end
        """
        self.telemetry.log_session_end(session_id)
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get stats for a session."""
        session = self.sessions.get(session_id)
        return session.get_stats() if session else None
    
    def get_telemetry_summary(self, **kwargs) -> Dict:
        """Get telemetry summary."""
        return self.telemetry.get_summary(**kwargs)
    
    def format_telemetry_summary(self, **kwargs) -> str:
        """Get formatted telemetry summary."""
        return self.telemetry.format_summary(**kwargs)
    
    def set_budget(self, tokens: int):
        """Update token budget."""
        self.budget.set_budget(tokens)
    
    def get_level(self) -> int:
        """Get current curation level (1-5)."""
        return self._level
    
    def set_level(self, level: Union[int, str]):
        """
        Set curation level.
        
        Args:
            level: Level 1-5, name ("minimal"/"balanced"/"full"), or "auto"
        """
        self._level = resolve_level(level)
        self._preset = get_preset(self._level)
        
        # Update components with new preset
        self.budget.set_budget(self._preset.token_budget)
        self.scorer.set_weights(recency=self._preset.recency_weight)
    
    def get_level_info(self) -> Dict:
        """Get information about current level settings."""
        return {
            "level": self._level,
            "name": self._preset.name,
            "description": self._preset.description,
            "token_budget": self._preset.token_budget,
            "min_score": self._preset.min_score,
            "max_results": self._preset.max_results,
            "recency_weight": self._preset.recency_weight
        }
    
    def set_scorer_weights(self, **kwargs):
        """Update scorer weights."""
        self.scorer.set_weights(**kwargs)
    
    def _get_all_memories(self) -> List[MemoryChunk]:
        """
        Get all memory chunks from the backend.
        
        Converts backend storage format to MemoryChunk objects.
        Uses cached embeddings when available.
        """
        # Check if we have cached memories
        if hasattr(self, '_memory_cache') and self._memory_cache:
            return self._memory_cache
        
        memories = []
        conn = self.memory._get_conn()
        cur = conn.cursor()
        
        # Try embedding backend table first (has embeddings)
        has_embeddings = False
        try:
            cur.execute("""
                SELECT d.doc_id, d.text, d.source, d.section, d.date
                FROM vec_docs d
            """)
            rows = cur.fetchall()
            has_embeddings = True
        except:
            rows = []
        
        # Fall back to FTS5 backend table
        if not rows:
            try:
                cur.execute("""
                    SELECT doc_id, text, source, section, date
                    FROM search_docs
                """)
                rows = cur.fetchall()
            except:
                pass
        
        # Convert to MemoryChunk objects
        # Batch embed for efficiency if needed
        texts_to_embed = []
        memory_data = []
        
        for row in rows:
            doc_id, text, source, section, date = row
            memory_data.append({
                'id': doc_id,
                'text': text,
                'source': source or "",
                'section': section,
                'date': date
            })
            texts_to_embed.append(text[:1000])  # Limit for embedding
        
        # Batch embed (much faster than one-by-one)
        if texts_to_embed:
            embeddings = self._embedder.embed_batch(texts_to_embed)
        else:
            embeddings = []
        
        for data, embedding in zip(memory_data, embeddings):
            memories.append(MemoryChunk(
                id=data['id'],
                text=data['text'],
                source=data['source'],
                embedding=embedding,
                section=data['section'],
                date=data['date'],
                token_count=0  # Will be calculated by budget manager
            ))
        
        # Cache for subsequent calls in this session
        self._memory_cache = memories
        return memories
    
    def invalidate_cache(self):
        """Invalidate the memory cache (call after indexing new content)."""
        self._memory_cache = None
    
    def _format_context(
        self,
        triggers: List,
        memories: List[MemoryChunk]
    ) -> str:
        """
        Format curated context for injection into prompt.
        
        Uses XML tags for clear delineation.
        """
        parts = []
        
        if triggers:
            parts.append("<triggered_context>")
            for t in triggers:
                parts.append(f"[{t.trigger}] {t.hint}")
            parts.append("</triggered_context>")
        
        if memories:
            parts.append("<relevant_memory>")
            for m in memories:
                snippet = m.text[:500] if len(m.text) > 500 else m.text
                parts.append(f"[{m.source}]")
                parts.append(snippet)
                parts.append("")  # Blank line between memories
            parts.append("</relevant_memory>")
        
        return "\n".join(parts)
    
    def close(self):
        """Clean up resources."""
        # End all active sessions
        for session_id in list(self.sessions.keys()):
            self.end_session(session_id)
        
        self.memory.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
