"""
Context Manager for memfas v4.

Handles active session context with:
- Pre-emptive compaction (trigger at 50%, not 90%)
- Relevance-based adaptive dropping
- Cold storage for recoverable content
- Full logging for observability

Usage:
    from agent_memfas.context import ContextManager
    
    ctx = ContextManager(
        memory_path="./memfas.json",
        config_path="./memfas.yaml"
    )
    
    # On every message
    ctx.on_message(message, context)
    
    # Before responding (check context health)
    status = ctx.before_response(max_tokens=100000)
    
    # After responding
    ctx.after_response()
    
    # Session end
    ctx.session_end()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import json
import time
from datetime import datetime

from .config import ContextConfig
from .relevance import RelevanceScorer
from .cold_storage import ColdStorage
from .logger import ContextLogger


@dataclass
class ContextStatus:
    """Current context health status."""
    current_tokens: int
    max_tokens: int
    pct_used: float
    chunks_count: int
    cold_storage_count: int
    needs_compaction: bool
    last_compaction: Optional[str] = None


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    tokens_before: int
    tokens_after: int
    chunks_dropped: int
    chunks_summarized: int
    token_savings: int


class ContextManager:
    """
    Main context management orchestrator.
    
    Integrates with agents to manage active context:
    - Tracks token usage in real-time
    - Scores chunks for relevance to current prompt
    - Drops low-relevance chunks to cold storage
    - Summarizes medium chunks when needed
    - Logs all activities for observability
    """
    
    def __init__(
        self,
        memory_path: str = "./memfas.json",
        config_path: Optional[str] = None,
        log_path: Optional[str] = None,
        cold_storage_path: Optional[str] = None,
        memfas_memory_path: Optional[str] = None,
        memfas_backend: str = "fts5",
        # Callbacks for agent integration
        get_context: Optional[Callable[[], list[str]]] = None,
        set_context: Optional[Callable[[list[str]], None]] = None,
        get_token_count: Optional[Callable[[], int]] = None,
        get_messages: Optional[Callable[[int], list[dict]]] = None,
    ):
        """
        Initialize context manager.
        
        Args:
            memory_path: Path to memfas memory store
            config_path: Path to context config YAML
            log_path: Path to log directory
            cold_storage_path: Path to cold storage directory
            memfas_memory_path: Path to memfas memory for similarity scoring
            memfas_backend: Search backend type ("fts5" or "embedding")
            get_context: Callback to get current context strings
            set_context: Callback to replace context
            get_token_count: Callback to count tokens
            get_messages: Callback to get last N messages
        """
        self.memory_path = Path(memory_path)
        self.config = ContextConfig.load(config_path)
        self.log_path = Path(log_path) if log_path else self.memory_path.parent / "logs" / "context"
        self.cold_storage_path = Path(cold_storage_path) if cold_storage_path else self.memory_path.parent / "cold-storage"
        
        # Callbacks for agent integration
        self.get_context = get_context or (lambda: [])
        self.set_context = set_context or (lambda x: None)
        self.get_token_count = get_token_count or (lambda: 0)
        self.get_messages = get_messages or (lambda n: [])
        
        # Initialize memfas integration for relevance scoring
        memfas_client = None
        if memfas_memory_path:
            try:
                from .memfas_integration import MemfasIntegration
                memfas_client = MemfasIntegration(
                    memory_path=memfas_memory_path,
                    backend_type=memfas_backend
                )
            except Exception as e:
                print(f"Warning: Could not initialize memfas integration: {e}")
        
        # Sub-components
        self.scorer = RelevanceScorer(self.config, memfas_client=memfas_client)
        self.cold_storage = ColdStorage(self.cold_storage_path)
        self.logger = ContextLogger(self.log_path)
        
        # Initialize summarizer once (not per-compact)
        self.summarizer = self._init_summarizer()
        
        # State
        self._token_count = 0
        self._last_status: Optional[ContextStatus] = None
        self._current_prompt: str = ""
        self._session_id: str = f"session_{int(time.time())}"
    
    def _init_summarizer(self):
        """Initialize summarizer based on config. Called once at startup."""
        if not self.config.summarize_medium_chunks:
            return None
        try:
            from .summarizer import create_summarizer
            import os
            api_key = os.getenv("MINIMAX_API_KEY", "")
            if api_key:
                return create_summarizer(
                    backend="minimax",
                    api_key=api_key,
                    model=self.config.summary_model,
                    target_tokens=self.config.summary_target_tokens,
                )
            else:
                return create_summarizer(backend="none")
        except Exception as e:
            print(f"Warning: Could not initialize summarizer: {e}")
            return None
        
    def on_message(self, message: str, context: list[str]):
        """
        Called when a new message is received.
        
        Updates token count and logs the message.
        """
        self._token_count = self.get_token_count()
        self.logger.log_message(message, len(context), self._token_count)
        
    def before_response(self, max_tokens: int = 100000) -> ContextStatus:
        """
        Called before generating a response.
        
        Checks context health, triggers compaction if needed.
        
        Returns:
            ContextStatus with current health metrics
        """
        self._token_count = self.get_token_count()
        pct_used = self._token_count / max_tokens if max_tokens > 0 else 0
        
        status = ContextStatus(
            current_tokens=self._token_count,
            max_tokens=max_tokens,
            pct_used=pct_used,
            chunks_count=len(self.get_context()),
            cold_storage_count=self.cold_storage.count(),
            needs_compaction=pct_used >= self.config.compaction_trigger_pct,
            last_compaction=self.logger.last_compaction_time()
        )
        
        self._last_status = status
        self.logger.log_health(status)
        
        # Auto-compact if needed
        if status.needs_compaction and self.config.auto_compact:
            result = self.compact()
            self.logger.log_compaction(result)
            
        return status
        
    def after_response(self):
        """Called after generating a response."""
        self.logger.log_activity("response_complete")
        
    def session_end(self):
        """Called when session ends. Archives context to cold storage."""
        context = self.get_context()
        messages = self.get_messages(20)
        
        self.logger.log_session_end(len(context), len(messages))
        
        # Archive recent context to cold storage for recovery
        self.cold_storage.archive_session(
            session_id=f"session_{int(time.time())}",
            messages=messages,
            context=context
        )
        
    def compact(self) -> CompactionResult:
        """
        Perform context compaction.
        
        Scores all chunks for relevance, drops low-score chunks,
        summarizes medium chunks using configured summarizer backend.
        
        Preserves original chunk ordering in the output.
        Enforces min_chunks_to_keep: if too many would be dropped,
        the lowest-scoring chunks are promoted to "summarize" instead.
        
        Returns:
            CompactionResult with metrics
        """
        context = self.get_context()
        tokens_before = self._token_count
        
        if not context:
            return CompactionResult(
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                chunks_dropped=0,
                chunks_summarized=0,
                token_savings=0,
            )
        
        # Score all chunks (preserving original index)
        scored_chunks = []
        for i, chunk in enumerate(context):
            score = self.scorer.score(chunk, self._current_prompt or "")
            scored_chunks.append((i, chunk, score))
        
        # Classify chunks
        keep = []       # (idx, chunk)
        drop = []       # (idx, chunk, score)
        summarize = []  # (idx, chunk)
        
        for idx, chunk, score in scored_chunks:
            if score >= self.config.relevance_keep_threshold:
                keep.append((idx, chunk))
            elif score <= self.config.relevance_cutoff:
                drop.append((idx, chunk, score))
            else:
                summarize.append((idx, chunk))
        
        # Enforce min_chunks_to_keep:
        # kept + summarized must be >= min_chunks_to_keep
        # If not, promote the highest-scoring drops to summarize
        retained_count = len(keep) + len(summarize)
        if retained_count < self.config.min_chunks_to_keep and drop:
            # Sort drops by score descending — promote best ones first
            drop.sort(key=lambda x: x[2], reverse=True)
            needed = self.config.min_chunks_to_keep - retained_count
            promoted = drop[:needed]
            drop = drop[needed:]
            for idx, chunk, _score in promoted:
                summarize.append((idx, chunk))
        
        # Process drops → cold storage + log each one
        for idx, chunk, score in drop:
            chunk_id = f"chunk_{idx}_{int(time.time())}"
            cold_path = self.cold_storage.store_dropped(
                chunk_id=chunk_id,
                content=chunk,
                relevance_score=score,
                prompt_at_drop=self._current_prompt or "",
                session_id=self._session_id,
            )
            self.logger.log_drop(
                chunk_id=chunk_id,
                session_id=self._session_id,
                relevance_score=score,
                reason="low_relevance",
                cold_storage_path=cold_path,
            )
        
        # Summarize medium chunks
        summarized = []  # (idx, summary_text)
        if self.summarizer and summarize:
            try:
                chunk_texts = [chunk for _, chunk in summarize]
                summaries = self.summarizer.summarize_batch(
                    chunks=chunk_texts,
                    prompt=self._current_prompt or "",
                )
                for (idx, _), summary in zip(summarize, summaries):
                    summarized.append((idx, summary))
            except Exception as e:
                print(f"Summarization error: {e}, keeping medium chunks as-is")
                summarized = [(idx, chunk) for idx, chunk in summarize]
        else:
            # No summarizer — keep medium chunks unchanged
            summarized = [(idx, chunk) for idx, chunk in summarize]
        
        # Build new context preserving original order
        # Merge kept and summarized by original index
        output_map = {}
        for idx, chunk in keep:
            output_map[idx] = chunk
        for idx, text in summarized:
            output_map[idx] = text
        
        new_context = [output_map[idx] for idx in sorted(output_map.keys())]
        
        # Update context via callback
        self.set_context(new_context)
        
        # Recalculate tokens
        self._token_count = self.get_token_count()
        
        result = CompactionResult(
            tokens_before=tokens_before,
            tokens_after=self._token_count,
            chunks_dropped=len(drop),
            chunks_summarized=len(summarized),
            token_savings=tokens_before - self._token_count,
        )
        
        return result
        
    def status(self) -> ContextStatus:
        """Get current context status."""
        if self._last_status:
            return self._last_status
        return self.before_response()
        
    def recover(self, query: str) -> list[str]:
        """
        Try to recover relevant chunks from cold storage.
        
        Args:
            query: Search query for relevance
            
        Returns:
            List of recovered content
        """
        return self.cold_storage.search_recover(query)
        
    def set_current_prompt(self, prompt: str):
        """Set the current prompt for relevance scoring."""
        self._current_prompt = prompt
        
    def health_check(self) -> dict:
        """Return health metrics for monitoring."""
        return {
            "tokens": self._token_count,
            "cold_storage_count": self.cold_storage.count(),
            "config": {
                "compaction_trigger_pct": self.config.compaction_trigger_pct,
                "relevance_cutoff": self.config.relevance_cutoff,
            }
        }
