"""Token budget management for context curation."""

from typing import List, Optional

from .types import MemoryChunk, ScoredMemory, CuratedContext


class TokenBudget:
    """
    Manages token allocation for memory context.
    
    Fills a fixed budget with highest-value memories,
    respecting token limits per chunk.
    """
    
    def __init__(
        self,
        total_budget: int = 8000,
        min_chunk_tokens: int = 20,
        max_chunk_tokens: int = 2000,
        encoding: str = "cl100k_base"
    ):
        """
        Initialize budget manager.
        
        Args:
            total_budget: Total tokens available for memory context
            min_chunk_tokens: Skip chunks smaller than this
            max_chunk_tokens: Truncate or skip chunks larger than this
            encoding: Tiktoken encoding name
        """
        self.total_budget = total_budget
        self.min_chunk = min_chunk_tokens
        self.max_chunk = max_chunk_tokens
        self._tokenizer = None
        self._encoding = encoding
    
    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self._encoding)
            except ImportError:
                # Fallback: rough estimate
                self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Uses tiktoken if available, otherwise estimates.
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate: ~4 chars per token
            return len(text) // 4
    
    def allocate(
        self,
        scored_memories: List[ScoredMemory],
        reserved: int = 0
    ) -> CuratedContext:
        """
        Fill budget with highest-scoring memories.
        
        Args:
            scored_memories: Memories sorted by score (descending)
            reserved: Tokens reserved for triggers, etc.
        
        Returns:
            CuratedContext with selected memories and stats
        """
        available = self.total_budget - reserved
        selected = []
        dropped = []
        total_tokens = 0
        
        for sm in scored_memories:
            mem = sm.memory
            
            # Get or calculate token count
            if mem.token_count == 0:
                mem.token_count = self.count_tokens(mem.text)
            
            tokens = mem.token_count
            
            # Skip too small
            if tokens < self.min_chunk:
                continue
            
            # Skip too large (could truncate instead)
            if tokens > self.max_chunk:
                dropped.append(mem)
                continue
            
            # Check if it fits
            if total_tokens + tokens <= available:
                selected.append(mem)
                total_tokens += tokens
            else:
                dropped.append(mem)
        
        utilization = total_tokens / self.total_budget if self.total_budget > 0 else 0
        
        return CuratedContext(
            memories=selected,
            total_tokens=total_tokens,
            budget=self.total_budget,
            utilization=utilization,
            dropped=dropped
        )
    
    def set_budget(self, tokens: int):
        """Update total budget."""
        self.total_budget = max(tokens, 100)  # Min 100 tokens
    
    def estimate_chunks_for_budget(
        self,
        avg_chunk_tokens: int = 300
    ) -> int:
        """Estimate how many chunks fit in budget."""
        return self.total_budget // avg_chunk_tokens
