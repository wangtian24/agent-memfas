"""
Summarization backends for context management.

Abstract interface with pluggable implementations:
- MiniMaxSummarizer: Uses MiniMax API
"""

from typing import List

from .summarizer_base import Summarizer
from .minimax_summarizer import (
    MiniMaxSummarizer,
    create_minimax_summarizer,
)

# Factory function for creating summarizers
def create_summarizer(
    backend: str = "minimax",
    api_key: str = None,
    config_path: str = None,
    **kwargs,
) -> Summarizer:
    """
    Factory function to create a summarizer by backend type.
    
    Args:
        backend: Backend type ("minimax", "openai", "local", "none")
        api_key: API key for the backend
        config_path: Path to config file
        **kwargs: Additional arguments for the backend
        
    Returns:
        Summarizer instance
        
    Example:
        # Use MiniMax
        summarizer = create_summarizer("minimax", api_key="...")
        
        # Use local/Ollama
        summarizer = create_summarizer("local", model="llama3")
        
        # No-op summarizer (pass-through)
        summarizer = create_summarizer("none")
    """
    if backend == "minimax":
        return create_minimax_summarizer(api_key=api_key, config_path=config_path, **kwargs)
    elif backend == "none":
        return NoOpSummarizer()
    else:
        print(f"Warning: Unknown summarizer backend '{backend}', using no-op")
        return NoOpSummarizer()


class NoOpSummarizer(Summarizer):
    """
    No-op summarizer that returns chunks unchanged.
    
    Use this when you don't want summarization but want the
    context manager to function normally.
    """
    
    def summarize(
        self,
        chunk: str,
        prompt: str = "",
        preserve_keywords: List[str] = None,
    ) -> str:
        """Return chunk unchanged."""
        return chunk
    
    def summarize_batch(
        self,
        chunks: List[str],
        prompt: str = "",
        preserve_keywords: List[str] = None,
    ) -> List[str]:
        """Return chunks unchanged."""
        return chunks
    
    def estimate_tokens(self, text: str) -> int:
        """Return rough token estimate."""
        return len(text) // 4


__all__ = [
    "Summarizer",           # Abstract base class
    "MiniMaxSummarizer",    # MiniMax implementation
    "create_minimax_summarizer",
    "create_summarizer",    # Factory function
    "NoOpSummarizer",       # Pass-through
]
