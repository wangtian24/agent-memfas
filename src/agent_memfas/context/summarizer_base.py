"""
Base summarizer interface for context condensation.

Abstract base class for summarization backends.
Different implementations can use different APIs or local models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class Summarizer(ABC):
    """
    Abstract base class for context summarization.
    
    Implementations:
    - MiniMaxSummarizer: Uses MiniMax API
    - OpenAISummarizer: Uses OpenAI API
    - LocalSummarizer: Uses local model (Ollama, etc.)
    """
    
    @abstractmethod
    def summarize(self, chunk: str, prompt: str = "", preserve_keywords: List[str] = None) -> str:
        """
        Summarize a single chunk.
        
        Args:
            chunk: Content to summarize
            prompt: Current user prompt (for context)
            preserve_keywords: Keywords to ensure are preserved
            
        Returns:
            Summarized content
        """
        pass
    
    @abstractmethod
    def summarize_batch(self, chunks: List[str], prompt: str = "", preserve_keywords: List[str] = None) -> List[str]:
        """
        Summarize multiple chunks.
        
        Args:
            chunks: List of chunks to summarize
            prompt: Current user prompt
            preserve_keywords: Keywords to preserve
            
        Returns:
            List of summarized chunks
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        pass
