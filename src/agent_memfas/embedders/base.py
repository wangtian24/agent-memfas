"""Base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """
    Base class for embedding providers.
    
    Implementations:
    - FastEmbedEmbedder: Local embeddings via FastEmbed (recommended)
    - OllamaEmbedder: Local embeddings via Ollama
    """
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return vector dimensions."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts. Override for efficiency.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        return [self.embed(t) for t in texts]
    
    def close(self) -> None:
        """Cleanup resources. Override if needed."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
