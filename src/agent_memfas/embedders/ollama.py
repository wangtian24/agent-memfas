"""Ollama embedder - local embeddings via Ollama."""

from typing import List

from .base import Embedder


class OllamaEmbedder(Embedder):
    """
    Local embeddings via Ollama.
    
    Requires Ollama running locally. Good for larger models.
    
    Recommended models:
    - nomic-embed-text: 768 dims, ~270MB, great quality
    - mxbai-embed-large: 1024 dims, higher quality
    - all-minilm: 384 dims, faster
    
    Install: brew install ollama && ollama pull nomic-embed-text
    
    Usage:
        embedder = OllamaEmbedder()
        vector = embedder.embed("Hello world")
    """
    
    # Model dimensions lookup
    MODEL_DIMS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama embedder.
        
        Args:
            model: Model name (default: nomic-embed-text)
            base_url: Ollama API URL (default: http://localhost:11434)
        """
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests not installed. Install with: pip install requests"
            )
        
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimensions = self.MODEL_DIMS.get(model, 768)
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    def embed(self, text: str) -> List[float]:
        """Embed single text via Ollama API."""
        resp = self._requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (sequential, Ollama doesn't batch)."""
        return [self.embed(t) for t in texts]
