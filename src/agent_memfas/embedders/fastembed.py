"""FastEmbed embedder - local embeddings via Qdrant's FastEmbed library."""

from typing import List

from .base import Embedder


class FastEmbedEmbedder(Embedder):
    """
    Local embeddings via FastEmbed (Qdrant).
    
    Uses ONNX runtime for fast CPU inference. Downloads model on first use.
    
    Recommended models:
    - BAAI/bge-small-en-v1.5: 384 dims, ~130MB, good quality
    - sentence-transformers/all-MiniLM-L6-v2: 384 dims, ~80MB, fast
    
    Install: pip install fastembed
    
    Usage:
        embedder = FastEmbedEmbedder()
        vector = embedder.embed("Hello world")
    """
    
    # Model dimensions lookup
    MODEL_DIMS = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "nomic-ai/nomic-embed-text-v1.5": 768,
    }
    
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize FastEmbed embedder.
        
        Args:
            model: Model name (default: BAAI/bge-small-en-v1.5)
        """
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "FastEmbed not installed. Install with: pip install fastembed"
            )
        
        self.model_name = model
        self._model = TextEmbedding(model_name=model)
        self._dimensions = self.MODEL_DIMS.get(model, 384)
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    def embed(self, text: str) -> List[float]:
        """Embed single text."""
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]
