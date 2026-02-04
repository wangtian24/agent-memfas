"""
memfas integration for context management.

Provides similarity scoring using memfas's Type 2 search backend.
"""

from typing import Optional, List, TYPE_CHECKING
import sys
from pathlib import Path

# Add src to path for memfas imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if TYPE_CHECKING:
    from agent_memfas.memory import Memory
    from agent_memfas.search.base import SearchResult


class MemfasIntegration:
    """
    Integration with agent-memfas for similarity scoring.
    
    Uses memfas's Type 2 search backend to score context chunks
    for relevance to the current prompt.
    """
    
    def __init__(
        self,
        memory_path: str = "./memfas.json",
        backend_type: str = "fts5",
    ):
        """
        Initialize memfas integration.
        
        Args:
            memory_path: Path to memfas memory store
            backend_type: Search backend type ("fts5" or "embedding")
        """
        self.memory_path = memory_path
        self.backend_type = backend_type
        self._memory: Optional["Memory"] = None
        
    def _get_memory(self) -> "Memory":
        """Lazy load memfas Memory instance."""
        if self._memory is None:
            from agent_memfas.memory import Memory
            from agent_memfas.config import Config
            
            # Create config with search backend
            config = Config(
                db_path=str(Path(self.memory_path).parent / "memfas.db"),
            )
            
            # Override search backend
            config.search = type('SearchConfig', (), {
                'backend': self.backend_type,
                'max_results': 10,
                'recency_weight': 0.3,
                'min_score': 0.0,
                'embedder_type': None,
                'embedder_model': None,
            })()
            
            self._memory = Memory(config)
        return self._memory
    
    def score_chunks(
        self,
        chunks: List[str],
        prompt: str,
        limit: int = 5,
    ) -> List[float]:
        """
        Score multiple chunks for relevance to prompt using embedding similarity.
        
        Each chunk is scored independently against the prompt by searching
        memfas with the chunk content and checking how well it aligns,
        OR by directly computing embedding cosine similarity if available.
        
        Falls back to keyword overlap if embeddings unavailable.
        
        Args:
            chunks: List of context chunks
            prompt: Current user prompt
            limit: Maximum chunks to search for
            
        Returns:
            List of relevance scores (0.0 to 1.0)
        """
        if not chunks:
            return []
            
        try:
            memory = self._get_memory()
            
            # If embedding backend is available, use direct cosine similarity
            if hasattr(memory, '_embedder') and memory._embedder is not None:
                return self._score_with_embeddings(memory, chunks, prompt)
            
            # Fallback: score each chunk by searching memfas with it
            # and seeing if the prompt appears in results
            return self._score_with_search(memory, chunks, prompt, limit)
            
        except Exception as e:
            print(f"memfas scoring error: {e}, using fallback")
            return [0.0] * len(chunks)
    
    def _score_with_embeddings(self, memory, chunks: List[str], prompt: str) -> List[float]:
        """Score chunks using direct embedding cosine similarity."""
        import numpy as np
        
        embedder = memory._embedder
        
        # Embed the prompt once
        prompt_emb = embedder.embed(prompt)
        prompt_arr = np.array(prompt_emb, dtype=np.float32)
        prompt_norm = np.linalg.norm(prompt_arr)
        if prompt_norm == 0:
            return [0.0] * len(chunks)
        prompt_arr = prompt_arr / prompt_norm
        
        scores = []
        for chunk in chunks:
            # Truncate very long chunks to first ~500 words for embedding
            chunk_text = " ".join(chunk.split()[:500])
            chunk_emb = embedder.embed(chunk_text)
            chunk_arr = np.array(chunk_emb, dtype=np.float32)
            chunk_norm = np.linalg.norm(chunk_arr)
            if chunk_norm == 0:
                scores.append(0.0)
                continue
            chunk_arr = chunk_arr / chunk_norm
            similarity = float(np.dot(prompt_arr, chunk_arr))
            scores.append(max(0.0, similarity))  # clamp negatives
            
        return scores
    
    def _score_with_search(self, memory, chunks: List[str], prompt: str, limit: int) -> List[float]:
        """Fallback: use FTS5 search to approximate relevance."""
        scores = []
        for chunk in chunks:
            # Search memfas using a blend of prompt + chunk keywords
            # Use first 50 words of chunk as query context
            chunk_words = chunk.split()[:50]
            query = prompt + " " + " ".join(chunk_words)
            try:
                results = memory.search(query=query, limit=1)
                if results:
                    # Normalize: memfas scores vary by backend
                    raw_score = getattr(results[0], 'score', 0.0)
                    scores.append(min(raw_score, 1.0))
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        return scores
    
    def find_relevant(
        self,
        chunks: List[str],
        prompt: str,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> List[int]:
        """
        Find indices of chunks above relevance threshold.
        
        Args:
            chunks: List of context chunks
            prompt: Current user prompt
            threshold: Minimum score to be considered relevant
            limit: Maximum results to return
            
        Returns:
            List of chunk indices above threshold
        """
        scores = self.score_chunks(chunks, prompt, limit=limit)
        
        relevant = []
        for i, score in enumerate(scores):
            if score >= threshold:
                relevant.append(i)
                
        return relevant
    
    def close(self):
        """Close memfas connection."""
        if self._memory:
            try:
                if hasattr(self._memory, 'close'):
                    self._memory.close()
            except Exception:
                pass
            self._memory = None


def create_memfas_scorer(
    memory_path: str = "./memfas.json",
    backend_type: str = "fts5",
) -> MemfasIntegration:
    """
    Factory function to create memfas integration.
    
    Args:
        memory_path: Path to memfas memory store
        backend_type: Search backend type
        
    Returns:
        MemfasIntegration instance
    """
    return MemfasIntegration(memory_path, backend_type)
