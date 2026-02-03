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
        Score multiple chunks for relevance to prompt.
        
        Args:
            chunks: List of context chunks
            prompt: Current user prompt
            limit: Maximum chunks to search for
            
        Returns:
            List of relevance scores (0.0 to 1.0+)
        """
        if not chunks:
            return []
            
        try:
            memory = self._get_memory()
            
            # Search for relevant chunks
            # memfas returns SearchResult with score
            results: List["SearchResult"] = memory.search(
                query=prompt,
                limit=min(limit, len(chunks))
            )
            
            # Create score map from results
            # Map source/chunk_id to score
            score_map = {}
            for result in results:
                source = getattr(result, 'source', '') or getattr(result, 'metadata', {}).get('source', '')
                score = getattr(result, 'score', 0.5)
                if source:
                    score_map[source] = score
            
            # Score each chunk
            scores = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{i}"
                score = score_map.get(chunk_id, 0.0)
                scores.append(score)
                
            return scores
            
        except Exception as e:
            # Return fallback scores on error
            print(f"memfas scoring error: {e}, using fallback")
            return [0.0] * len(chunks)
    
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
