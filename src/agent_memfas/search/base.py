"""Base classes for search backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SearchResult:
    """Universal search result from any backend."""
    text: str
    score: float
    source: str
    section: Optional[str] = None
    date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        snippet = self.text[:200] + "..." if len(self.text) > 200 else self.text
        return f"[{self.source}] (score: {self.score:.3f})\n  > {snippet}"


class SearchBackend(ABC):
    """
    Base class for Type 2 search backends.
    
    Implementations:
    - FTS5Backend: SQLite FTS5 full-text search (default, zero deps)
    - EmbeddingBackend: Vector similarity search (optional deps)
    """
    
    @abstractmethod
    def index(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Index a document.
        
        Args:
            doc_id: Unique document identifier
            text: Document text to index
            metadata: Optional metadata (source, section, date, etc.)
        """
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Search for documents.
        
        Args:
            query: Search query
            limit: Maximum results to return
        
        Returns:
            List of SearchResult objects, sorted by relevance
        """
        pass
    
    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove a document from the index."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed documents."""
        pass
    
    def count(self) -> int:
        """Return number of indexed documents. Override for efficiency."""
        return 0
    
    def close(self) -> None:
        """Cleanup resources. Override if needed."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
