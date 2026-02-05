"""Search backends for agent-memfas."""

from .base import SearchBackend, SearchResult
from .fts5 import FTS5Backend

__all__ = ["SearchBackend", "SearchResult", "FTS5Backend"]

# Lazy imports for optional backends
def get_embedding_backend():
    """Get EmbeddingBackend (requires sqlite-vec)."""
    from .embedding import EmbeddingBackend
    return EmbeddingBackend

def get_journal_backend():
    """Get JournalSearchBackend (requires ollama running)."""
    from .journal import JournalSearchBackend
    return JournalSearchBackend
