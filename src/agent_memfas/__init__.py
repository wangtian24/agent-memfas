"""
agent-memfas: Memory Fast and Slow for AI Agents

Dual-store memory with:
- Type 1 (Fast): Keyword triggers for instant recall
- Type 2 (Slow): Pluggable search (FTS5 or embeddings)

Usage:
    from agent_memfas import Memory, Config
    
    # Default (FTS5, zero deps)
    mem = Memory("./memfas.yaml")
    context = mem.recall("How's the family?")
    
    # With embeddings (optional deps)
    from agent_memfas.embedders.fastembed import FastEmbedEmbedder
    mem = Memory("./memfas.yaml", embedder=FastEmbedEmbedder())
"""

from .config import Config
from .memory import Memory, MemoryResult
from .search.base import SearchBackend, SearchResult

__version__ = "0.2.0"
__all__ = [
    "Memory",
    "MemoryResult", 
    "Config",
    "SearchBackend",
    "SearchResult",
]
