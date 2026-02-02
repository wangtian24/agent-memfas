"""Embedding providers for agent-memfas."""

from .base import Embedder

__all__ = ["Embedder"]

# Lazy imports for optional embedders
def get_fastembed_embedder():
    """Get FastEmbedEmbedder (requires fastembed)."""
    from .fastembed import FastEmbedEmbedder
    return FastEmbedEmbedder

def get_ollama_embedder():
    """Get OllamaEmbedder (requires requests)."""
    from .ollama import OllamaEmbedder
    return OllamaEmbedder
