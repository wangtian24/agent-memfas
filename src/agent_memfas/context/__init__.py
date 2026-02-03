# Context Management Module
# Part of agent-memfas v4

from .context_manager import ContextManager, ContextStatus, CompactionResult
from .config import ContextConfig
from .relevance import RelevanceScorer
from .cold_storage import ColdStorage, DroppedChunk
from .logger import ContextLogger
from .memfas_integration import MemfasIntegration, create_memfas_scorer
from .summarizer import (
    Summarizer,           # Abstract base class
    NoOpSummarizer,       # Pass-through
    MiniMaxSummarizer,    # MiniMax implementation
    create_summarizer,    # Factory function
)
from .minimax_summarizer import create_minimax_summarizer

__all__ = [
    "ContextManager",
    "ContextStatus",
    "CompactionResult",
    "ContextConfig",
    "RelevanceScorer",
    "ColdStorage",
    "DroppedChunk",
    "ContextLogger",
    "MemfasIntegration",
    "create_memfas_scorer",
    "Summarizer",           # Abstract base class
    "NoOpSummarizer",       # Pass-through
    "MiniMaxSummarizer",    # MiniMax implementation
    "create_summarizer",    # Factory function
    "create_minimax_summarizer",
]
