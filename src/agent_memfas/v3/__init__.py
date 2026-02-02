"""
agent-memfas v3: Dynamic Context Engineering

Proactively curates memory context each turn instead of
reactively recovering after compaction.

Usage:
    from agent_memfas.v3 import ContextCurator
    
    curator = ContextCurator("./memfas.yaml")
    result = curator.get_context(
        query="what's the project status?",
        session_id="main",
        baseline_tokens=85000  # Current context size
    )
    print(result.context)  # Curated, focused context
    print(result.tokens_saved)  # How much we saved
"""

from .types import (
    Topic,
    MemoryChunk,
    ScoredMemory,
    CuratedContext,
    ContextResponse
)
from .topic import TopicDetector
from .scorer import RelevanceScorer
from .budget import TokenBudget
from .session import SessionState
from .telemetry import TelemetryLogger, TurnMetrics, SessionMetrics
from .curator import ContextCurator

__all__ = [
    "ContextCurator",
    "ContextResponse",
    "TopicDetector",
    "Topic",
    "RelevanceScorer",
    "ScoredMemory",
    "TokenBudget",
    "CuratedContext",
    "SessionState",
    "TelemetryLogger",
    "TurnMetrics",
    "SessionMetrics",
]
