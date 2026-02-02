"""
agent-memfas v3: Dynamic Context Engineering

Proactively curates memory context each turn instead of
reactively recovering after compaction.

Usage:
    from agent_memfas.v3 import ContextCurator
    
    curator = ContextCurator("./memfas.yaml", level=3)  # or "balanced"
    result = curator.get_context(
        query="what's the project status?",
        session_id="main",
        baseline_tokens=85000  # Current context size
    )
    print(result.context)  # Curated, focused context
    print(result.tokens_saved)  # How much we saved

Curation Levels (1-5):
    1. minimal  - ~300 tokens, threshold 0.75 - Only near-exact matches
    2. lean     - ~800 tokens, threshold 0.55 - High-confidence only  
    3. balanced - ~1500 tokens, threshold 0.40 - Default, good coverage
    4. rich     - ~3000 tokens, threshold 0.25 - Include "probably relevant"
    5. full     - ~5000 tokens, threshold 0.10 - Kitchen sink
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
from .levels import (
    LevelPreset,
    LEVEL_PRESETS,
    LEVEL_NAMES,
    DEFAULT_LEVEL,
    resolve_level,
    get_preset,
    describe_levels
)

__all__ = [
    # Main entry point
    "ContextCurator",
    "ContextResponse",
    
    # Levels
    "LevelPreset",
    "LEVEL_PRESETS",
    "LEVEL_NAMES", 
    "DEFAULT_LEVEL",
    "resolve_level",
    "get_preset",
    "describe_levels",
    
    # Components
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
