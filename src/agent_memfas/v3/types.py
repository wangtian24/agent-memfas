"""Core data types for v3."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Topic:
    """Detected conversation topic."""
    name: str
    embedding: List[float]
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MemoryChunk:
    """A chunk of memory with metadata."""
    id: str
    text: str
    source: str
    embedding: List[float]
    section: Optional[str] = None
    date: Optional[str] = None
    token_count: int = 0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredMemory:
    """Memory with relevance score."""
    memory: MemoryChunk
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    # breakdown: {semantic: 0.8, recency: 0.6, access: 0.3, topic: 0.7}


@dataclass
class CuratedContext:
    """Result of budget allocation."""
    memories: List[MemoryChunk]
    total_tokens: int
    budget: int
    utilization: float  # total_tokens / budget
    dropped: List[MemoryChunk] = field(default_factory=list)


@dataclass
class ContextResponse:
    """Response from get_context()."""
    context: str  # Formatted context to inject
    tokens_used: int
    budget: int
    
    # What was included
    memories_included: int
    memories_dropped: int  # Didn't fit in budget
    memories_filtered: int = 0  # Below min_score threshold
    triggers_matched: int = 0
    
    # Topic info
    topic: str = ""
    topic_shifted: bool = False
    
    # Savings
    baseline_tokens: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 1.0
    
    # Performance
    latency_ms: float = 0.0
    
    # Curation level
    curation_level: int = 3
    curation_level_name: str = "balanced"
    
    # Debug info
    top_memories: List[Dict[str, Any]] = field(default_factory=list)
