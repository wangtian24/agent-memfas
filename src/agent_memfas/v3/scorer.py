"""Relevance scoring for memory curation."""

import math
from typing import List, Dict, Optional, Set, TYPE_CHECKING
from datetime import datetime

from .types import MemoryChunk, ScoredMemory, Topic
from .topic import cosine_similarity

if TYPE_CHECKING:
    from ..embedders.base import Embedder


class RelevanceScorer:
    """
    Multi-factor relevance scoring for memory chunks.
    
    Combines:
    - Semantic similarity (embedding distance)
    - Recency decay (when was it created/accessed)
    - Access frequency (hot memories)
    - Topic coherence (matches current topic)
    """
    
    def __init__(
        self,
        embedder: "Embedder",
        semantic_weight: float = 0.6,
        recency_weight: float = 0.2,
        access_weight: float = 0.1,
        topic_weight: float = 0.1,
        recency_half_life_days: float = 7.0
    ):
        """
        Initialize scorer with configurable weights.
        
        Args:
            embedder: Embedder for query embedding
            semantic_weight: Weight for semantic similarity [0-1]
            recency_weight: Weight for recency [0-1]
            access_weight: Weight for access frequency [0-1]
            topic_weight: Weight for topic coherence [0-1]
            recency_half_life_days: Half-life for recency decay
        """
        self.embedder = embedder
        self.weights = {
            "semantic": semantic_weight,
            "recency": recency_weight,
            "access": access_weight,
            "topic": topic_weight
        }
        self.recency_half_life = recency_half_life_days
        
        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize
            for k in self.weights:
                self.weights[k] /= total
    
    def score(
        self,
        query: str,
        memories: List[MemoryChunk],
        current_topic: Optional[Topic] = None,
        boost_ids: Optional[Set[str]] = None,
        boost_factor: float = 1.2
    ) -> List[ScoredMemory]:
        """
        Score all memories by relevance to query.
        
        Args:
            query: Current user query
            memories: All memory chunks to score
            current_topic: Current conversation topic (optional)
            boost_ids: Memory IDs to boost (from session state)
            boost_factor: Multiplier for boosted memories
        
        Returns:
            List of ScoredMemory, sorted by score descending
        """
        if not memories:
            return []
        
        # Embed query
        query_embedding = self.embedder.embed(query)
        now = datetime.utcnow()
        boost_ids = boost_ids or set()
        
        scored = []
        
        for mem in memories:
            # 1. Semantic similarity
            semantic = cosine_similarity(query_embedding, mem.embedding)
            
            # 2. Recency decay
            recency = self._calculate_recency(mem, now)
            
            # 3. Access frequency (log scale, capped)
            access = min(math.log1p(mem.access_count) / 5.0, 1.0)
            
            # 4. Topic coherence
            topic_score = 0.5  # Default if no topic
            if current_topic and mem.embedding:
                topic_score = cosine_similarity(
                    current_topic.embedding,
                    mem.embedding
                )
            
            # Weighted combination
            final_score = (
                self.weights["semantic"] * semantic +
                self.weights["recency"] * recency +
                self.weights["access"] * access +
                self.weights["topic"] * topic_score
            )
            
            # Apply boost for hot memories
            if mem.id in boost_ids:
                final_score *= boost_factor
            
            scored.append(ScoredMemory(
                memory=mem,
                score=final_score,
                breakdown={
                    "semantic": semantic,
                    "recency": recency,
                    "access": access,
                    "topic": topic_score,
                    "boosted": mem.id in boost_ids
                }
            ))
        
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored
    
    def _calculate_recency(
        self,
        memory: MemoryChunk,
        now: datetime
    ) -> float:
        """
        Calculate recency score with exponential decay.
        
        Score of 1.0 for today, 0.5 after half_life days, etc.
        """
        if memory.last_accessed:
            ref_time = memory.last_accessed
        elif memory.date:
            try:
                ref_time = datetime.fromisoformat(memory.date.replace('Z', '+00:00'))
            except:
                return 0.5  # Unknown date
        else:
            return 0.5  # Unknown date
        
        # Handle timezone-naive comparison
        if ref_time.tzinfo is not None:
            ref_time = ref_time.replace(tzinfo=None)
        
        days_old = (now - ref_time).days
        
        # Exponential decay with half-life
        return 0.5 ** (days_old / self.recency_half_life)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current scoring weights."""
        return self.weights.copy()
    
    def set_weights(
        self,
        semantic: float = None,
        recency: float = None,
        access: float = None,
        topic: float = None
    ):
        """
        Update scoring weights.
        
        Weights are normalized to sum to 1.0.
        """
        if semantic is not None:
            self.weights["semantic"] = semantic
        if recency is not None:
            self.weights["recency"] = recency
        if access is not None:
            self.weights["access"] = access
        if topic is not None:
            self.weights["topic"] = topic
        
        # Normalize
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total
