"""Topic detection for context curation."""

import re
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

from .types import Topic

if TYPE_CHECKING:
    from ..embedders.base import Embedder


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TopicDetector:
    """
    Lightweight topic detection using embeddings.
    
    Detects current conversation topic and tracks topic shifts.
    Uses the same embedder as the memory system - no new models needed.
    """
    
    def __init__(
        self,
        embedder: "Embedder",
        shift_threshold: float = 0.6,
        history_size: int = 10
    ):
        """
        Initialize topic detector.
        
        Args:
            embedder: Embedder for generating topic embeddings
            shift_threshold: Similarity below this = topic shift
            history_size: Max topics to track
        """
        self.embedder = embedder
        self.shift_threshold = shift_threshold
        self.history_size = history_size
        self.topic_history: List[Topic] = []
    
    def detect(
        self,
        message: str,
        recent_history: Optional[List[str]] = None
    ) -> Topic:
        """
        Detect current topic from message and recent context.
        
        Args:
            message: Current user message
            recent_history: Last few messages for context
        
        Returns:
            Topic with name, embedding, and confidence
        """
        # Combine message with recent context for better signal
        context_parts = []
        if recent_history:
            context_parts.extend(recent_history[-3:])
        context_parts.append(message)
        context = "\n".join(context_parts)
        
        # Generate embedding
        embedding = self.embedder.embed(context)
        
        # Extract topic name from message
        name = self._extract_topic_name(message)
        
        # Calculate confidence based on clarity of topic
        confidence = self._calculate_confidence(message, embedding)
        
        topic = Topic(
            name=name,
            embedding=embedding,
            confidence=confidence
        )
        
        # Add to history (keep bounded)
        self.topic_history.append(topic)
        if len(self.topic_history) > self.history_size:
            self.topic_history.pop(0)
        
        return topic
    
    def detect_shift(self) -> bool:
        """
        Check if topic shifted from previous turn.
        
        Returns:
            True if topic changed significantly
        """
        if len(self.topic_history) < 2:
            return False
        
        prev = self.topic_history[-2]
        curr = self.topic_history[-1]
        
        similarity = cosine_similarity(prev.embedding, curr.embedding)
        return similarity < self.shift_threshold
    
    def get_current_topic(self) -> Optional[Topic]:
        """Get the current topic if any."""
        return self.topic_history[-1] if self.topic_history else None
    
    def _extract_topic_name(self, message: str) -> str:
        """
        Extract key topic from message.
        
        Simple heuristic: find most distinctive noun phrase or entity.
        """
        # Remove common question words
        cleaned = re.sub(
            r'\b(what|how|when|where|why|who|is|are|the|a|an|can|could|would|should|do|does|did)\b',
            '',
            message.lower()
        )
        
        # Find capitalized words (likely entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', message)
        if entities:
            return entities[0].lower()
        
        # Find longest remaining word as topic
        words = [w for w in cleaned.split() if len(w) > 3]
        if words:
            return max(words, key=len)
        
        # Fallback
        return "general"
    
    def _calculate_confidence(
        self,
        message: str,
        embedding: List[float]
    ) -> float:
        """
        Calculate confidence in topic detection.
        
        Higher confidence for:
        - Specific entities
        - Clear questions
        - Matches to previous topics
        """
        confidence = 0.5  # Base confidence
        
        # Boost for capitalized entities
        if re.search(r'\b[A-Z][a-z]+\b', message):
            confidence += 0.2
        
        # Boost for specific question patterns
        if re.search(r'\b(status|update|progress|about|working on)\b', message.lower()):
            confidence += 0.15
        
        # Boost if similar to recent topics
        if len(self.topic_history) > 0:
            recent = self.topic_history[-1]
            sim = cosine_similarity(embedding, recent.embedding)
            if sim > 0.8:
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def reset(self):
        """Clear topic history."""
        self.topic_history.clear()
