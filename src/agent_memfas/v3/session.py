"""Session state management for context curation."""

import re
from typing import List, Dict, Set, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .types import Topic
from .topic import cosine_similarity


@dataclass
class SessionState:
    """
    Maintains state across a conversation session.
    
    Tracks:
    - Current and historical topics
    - Active entities (people, projects mentioned)
    - Memory access patterns (what gets used)
    - Turn count and timing
    """
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Topic tracking
    current_topic: Optional[Topic] = None
    topic_history: List[Topic] = field(default_factory=list)
    
    # Entity tracking
    active_entities: Set[str] = field(default_factory=set)
    entity_mentions: Dict[str, int] = field(default_factory=dict)  # entity -> count
    
    # Memory access tracking
    accessed_memories: Dict[str, int] = field(default_factory=dict)  # memory_id -> count
    last_used_memories: List[str] = field(default_factory=list)  # Recent memory IDs
    
    # Turn tracking
    turn_count: int = 0
    last_turn_at: Optional[datetime] = None
    
    def on_turn(
        self,
        query: str,
        topic: Topic,
        used_memory_ids: List[str]
    ):
        """
        Update state after each turn.
        
        Args:
            query: User's query
            topic: Detected topic
            used_memory_ids: Memory IDs included in context
        """
        self.turn_count += 1
        self.last_turn_at = datetime.utcnow()
        
        # Update topic
        self.current_topic = topic
        self.topic_history.append(topic)
        
        # Keep topic history bounded
        if len(self.topic_history) > 20:
            self.topic_history.pop(0)
        
        # Track memory access
        for mem_id in used_memory_ids:
            self.accessed_memories[mem_id] = \
                self.accessed_memories.get(mem_id, 0) + 1
        
        # Track recent memories
        self.last_used_memories = used_memory_ids[:10]
        
        # Extract and track entities
        entities = self._extract_entities(query)
        self.active_entities.update(entities)
        for entity in entities:
            self.entity_mentions[entity] = \
                self.entity_mentions.get(entity, 0) + 1
    
    def get_boost_ids(self, min_access: int = 2) -> Set[str]:
        """
        Get memory IDs that should be boosted.
        
        Memories accessed multiple times in this session
        are likely important and should score higher.
        
        Args:
            min_access: Minimum access count to qualify
        
        Returns:
            Set of memory IDs to boost
        """
        return {
            mid for mid, count in self.accessed_memories.items()
            if count >= min_access
        }
    
    def get_hot_entities(self, min_mentions: int = 2) -> Set[str]:
        """
        Get entities mentioned multiple times.
        
        These can be used to boost related memories.
        """
        return {
            entity for entity, count in self.entity_mentions.items()
            if count >= min_mentions
        }
    
    def detect_topic_shift(self, threshold: float = 0.6) -> bool:
        """
        Detect if topic changed significantly from previous turn.
        
        Args:
            threshold: Similarity below this = shift
        
        Returns:
            True if topic shifted
        """
        if len(self.topic_history) < 2:
            return False
        
        prev = self.topic_history[-2]
        curr = self.topic_history[-1]
        
        similarity = cosine_similarity(prev.embedding, curr.embedding)
        return similarity < threshold
    
    def get_session_duration_minutes(self) -> float:
        """Get session duration in minutes."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 60
    
    def get_stats(self) -> Dict:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "duration_minutes": self.get_session_duration_minutes(),
            "unique_topics": len(set(t.name for t in self.topic_history)),
            "active_entities": len(self.active_entities),
            "memories_accessed": len(self.accessed_memories),
            "hot_memories": len(self.get_boost_ids())
        }
    
    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract entities from text.
        
        Simple heuristic: capitalized words, project names, etc.
        Could be enhanced with NER.
        """
        entities = set()
        
        # Capitalized words (likely names, projects)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(w.lower() for w in caps)
        
        # Words after "the" that might be project names
        the_words = re.findall(r'\bthe\s+(\w+)\b', text.lower())
        entities.update(the_words)
        
        # Filter out common words
        stopwords = {'the', 'this', 'that', 'what', 'how', 'when', 'where', 'who'}
        entities -= stopwords
        
        return entities
    
    def reset(self):
        """Reset session state."""
        self.current_topic = None
        self.topic_history.clear()
        self.active_entities.clear()
        self.entity_mentions.clear()
        self.accessed_memories.clear()
        self.last_used_memories.clear()
        self.turn_count = 0
