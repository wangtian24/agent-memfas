"""
Cold storage for dropped context chunks.

Provides persistent storage for chunks that are dropped from active context
but can be recovered if they become relevant again.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import json
import time
from datetime import datetime, timedelta


@dataclass
class DroppedChunk:
    """A chunk that was dropped from active context."""
    chunk_id: str
    session_id: str
    content: str
    relevance_score: float
    prompt_at_drop: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tokens: int = 0
    recoverable_until: str = field(default_factory=lambda: (
        datetime.now() + timedelta(days=30)
    ).isoformat())
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "prompt_at_drop": self.prompt_at_drop,
            "timestamp": self.timestamp,
            "tokens": self.tokens,
            "recoverable_until": self.recoverable_until,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DroppedChunk":
        return cls(**data)


class ColdStorage:
    """
    Storage for dropped context chunks.
    
    Stores dropped chunks in a session-based directory structure,
    indexed for recovery search.
    """
    
    def __init__(self, storage_path: str = "./cold-storage/"):
        """
        Initialize cold storage.
        
        Args:
            storage_path: Path to cold storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Index for recovery search (simple text index)
        self.index_path = self.storage_path / "index.jsonl"
        
    def store_dropped(
        self,
        chunk_id: str,
        content: str,
        relevance_score: float,
        prompt_at_drop: str,
        session_id: Optional[str] = None,
        tokens: int = 0,
        recoverable_days: int = 30,
    ) -> str:
        """
        Store a dropped chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            content: The dropped content
            relevance_score: Score at time of drop
            prompt_at_drop: The prompt that triggered the drop
            session_id: Optional session identifier
            tokens: Number of tokens in chunk
            recoverable_days: Days to keep before expiry
            
        Returns:
            Path to stored chunk file
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{int(time.time())}"
            
        # Create session directory
        session_dir = self.storage_path / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Create chunk record
        chunk = DroppedChunk(
            chunk_id=chunk_id,
            session_id=session_id,
            content=content,
            relevance_score=relevance_score,
            prompt_at_drop=prompt_at_drop,
            tokens=tokens,
            recoverable_until=(
                datetime.now() + timedelta(days=recoverable_days)
            ).isoformat()
        )
        
        # Write to file
        chunk_path = session_dir / f"{chunk_id}.jsonl"
        with open(chunk_path, "w") as f:
            f.write(json.dumps(chunk.to_dict()) + "\n")
            
        # Update index
        self._add_to_index(chunk)
        
        return str(chunk_path)
    
    def _add_to_index(self, chunk: DroppedChunk):
        """Add chunk to search index."""
        with open(self.index_path, "a") as f:
            index_entry = {
                "chunk_id": chunk.chunk_id,
                "session_id": chunk.session_id,
                "timestamp": chunk.timestamp,
                "recoverable_until": chunk.recoverable_until,
                # Simple keyword index
                "keywords": list(set(
                    w.lower() for w in chunk.content.split()[:100]
                )),
            }
            f.write(json.dumps(index_entry) + "\n")
    
    def search_recover(self, query: str, limit: int = 5) -> list[str]:
        """
        Search cold storage for relevant chunks.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of recovered content strings
        """
        # Simple keyword search
        query_words = set(query.lower().split())
        results = []
        
        if not self.index_path.exists():
            return []
            
        with open(self.index_path) as f:
            for line in f:
                entry = json.loads(line)
                
                # Check if query words match keywords
                keywords = set(entry.get("keywords", []))
                if keywords & query_words:
                    # Load the actual chunk
                    chunk_path = (
                        self.storage_path / 
                        entry["session_id"] / 
                        f"{entry['chunk_id']}.jsonl"
                    )
                    if chunk_path.exists():
                        with open(chunk_path) as cf:
                            chunk = DroppedChunk.from_dict(
                                json.loads(cf.readline())
                            )
                            # Check if still recoverable
                            if datetime.fromisoformat(chunk.recoverable_until) > datetime.now():
                                results.append(chunk.content)
                                
                if len(results) >= limit:
                    break
                    
        return results
    
    def archive_session(
        self,
        session_id: str,
        messages: list[dict],
        context: list[str],
    ) -> int:
        """
        Archive an entire session to cold storage.
        
        Args:
            session_id: Session identifier
            messages: List of session messages
            context: List of context strings
            
        Returns:
            Number of chunks archived
        """
        session_dir = self.storage_path / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Archive messages
        messages_path = session_dir / "messages.jsonl"
        with open(messages_path, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")
                
        # Archive context chunks
        archived = 0
        for i, chunk in enumerate(context):
            if len(chunk) > 100:  # Skip tiny chunks
                self.store_dropped(
                    chunk_id=f"chunk_{i}",
                    content=chunk,
                    relevance_score=0.0,  # Unknown at archive time
                    prompt_at_drop="session_archive",
                    session_id=session_id,
                    tokens=len(chunk.split()) * 4,  # Rough estimate
                    recoverable_days=7,  # Shorter for archives
                )
                archived += 1
                
        return archived
    
    def count(self) -> int:
        """Return count of stored chunks."""
        count = 0
        for session_dir in self.storage_path.iterdir():
            if session_dir.is_dir():
                count += len(list(session_dir.glob("*.jsonl")))
        return count
    
    def cleanup_expired(self) -> int:
        """
        Remove expired chunks from cold storage.
        
        Returns:
            Number of chunks removed
        """
        removed = 0
        now = datetime.now()
        
        for session_dir in self.storage_path.iterdir():
            if session_dir.is_dir():
                for chunk_file in session_dir.glob("*.jsonl"):
                    with open(chunk_file) as f:
                        chunk = DroppedChunk.from_dict(json.loads(f.readline()))
                    if datetime.fromisoformat(chunk.recoverable_until) < now:
                        chunk_file.unlink()
                        removed += 1
                        
        return removed
    
    def get_stats(self) -> dict:
        """Get storage statistics."""
        total_chunks = self.count()
        
        # Calculate age distribution
        now = datetime.now()
        ages = {"day": 0, "week": 0, "older": 0}
        
        for session_dir in self.storage_path.iterdir():
            if session_dir.is_dir():
                for chunk_file in session_dir.glob("*.jsonl"):
                    with open(chunk_file) as f:
                        chunk = DroppedChunk.from_dict(json.loads(f.readline()))
                    age = now - datetime.fromisoformat(chunk.timestamp)
                    if age.days < 1:
                        ages["day"] += 1
                    elif age.days < 7:
                        ages["week"] += 1
                    else:
                        ages["older"] += 1
                        
        return {
            "total_chunks": total_chunks,
            "age_distribution": ages,
            "storage_path": str(self.storage_path),
        }
