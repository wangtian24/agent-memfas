"""
Logging utilities for context management.
"""

from pathlib import Path
from typing import Optional, Any
import json
import time
from datetime import datetime


class ContextLogger:
    """
    Logs all context management activities to JSONL files.
    
    Files:
    - compaction.jsonl: Compaction operations
    - curation.jsonl: memfas curation results
    - drops.jsonl: Chunk drops to cold storage
    - health.jsonl: Context health metrics
    - recovery.jsonl: Chunk recoveries
    """
    
    def __init__(self, log_path: str = "./logs/context/"):
        """
        Initialize logger.
        
        Args:
            log_path: Path to log directory
        """
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Track last compaction time
        self._last_compaction: Optional[str] = None
        
    def _log(self, file: str, entry: dict):
        """Write a log entry to file."""
        log_file = self.log_path / file
        entry["timestamp"] = datetime.now().isoformat()
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def log_message(self, message: str, context_count: int, token_count: int):
        """Log a new message."""
        self._log("health.jsonl", {
            "event": "message",
            "context_chunks": context_count,
            "token_count": token_count,
        })
        
    def log_health(self, status):
        """Log context health check."""
        self._log("health.jsonl", {
            "event": "health_check",
            "current_tokens": status.current_tokens,
            "max_tokens": status.max_tokens,
            "pct_used": status.pct_used,
            "chunks_count": status.chunks_count,
            "cold_storage_count": status.cold_storage_count,
            "needs_compaction": status.needs_compaction,
        })
        
    def log_compaction(self, result):
        """Log a compaction operation."""
        self._last_compaction = datetime.now().isoformat()
        self._log("compaction.jsonl", {
            "event": "compaction",
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "token_savings": result.token_savings,
            "chunks_dropped": result.chunks_dropped,
            "chunks_summarized": result.chunks_summarized,
        })
        
    def log_curation(
        self,
        query: str,
        hits_found: int,
        total_before: int,
        token_savings: int,
        drops_triggered: int,
    ):
        """Log a memfas curation operation."""
        self._log("curation.jsonl", {
            "event": "curation",
            "query": query,
            "hits_found": hits_found,
            "total_before_curation": total_before,
            "token_savings": token_savings,
            "drops_triggered": drops_triggered,
        })
        
    def log_drop(
        self,
        chunk_id: str,
        session_id: str,
        relevance_score: float,
        reason: str,
        cold_storage_path: str,
    ):
        """Log a chunk drop to cold storage."""
        self._log("drops.jsonl", {
            "event": "chunk_dropped",
            "chunk_id": chunk_id,
            "session_id": session_id,
            "relevance_score": relevance_score,
            "reason": reason,
            "cold_storage_path": cold_storage_path,
        })
        
    def log_recovery(self, query: str, chunks_recovered: int):
        """Log a recovery from cold storage."""
        self._log("recovery.jsonl", {
            "event": "recovery",
            "query": query,
            "chunks_recovered": chunks_recovered,
        })
        
    def log_session_end(self, context_count: int, message_count: int):
        """Log session end."""
        self._log("health.jsonl", {
            "event": "session_end",
            "context_chunks": context_count,
            "message_count": message_count,
        })
        
    def log_activity(self, activity: str, **kwargs):
        """Log a generic activity."""
        self._log("health.jsonl", {
            "event": activity,
            **kwargs
        })
        
    def last_compaction_time(self) -> Optional[str]:
        """Get timestamp of last compaction."""
        return self._last_compaction
        
    def get_compaction_stats(self, hours: int = 24) -> dict:
        """Get compaction statistics for the last N hours."""
        log_file = self.log_path / "compaction.jsonl"
        if not log_file.exists():
            return {}
            
        since = datetime.now().timestamp() - (hours * 3600)
        count = 0
        total_savings = 0
        
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if ts > since:
                    count += 1
                    total_savings += entry.get("token_savings", 0)
                    
        return {
            "compaction_count": count,
            "total_token_savings": total_savings,
        }
        
    def get_drop_stats(self, hours: int = 24) -> dict:
        """Get drop statistics for the last N hours."""
        log_file = self.log_path / "drops.jsonl"
        if not log_file.exists():
            return {}
            
        since = datetime.now().timestamp() - (hours * 3600)
        by_reason = {}
        total = 0
        
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if ts > since:
                    total += 1
                    reason = entry.get("reason", "unknown")
                    by_reason[reason] = by_reason.get(reason, 0) + 1
                    
        return {
            "total_drops": total,
            "drops_by_reason": by_reason,
        }
