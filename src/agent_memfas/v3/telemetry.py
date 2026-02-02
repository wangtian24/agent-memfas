"""Telemetry and logging for v3 context curation."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from statistics import mean


@dataclass
class TurnMetrics:
    """Metrics for a single turn."""
    timestamp: str
    session_id: str
    turn_number: int
    
    # Query info
    query: str
    query_tokens: int
    detected_topic: str
    topic_shifted: bool
    
    # Context sizes
    baseline_context_tokens: int  # What WOULD have been used
    curated_context_tokens: int   # What memfas provided
    
    # Memory stats
    memories_scored: int
    memories_included: int
    memories_dropped: int
    triggers_matched: int
    
    # Savings
    tokens_saved: int
    compression_ratio: float  # curated / baseline (lower = better)
    
    # Performance
    latency_ms: float
    
    # Debug info
    top_memories: List[Dict[str, Any]] = field(default_factory=list)
    score_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""
    session_id: str
    start_time: str
    end_time: str
    total_turns: int
    
    # Totals
    total_baseline_tokens: int
    total_curated_tokens: int
    total_tokens_saved: int
    avg_compression_ratio: float
    
    # Topic tracking
    topics_detected: List[str]
    topic_shifts: int
    
    # Memory usage
    unique_memories_used: int
    most_accessed_memories: List[Dict[str, Any]]
    
    # Performance
    avg_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float


class TelemetryLogger:
    """
    Logs structured telemetry for analysis.
    
    Writes JSONL (one JSON object per line) for easy parsing
    by agents, scripts, or data tools.
    """
    
    def __init__(
        self,
        log_path: str = "./memfas-telemetry.jsonl",
        max_file_size_mb: int = 50
    ):
        """
        Initialize telemetry logger.
        
        Args:
            log_path: Path to JSONL log file
            max_file_size_mb: Rotate file when it exceeds this size
        """
        self.log_path = Path(log_path)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self._session_turns: Dict[str, List[TurnMetrics]] = {}
    
    def log_turn(self, metrics: TurnMetrics):
        """
        Log a single turn's metrics.
        
        Args:
            metrics: TurnMetrics for this turn
        """
        # Rotate if needed
        self._maybe_rotate()
        
        # Write to file
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            entry = {
                "type": "turn",
                **asdict(metrics)
            }
            f.write(json.dumps(entry) + "\n")
        
        # Track for session aggregation
        if metrics.session_id not in self._session_turns:
            self._session_turns[metrics.session_id] = []
        self._session_turns[metrics.session_id].append(metrics)
    
    def log_session_end(self, session_id: str) -> Optional[SessionMetrics]:
        """
        Log session summary when session ends.
        
        Args:
            session_id: Session to summarize
        
        Returns:
            SessionMetrics if session had data
        """
        turns = self._session_turns.get(session_id, [])
        if not turns:
            return None
        
        # Calculate aggregates
        metrics = SessionMetrics(
            session_id=session_id,
            start_time=turns[0].timestamp,
            end_time=turns[-1].timestamp,
            total_turns=len(turns),
            total_baseline_tokens=sum(t.baseline_context_tokens for t in turns),
            total_curated_tokens=sum(t.curated_context_tokens for t in turns),
            total_tokens_saved=sum(t.tokens_saved for t in turns),
            avg_compression_ratio=mean(t.compression_ratio for t in turns) if turns else 0,
            topics_detected=list(set(t.detected_topic for t in turns)),
            topic_shifts=sum(1 for t in turns if t.topic_shifted),
            unique_memories_used=self._count_unique_memories(turns),
            most_accessed_memories=self._get_most_accessed(turns),
            avg_latency_ms=mean(t.latency_ms for t in turns) if turns else 0,
            p95_latency_ms=self._percentile([t.latency_ms for t in turns], 95),
            max_latency_ms=max(t.latency_ms for t in turns) if turns else 0
        )
        
        # Write summary
        with open(self.log_path, "a") as f:
            entry = {
                "type": "session_summary",
                **asdict(metrics)
            }
            f.write(json.dumps(entry) + "\n")
        
        # Clean up
        del self._session_turns[session_id]
        
        return metrics
    
    def get_summary(
        self,
        since: Optional[str] = None,
        session_id: Optional[str] = None,
        last_n_turns: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Args:
            since: ISO timestamp to filter from
            session_id: Filter to specific session
            last_n_turns: Only include last N turns
        
        Returns:
            Summary dict suitable for agent to read and report
        """
        entries = self._read_entries(since, session_id)
        turns = [e for e in entries if e.get("type") == "turn"]
        
        if last_n_turns:
            turns = turns[-last_n_turns:]
        
        if not turns:
            return {
                "status": "no_data",
                "message": "No telemetry data found for the specified filters"
            }
        
        latencies = [t["latency_ms"] for t in turns]
        
        return {
            "status": "ok",
            "period": {
                "start": turns[0]["timestamp"],
                "end": turns[-1]["timestamp"],
                "total_turns": len(turns)
            },
            "compression": {
                "total_baseline_tokens": sum(t["baseline_context_tokens"] for t in turns),
                "total_curated_tokens": sum(t["curated_context_tokens"] for t in turns),
                "total_tokens_saved": sum(t["tokens_saved"] for t in turns),
                "avg_compression_ratio": mean(t["compression_ratio"] for t in turns),
                "best_compression": min(t["compression_ratio"] for t in turns),
                "worst_compression": max(t["compression_ratio"] for t in turns)
            },
            "memory_usage": {
                "total_memories_scored": sum(t["memories_scored"] for t in turns),
                "total_memories_included": sum(t["memories_included"] for t in turns),
                "total_memories_dropped": sum(t["memories_dropped"] for t in turns),
                "avg_memories_per_turn": mean(t["memories_included"] for t in turns),
                "avg_triggers_per_turn": mean(t["triggers_matched"] for t in turns)
            },
            "topics": {
                "unique_topics": list(set(t["detected_topic"] for t in turns)),
                "topic_shifts": sum(1 for t in turns if t["topic_shifted"]),
                "shift_rate": sum(1 for t in turns if t["topic_shifted"]) / len(turns)
            },
            "performance": {
                "avg_latency_ms": mean(latencies),
                "p50_latency_ms": self._percentile(latencies, 50),
                "p95_latency_ms": self._percentile(latencies, 95),
                "max_latency_ms": max(latencies)
            }
        }
    
    def format_summary(
        self,
        summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Format summary as human-readable text.
        
        Args:
            summary: Pre-computed summary, or compute with kwargs
        
        Returns:
            Formatted string for display
        """
        if summary is None:
            summary = self.get_summary(**kwargs)
        
        if summary.get("status") == "no_data":
            return "📊 No telemetry data found."
        
        c = summary["compression"]
        m = summary["memory_usage"]
        t = summary["topics"]
        p = summary["performance"]
        
        saved_pct = (c["total_tokens_saved"] / max(c["total_baseline_tokens"], 1)) * 100
        
        return f"""📊 Memfas Performance Summary
{'═' * 40}

Period: {summary['period']['start'][:10]} to {summary['period']['end'][:10]}
Turns:  {summary['period']['total_turns']}

Compression:
  Baseline tokens:  {c['total_baseline_tokens']:,}
  Curated tokens:   {c['total_curated_tokens']:,}
  Tokens saved:     {c['total_tokens_saved']:,} ({saved_pct:.1f}% reduction)
  Avg compression:  {c['avg_compression_ratio']:.2f}x

Memory Usage:
  Memories scored:  {m['total_memories_scored']:,}
  Included:         {m['total_memories_included']:,} ({m['avg_memories_per_turn']:.1f}/turn)
  Dropped:          {m['total_memories_dropped']:,}
  Triggers:         {m['avg_triggers_per_turn']:.1f}/turn

Topics:
  Unique:           {len(t['unique_topics'])}
  Shifts:           {t['topic_shifts']} ({t['shift_rate']*100:.1f}%)

Latency:
  Average:          {p['avg_latency_ms']:.1f}ms
  P95:              {p['p95_latency_ms']:.1f}ms
  Max:              {p['max_latency_ms']:.1f}ms
"""
    
    def _read_entries(
        self,
        since: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict]:
        """Read and filter log entries."""
        if not self.log_path.exists():
            return []
        
        entries = []
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Filter by timestamp
                    if since and entry.get("timestamp", "") < since:
                        continue
                    
                    # Filter by session
                    if session_id and entry.get("session_id") != session_id:
                        continue
                    
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        return entries
    
    def _maybe_rotate(self):
        """Rotate log file if too large."""
        if not self.log_path.exists():
            return
        
        if self.log_path.stat().st_size > self.max_file_size:
            # Rotate: rename current to .old, start fresh
            old_path = self.log_path.with_suffix(".old.jsonl")
            if old_path.exists():
                old_path.unlink()
            self.log_path.rename(old_path)
    
    def _count_unique_memories(self, turns: List[TurnMetrics]) -> int:
        """Count unique memories across turns."""
        all_ids = set()
        for turn in turns:
            for mem in turn.top_memories:
                if "id" in mem:
                    all_ids.add(mem["id"])
        return len(all_ids)
    
    def _get_most_accessed(
        self,
        turns: List[TurnMetrics],
        top_n: int = 5
    ) -> List[Dict]:
        """Get most frequently accessed memories."""
        counts: Dict[str, Dict] = {}
        for turn in turns:
            for mem in turn.top_memories:
                mid = mem.get("id", "")
                if mid:
                    if mid not in counts:
                        counts[mid] = {
                            "id": mid,
                            "source": mem.get("source", ""),
                            "count": 0
                        }
                    counts[mid]["count"] += 1
        
        sorted_mems = sorted(
            counts.values(),
            key=lambda x: x["count"],
            reverse=True
        )
        return sorted_mems[:top_n]
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def clear(self):
        """Clear telemetry log."""
        if self.log_path.exists():
            self.log_path.unlink()
        self._session_turns.clear()
