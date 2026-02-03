"""
Context configuration for memfas v4.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ContextConfig:
    """Configuration for context management."""
    
    # Context Limits
    compaction_trigger_pct: float = 0.50  # Trigger at 50%
    max_context_pct: float = 0.85          # Hard limit at 85%
    auto_compact: bool = True              # Auto-compact when triggered
    
    # Relevance Scoring
    relevance_cutoff: float = 0.3          # Below this, DROP
    relevance_keep_threshold: float = 0.7  # Above this, KEEP
    memfas_weight: float = 0.6             # memfas vs recency tradeoff
    recency_bonus: float = 0.2             # Boost per hour
    importance_bonus: float = 0.1          # Boost for important content
    min_chunks_to_keep: int = 5            # Always keep at least N chunks
    
    # memfas Integration
    memfas_min_score: float = 0.05         # memfas similarity threshold
    curation_trigger_pct: float = 0.40     # Run memfas curation at 40%
    
    # Cold Storage
    cold_storage_enabled: bool = True
    cold_storage_path: str = "./cold-storage/"
    recoverable_days: int = 30
    
    # Summarization
    summarize_medium_chunks: bool = True
    summary_target_tokens: int = 500
    summary_model: str = "minimax/MiniMax-M2.1"
    
    # Logging
    log_path: str = "./logs/context/"
    log_compaction: bool = True
    log_curation: bool = True
    log_drops: bool = True
    log_health: bool = True
    log_recovery: bool = True
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ContextConfig":
        """Load config from YAML file or return defaults."""
        if not config_path:
            # Try default locations
            default_paths = [
                "./memfas.yaml",
                "./memfas-context.yaml",
                str(Path.home() / ".clawdbot" / "memfas.yaml"),
            ]
            for path in default_paths:
                if Path(path).exists():
                    config_path = path
                    break
                    
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
                
            # Handle nested context config
            if "context" in data:
                data = data["context"]
                
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
            
        return cls()
    
    def to_dict(self) -> dict:
        """Export config to dict."""
        return {
            "compaction_trigger_pct": self.compaction_trigger_pct,
            "max_context_pct": self.max_context_pct,
            "auto_compact": self.auto_compact,
            "relevance_cutoff": self.relevance_cutoff,
            "relevance_keep_threshold": self.relevance_keep_threshold,
            "memfas_weight": self.memfas_weight,
            "recency_bonus": self.recency_bonus,
            "importance_bonus": self.importance_bonus,
            "min_chunks_to_keep": self.min_chunks_to_keep,
            "memfas_min_score": self.memfas_min_score,
            "curation_trigger_pct": self.curation_trigger_pct,
            "cold_storage_enabled": self.cold_storage_enabled,
            "cold_storage_path": self.cold_storage_path,
            "recoverable_days": self.recoverable_days,
            "summarize_medium_chunks": self.summarize_medium_chunks,
            "summary_target_tokens": self.summary_target_tokens,
            "summary_model": self.summary_model,
            "log_path": self.log_path,
            "log_compaction": self.log_compaction,
            "log_curation": self.log_curation,
            "log_drops": self.log_drops,
            "log_health": self.log_health,
            "log_recovery": self.log_recovery,
        }
    
    def save(self, path: str):
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
