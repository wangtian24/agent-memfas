"""Configuration loader for agent-memfas."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Try YAML, fall back to JSON if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import json


@dataclass
class SourceConfig:
    """Configuration for a memory source."""
    path: str
    type: str = "markdown"  # markdown, json, text
    always_load: bool = False
    
    
@dataclass 
class SearchConfig:
    """Configuration for search behavior."""
    max_results: int = 5
    recency_weight: float = 0.3
    min_score: float = 0.0  # BM25 scores are tiny, don't filter by default


@dataclass
class TriggerConfig:
    """Configuration for a keyword trigger."""
    keyword: str
    hint: str
    memory_ids: list = field(default_factory=list)


@dataclass
class Config:
    """Main configuration for agent-memfas."""
    db_path: str = "./memfas.db"
    sources: list[SourceConfig] = field(default_factory=list)
    triggers: list[TriggerConfig] = field(default_factory=list)
    triggers_file: Optional[str] = None
    search: SearchConfig = field(default_factory=SearchConfig)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        content = path.read_text()
        
        if path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("PyYAML required for YAML config. Install with: pip install pyyaml")
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary."""
        sources = [
            SourceConfig(**s) if isinstance(s, dict) else s
            for s in data.get("sources", [])
        ]
        
        triggers = [
            TriggerConfig(**t) if isinstance(t, dict) else t
            for t in data.get("triggers", [])
        ]
        
        search_data = data.get("search", {})
        search = SearchConfig(**search_data) if isinstance(search_data, dict) else SearchConfig()
        
        return cls(
            db_path=data.get("db_path", "./memfas.db"),
            sources=sources,
            triggers=triggers,
            triggers_file=data.get("triggers_file"),
            search=search,
        )
    
    @classmethod
    def default(cls, base_dir: str = ".") -> "Config":
        """Create default configuration for a directory."""
        base = Path(base_dir)
        
        sources = []
        
        # Check for common memory file patterns
        if (base / "MEMORY.md").exists():
            sources.append(SourceConfig(path=str(base / "MEMORY.md")))
        
        if (base / "memory").is_dir():
            sources.append(SourceConfig(path=str(base / "memory" / "*.md")))
        
        if (base / "intuition.md").exists():
            sources.append(SourceConfig(
                path=str(base / "intuition.md"),
                always_load=True
            ))
        
        return cls(
            db_path=str(base / "memfas.db"),
            sources=sources,
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "db_path": self.db_path,
            "sources": [
                {"path": s.path, "type": s.type, "always_load": s.always_load}
                for s in self.sources
            ],
            "triggers": [
                {"keyword": t.keyword, "hint": t.hint, "memory_ids": t.memory_ids}
                for t in self.triggers
            ],
            "triggers_file": self.triggers_file,
            "search": {
                "max_results": self.search.max_results,
                "recency_weight": self.search.recency_weight,
                "min_score": self.search.min_score,
            }
        }
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("PyYAML required for YAML config")
            content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        else:
            content = json.dumps(data, indent=2)
        
        path.write_text(content)
