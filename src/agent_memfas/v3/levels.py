"""Curation level presets for controlling context aggressiveness."""

from dataclasses import dataclass
from typing import Dict, Union, Optional


@dataclass
class LevelPreset:
    """Configuration preset for a curation level."""
    name: str
    token_budget: int
    min_score: float
    max_results: int
    recency_weight: float  # How much to favor recent memories
    description: str


# Level presets: 1 (minimal) to 5 (full)
LEVEL_PRESETS: Dict[int, LevelPreset] = {
    1: LevelPreset(
        name="minimal",
        token_budget=300,
        min_score=0.75,
        max_results=3,
        recency_weight=0.1,
        description="Only near-exact matches. Fast, cheap, risky."
    ),
    2: LevelPreset(
        name="lean",
        token_budget=800,
        min_score=0.55,
        max_results=6,
        recency_weight=0.15,
        description="High-confidence matches only."
    ),
    3: LevelPreset(
        name="balanced",
        token_budget=1500,
        min_score=0.40,
        max_results=10,
        recency_weight=0.2,
        description="Default. Good coverage, reasonable cost."
    ),
    4: LevelPreset(
        name="rich",
        token_budget=3000,
        min_score=0.25,
        max_results=18,
        recency_weight=0.25,
        description="Include 'probably relevant' stuff."
    ),
    5: LevelPreset(
        name="full",
        token_budget=5000,
        min_score=0.10,
        max_results=30,
        recency_weight=0.3,
        description="Kitchen sink. Safe but expensive."
    ),
}

# Name aliases
LEVEL_NAMES: Dict[str, int] = {
    "minimal": 1,
    "lean": 2,
    "balanced": 3,
    "rich": 4,
    "full": 5,
    "auto": 3,  # Auto defaults to balanced for now
}

DEFAULT_LEVEL = 3


def resolve_level(level: Union[int, str, None]) -> int:
    """
    Resolve a level specification to a numeric level.
    
    Args:
        level: Level as int (1-5), name ("minimal", "balanced", etc.), 
               "auto", or None (uses default)
    
    Returns:
        Numeric level 1-5
    """
    if level is None:
        return DEFAULT_LEVEL
    
    if isinstance(level, int):
        return max(1, min(5, level))  # Clamp to 1-5
    
    if isinstance(level, str):
        level_lower = level.lower().strip()
        if level_lower in LEVEL_NAMES:
            return LEVEL_NAMES[level_lower]
        # Try parsing as int
        try:
            return max(1, min(5, int(level_lower)))
        except ValueError:
            pass
    
    return DEFAULT_LEVEL


def get_preset(level: Union[int, str, None]) -> LevelPreset:
    """
    Get the preset for a given level.
    
    Args:
        level: Level specification (int, name, or None)
    
    Returns:
        LevelPreset with all parameters
    """
    numeric = resolve_level(level)
    return LEVEL_PRESETS[numeric]


def describe_levels() -> str:
    """Get a human-readable description of all levels."""
    lines = ["Curation Levels:", ""]
    for num, preset in sorted(LEVEL_PRESETS.items()):
        lines.append(f"  {num}. {preset.name.capitalize()}")
        lines.append(f"     {preset.description}")
        lines.append(f"     Budget: ~{preset.token_budget} tokens, "
                    f"threshold: {preset.min_score:.2f}")
        lines.append("")
    return "\n".join(lines)
