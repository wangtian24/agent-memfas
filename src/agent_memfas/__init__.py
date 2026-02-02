"""
agent-memfas: Memory Fast and Slow for AI Agents

A dual-store memory system inspired by Kahneman's thinking fast/slow:
- Type 1 (Fast): Keyword triggers for instant pattern matching
- Type 2 (Slow): FTS5 semantic search for deliberate recall

Usage:
    from agent_memfas import Memory
    
    mem = Memory("./memfas.yaml")
    context = mem.recall("How's the family?")
    mem.add_trigger("tahoe", "Family ski trips")
"""

__version__ = "0.1.0"

from .memory import Memory
from .config import Config

__all__ = ["Memory", "Config", "__version__"]
