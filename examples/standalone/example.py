#!/usr/bin/env python3
"""
Standalone example of agent-memfas usage.

Run from this directory:
    python example.py
"""

from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_memfas import Memory, Config


def main():
    # Create a temporary config
    config = Config(
        db_path="./example.db",
        sources=[],
        triggers=[]
    )
    
    # Initialize memory
    mem = Memory(config)
    
    print("=== agent-memfas Example ===\n")
    
    # Add some triggers
    print("Adding triggers...")
    mem.add_trigger("python", "User's favorite programming language")
    mem.add_trigger("coffee", "User likes oat milk lattes")
    mem.add_trigger("meeting", "Weekly standup is Monday 10am")
    
    # List triggers
    print("\nCurrent triggers:")
    for t in mem.list_triggers():
        print(f"  [{t['keyword']}] {t['hint']}")
    
    # Index some sample content
    print("\nIndexing sample content...")
    sample_memory = """
# Project Notes

## Python Project
Working on a new FastAPI service for user authentication.
Using SQLAlchemy for the database layer.

## Meeting Notes
- Discussed roadmap for Q2
- Need to finish the auth service by March
- Coffee budget approved for team
    """
    
    # Write temp file and index it
    temp_file = Path("./sample_memory.md")
    temp_file.write_text(sample_memory)
    mem.index_file(str(temp_file), "markdown")
    temp_file.unlink()  # Clean up
    
    # Show stats
    print("\nStats:")
    stats = mem.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Test recall with trigger
    print("\n--- Testing Recall ---\n")
    
    print("Query: 'What about python?'")
    result = mem.recall("What about python?")
    print(result or "(no results)")
    
    print("\nQuery: 'Tell me about the meeting'")
    result = mem.recall("Tell me about the meeting")
    print(result or "(no results)")
    
    # Test search without trigger
    print("\n--- Testing Search ---\n")
    
    print("Search: 'FastAPI authentication'")
    results = mem.search("FastAPI authentication", limit=3)
    for r in results:
        print(f"  [{r.source}] score={r.score:.2f}")
        print(f"    {r.text[:100]}...")
    
    # Clean up
    mem.close()
    Path("./example.db").unlink(missing_ok=True)
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
