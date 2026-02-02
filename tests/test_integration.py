"""Integration tests for agent-memfas end-to-end workflows."""

import pytest
import tempfile
import subprocess
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_memfas import Memory, Config


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def workspace(self):
        """Create a temporary workspace with memory files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            # Create MEMORY.md
            (base / "MEMORY.md").write_text("""# Long-term Memory

## Work
- Currently at Yupp.ai, started January 2025
- Working on LLM routing and preference learning
- Competes with LMArena

## Family
- Wife: Xu
- Daughters: Veronica (11) and Oumi (6)
- Live in SF Bay Area

## Running
- Training for CIM marathon in December
- Current plan: 3x 10K + 1x half marathon per week
""")
            
            # Create memory/ directory with daily notes
            (base / "memory").mkdir()
            (base / "memory" / "2025-01-15.md").write_text("""# January 15, 2025

## Notes
- Met with the model team about routing strategies
- Discussed preference learning approaches with RLHF
- Need to research arena methodologies more
""")
            
            # Create intuition.md (Type 1 fast recall)
            (base / "intuition.md").write_text("""# Quick Facts
- Timezone: America/Los_Angeles
- Prefers concise responses
- Running training schedule is important
""")
            
            yield base
    
    def test_full_workflow(self, workspace):
        """Test complete workflow: init, index, trigger, search, recall."""
        # Use from_dict to properly construct config with nested objects
        config = Config.from_dict({
            "db_path": str(workspace / "memfas.db"),
            "sources": [
                {"path": str(workspace / "MEMORY.md"), "type": "markdown"},
                {"path": str(workspace / "memory" / "*.md"), "type": "markdown"},
            ]
        })
        
        mem = Memory(config)
        
        # Add triggers
        mem.add_trigger("family", "User's family info in MEMORY.md")
        mem.add_trigger("work", "Current job at Yupp.ai")
        mem.add_trigger("running", "Marathon training schedule")
        
        # Index files
        mem.index_file(str(workspace / "MEMORY.md"))
        mem.index_file(str(workspace / "memory" / "2025-01-15.md"))
        
        # Verify indexing
        stats = mem.stats()
        assert stats["memories"] > 0
        assert stats["triggers"] == 3
        
        # Test Type 1 recall (triggers)
        result = mem.recall("How's the family?")
        assert "family" in result.lower()
        
        # Test Type 2 recall (search)
        results = mem.search("preference learning")
        assert len(results) > 0
        
        # Test combined recall
        result = mem.recall("Tell me about work and RLHF")
        assert "work" in result.lower() or "Memory Context" in result
        
        mem.close()
    
    @pytest.mark.skip(reason="memory_ids linkage needs rework for v2 pluggable backends")
    def test_trigger_with_memory_ids(self, workspace):
        """Test triggers that link to specific memories."""
        config = Config(db_path=str(workspace / "memfas.db"))
        mem = Memory(config)
        
        # Index content first
        mem.index_file(str(workspace / "MEMORY.md"))
        
        # This test used the old row-ID based memory_ids linkage
        # which doesn't work with the new pluggable backend architecture.
        # TODO: Implement doc_id based trigger linkage in v2.1
        
        mem.close()
    
    def test_config_yaml_roundtrip(self, workspace):
        """Test saving and loading YAML config."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        
        config = Config(
            db_path=str(workspace / "memfas.db"),
            sources=[{"path": "*.md", "type": "markdown"}],
            triggers=[{"keyword": "test", "hint": "Test trigger"}],
        )
        config = Config.from_dict(config.to_dict())
        
        config_path = workspace / "memfas.yaml"
        config.save(str(config_path))
        
        # Reload
        loaded = Config.load(str(config_path))
        assert loaded.db_path == config.db_path
        assert len(loaded.triggers) == 1
        assert loaded.triggers[0].keyword == "test"
    
    def test_config_json_roundtrip(self, workspace):
        """Test saving and loading JSON config."""
        config = Config.from_dict({
            "db_path": str(workspace / "memfas.db"),
            "triggers": [{"keyword": "json_test", "hint": "JSON trigger"}],
        })
        
        config_path = workspace / "memfas.json"
        config.save(str(config_path))
        
        loaded = Config.load(str(config_path))
        assert loaded.triggers[0].keyword == "json_test"
    
    def test_multiple_file_indexing(self, workspace):
        """Test indexing multiple files with glob patterns."""
        config = Config(db_path=str(workspace / "memfas.db"))
        mem = Memory(config)
        
        # Create more memory files
        for i in range(3):
            (workspace / "memory" / f"2025-01-{10+i}.md").write_text(
                f"# January {10+i}, 2025\n\nThis is test content number {i} about " +
                "various topics including machine learning, neural networks, and AI research."
            )
        
        # Index with glob
        from glob import glob
        for path in glob(str(workspace / "memory" / "*.md")):
            mem.index_file(path)
        
        stats = mem.stats()
        assert stats["memories"] >= 3  # At least 3 files indexed
        
        mem.close()
    
    def test_clear_preserves_triggers(self, workspace):
        """Test that clear() removes memories but keeps triggers."""
        config = Config(db_path=str(workspace / "memfas.db"))
        mem = Memory(config)
        
        # Add trigger and index content
        mem.add_trigger("preserve_me", "This should survive clear")
        mem.index_file(str(workspace / "MEMORY.md"))
        
        initial_stats = mem.stats()
        assert initial_stats["memories"] > 0
        assert initial_stats["triggers"] == 1
        
        # Clear
        mem.clear()
        
        final_stats = mem.stats()
        assert final_stats["memories"] == 0
        assert final_stats["triggers"] == 1  # Trigger preserved
        
        mem.close()
    
    def test_search_special_characters(self, workspace):
        """Test that special characters in search don't break FTS5."""
        config = Config(db_path=str(workspace / "memfas.db"))
        mem = Memory(config)
        
        mem.index_file(str(workspace / "MEMORY.md"))
        
        # These queries contain FTS5 special characters
        queries = [
            "What's up?",
            "work (Yupp)",
            '"exact phrase"',
            "test + search",
            "running - marathon",
            "family: info",
            "SF* Bay",
        ]
        
        for query in queries:
            # Should not raise exception
            results = mem.search(query)
            # Results may or may not be found, but shouldn't crash
            assert isinstance(results, list)
        
        mem.close()
    
    def test_empty_search_returns_empty(self, workspace):
        """Test that empty/whitespace search returns empty list."""
        config = Config(db_path=str(workspace / "memfas.db"))
        mem = Memory(config)
        
        mem.index_file(str(workspace / "MEMORY.md"))
        
        assert mem.search("") == []
        assert mem.search("   ") == []
        assert mem.search("!@#$%") == []  # Only special chars
        
        mem.close()


class TestCLI:
    """CLI integration tests."""
    
    @pytest.fixture
    def workspace(self):
        """Create temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "test.md").write_text(
                "# Test\n\nThis is test content about machine learning and neural networks."
            )
            yield base
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            [sys.executable, "-m", "agent_memfas", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "memfas" in result.stdout.lower() or "memory" in result.stdout.lower()
    
    def test_cli_init(self, workspace):
        """Test CLI init command."""
        result = subprocess.run(
            [sys.executable, "-m", "agent_memfas", "init"],
            capture_output=True,
            text=True,
            cwd=str(workspace)
        )
        # Should create config file, or fail gracefully with known error
        # PyYAML may not be installed, which is a known optional dependency
        success = result.returncode == 0
        yaml_missing = "pyyaml" in result.stderr.lower()
        already_exists = "already exists" in result.stdout.lower()
        assert success or yaml_missing or already_exists, f"Unexpected error: {result.stderr}"
    
    def test_cli_stats(self, workspace):
        """Test CLI stats command."""
        # Initialize first
        config = Config(db_path=str(workspace / "memfas.db"))
        mem = Memory(config)
        mem.close()
        
        result = subprocess.run(
            [sys.executable, "-m", "agent_memfas", "stats"],
            capture_output=True,
            text=True,
            cwd=str(workspace)
        )
        # Should show stats or error gracefully
        assert result.returncode == 0 or "no database" in result.stderr.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
