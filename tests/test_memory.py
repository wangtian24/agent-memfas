"""Tests for agent-memfas."""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_memfas import Memory, Config
from agent_memfas.config import TriggerConfig, SourceConfig, SearchConfig


class TestConfig:
    """Test configuration loading."""
    
    def test_default_config(self):
        config = Config.default(".")
        # Path may or may not have ./ prefix depending on normalization
        assert config.db_path.endswith("memfas.db")
        assert isinstance(config.search, SearchConfig)
    
    def test_from_dict(self):
        data = {
            "db_path": "./test.db",
            "triggers": [
                {"keyword": "test", "hint": "Test trigger"}
            ],
            "search": {
                "max_results": 10
            }
        }
        config = Config.from_dict(data)
        assert config.db_path == "./test.db"
        assert len(config.triggers) == 1
        assert config.search.max_results == 10


class TestMemory:
    """Test Memory class."""
    
    @pytest.fixture
    def mem(self):
        """Create a temporary memory instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(db_path=f"{tmpdir}/test.db")
            m = Memory(config)
            yield m
            m.close()
    
    def test_add_trigger(self, mem):
        mem.add_trigger("test", "Test trigger hint")
        triggers = mem.list_triggers()
        assert len(triggers) == 1
        assert triggers[0]["keyword"] == "test"
        assert triggers[0]["hint"] == "Test trigger hint"
    
    def test_remove_trigger(self, mem):
        mem.add_trigger("test", "Test")
        mem.remove_trigger("test")
        triggers = mem.list_triggers()
        assert len(triggers) == 0
    
    def test_trigger_matching(self, mem):
        mem.add_trigger("family", "User's family")
        result = mem.recall("How's the family?")
        assert "family" in result.lower()
    
    def test_index_markdown(self, mem):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            # Content must be > 100 chars to be indexed (tiny chunks are skipped)
            f.write("# Test\n\nThis is test content about machine learning and neural networks. "
                    "We're exploring deep learning architectures including transformers, "
                    "convolutional networks, and recurrent neural networks for various tasks.")
            f.flush()
            mem.index_file(f.name, "markdown")
        
        stats = mem.stats()
        assert stats["memories"] > 0
    
    def test_search(self, mem):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            # Content must be > 100 chars to be indexed
            f.write("# Notes\n\nDeep learning and neural networks are fascinating topics. "
                    "Modern architectures like GPT and BERT have revolutionized natural "
                    "language processing. Convolutional neural networks dominate computer vision tasks.")
            f.flush()
            mem.index_file(f.name, "markdown")
        
        results = mem.search("neural networks")
        assert len(results) > 0
        assert "neural" in results[0].text.lower()
    
    def test_special_characters_in_search(self, mem):
        """Test that special characters don't break FTS5."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n\nSome content here.")
            f.flush()
            mem.index_file(f.name, "markdown")
        
        # These should not raise exceptions
        mem.search("What's up?")
        mem.search("test (with parens)")
        mem.search('test "with quotes"')
        mem.search("test + - * special")
    
    def test_stats(self, mem):
        stats = mem.stats()
        assert "memories" in stats
        assert "triggers" in stats
        assert "backend" in stats
    
    def test_clear(self, mem):
        mem.add_trigger("test", "Test")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n\nContent")
            f.flush()
            mem.index_file(f.name, "markdown")
        
        mem.clear()
        stats = mem.stats()
        assert stats["memories"] == 0
        # Triggers should be preserved
        assert stats["triggers"] == 1


class TestSuggestTriggers:
    """Test auto-trigger suggestion."""
    
    @pytest.fixture
    def mem_with_content(self):
        """Create memory with indexed content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(db_path=f"{tmpdir}/test.db")
            m = Memory(config)
            
            # Add content with repeated terms
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write("""# Project Notes
                
Working on the Acme project with Alice and Bob.
Alice is leading the frontend work.
Bob handles the backend systems.

The Acme project uses machine learning for recommendations.
Machine learning models need training data.
Alice found good training data sources.

Bob integrated the machine learning pipeline.
The Acme dashboard shows model performance.
""")
                f.flush()
                m.index_file(f.name, "markdown")
            
            yield m
            m.close()
    
    def test_suggest_finds_entities(self, mem_with_content):
        suggestions = mem_with_content.suggest_triggers(min_occurrences=2)
        terms = [s["term"] for s in suggestions]
        
        # Should find capitalized names
        assert "alice" in terms or "bob" in terms or "acme" in terms
    
    def test_suggest_finds_frequent_terms(self, mem_with_content):
        suggestions = mem_with_content.suggest_triggers(min_occurrences=2)
        terms = [s["term"] for s in suggestions]
        
        # Should find repeated terms
        assert "machine" in terms or "learning" in terms or "project" in terms
    
    def test_suggest_excludes_existing_triggers(self, mem_with_content):
        # Add a trigger
        mem_with_content.add_trigger("alice", "Team member")
        
        suggestions = mem_with_content.suggest_triggers(min_occurrences=2)
        terms = [s["term"] for s in suggestions]
        
        # Should not suggest existing trigger
        assert "alice" not in terms
    
    def test_suggest_empty_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(db_path=f"{tmpdir}/test.db")
            mem = Memory(config)
            
            suggestions = mem.suggest_triggers()
            assert suggestions == []
            
            mem.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
