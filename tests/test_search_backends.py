"""Tests for search backends."""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_memfas.search.base import SearchBackend, SearchResult
from agent_memfas.search.fts5 import FTS5Backend


class TestFTS5Backend:
    """Test FTS5 full-text search backend."""
    
    @pytest.fixture
    def backend(self):
        """Create a temporary FTS5 backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            b = FTS5Backend(f"{tmpdir}/test.db")
            yield b
            b.close()
    
    def test_index_and_search(self, backend):
        backend.index("doc1", "Hello world, this is a test document about AI")
        backend.index("doc2", "Machine learning is a subset of AI")
        
        results = backend.search("AI")
        assert len(results) >= 1
        assert any("AI" in r.text for r in results)
    
    def test_search_returns_search_result(self, backend):
        backend.index("doc1", "Neural networks are powerful", {"source": "test.md"})
        
        results = backend.search("neural")
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "Neural networks are powerful"
        assert results[0].source == "test.md"
        assert results[0].score > 0
    
    def test_delete(self, backend):
        backend.index("doc1", "Document to delete")
        assert backend.count() == 1
        
        backend.delete("doc1")
        assert backend.count() == 0
    
    def test_clear(self, backend):
        backend.index("doc1", "First document")
        backend.index("doc2", "Second document")
        assert backend.count() == 2
        
        backend.clear()
        assert backend.count() == 0
    
    def test_metadata(self, backend):
        backend.index("doc1", "Test content", {
            "source": "notes.md",
            "section": "chapter1",
            "date": "2025-01-01T00:00:00"
        })
        
        results = backend.search("test")
        assert len(results) == 1
        assert results[0].source == "notes.md"
        assert results[0].section == "chapter1"
        assert results[0].date == "2025-01-01T00:00:00"
    
    def test_special_characters(self, backend):
        backend.index("doc1", "Test with special chars")
        
        # These should not raise
        backend.search("What's this?")
        backend.search("test (parens)")
        backend.search('test "quotes"')
        backend.search("test + - * ~")
    
    def test_empty_query(self, backend):
        backend.index("doc1", "Some content")
        
        results = backend.search("")
        assert results == []
        
        results = backend.search("   ")
        assert results == []


class TestEmbeddingBackend:
    """Test embedding backend with mock embedder."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Create a simple mock embedder."""
        from agent_memfas.embedders.base import Embedder
        
        class MockEmbedder(Embedder):
            """Mock embedder that returns deterministic vectors."""
            
            @property
            def dimensions(self) -> int:
                return 4
            
            def embed(self, text: str) -> list:
                # Simple hash-based embedding for testing
                h = hash(text.lower())
                return [
                    (h % 100) / 100.0,
                    ((h >> 8) % 100) / 100.0,
                    ((h >> 16) % 100) / 100.0,
                    ((h >> 24) % 100) / 100.0,
                ]
        
        return MockEmbedder()
    
    @pytest.fixture
    def backend(self, mock_embedder):
        """Create a temporary embedding backend with mock embedder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                from agent_memfas.search.embedding import EmbeddingBackend
                b = EmbeddingBackend(f"{tmpdir}/test.db", mock_embedder)
                yield b
                b.close()
            except ImportError:
                pytest.skip("sqlite-vec not installed")
    
    def test_index_and_search(self, backend):
        backend.index("doc1", "Hello world")
        backend.index("doc2", "Goodbye world")
        
        # Search should return results (exact matching depends on mock embedder)
        results = backend.search("hello")
        assert len(results) >= 1
    
    def test_search_returns_search_result(self, backend):
        backend.index("doc1", "Test document content", {"source": "test.md"})
        
        results = backend.search("test")
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "Test document content"
        assert results[0].source == "test.md"
        assert results[0].score > 0
    
    def test_delete(self, backend):
        backend.index("doc1", "Document to delete")
        assert backend.count() == 1
        
        backend.delete("doc1")
        assert backend.count() == 0
    
    def test_clear(self, backend):
        backend.index("doc1", "First")
        backend.index("doc2", "Second")
        assert backend.count() == 2
        
        backend.clear()
        assert backend.count() == 0
    
    def test_empty_query(self, backend):
        backend.index("doc1", "Some content")
        
        results = backend.search("")
        assert results == []


class TestBackendInterface:
    """Test that backends conform to the interface."""
    
    def test_fts5_is_search_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            b = FTS5Backend(f"{tmpdir}/test.db")
            assert isinstance(b, SearchBackend)
            b.close()
    
    def test_embedding_is_search_backend(self):
        try:
            from agent_memfas.search.embedding import EmbeddingBackend
            from agent_memfas.embedders.base import Embedder
            
            class DummyEmbedder(Embedder):
                @property
                def dimensions(self) -> int:
                    return 4
                
                def embed(self, text: str) -> list:
                    return [0.0, 0.0, 0.0, 0.0]
            
            with tempfile.TemporaryDirectory() as tmpdir:
                b = EmbeddingBackend(f"{tmpdir}/test.db", DummyEmbedder())
                assert isinstance(b, SearchBackend)
                b.close()
        except ImportError:
            pytest.skip("sqlite-vec not installed")


class TestConfigBackendIntegration:
    """Test Memory with different backend configs."""
    
    def test_default_uses_fts5(self):
        from agent_memfas import Memory, Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(db_path=f"{tmpdir}/test.db")
            mem = Memory(config)
            
            assert type(mem.backend).__name__ == "FTS5Backend"
            mem.close()
    
    def test_config_embedding_backend(self):
        """Test that config.search.backend is respected."""
        from agent_memfas import Memory, Config
        from agent_memfas.config import SearchConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                db_path=f"{tmpdir}/test.db",
                search=SearchConfig(
                    backend="embedding",
                    embedder_type="fastembed"
                )
            )
            
            try:
                mem = Memory(config)
                assert type(mem.backend).__name__ == "EmbeddingBackend"
                mem.close()
            except ImportError:
                pytest.skip("fastembed or sqlite-vec not installed")
    
    def test_explicit_embedder_overrides_config(self):
        """Test that passing embedder= takes precedence."""
        from agent_memfas import Memory, Config
        from agent_memfas.embedders.base import Embedder
        
        class CustomEmbedder(Embedder):
            @property
            def dimensions(self) -> int:
                return 8
            
            def embed(self, text: str) -> list:
                return [0.0] * 8
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(db_path=f"{tmpdir}/test.db")
            
            try:
                mem = Memory(config, search_backend="embedding", embedder=CustomEmbedder())
                assert type(mem.backend).__name__ == "EmbeddingBackend"
                mem.close()
            except ImportError:
                pytest.skip("sqlite-vec not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
