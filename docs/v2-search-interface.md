# v2 Search Interface Design

## Goals
1. **Pluggable backends** — swap FTS5 for embeddings without changing app code
2. **Simple interface** — adapters are easy to write
3. **Optional deps** — heavy stuff only if you opt-in
4. **Contrib-friendly** — community can add backends

---

## Interface

```python
# agent_memfas/search/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    """Universal search result."""
    text: str
    score: float
    source: str
    metadata: dict = None

class SearchBackend(ABC):
    """Base class for all Type 2 search backends."""
    
    @abstractmethod
    def index(self, doc_id: str, text: str, metadata: dict = None) -> None:
        """Index a document."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for documents."""
        pass
    
    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove a document from index."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed documents."""
        pass
    
    def close(self) -> None:
        """Cleanup resources."""
        pass
```

---

## Built-in Backends

### 1. FTS5 (Default, Zero Deps)

```python
# agent_memfas/search/fts5.py

class FTS5Backend(SearchBackend):
    """SQLite FTS5 full-text search. No dependencies."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._setup_tables()
    
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        # BM25 ranking, same as v0.1
        ...
```

### 2. Embedding + sqlite-vec (Optional)

```python
# agent_memfas/search/embedding.py

class EmbeddingBackend(SearchBackend):
    """Vector similarity search with embeddings."""
    
    def __init__(
        self,
        db_path: str,
        embedder: "Embedder",  # Pluggable embedder
        dimensions: int = 384
    ):
        self.embedder = embedder
        self._setup_vector_table(dimensions)
    
    def index(self, doc_id: str, text: str, metadata: dict = None):
        vector = self.embedder.embed(text)
        # Store in sqlite-vec
        ...
    
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        query_vec = self.embedder.embed(query)
        # KNN search via sqlite-vec
        ...
```

---

## Embedder Interface

```python
# agent_memfas/embedders/base.py

class Embedder(ABC):
    """Base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensions."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed single text."""
        pass
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts. Override for efficiency."""
        return [self.embed(t) for t in texts]
```

---

## Local Embedding Options (No API Cost)

| Model | Size | Dimensions | Quality | Notes |
|-------|------|------------|---------|-------|
| **all-MiniLM-L6-v2** | ~80MB | 384 | Good | Popular, fast |
| **BGE-small-en** | ~130MB | 384 | Better | MTEB top performer |
| **nomic-embed-text** | ~270MB | 768 | Great | Via Ollama |
| **gte-small** | ~60MB | 384 | Good | Alibaba, tiny |

### Recommended: FastEmbed (Qdrant)

Lightweight wrapper, auto-downloads models, ONNX runtime:

```python
# agent_memfas/embedders/fastembed.py

class FastEmbedEmbedder(Embedder):
    """Local embeddings via FastEmbed (Qdrant). ~80MB model."""
    
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model)
        self._dimensions = 384
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    def embed(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0].tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self.model.embed(texts)]
```

**Install:** `pip install fastembed`  (~15MB, downloads model on first use)

### Alternative: Ollama

```python
# agent_memfas/embedders/ollama.py

class OllamaEmbedder(Embedder):
    """Local embeddings via Ollama. Requires Ollama running."""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._dimensions = 768  # nomic-embed-text
    
    def embed(self, text: str) -> List[float]:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        return resp.json()["embedding"]
```

**Requires:** Ollama installed + `ollama pull nomic-embed-text`

---

## Vector Storage: sqlite-vec

SQLite extension for vector similarity search.

```python
# agent_memfas/search/embedding.py

def _setup_vector_table(self, dimensions: int):
    import sqlite_vec
    
    self.conn.enable_load_extension(True)
    sqlite_vec.load(self.conn)
    
    self.conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories 
        USING vec0(embedding float[{dimensions}])
    """)
```

**Install:** `pip install sqlite-vec`

**KNN Search:**
```sql
SELECT rowid, distance
FROM vec_memories
WHERE embedding MATCH ?
ORDER BY distance
LIMIT ?
```

---

## Directory Structure

```
agent_memfas/
├── search/
│   ├── __init__.py
│   ├── base.py          # SearchBackend ABC
│   ├── fts5.py          # Default, no deps
│   └── embedding.py     # Vector search, optional deps
├── embedders/
│   ├── __init__.py
│   ├── base.py          # Embedder ABC
│   ├── fastembed.py     # Recommended local
│   └── ollama.py        # Alternative local
└── contrib/
    └── embedders/
        ├── openai.py    # Community: OpenAI
        ├── cohere.py    # Community: Cohere
        └── voyager.py   # Community: Voyager
```

---

## Usage

### Default (FTS5, no deps)
```python
from agent_memfas import Memory

mem = Memory("./memfas.json")
mem.search("machine learning")  # Uses FTS5
```

### With Embeddings (opt-in)
```python
from agent_memfas import Memory
from agent_memfas.embedders.fastembed import FastEmbedEmbedder

mem = Memory(
    "./memfas.json",
    search_backend="embedding",
    embedder=FastEmbedEmbedder()
)
mem.search("machine learning")  # Uses vector similarity
```

### Config-based
```json
{
  "search": {
    "backend": "embedding",
    "embedder": {
      "type": "fastembed",
      "model": "BAAI/bge-small-en-v1.5"
    }
  }
}
```

---

## Optional Dependencies

```toml
# pyproject.toml

[project.optional-dependencies]
embeddings = [
    "fastembed>=0.2.0",
    "sqlite-vec>=0.1.0"
]
ollama = [
    "requests>=2.28.0",
    "sqlite-vec>=0.1.0"
]
all = [
    "fastembed>=0.2.0",
    "sqlite-vec>=0.1.0",
    "requests>=2.28.0",
    "pyyaml>=6.0"
]
```

**Install:**
```bash
pip install agent-memfas                    # FTS5 only
pip install agent-memfas[embeddings]       # + FastEmbed + sqlite-vec
pip install agent-memfas[ollama]           # + Ollama support
pip install agent-memfas[all]              # Everything
```

---

## Writing a Custom Adapter

```python
# contrib/embedders/my_embedder.py

from agent_memfas.embedders.base import Embedder

class MyCustomEmbedder(Embedder):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._dimensions = 1024
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    def embed(self, text: str) -> List[float]:
        # Call your API
        return my_api_call(text, self.api_key)

# Usage:
from contrib.embedders.my_embedder import MyCustomEmbedder
mem = Memory(config, embedder=MyCustomEmbedder(api_key="..."))
```

---

## Migration Path

1. **v0.1 users**: No changes needed, FTS5 remains default
2. **v0.2 upgrade**: `pip install agent-memfas[embeddings]`, set config
3. **Re-index**: `memfas reindex --backend embedding` (one-time)

---

## Open Questions

1. **Hybrid search** — Combine FTS5 + embedding scores? (defer to v0.3?)
2. **Chunking** — Re-chunk for embeddings? (current 100-char min too small)
3. **Caching** — Cache embeddings to avoid re-computing? (yes, in sqlite)
