# agent-memfas Roadmap

## v0.1.0 ✅
- Type 1: Keyword triggers
- Type 2: FTS5 + BM25
- Zero dependencies
- CLI + Python API
- Auto-suggest triggers

---

## v0.2.0 ✅ (Current)

### Pluggable Search Backends

Implemented a clean abstraction for search backends:

```python
# Default: FTS5 (zero deps)
mem = Memory("./memfas.yaml")

# Opt-in: Local embeddings via FastEmbed
from agent_memfas.embedders.fastembed import FastEmbedEmbedder
mem = Memory(config, search_backend="embedding", embedder=FastEmbedEmbedder())

# Opt-in: Local embeddings via Ollama
from agent_memfas.embedders.ollama import OllamaEmbedder
mem = Memory(config, search_backend="embedding", embedder=OllamaEmbedder())
```

### New Components

- `agent_memfas.search.base` — `SearchBackend` ABC, `SearchResult` dataclass
- `agent_memfas.search.fts5` — `FTS5Backend` (default, zero deps)
- `agent_memfas.search.embedding` — `EmbeddingBackend` (sqlite-vec)
- `agent_memfas.embedders.base` — `Embedder` ABC
- `agent_memfas.embedders.fastembed` — `FastEmbedEmbedder` (BAAI/bge-small)
- `agent_memfas.embedders.ollama` — `OllamaEmbedder` (nomic-embed-text)

### Optional Dependencies

```bash
pip install agent-memfas                 # FTS5 only
pip install agent-memfas[embeddings]     # + FastEmbed + sqlite-vec
pip install agent-memfas[ollama]         # + Ollama support
pip install agent-memfas[all]            # Everything
```

### CLI Reindex

```bash
# Re-index with embedding backend
memfas reindex -b embedding -e fastembed -y --save-config
```

### Config-based Backend

```yaml
search:
  backend: embedding
  embedder_type: fastembed
  embedder_model: BAAI/bge-small-en-v1.5
```

### Migration

- v0.1 → v0.2: No changes needed, FTS5 remains default
- To upgrade to embeddings: `memfas reindex -b embedding -e fastembed`

---

## v0.3.0 — Hybrid Search + Polish

### 1. Hybrid Search (FTS5 + Embeddings)

Combine keyword and semantic for best results:

```python
{
  "search": {
    "mode": "hybrid",
    "hybrid_alpha": 0.5  # Balance FTS5 vs embedding
  }
}
```

### 2. Memory IDs for Triggers

Currently broken in v0.2 due to schema change. Fix with doc_id based linking:

```python
mem.add_trigger("project", "Current project", doc_ids=["abc123", "def456"])
```

### 3. CLI Embedding Support ✅ (Partial)

```bash
# Implemented in v0.2:
memfas reindex -b embedding -e fastembed --save-config

# TODO for v0.3:
memfas index ./memory/ --backend embedding
memfas search "concepts" --semantic
```

### 4. Better Chunking

Configurable chunk size for embeddings (current 100-char min is too small):

```python
{
  "indexing": {
    "chunk_size": 500,
    "chunk_overlap": 50
  }
}
```

---

## v0.4.0 — MCP Server + Auto-Integration

### 1. MCP Server
Expose memfas as MCP tools for any agent framework:
```python
# memfas-mcp-server
tools:
  - get_context(query, token_budget) → curated context
  - recall(query) → basic search
  - add_trigger(keyword, hint) → add trigger
```

### 2. Clawdbot Integration
- Auto-inject curated context each turn
- No manual `recall` needed
- Gateway plugin or hook

### 3. Auto-Compaction Hook
Detect when context is large → proactively curate before compaction hits.

---

## v0.5.0 — Advanced Features

### 1. Trigger Learning
Auto-suggest promotions from frequent Type 2 queries:
```python
mem.suggest_promotions()
# → "preference learning" searched 5x this week
```

### 2. Memory Decay
Configurable forgetting:
```python
{
  "decay": {
    "half_life_days": 90,
    "access_refresh": true
  }
}
```

### 3. Source Watching
Auto-reindex on file changes:
```bash
memfas watch ./memory/
```

### 4. Context Budget
Help agents budget context:
```python
result = mem.recall("family", max_tokens=500)
print(result.token_estimate)
```

---

## v1.0.0 — Production Ready

- [ ] Comprehensive docs site
- [ ] Benchmarks (speed, recall quality)
- [ ] Multiple embedding provider tests
- [ ] Async API support
- [ ] Memory namespaces (multi-agent)
- [ ] Export/import (backup, migration)
- [ ] Web UI for browsing memories

---

## Non-Goals (for now)

- **Multi-user auth** — Agent-local memory, not a service
- **Real-time sync** — Files are source of truth
- **Complex ontologies** — Keep it simple

---

## Contributing

Ideas? Open an issue: https://github.com/wangtianthu/agent-memfas/issues

Community embedders welcome in `contrib/embedders/`:
- OpenAI
- Cohere
- Voyager
