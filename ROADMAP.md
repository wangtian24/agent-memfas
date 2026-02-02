# agent-memfas Roadmap

## v0.1.0 (Current) ✅
- Type 1: Keyword triggers
- Type 2: FTS5 + BM25
- Zero dependencies
- CLI + Python API
- Auto-suggest triggers

---

## v0.2.0 — Embedding Support (RAG)

### The Case for Embeddings

Current FTS5 limitations:
- **Lexical only** — "running" won't match "jogging" or "marathon training"
- **No semantic similarity** — can't find conceptually related memories
- **Keyword-dependent** — user must guess the right words

Embeddings solve this:
- **Semantic search** — meaning-based, not word-based
- **Better recall** — finds related content even with different vocabulary
- **Standard RAG pattern** — familiar to most developers

### Design Decisions

#### 1. Embedding Provider Options

| Option | Pros | Cons |
|--------|------|------|
| **OpenAI** | Best quality, easy API | Requires API key, costs $, network dependency |
| **Local (sentence-transformers)** | Free, offline, private | Adds ~500MB dependency, slower first load |
| **Ollama** | Local, good quality | Requires Ollama running |
| **Pluggable** | User choice | More complex API |

**Recommendation**: Pluggable with sensible defaults
```python
# Default: FTS5 (no deps)
mem = Memory(config)

# Opt-in: OpenAI embeddings
mem = Memory(config, embedder="openai")

# Opt-in: Local embeddings
mem = Memory(config, embedder="local")  # uses sentence-transformers
```

#### 2. Vector Storage Options

| Option | Pros | Cons |
|--------|------|------|
| **SQLite + numpy** | Zero new deps, simple | Slower for large datasets |
| **sqlite-vec** | Fast, SQLite native | Requires extension install |
| **Chroma** | Full-featured, popular | Heavy dependency |
| **FAISS** | Very fast | Complex, C++ dependency |

**Recommendation**: SQLite + numpy for v0.2, optional sqlite-vec for v0.3
```sql
CREATE TABLE embeddings (
    memory_id INTEGER PRIMARY KEY,
    vector BLOB  -- numpy array serialized
);
```

#### 3. Hybrid Search

Best of both worlds — combine FTS5 + embeddings:

```
Query: "How's marathon training going?"
         │
         ├─→ FTS5: "marathon" "training" → BM25 scores
         │
         └─→ Embeddings: semantic similarity → cosine scores
         
         Combined: α * BM25 + (1-α) * cosine
         (α configurable, default 0.5)
```

### API Changes

```python
# Config additions
{
  "search": {
    "mode": "hybrid",  # "fts5" | "embedding" | "hybrid"
    "embedding": {
      "provider": "openai",  # "openai" | "local" | "ollama"
      "model": "text-embedding-3-small",
      "dimensions": 1536
    },
    "hybrid_alpha": 0.5  # Weight for FTS5 vs embedding
  }
}

# Memory API
mem.search("marathon training", mode="hybrid")  # Uses config default
mem.search("marathon training", mode="embedding")  # Force embedding only
```

### CLI Changes

```bash
# Index with embeddings
memfas index ./memory/ --embed

# Search modes
memfas search "marathon training"           # Uses config default
memfas search "marathon training" --semantic  # Force embedding search

# Re-embed existing content
memfas reindex --embed
```

### Migration Path

1. Existing users: No changes needed, FTS5 remains default
2. Opt-in: Set `search.mode: "hybrid"` and configure embedding provider
3. Gradual: Can embed incrementally, hybrid search works with partial embeddings

---

## v0.3.0 — Advanced Features

### 1. Trigger Learning
Auto-promote frequent Type 2 queries to Type 1 triggers:
```python
# Track query patterns
mem.recall("family")  # Type 1 hit
mem.recall("preference learning")  # Type 2 search, 5th time this week

# Suggest promotion
mem.suggest_promotions()
# → "preference learning" searched 5x, consider: 
#   memfas remember "preference learning" --hint "..."
```

### 2. Memory Decay
Configurable forgetting — old, unaccessed memories fade:
```python
{
  "decay": {
    "enabled": true,
    "half_life_days": 90,  # Memories lose 50% relevance after 90 days
    "access_refresh": true  # Accessing resets decay
  }
}
```

### 3. Memory Graphs
Link related memories:
```python
mem.link(memory_id_1, memory_id_2, relation="related_to")
mem.get_related(memory_id)  # Graph traversal
```

### 4. Source Watching
Auto-reindex when files change:
```bash
memfas watch ./memory/  # Watches for changes, auto-reindexes
```

### 5. Context Window Estimation
Help agents budget context:
```python
result = mem.recall("family", max_tokens=500)
print(result.token_estimate)  # ~340 tokens
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

- **Multi-user auth** — This is agent-local memory, not a service
- **Real-time sync** — Files are the source of truth
- **Complex ontologies** — Keep it simple, keyword + search

---

## Open Questions

1. **Embedding cost**: Should we cache embeddings aggressively? Re-embed on content change only?

2. **Chunk size**: Current 100-char minimum is arbitrary. Optimal chunk size for embeddings is typically 200-500 tokens. Make configurable?

3. **Hybrid ranking**: How to normalize BM25 (unbounded) with cosine similarity (0-1)? Research needed.

4. **Ollama integration**: Auto-detect if Ollama is running? Fallback to FTS5?

5. **Privacy**: Local-only embedding option is critical for sensitive memories. Make this the default?

---

## Contributing

Ideas? Open an issue: https://github.com/wangtian24/agent-memfas/issues

Priority for v0.2:
1. Pluggable embedding interface
2. OpenAI provider (most common)
3. Local provider (sentence-transformers)
4. Hybrid search
