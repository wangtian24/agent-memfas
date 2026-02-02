# Changelog

All notable changes to agent-memfas will be documented in this file.

## [0.2.0] - 2025-02-02

### Added

- **Pluggable search backends** — Swap between FTS5 and embedding search without changing code
- **Embedding backend** — Vector similarity search using sqlite-vec
- **FastEmbed embedder** — Local embeddings via BAAI/bge-small-en-v1.5 (384 dims, ~130MB)
- **Ollama embedder** — Local embeddings via nomic-embed-text (768 dims)
- **Config-based backend selection** — Set `search.backend: embedding` in config
- **Config-based embedder** — Set `embedder_type` and `embedder_model` in config
- **`memfas reindex` CLI** — Migrate data between backends
- **`memfas suggest` CLI** — Auto-suggest triggers from indexed content
- **Backend tests** — Comprehensive tests for FTS5 and embedding backends

### Changed

- `SearchConfig` now includes `backend`, `embedder_type`, and `embedder_model` fields
- `Memory` class accepts `search_backend` and `embedder` parameters
- `Config` dataclass now auto-converts dicts to proper types via `__post_init__`

### Fixed

- sqlite-vec KNN query syntax (`k=?` constraint instead of `LIMIT`)
- Config dict-to-dataclass conversion for sources and triggers
- Reindex function now properly reads from database tables

### Dependencies

New optional dependencies:
```bash
pip install agent-memfas[embeddings]  # fastembed + sqlite-vec
pip install agent-memfas[ollama]      # requests + sqlite-vec  
pip install agent-memfas[all]         # everything
```

## [0.1.0] - 2025-02-02

### Added

- Initial release
- **Type 1 (Fast)** — Keyword triggers for instant O(1) recall
- **Type 2 (Slow)** — FTS5 full-text search with BM25 ranking
- **CLI** — `memfas init/recall/search/remember/forget/index/stats`
- **Python API** — `Memory` class with `recall()`, `search()`, `add_trigger()`
- **Zero dependencies** — Works out of the box with SQLite
- **YAML/JSON config** — Flexible configuration format
- **Recency decay** — Recent memories score higher
- **Auto-suggest** — `suggest_triggers()` for finding trigger candidates
