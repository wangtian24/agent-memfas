# Changelog

All notable changes to agent-memfas will be documented in this file.

## [0.3.1] - 2026-02-02

### Added

- **Curation levels 1-5** — Slider for controlling context aggressiveness
  - Level 1 (minimal): ~300 tokens, threshold 0.75
  - Level 2 (lean): ~800 tokens, threshold 0.55
  - Level 3 (balanced): ~1500 tokens, threshold 0.40 — **default**
  - Level 4 (rich): ~3000 tokens, threshold 0.25
  - Level 5 (full): ~5000 tokens, threshold 0.10
- **Level names** — Use `"balanced"` or `3` interchangeably
- **`level="auto"`** — Defaults to 3, ready for smart auto-selection
- **Per-query level override** — `get_context(query, level=4)`
- **CLI level support** — `memfas curate --level 2`
- **`--describe-levels`** — Show level descriptions

### Changed

- `ContextCurator.__init__` accepts `level` parameter
- `ContextResponse` includes `curation_level` and `curation_level_name`
- `TurnMetrics` tracks level and min_score_threshold

## [0.3.0] - 2026-02-02

### Added

- **v3: Dynamic Context Curation** — Proactive memory curation each turn
- **ContextCurator** — Main entry point for v3 API
- **Topic detection** — Tracks conversation topic and shifts
- **Multi-factor scoring** — Semantic, recency, access, topic coherence
- **Token budget management** — Fill budget with highest-value memories
- **Session state** — Track per-session context across turns
- **Telemetry logging** — JSONL logs for performance analysis
- **84% token reduction** — Compared to loading all memories

### Usage

```python
from agent_memfas.v3 import ContextCurator

curator = ContextCurator("./memfas.yaml", level=3)
result = curator.get_context("what's the status?", session_id="main")
print(result.context)  # Curated context to inject
print(result.tokens_saved)  # How much we saved
```

## [0.2.0] - 2026-02-02

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
