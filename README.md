# agent-memfas

**Memory Fast and Slow for AI Agents**

A dual-store memory system inspired by Kahneman's "Thinking, Fast and Slow":

- **Type 1 (Fast)**: Keyword triggers for instant pattern matching - O(1) lookup
- **Type 2 (Slow)**: Pluggable search backends - FTS5 (default) or embeddings

Zero external dependencies for the default. Optional embedding support with local models.

## Installation

```bash
# Core (FTS5 only, zero deps)
pip install agent-memfas

# With local embeddings (recommended)
pip install agent-memfas[embeddings]

# With Ollama support
pip install agent-memfas[ollama]

# Everything
pip install agent-memfas[all]
```

## Quick Start

### CLI Usage

```bash
# Initialize in your agent's workspace
cd ~/my-agent
memfas init

# Add keyword triggers (Type 1 - fast)
memfas remember family --hint "User's family context"
memfas remember yupp --hint "Current company, LLM routing"
memfas remember running --hint "Marathon training"

# Index your memory files (Type 2 - slow)
memfas index ./memory/
memfas index ./MEMORY.md

# Recall memories for context
memfas recall "How's the family doing?"
memfas recall "What papers were interesting?"

# Search directly
memfas search "preference learning RLHF"

# Check stats
memfas stats

# Auto-suggest triggers from your content
memfas suggest
```

### Python Usage

```python
from agent_memfas import Memory

# Default (FTS5, zero deps)
mem = Memory("./memfas.yaml")

# Add triggers
mem.add_trigger("family", "User's family context")
mem.add_trigger("work", "Current projects and job")

# Index files
mem.index_file("./MEMORY.md")
mem.index_sources()  # Index all sources from config

# Recall (combines Type 1 + Type 2)
context = mem.recall("How's the family?")
print(context)

# Search only (Type 2)
results = mem.search("preference learning", limit=5)
for r in results:
    print(f"[{r.source}] {r.text[:100]}...")

mem.close()
```

### With Embeddings (Semantic Search)

```python
from agent_memfas import Memory
from agent_memfas.embedders.fastembed import FastEmbedEmbedder

# Local embeddings via FastEmbed (~130MB model, runs on CPU)
embedder = FastEmbedEmbedder()  # Uses BAAI/bge-small-en-v1.5
mem = Memory("./memfas.yaml", search_backend="embedding", embedder=embedder)

# Now search uses semantic similarity
results = mem.search("machine learning concepts")  # Finds related content
```

Or with Ollama:

```python
from agent_memfas.embedders.ollama import OllamaEmbedder

# Requires: ollama pull nomic-embed-text
embedder = OllamaEmbedder(model="nomic-embed-text")
mem = Memory("./memfas.yaml", search_backend="embedding", embedder=embedder)
```

## Search Backends

### FTS5 (Default)

- **Zero dependencies** - uses SQLite's built-in FTS5
- **BM25 ranking** with recency decay
- Great for exact keyword matching
- Best for: most use cases, low resource environments

### Embedding (Optional)

- **Semantic search** - finds conceptually related content
- Uses local models via FastEmbed or Ollama
- Requires: `pip install agent-memfas[embeddings]`
- Best for: finding related concepts, multilingual content

| Backend | Dependencies | Quality | Speed | Use Case |
|---------|-------------|---------|-------|----------|
| FTS5 | None | Good for keywords | Fast | Default, most cases |
| Embedding | fastembed, sqlite-vec | Better semantics | Slower | Semantic search |

### Embedder Options

| Embedder | Install | Model Size | Notes |
|----------|---------|------------|-------|
| **FastEmbed** | `pip install fastembed` | ~130MB | Recommended, runs on CPU |
| **Ollama** | `ollama pull nomic-embed-text` | ~270MB | Good if you already use Ollama |

## Configuration

Create `memfas.yaml`:

```yaml
db_path: ./memfas.db

sources:
  - path: ./MEMORY.md
    type: markdown
  - path: ./memory/*.md
    type: markdown
  - path: ./intuition.md
    type: markdown
    always_load: true  # Include in every recall

triggers:
  - keyword: family
    hint: "User's family context"
  - keyword: work
    hint: "Current job and projects"

search:
  backend: fts5  # or "embedding"
  max_results: 5
  recency_weight: 0.3  # Higher = favor recent memories
  min_score: 0.1
  
  # For embedding backend:
  # embedder_type: fastembed  # or "ollama"
  # embedder_model: BAAI/bge-small-en-v1.5
```

Or use JSON:

```json
{
  "db_path": "./memfas.db",
  "sources": [
    {"path": "./MEMORY.md", "type": "markdown"}
  ],
  "triggers": [
    {"keyword": "family", "hint": "User's family context"}
  ]
}
```

## How It Works

### Type 1: Keyword Triggers (Fast Path)

When you call `recall()`, it first checks for keyword matches:

```
Input: "How's the family doing?"
       ↓
Triggers table: family → "User's family context"
       ↓
Match found! Return instantly (O(1))
```

### Type 2: Search (Slow Path)

If no triggers match, or you want more context, it searches:

**FTS5 Backend:**
```
Input: "What papers were interesting?"
       ↓
FTS5 query with BM25 ranking
       ↓
Apply recency decay
       ↓
Top results returned
```

**Embedding Backend:**
```
Input: "machine learning concepts"
       ↓
Generate query embedding
       ↓
KNN search via sqlite-vec
       ↓
Top results by similarity
```

### Recency Decay (FTS5)

Recent memories score higher:

```python
recency_score = 1.0 / (1.0 + days_old * recency_weight * 0.01)
final_score = bm25_score * recency_score
```

## API Reference

### Memory Class

```python
Memory(
    config: str | Config | None = None,
    search_backend: str = None,  # "fts5" or "embedding"
    embedder: Embedder = None    # Required for embedding backend
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `recall(context)` | Main entry - combines Type 1 + Type 2 |
| `search(query, limit)` | Type 2 only - search |
| `add_trigger(keyword, hint)` | Add Type 1 trigger |
| `remove_trigger(keyword)` | Remove trigger |
| `list_triggers()` | List all triggers |
| `index_file(path, type)` | Index a single file |
| `index_sources()` | Index all config sources |
| `clear()` | Clear all indexed memories |
| `stats()` | Get statistics |
| `suggest_triggers()` | Auto-suggest triggers |
| `reindex(backend, embedder)` | Re-index with new backend |
| `close()` | Close database connection |

### Search Backends

```python
from agent_memfas.search.fts5 import FTS5Backend
from agent_memfas.search.embedding import EmbeddingBackend

# Direct backend usage
backend = FTS5Backend("./memfas.db")
backend.index("doc1", "Hello world", {"source": "test.md"})
results = backend.search("hello")
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `memfas init` | Initialize in current directory |
| `memfas recall <context>` | Recall memories |
| `memfas search <query>` | Search memories |
| `memfas remember <kw> --hint <h>` | Add trigger |
| `memfas forget <keyword>` | Remove trigger |
| `memfas triggers` | List triggers |
| `memfas index <paths...>` | Index files/dirs |
| `memfas stats` | Show statistics |
| `memfas clear` | Clear memories |
| `memfas suggest` | Auto-suggest triggers |
| `memfas reindex -b <backend>` | Re-index with new backend |

## Use Case: Surviving Context Compaction

AI agents running long conversations hit context limits. When the window fills up, older messages get **compacted** and critical context can be lost.

memfas maintains persistent memory outside the context window:

```bash
# Agent's workspace setup
cd ~/agent-workspace
memfas init

# Add triggers for active work
memfas remember project --hint "Building agent-memfas"
memfas remember "working on" --hint "Memory system for AI agents"

# Index memory files  
memfas index ./memory/ ./MEMORY.md

# After compaction, agent recovers context:
$ memfas recall "what were we working on"

📚 Memory Context:

**Triggered Memories:**
[project] Building agent-memfas
[working on] Memory system for AI agents
```

## Migrating from v0.1

v0.2 is backward compatible. Your existing FTS5 data keeps working.

To upgrade to embeddings via CLI:

```bash
# Install embedding dependencies
pip install agent-memfas[embeddings]

# Re-index with embedding backend
memfas reindex -b embedding -e fastembed --save-config
```

Or via Python:

```python
from agent_memfas import Memory
from agent_memfas.embedders.fastembed import FastEmbedEmbedder

# Load existing memory with new embedding backend
mem = Memory("./memfas.yaml")
mem.reindex(new_backend="embedding", embedder=FastEmbedEmbedder())
```

## License

MIT
