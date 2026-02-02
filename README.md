# agent-memfas

**Memory Fast and Slow for AI Agents**

A dual-store memory system inspired by Kahneman's "Thinking, Fast and Slow" — giving AI agents persistent, intelligent memory that survives context window limits.

[![PyPI version](https://badge.fury.io/py/agent-memfas.svg)](https://badge.fury.io/py/agent-memfas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Why memfas?

AI agents lose context. When conversations get long, older messages get compacted or dropped. Critical information vanishes:

```
User: "Let's continue the project"
Agent: "I apologize, but I don't have context about what project..."
```

**memfas fixes this** with persistent memory that lives outside the context window.

---

## ✨ Features at a Glance

### 🧠 Dual-Store Memory (v0.1+)
| Type | Speed | Use Case |
|------|-------|----------|
| **Type 1 (Fast)** | O(1) | Keyword triggers → instant recall |
| **Type 2 (Slow)** | O(log n) | Full-text search with BM25 ranking |

### 🔌 Pluggable Search Backends (v0.2+)
| Backend | Dependencies | Best For |
|---------|-------------|----------|
| **FTS5** | None (SQLite built-in) | Keyword matching, low resources |
| **Embeddings** | fastembed, sqlite-vec | Semantic search, "find similar" |

### 🎯 Dynamic Context Curation (v0.3+)
| Feature | Benefit |
|---------|---------|
| **Topic Detection** | Understands what you're talking about |
| **Relevance Scoring** | Multi-factor: semantic + recency + access patterns |
| **Token Budget** | Fills fixed budget with top memories |
| **84% Token Reduction** | 50K baseline → 7.8K curated |

### 📊 Telemetry & Analytics (v0.3+)
- JSONL logging of every memory operation
- Track compression ratios, latency, topic shifts
- `memfas telemetry summary` for insights

---

## 🚀 Quick Start

### Installation

```bash
pip install agent-memfas                 # Core (FTS5, zero deps)
pip install agent-memfas[embeddings]     # + semantic search
pip install agent-memfas[v3]             # + dynamic curation
pip install agent-memfas[all]            # Everything
```

### Basic Usage (30 seconds)

```bash
# Initialize
cd ~/my-agent && memfas init

# Add keyword triggers (Type 1)
memfas remember family --hint "Wife Xu, daughters Veronica & Oumi"
memfas remember project --hint "Building agent-memfas memory system"

# Index your memory files (Type 2)
memfas index ./MEMORY.md ./memory/

# Recall context
memfas recall "How's the family?"
# → Returns triggered + searched memories
```

### Python API

```python
from agent_memfas import Memory

# Initialize
mem = Memory("./memfas.yaml")

# Type 1: Instant triggers
mem.add_trigger("family", "User's family context")

# Type 2: Index and search
mem.index_file("./MEMORY.md")
results = mem.search("preference learning", limit=5)

# Combined recall
context = mem.recall("How's the family?")
print(context)  # Ready to inject into LLM prompt
```

### With Semantic Search (v0.2+)

```python
from agent_memfas import Memory
from agent_memfas.embedders.fastembed import FastEmbedEmbedder

# Local embeddings (~130MB model, runs on CPU)
mem = Memory(
    "./memfas.yaml",
    search_backend="embedding",
    embedder=FastEmbedEmbedder()
)

# Now finds conceptually related content
results = mem.search("machine learning concepts")
```

### With Dynamic Curation (v0.3+)

```python
from agent_memfas.v3 import ContextCurator

curator = ContextCurator("./memfas.yaml")

# Get curated context within token budget
result = curator.get_context(
    query="what's the project status?",
    session_id="main",
    baseline_tokens=50000  # Your context limit
)

print(f"Curated: {result.curated_tokens} tokens")
print(f"Saved: {result.tokens_saved} ({result.compression_ratio:.0%})")
print(result.context)  # Inject this into your prompt
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      agent-memfas                           │
├─────────────────────────────────────────────────────────────┤
│  v0.3: Context Curation                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Topic   │  │Relevance │  │  Token   │  │ Session  │   │
│  │ Detector │→ │  Scorer  │→ │  Budget  │→ │  State   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
├─────────────────────────────────────────────────────────────┤
│  v0.2: Search Backends                                      │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   FTS5Backend   │    │EmbeddingBackend │                │
│  │  (zero deps)    │    │ (sqlite-vec)    │                │
│  └─────────────────┘    └─────────────────┘                │
│           ↑                      ↑                          │
│           └──────┬───────────────┘                          │
│                  │                                          │
│         ┌───────────────┐                                   │
│         │ SearchBackend │  ← Pluggable interface            │
│         │     ABC       │                                   │
│         └───────────────┘                                   │
├─────────────────────────────────────────────────────────────┤
│  v0.1: Core Memory                                          │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Type 1: Fast  │    │   Type 2: Slow  │                │
│  │    Triggers     │    │     Search      │                │
│  │     O(1)        │    │    O(log n)     │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📖 Documentation

### Configuration

Create `memfas.yaml`:

```yaml
db_path: ./memfas.db

sources:
  - path: ./MEMORY.md
    type: markdown
  - path: ./memory/*.md
    type: markdown

triggers:
  - keyword: family
    hint: "User's family context"
  - keyword: work
    hint: "Current projects"

search:
  max_results: 5
  recency_weight: 0.3  # Favor recent memories
  min_score: 0.1
```

### CLI Reference

| Command | Description |
|---------|-------------|
| `memfas init` | Initialize in current directory |
| `memfas recall <context>` | Recall memories (Type 1 + Type 2) |
| `memfas search <query>` | Search only (Type 2) |
| `memfas remember <kw> --hint <h>` | Add trigger |
| `memfas forget <keyword>` | Remove trigger |
| `memfas triggers` | List all triggers |
| `memfas index <paths...>` | Index files/directories |
| `memfas suggest` | Auto-suggest triggers from content |
| `memfas stats` | Show statistics |
| `memfas clear` | Clear indexed memories |
| `memfas curate <query>` | Get curated context (v0.3) |
| `memfas telemetry summary` | View performance stats (v0.3) |

### Embedder Options

| Embedder | Install | Model | Notes |
|----------|---------|-------|-------|
| **FastEmbed** | `pip install fastembed` | bge-small-en | Recommended, ~130MB |
| **Ollama** | `ollama pull nomic-embed-text` | nomic-embed | Good if using Ollama |

---

## 🔬 How It Works

### Type 1: Keyword Triggers (Fast Path)

```
Input: "How's the family doing?"
         ↓
Trigger table scan: "family" → match!
         ↓
Return hint + linked memories instantly
```

### Type 2: Search (Slow Path)

**FTS5 (default):**
```
Input: "preference learning papers"
         ↓
BM25 ranking + recency decay
         ↓
Top results by relevance
```

**Embeddings:**
```
Input: "machine learning concepts"
         ↓
Generate query embedding
         ↓
KNN search (cosine similarity)
         ↓
Semantically related results
```

### v0.3: Dynamic Curation

```
Context: "Let's continue the project discussion"
                    ↓
┌─────────────────────────────────────┐
│ 1. Detect topic: "project"          │
│ 2. Score all memories:              │
│    - Semantic relevance: 0.85       │
│    - Recency: 0.92                  │
│    - Topic continuity: 0.78         │
│    - Access pattern: 0.65           │
│ 3. Fill 8000 token budget           │
│ 4. Return curated context           │
└─────────────────────────────────────┘
                    ↓
Result: 84% token reduction, focused context
```

---

## 🧪 Performance

| Metric | v0.1 | v0.2 | v0.3 |
|--------|------|------|------|
| Trigger lookup | O(1) | O(1) | O(1) |
| FTS5 search | O(log n) | O(log n) | O(log n) |
| Embedding search | - | O(n) | O(n) cached |
| Token reduction | - | - | **84%** |
| Warm query latency | - | - | **8ms** (296x speedup) |

---

## 🤝 Integration

### Clawdbot

```markdown
## Memory (in AGENTS.md)

Before answering about prior work:
1. Run `memfas recall "<context>"`
2. Include returned context in reasoning

After compaction:
1. Run `memfas recall "current project"`
2. Check `memfas triggers`
```

### Custom Agents

```python
# In your agent loop
from agent_memfas.v3 import ContextCurator

curator = ContextCurator("./memfas.yaml")

def get_response(user_message):
    # Get curated memory context
    mem_result = curator.get_context(
        query=user_message,
        session_id="main",
        baseline_tokens=100000
    )
    
    # Inject into prompt
    prompt = f"""
{mem_result.context}

User: {user_message}
"""
    return llm.complete(prompt)
```

---

## 📚 Resources

- **Design Docs**: See `/docs` for architecture decisions
- **Changelog**: See releases for version history
- **Issues**: [GitHub Issues](https://github.com/wangtian24/agent-memfas/issues)

---

## 📄 License

MIT

---

*Built for AI agents that need to remember. Inspired by losing context while building a memory system.*
