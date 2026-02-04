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

- **[v0.1] Core Memory**
  - Type 1 (Fast) — O(1) keyword triggers for instant recall
  - Type 2 (Slow) — FTS5 full-text search with BM25 ranking
  - Zero dependencies — works with SQLite built-in

- **[v0.2] Pluggable Backends**
  - Swappable search backends — FTS5 or embeddings
  - Semantic search — FastEmbed or Ollama embeddings
  - Auto-suggest triggers from indexed content
  - `memfas reindex` — migrate between backends

- **[v0.3] Dynamic Context Curation**
  - Proactive memory selection each turn
  - Topic detection — tracks conversation topic and shifts
  - Multi-factor relevance scoring — semantic + recency + access patterns
  - Token budget management — fills budget with highest-value memories
  - 84% token reduction — 50K baseline → 7.8K curated
  - Telemetry — JSONL logging, compression stats, latency tracking

- **[v0.3.1] Curation Levels**
  - 5-level slider from minimal to full context
  - Level names: minimal / lean / balanced / rich / full
  - Per-query level override
  - `auto` level ready for smart selection

- **[v0.4] Context Management**
  - Pre-emptive compaction — triggers at 50%, not 90%
  - Three-way classification — KEEP / SUMMARIZE / DROP
  - Relevance scoring — embeddings + keyword fallback + recency decay
  - Cold storage — dropped chunks recoverable for 30 days
  - Pluggable summarization — MiniMax API or custom backends
  - Full observability — JSONL logging for all events
  - Agent-agnostic — plugs into any agent loop via callbacks

---

## 🚀 Quick Start

### Installation

```bash
pip install agent-memfas                 # Core (FTS5, zero deps)
pip install agent-memfas[embeddings]     # + semantic search
pip install agent-memfas[v3]             # + dynamic curation
pip install agent-memfas[context]        # + context management (v0.4)
pip install agent-memfas[all]            # Everything
```

### Basic Usage (30 seconds)

```bash
# Initialize
cd ~/my-agent && memfas init

# Add keyword triggers (Type 1)
memfas remember alice --hint "Project manager, prefers async communication"
memfas remember acme --hint "Client project, due Q2, React frontend"

# Index your memory files (Type 2)
memfas index ./MEMORY.md ./memory/

# Recall context
memfas recall "What did Alice say about the deadline?"
# → Returns triggered + searched memories
```

### Python API

```python
from agent_memfas import Memory

# Initialize
mem = Memory("./memfas.yaml")

# Type 1: Instant triggers
mem.add_trigger("alice", "Project manager, prefers async")

# Type 2: Index and search
mem.index_file("./MEMORY.md")
results = mem.search("preference learning", limit=5)

# Combined recall
context = mem.recall("What did Alice say about the deadline?")
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

### With Context Management (v0.4+)

```python
from agent_memfas.context import ContextManager

# Initialize with callbacks for your agent's context
ctx = ContextManager(
    config_path="./memfas.yaml",
    memfas_memory_path="./memfas.yaml",   # enables embedding-based scoring
    get_context=lambda: current_messages,
    set_context=lambda msgs: replace_messages(msgs),
    get_token_count=lambda: count_tokens(current_messages),
    get_messages=lambda n: current_messages[-n:],
)

# In your agent loop:

# 1. New message arrived
ctx.on_message(user_message, current_messages)
ctx.set_current_prompt(user_message)

# 2. Before generating response — checks health, auto-compacts if needed
status = ctx.before_response(max_tokens=100000)
print(f"Context: {status.pct_used:.0%} used, compaction: {status.needs_compaction}")

# 3. After response
ctx.after_response()

# 4. Session ending — archives to cold storage
ctx.session_end()

# 5. Need something back from cold storage?
recovered = ctx.recover("what was the database migration plan")
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      agent-memfas                           │
├─────────────────────────────────────────────────────────────┤
│  v0.4: Context Management                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Context  │  │Relevance │  │  Cold    │  │Summarizer│   │
│  │ Manager  │→ │  Scorer  │→ │ Storage  │→ │ (MiniMax)│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       ↓                                                     │
│  KEEP / SUMMARIZE / DROP → Recoverable for 30 days         │
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
  - keyword: alice
    hint: "Project manager, prefers async"
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

**Context Management (v0.4)** — Python API only for now:

```python
from agent_memfas.context import ContextManager, ContextConfig

# Status check
ctx.status()  # Returns ContextStatus with tokens, chunks, pct_used

# Manual compaction
result = ctx.compact()  # Returns CompactionResult

# Cold storage recovery
chunks = ctx.recover("database migration")

# Health metrics
ctx.health_check()  # Dict with tokens, cold storage count, config
```

### Embedder Options

| Embedder | Install | Model | Notes |
|----------|---------|-------|-------|
| **FastEmbed** | `pip install fastembed` | bge-small-en | Recommended, ~130MB |
| **Ollama** | `ollama pull nomic-embed-text` | nomic-embed | Good if using Ollama |

---

## 🔬 How It Works

### Type 1: Keyword Triggers (Fast Path)

```
Input: "What's the status on the acme project?"
         ↓
Trigger table scan: "alice" → match!
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

### v0.4: Context Management

```
Context window at 50% capacity
                    ↓
┌─────────────────────────────────────┐
│ 1. Score each chunk:                │
│    score = (embedding_sim × 0.6)    │
│          + (recency_decay × 0.2)    │
│          + (importance × 0.1)       │
│                                     │
│ 2. Classify:                        │
│    ≥ 0.7  → KEEP                    │
│    0.3-0.7 → SUMMARIZE (MiniMax)    │
│    ≤ 0.3  → DROP to cold storage    │
│                                     │
│ 3. Enforce min_chunks_to_keep       │
│ 4. Log all drops for debugging      │
└─────────────────────────────────────┘
                    ↓
Result: Pre-emptive compaction, recoverable drops
```

**Recency decay:** Exponential with ~4h half-life. A chunk added 6h ago retains 37% of recency bonus.

**Cold storage recovery:** Jaccard-ranked search with stopword filtering. Chunks recoverable for 30 days.

---

## 🧪 Performance

| Metric | v0.1 | v0.2 | v0.3 | v0.4 |
|--------|------|------|------|------|
| Trigger lookup | O(1) | O(1) | O(1) | O(1) |
| FTS5 search | O(log n) | O(log n) | O(log n) | O(log n) |
| Embedding search | - | O(n) | O(n) cached | O(n) cached |
| Token reduction | - | - | **84%** | dynamic |
| Warm query latency | - | - | **8ms** | **<10ms** |
| Cold storage recovery | - | - | - | Jaccard O(n) |
| Summarization | - | - | - | MiniMax API |

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
# In your agent loop — with v0.3 Curation
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

### Full Agent Loop with Context Management (v0.4)

```python
from agent_memfas.context import ContextManager

class Agent:
    def __init__(self):
        self.messages = []
        self.ctx = ContextManager(
            config_path="./memfas.yaml",
            memfas_memory_path="./memfas.yaml",
            get_context=lambda: self.messages,
            set_context=lambda m: setattr(self, 'messages', m),
            get_token_count=lambda: sum(len(m['content'])//4 for m in self.messages),
            get_messages=lambda n: self.messages[-n:],
        )
    
    def handle_message(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        
        # Pre-response: check health, auto-compact if needed
        self.ctx.set_current_prompt(user_input)
        status = self.ctx.before_response(max_tokens=100000)
        
        # Generate response (your LLM call here)
        response = self.generate(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        
        self.ctx.after_response()
        return response
    
    def end_session(self):
        self.ctx.session_end()  # Archives to cold storage
```

### Context Config (v0.4)

Add to your `memfas.yaml`:

```yaml
context:
  compaction_trigger_pct: 0.50    # Trigger early at 50%
  relevance_cutoff: 0.3           # DROP below this score
  relevance_keep_threshold: 0.7   # KEEP above this score
  min_chunks_to_keep: 5           # Safety floor
  
  # Scoring weights
  memfas_weight: 0.6              # Embedding similarity weight
  recency_bonus: 0.2              # Max recency boost (decays over ~4h)
  importance_bonus: 0.1           # Flat boost for important chunks
  
  # Cold storage
  cold_storage_enabled: true
  cold_storage_path: "./cold-storage/"
  recoverable_days: 30
  
  # Summarization (optional)
  summarize_medium_chunks: true
  summary_model: "minimax/MiniMax-M2.1"  # Set MINIMAX_API_KEY env var
  
  # Logging
  log_path: "./logs/context/"
  log_compaction: true
  log_drops: true
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

