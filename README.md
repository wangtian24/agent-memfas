# agent-memfas

**Memory Fast and Slow for AI Agents**

A dual-store memory system inspired by Kahneman's "Thinking, Fast and Slow":

- **Type 1 (Fast)**: Keyword triggers for instant pattern matching - O(1) lookup
- **Type 2 (Slow)**: FTS5 semantic search for deliberate recall - BM25 ranking with recency decay

Zero external dependencies. Just Python 3.10+ and SQLite (built-in).

## Installation

```bash
pip install agent-memfas

# Or with YAML config support:
pip install agent-memfas[yaml]
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
# 💡 Suggested triggers (min 3 occurrences):
# Entities: alice, bob, acme
# Terms: project, machine, learning
```

### Python Usage

```python
from agent_memfas import Memory

# Initialize
mem = Memory("./memfas.yaml")  # or Memory() for auto-detect

# Add triggers
mem.add_trigger("family", "User's family context")
mem.add_trigger("work", "Current projects and job")

# Index files
mem.index_file("./MEMORY.md")
mem.index_sources()  # Index all sources from config

# Recall (combines Type 1 + Type 2)
context = mem.recall("How's the family?")
print(context)
# 📚 Memory Context:
# **Triggered Memories:**
# [family] User's family context
#   > Wife: Xu. Daughters: Veronica (11), Oumi (6)...

# Search only (Type 2)
results = mem.search("preference learning", limit=5)
for r in results:
    print(f"[{r.source}] {r.text[:100]}...")

# Stats
print(mem.stats())

# Auto-suggest triggers from indexed content
suggestions = mem.suggest_triggers(min_occurrences=3)
for s in suggestions:
    print(f"{s['term']} ({s['count']}x) - {s['type']}")
# {'memories': 150, 'triggers': 12, 'total_chars': 45000, ...}
```

## Configuration

Create `memfas.yaml`:

```yaml
db_path: ./memfas.db

sources:
  - path: ./MEMORY.md
    type: markdown
    always_load: false
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
  max_results: 5
  recency_weight: 0.3  # Higher = favor recent memories
  min_score: 0.1
```

Or use JSON if you prefer:

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

## Clawdbot Integration

For [Clawdbot](https://github.com/clawdbot/clawdbot) agents, add to your `AGENTS.md`:

```markdown
## Memory

Before answering questions about prior work, decisions, or preferences:
1. Run `memfas recall "<context>"` to get relevant memories
2. Include the returned context in your reasoning

To remember something important:
- `memfas remember <keyword> --hint "<description>"`
- Or add to memfas.yaml triggers section
```

Example skill integration:

```bash
# In your agent's workspace
memfas init

# Create a retrieval wrapper
cat > scripts/memory-recall.sh << 'EOF'
#!/bin/bash
cd ~/clawd
memfas recall "$*"
EOF
chmod +x scripts/memory-recall.sh
```

Then in conversations, the agent can:

```bash
# Recall context before answering
./scripts/memory-recall.sh "What did we discuss about the project?"
```

## How It Works

### Type 1: Keyword Triggers (Fast Path)

When you call `recall()`, it first checks for keyword matches:

```
Input: "How's the family doing?"
       ↓
Triggers table: family → "User's family context"
       ↓
Match found! Return instantly.
```

This is O(n) where n = number of triggers (typically < 100), so effectively instant.

### Type 2: FTS5 Search (Slow Path)

If no triggers match, or you want more context, it falls back to full-text search:

```
Input: "What papers were interesting?"
       ↓
FTS5 query: "papers interesting"
       ↓
BM25 ranking + recency decay
       ↓
Top 5 results returned
```

### Recency Decay

Recent memories score higher:

```python
recency_score = 1.0 / (1.0 + days_old * recency_weight * 0.01)
final_score = bm25_score * recency_score
```

With `recency_weight: 0.3`:
- Today: 1.0x
- 30 days ago: 0.91x
- 100 days ago: 0.77x
- 1 year ago: 0.48x

## Design Philosophy

1. **Two speeds of thought**: Fast pattern matching for common queries, deliberate search for exploration
2. **Zero dependencies**: Works with just Python stdlib + SQLite
3. **Agent-friendly output**: Returns markdown formatted for LLM context injection
4. **Recency matters**: Recent memories naturally surface more
5. **Simple to extend**: Add your own indexers for custom formats

## API Reference

### Memory Class

```python
Memory(config: str | Config | None = None)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `recall(context)` | Main entry - combines Type 1 + Type 2 |
| `search(query, limit)` | Type 2 only - FTS5 search |
| `add_trigger(keyword, hint)` | Add Type 1 trigger |
| `remove_trigger(keyword)` | Remove trigger |
| `list_triggers()` | List all triggers |
| `index_file(path, type)` | Index a single file |
| `index_sources()` | Index all config sources |
| `clear()` | Clear all indexed memories |
| `stats()` | Get statistics |
| `suggest_triggers(min_occurrences, limit)` | Auto-suggest triggers from content |
| `close()` | Close database connection |

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

## Real-World Use Case: Surviving Context Compaction

*The problem that inspired this project.*

AI agents running long conversations eventually hit context limits. When the context window fills up, older messages get **compacted** (summarized and dropped). Critical context can be lost:

```
User: "Let's continue working on the project"
Agent: "I apologize, but I don't have context about what project we were working on..."
```

**memfas solves this** by maintaining persistent memory outside the context window:

```bash
# Agent's workspace setup
cd ~/agent-workspace
memfas init

# Add triggers for active work
memfas remember project --hint "Building agent-memfas at ~/workspace/agent-memfas"
memfas remember "working on" --hint "Memory system for AI agents - Type 1 + Type 2"

# Index memory files  
memfas index ./memory/ ./MEMORY.md

# After compaction, agent recovers context:
$ memfas recall "what were we working on"

📚 Memory Context:

**Triggered Memories:**
[project] Building agent-memfas at ~/workspace/agent-memfas
[working on] Memory system for AI agents - Type 1 + Type 2
```

Add this to your agent's instructions:
```markdown
## After Compaction
If you see "Summary unavailable" or feel confused:
1. Run: `memfas recall "current project"`
2. Check: `memfas triggers`
```

*"I lost context while building a memory system. So I used that memory system to never lose context again."*

## License

MIT
