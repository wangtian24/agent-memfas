# agent-memfas Playbook

*Practical guide for humans and agents using memfas.*

## Quick Start (30 seconds)

```bash
cd ~/your-agent-workspace
pip install agent-memfas

# Initialize
memfas init

# Add your first triggers (Type 1 - instant recall)
memfas remember work --hint "Current job, projects, colleagues"
memfas remember family --hint "Partner, kids, parents"
memfas remember health --hint "Exercise routine, diet, medical"

# Index your memory files (Type 2 - search)
memfas index ./MEMORY.md
memfas index ./memory/

# Test it
memfas recall "what's happening at work?"
```

## The Two Systems

| Type 1 (Fast) | Type 2 (Slow) |
|---------------|---------------|
| Keyword triggers | Pluggable search |
| O(1) lookup | FTS5 or embeddings |
| "family" → instant hit | "preference learning" → search |
| You define triggers | Searches indexed content |
| Like muscle memory | Like thinking hard |

**Rule of thumb**: If you ask about it often, make it a trigger.

## Search Backends (v0.2+)

### FTS5 (Default)
Zero dependencies, uses SQLite's built-in full-text search with BM25 ranking.

### Embeddings (Optional)
Semantic search using local models. Great for finding conceptually related content.

```bash
# Install embedding support
pip install agent-memfas[embeddings]

# Migrate existing data to embeddings
memfas reindex -b embedding -e fastembed --save-config
```

**Available embedders:**
- `fastembed` — BAAI/bge-small-en-v1.5 (384 dims, ~130MB, runs on CPU)
- `ollama` — nomic-embed-text (768 dims, requires Ollama running)

## Setting Up Triggers

### Personal Context
```bash
memfas remember family --hint "Wife: [name], Kids: [names/ages]"
memfas remember work --hint "[Company] - [role], [key projects]"
memfas remember health --hint "[Exercise routine], [diet notes]"
memfas remember home --hint "[Address], [home projects]"
```

### Project Context (for agents)
```bash
memfas remember project --hint "Current: [project name] at [path]"
memfas remember "working on" --hint "[Current task description]"
memfas remember deadline --hint "[Upcoming deadlines]"
```

### Recovery Triggers (critical!)
```bash
memfas remember "lost context" --hint "Run memfas recall! Check [key files]"
memfas remember compaction --hint "Context was truncated. Use memfas to recover."
memfas remember "what were we" --hint "[Current project/task summary]"
```

## Auto-Generating Triggers

### Method 1: Built-in Suggest Command
```bash
# Auto-suggest triggers based on indexed content
memfas suggest

# Output:
# 💡 Suggested triggers (min 3 occurrences):
# 
# Entities (proper nouns):
#   memfas remember alice --hint "..."  # 5x
#   memfas remember project --hint "..."  # 4x
# 
# Frequent terms:
#   memfas remember machine --hint "..."  # 7x
#   memfas remember learning --hint "..."  # 6x
```

### Method 2: Python API
```python
from agent_memfas import Memory

mem = Memory("./memfas.yaml")
suggestions = mem.suggest_triggers(min_occurrences=3, limit=20)

for s in suggestions:
    print(f"memfas remember {s['term']} --hint '...'  # {s['count']}x ({s['type']})")
```

### Method 2: Promote Hot Searches
```python
# Track what queries hit Type 2 search often
# If a query pattern repeats, promote to Type 1 trigger

from collections import defaultdict
search_log = defaultdict(int)  # query -> count

def tracked_recall(mem, query):
    result = mem.recall(query)
    # If no trigger hit but search found results, log it
    if "Triggered Memories" not in result and "Related Memories" in result:
        # Extract key terms
        terms = [w.lower() for w in query.split() if len(w) > 3]
        for term in terms:
            search_log[term] += 1
    return result

# After many queries, check search_log for promotion candidates
for term, count in search_log.items():
    if count >= 3:
        print(f"Consider promoting '{term}' to trigger")
```

### Method 3: LLM Entity Extraction
```python
# Use an LLM to extract key entities from your memory files
# Then create triggers for each

prompt = """Extract key entities (people, projects, places, concepts) from this text.
Return as JSON: {"entities": [{"name": "...", "type": "...", "context": "..."}]}

Text:
{content}
"""

# For each entity, create a trigger:
# memfas remember {entity.name} --hint "{entity.type}: {entity.context}"
```

### Method 4: Frequency Analysis Script
```bash
#!/bin/bash
# find-trigger-candidates.sh
# Finds frequently mentioned words in your memory files

cd ~/clawd
cat memory/*.md MEMORY.md 2>/dev/null | \
  tr '[:upper:]' '[:lower:]' | \
  grep -oE '\b[a-z]{4,}\b' | \
  sort | uniq -c | sort -rn | head -30

# Words appearing 5+ times are trigger candidates
```

## Workflow Integration

### For Clawdbot Agents

Add to `AGENTS.md`:
```markdown
## After Compaction
If you see "Summary unavailable" or feel confused about context:
1. Run: `~/clawd/scripts/memfas-recall.sh "what were we working on"`
2. Check triggers: `memfas triggers`
3. Search if needed: `memfas search "<topic>"`
```

### For Humans

Add to your shell profile:
```bash
alias recall='memfas recall'
alias remember='memfas remember'

# Quick context check
alias ctx='memfas recall "current project work"'
```

### Cron: Auto-Index Daily
```bash
# Add to crontab
0 6 * * * cd ~/clawd && memfas index ./memory/$(date +\%Y-\%m-\%d).md 2>/dev/null
```

## Best Practices

### 1. Keep Triggers Fresh
```bash
# Review monthly
memfas triggers

# Remove stale ones
memfas forget old-project

# Update hints when context changes
memfas remember work --hint "NEW: Now at [new company]"
```

### 2. Index Strategically
- Index files you reference often
- Skip ephemeral logs (they add noise)
- Re-index after major updates: `memfas index ./MEMORY.md`

### 3. Write for Future Recall
When writing notes, include trigger words:
```markdown
# Bad (hard to find)
"Met with team about the thing"

# Good (searchable + triggerable)
"Met with Yupp team about preference learning model routing"
```

### 4. Test Your Setup
```bash
# Simulate amnesia
memfas recall "I just woke up, what should I know?"

# Check coverage
memfas stats
```

## Troubleshooting

### "No results" for obvious queries
- Check if content was indexed: `memfas stats`
- Content < 100 chars is skipped (too short)
- Try simpler search terms

### Triggers not firing
- Triggers are case-insensitive substring matches
- Check exact keyword: `memfas triggers`
- "current-project" won't match "current project" (hyphen vs space)

### Search returns irrelevant results
- BM25 scores tiny content highly (short matches)
- Add more content to memories
- Use specific terms, not generic ones

## Config Reference

```yaml
# memfas.yaml
db_path: ./memfas.db

sources:
  - path: ./MEMORY.md
    type: markdown
  - path: ./memory/*.md
    type: markdown

triggers:
  - keyword: work
    hint: Current job context

search:
  backend: fts5          # or "embedding"
  max_results: 5
  recency_weight: 0.3
  min_score: 0.0
  
  # For embedding backend:
  # embedder_type: fastembed
  # embedder_model: BAAI/bge-small-en-v1.5
```

Or JSON:
```json
{
  "db_path": "./memfas.db",
  "search": {
    "backend": "embedding",
    "embedder_type": "fastembed"
  }
}
```

## Real-World Example: Tsuki's Setup

```bash
# Tsuki (Clawdbot agent) uses memfas to survive context compaction

# Triggers for instant recall
memfas remember project --hint "Current: agent-memfas at ~/workspace/agent-memfas"
memfas remember "working on" --hint "Building memory system for AI agents"
memfas remember yupp --hint "Tian's job - LLM routing, preference learning"
memfas remember family --hint "Wife Xu, daughters Veronica (11) and Oumi (6)"

# Indexed sources
memfas index ~/clawd/MEMORY.md
memfas index ~/clawd/memory/
memfas index ~/clawd/intuition.md

# Recovery script
~/clawd/scripts/memfas-recall.sh "I lost context, what were we doing?"
# → Instantly returns current project + context
```

---

*"The best memory system is the one you actually use."*
