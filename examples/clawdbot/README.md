# Clawdbot Integration Example

This shows how to integrate agent-memfas with a Clawdbot agent.

## Setup

1. Install memfas in your agent's workspace:

```bash
cd ~/clawd
pip install agent-memfas
```

2. Initialize:

```bash
memfas init
```

3. Configure your memory sources in `memfas.yaml`:

```yaml
db_path: ./memfas.db

sources:
  - path: ./MEMORY.md
    type: markdown
  - path: ./memory/*.md
    type: markdown
  - path: ./intuition.md
    type: markdown
    always_load: true

triggers:
  # Add triggers for things you want instant recall on
  - keyword: family
    hint: "User's family members and context"
  - keyword: work
    hint: "Current job and projects"
```

4. Index your existing memory files:

```bash
memfas index ./MEMORY.md ./memory/
```

## Usage in AGENTS.md

Add to your agent's `AGENTS.md`:

```markdown
## Memory Recall

Before answering questions about prior work, decisions, preferences, or people:

1. Run: `memfas recall "<the user's question or context>"`
2. Review the returned memories
3. Use relevant context in your response

To add new keyword triggers:
- `memfas remember <keyword> --hint "<description>"`

To re-index after updating memory files:
- `memfas index ./memory/`
```

## Example Session

```
User: What do you know about my running goals?

Agent thinking:
1. Check memory: `memfas recall "running goals"`
2. Got context about marathon training, CIM December 2025
3. Respond with relevant info
```

## Programmatic Usage

If you want to use memfas from Python code in a skill:

```python
#!/usr/bin/env python3
# scripts/memory-lookup.py

import sys
from agent_memfas import Memory

def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not query:
        print("Usage: memory-lookup.py <query>")
        sys.exit(1)
    
    mem = Memory("./memfas.yaml")
    context = mem.recall(query)
    
    if context:
        print(context)
    else:
        print("No relevant memories found.")
    
    mem.close()

if __name__ == "__main__":
    main()
```

## Tips

1. **Start with triggers**: Add triggers for the most common topics first
2. **Index regularly**: Re-index after significant memory updates
3. **Check stats**: Use `memfas stats` to see what's indexed
4. **Tune recency**: Adjust `recency_weight` if old memories matter more/less
