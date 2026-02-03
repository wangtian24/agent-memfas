# Context Management Plan (memfas v4)

## Overview
Part of agent-memfas as the `context/` module. Handles active session context with aggressive compaction, relevance-based dropping, and cold storage.

## Integration
- **Agent workflow:** Agent handles context calls in workflow (no PR dependency)
- **AGENTS.md:** Documents agent responsibilities
- **Framework-agnostic:** Any agent can call the API
- **Clawdbot hook:** Future integration after PR merges

## Goals
1. Pre-emptive compaction (trigger at 50%, not 90%)
2. Adaptive relevance-based dropping
3. Cold storage for recoverable content
4. Full observability for tuning

---

## Config Knobs (`~/.clawdbot/context-config.yaml`)

```yaml
# Context Limits
COMPACTION_TRIGGER_PCT: 0.50      # Trigger at 50% of context limit
MAX_CONTEXT_PCT: 0.85             # Hard limit at 85%

# Relevance Scoring
RELEVANCE_CUTOFF: 0.3             # Below this, DROP
RECENCY_BONUS: 0.2                # Boost per hour of recency
IMPORTANCE_BOOT: 0.1              # Boost for "important" marked content
MIN_CHUNKS_TO_KEEP: 5             # Always keep at least this many

# memfas Integration
MEMFAS_MIN_SCORE: 0.05            # memfas similarity threshold
MEMFAS_WEIGHT: 0.6                # How much memfas counts in relevance
CURATION_TRIGGER_PCT: 0.40        # Run memfas curation at 40%

# Cold Storage
COLD_STORAGE_ENABLED: true
COLD_STORAGE_PATH: ~/.clawdbot/cold-storage/
RECOVERABLE_FOR_DAYS: 30          # Keep dropped content searchable

# Summarization
SUMMARIZE_MEDIUM_CHUNKS: true
SUMMARY_TARGET_TOKENS: 500        # Target size for summaries
SUMMARY_MODEL: minimax/MiniMax-M2.1

# Logging
LOG_PATH: ~/.clawdbot/logs/context/
LOG_COMPACTION: true
LOG_CURATION: true
LOG_DROPS: true
LOG_RECOVERY: true
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Context Manager                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Token Tracker │ → │ Relevanc     │ → │ Adaptive     │      │
│  │ (realtime)    │    │ Scorer       │    │ Dropper      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │            │
│         ↓                    ↓                    ↓            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ memfas       │    │ Relevance    │    │ Cold         │      │
│  │ Curation     │    │ Calculator   │    │ Storage      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Summary Generator (MiniMax)                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Logger (all activities to JSONL)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Relevance Scoring Formula

```
relevance_score(chunk, current_prompt) = 
    (memfas_similarity × MEMFAS_WEIGHT) +
    (recency_hours × RECENCY_BONUS) +
    (importance_flag × IMPORTANCE_BONUS)
```

**Scoring pipeline:**
1. Get memfas similarity score for chunk vs prompt
2. Calculate recency bonus (hours since chunk added)
3. Check if chunk has "important" flag
4. Compute weighted sum
5. Apply thresholds

---

## Adaptive Drop Logic

```
For each context chunk:
    score = relevance_score(chunk, current_prompt)
    
    if score >= THRESHOLD_KEEP:
        → KEEP
    elif score <= RELEVANCE_CUTOFF:
        → DROP to cold storage (log it)
    else:
        → SUMMARIZE if enabled
        → Or keep condensed version
```

**Edge cases:**
- Always keep last N chunks (MIN_CHUNKS_TO_KEEP)
- Always keep "important" flagged chunks
- Recent chunks (within 1 hour) get recency boost

---

## Cold Storage Format

```
~/.clawdbot/cold-storage/{session_id}/{chunk_id}.jsonl
```

```json
{
  "chunk_id": "abc123",
  "session_id": "xyz789",
  "timestamp": "2026-02-02T22:55:00Z",
  "content": "...",
  "tokens": 1500,
  "drop_reason": "low_relevance",
  "relevance_score": 0.15,
  "prompt_at_drop": "journal search",
  "recoverable_until": "2026-03-02T22:55:00Z"
}
```

**Recovery:**
- memfas can still index cold storage
- If future prompt is relevant → auto-recover
- Recovery logged for tuning

---

## Logging Schema

### `compaction.jsonl`
```json
{
  "timestamp": "2026-02-02T22:55:00Z",
  "event": "compaction_triggered",
  "trigger": "token_threshold",
  "tokens_before": 80000,
  "tokens_after": 40000,
  "chunks_dropped": 15,
  "chunks_summarized": 8,
  "config_used": "COMPACTION_TRIGGER_PCT=0.50"
}
```

### `curation.jsonl`
```json
{
  "timestamp": "2026-02-02T22:55:00Z",
  "event": "memfas_curation",
  "query": "journal search",
  "hits_found": 5,
  "total_hits_before_curation": 12,
  "token_savings": 25000,
  "drops_triggered": 7
}
```

### `drops.jsonl`
```json
{
  "timestamp": "2026-02-02T22:55:00Z",
  "event": "chunk_dropped",
  "chunk_id": "abc123",
  "session_id": "xyz789",
  "reason": "low_relevance",
  "relevance_score": 0.15,
  "threshold_used": 0.30,
  "cold_storage_path": "~/.clawdbot/cold-storage/..."
}
```

### `context-health.jsonl`
```json
{
  "timestamp": "2026-02-02T22:55:00Z",
  "current_tokens": 45000,
  "limit": 100000,
  "pct_used": 0.45,
  "next_compaction_estimate": 50000,
  "memcached_chunks": 23,
  "cold_storage_chunks": 156
}
```

---

## Implementation Phases

### Phase 1: Config + Token Tracking
- Create `context-config.yaml`
- Real-time token counter hook
- Test logging infrastructure

### Phase 2: Relevance Scoring + memfas Integration
- Relevance scoring function
- memfas curation trigger
- Test on development session

### Phase 3: Adaptive Drop + Cold Storage
- Drop logic implementation
- Cold storage write/read
- Recovery mechanism

### Phase 4: Summary Generation
- MiniMax integration for summarization
- Condensed chunk format
- Integration with drop logic

### Phase 5: Tuning
- Analyze logs
- Adjust thresholds
- Document optimal configurations

---

## CLI Commands for Debugging

```bash
# View context health
context-manager status

# View recent compactions
tail -20 ~/.clawdbot/logs/context/compaction.jsonl

# View drops for a session
context-manager drops --session xyz789

# Force compaction
context-manager compact --now

# View cold storage contents
context-manager cold-storage list

# Recover a chunk from cold storage
context-manager recover --chunk-id abc123

# Reset all (dangerous)
context-manager reset --force
```

---

## Tuning Workflow

1. Run normally with logging
2. After day/week, analyze logs:
   ```
   # What got dropped?
   cat drops.jsonl | jq '.reason' | sort | uniq -c
   
   # Relevance scores distribution?
   cat drops.jsonl | jq '.relevance_score' | histogram
   
   # Token savings from curation?
   cat curation.jsonl | jq '.token_savings' | stats
   ```
3. Adjust `RELEVANCE_CUTOFF`, `MEMFAS_WEIGHT`, etc.
4. Re-run and compare

---

## Files to Create

```
~/.clawdbot/
├── context-config.yaml           # Config file
├── logs/context/
│   ├── compaction.jsonl
│   ├── curation.jsonl
│   ├── drops.jsonl
│   └── context-health.jsonl
└── cold-storage/                 # Dropped content
    └── {session_id}/
        └── {chunk_id}.jsonl

~/clawd/scripts/
├── context-manager.py            # Main CLI
├── relevance-scorer.py           # Scoring logic
├── cold-storage-handler.py       # Drop/recover
└── context-logger.py             # Logging utilities
```

---

## Integration with Clawdbot

**Hooks needed:**
1. `message-received` → update token count
2. `pre-response` → check if compaction needed
3. `post-response` → run curation if threshold hit
4. `session-end` → archive to cold storage

**Or via Gateway cron:**
```yaml
cron:
  - id: context-health-check
    interval: 60000  # Every minute
    command: context-manager health --notify-if-pct > 0.50
```

---

## Next Steps

1. Review this plan
2. Adjust config knobs if needed
3. I create Phase 1 (config + token tracking)
4. Test on development
5. Proceed to Phase 2

---

## ✅ IMPLEMENTATION COMPLETE: Phase 1

The context module is now implemented in `agent-memfas`:

```bash
pip install agent-memfas[context]
```

**Files created:**
```
agent-memfas/src/agent_memfas/context/
├── __init__.py              # Exports
├── config.py                # Configuration
├── context_manager.py       # Main orchestrator
├── relevance.py             # Scoring logic
├── cold_storage.py          # Drop/recover storage
└── logger.py                # JSONL logging
```

**Quick test:**
```python
from agent_memfas.context import ContextManager

ctx = ContextManager(
    memory_path="./memfas.json",
    get_context=lambda: current_context,
    set_context=lambda c: replace_context(c),
    get_token_count=lambda: count_tokens(),
    get_messages=lambda n: get_last_n(n)
)

status = ctx.before_response(max_tokens=100000)
if status.needs_compaction:
    ctx.compact()

ctx.session_end()
```

**Status:** Phase 1 (config + core structure) complete. Phase 2+ (memfas integration, MiniMax summarization) pending.
