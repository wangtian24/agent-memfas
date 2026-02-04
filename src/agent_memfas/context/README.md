# context/ — Context Management (v4)

Pre-emptive, relevance-based context compaction with cold storage and optional summarization. Sits on top of memfas core and plugs into any agent loop via callbacks.

---

## What it does

When an agent's context window fills up, most systems wait until it's 90%+ full, then do a blind summarization. This module does it differently:

- **Triggers early** (50% by default) — before quality degrades
- **Scores every chunk** — uses memfas embeddings (or keyword fallback) + recency decay + importance flags
- **Three-way classify**: KEEP / SUMMARIZE / DROP
- **Drops go to cold storage** — searchable, recoverable for 30 days
- **Summarization is pluggable** — MiniMax by default, NoOp if no API key, easy to add others
- **Full JSONL logging** — compactions, drops, recoveries, health checks

---

## Module layout

```
context/
├── __init__.py              # Public exports
├── config.py                # ContextConfig dataclass + YAML load/save
├── context_manager.py       # Main orchestrator (the entry point)
├── relevance.py             # RelevanceScorer — scores chunks for keep/drop/summarize
├── cold_storage.py          # ColdStorage — drop/recover/search/archive
├── logger.py                # ContextLogger — JSONL event logging
├── memfas_integration.py    # MemfasIntegration — embedding similarity via memfas
├── summarizer_base.py       # Summarizer ABC
├── summarizer.py            # Factory (create_summarizer) + NoOpSummarizer
└── minimax_summarizer.py    # MiniMaxSummarizer — HTTP calls to MiniMax API
```

---

## Quick start

```python
from agent_memfas.context import ContextManager

ctx = ContextManager(
    config_path="./memfas.yaml",          # optional, has sane defaults
    memfas_memory_path="./memfas.yaml",   # enables embedding-based scoring
    # Wire these to your agent's actual context:
    get_context=lambda: current_messages,
    set_context=lambda msgs: replace_messages(msgs),
    get_token_count=lambda: count_tokens(current_messages),
    get_messages=lambda n: current_messages[-n:],
)

# --- in your agent loop ---

# 1. New message arrived
ctx.on_message(user_message, current_messages)
ctx.set_current_prompt(user_message)

# 2. Before generating a response — checks health, auto-compacts if needed
status = ctx.before_response(max_tokens=100000)
# status.needs_compaction, status.pct_used, etc.

# 3. After response
ctx.after_response()

# 4. Session ending — archives to cold storage
ctx.session_end()

# 5. Need something back from cold storage?
recovered = ctx.recover("what was the database migration plan")
```

---

## Config reference

All values have defaults. Override via YAML (`context:` key) or constructor:

```yaml
context:
  # When to trigger compaction
  compaction_trigger_pct: 0.50    # 50% of max_tokens
  max_context_pct: 0.85           # hard ceiling
  auto_compact: true

  # Relevance scoring
  relevance_cutoff: 0.3           # score <= this → DROP
  relevance_keep_threshold: 0.7   # score >= this → KEEP (else SUMMARIZE)
  memfas_weight: 0.6              # weight of embedding similarity component
  recency_bonus: 0.2              # max bonus for a just-added chunk (decays exponentially, half-life ~4h)
  importance_bonus: 0.1           # flat bonus for chunks flagged important
  min_chunks_to_keep: 5           # never drop below this many chunks total

  # memfas integration
  memfas_min_score: 0.05
  curation_trigger_pct: 0.40

  # Cold storage
  cold_storage_enabled: true
  cold_storage_path: "./cold-storage/"
  recoverable_days: 30

  # Summarization
  summarize_medium_chunks: true
  summary_target_tokens: 500
  summary_model: "minimax/MiniMax-M2.1"   # set MINIMAX_API_KEY env var

  # Logging (all JSONL)
  log_path: "./logs/context/"
  log_compaction: true
  log_drops: true
  log_health: true
  log_recovery: true
```

---

## Relevance scoring

```
score = (embedding_similarity × memfas_weight)
      + (exp(-hours_since_added / 6) × recency_bonus)
      + (importance_flag × importance_bonus)
```

- **Embedding similarity**: cosine similarity between chunk and current prompt embeddings (via memfas). Falls back to keyword overlap if embeddings unavailable.
- **Recency decay**: exponential with a ~4-hour half-life. A chunk added 6h ago retains 37% of its recency bonus; at 24h it's ~2%.
- **Importance**: binary flag, adds a flat bonus. Useful for system-prompt-style chunks that should never drop.

Classification:
| Score | Action |
|---|---|
| ≥ 0.7 | KEEP as-is |
| 0.3 – 0.7 | SUMMARIZE (condense via MiniMax or keep if no summarizer) |
| ≤ 0.3 | DROP to cold storage |

`min_chunks_to_keep` is enforced after classification: if the KEEP + SUMMARIZE count would fall below it, the highest-scoring DROPs get promoted to SUMMARIZE.

---

## Cold storage

Dropped chunks land in `cold-storage/{session_id}/{chunk_id}.jsonl`. Each record:

```json
{
  "chunk_id": "chunk_3_1706900000",
  "session_id": "session_1706899000",
  "content": "...",
  "relevance_score": 0.18,
  "prompt_at_drop": "tell me about the API design",
  "timestamp": "2026-02-04T10:30:00",
  "recoverable_until": "2026-03-06T10:30:00"
}
```

**Recovery search** uses Jaccard overlap on meaningful keywords (stopwords stripped at both index-write and query time), ranked best-match-first. Call `ctx.recover("query")` to pull content back.

**Expiry**: `cleanup_expired()` removes chunks past `recoverable_until`.

---

## Summarization

Pluggable via `create_summarizer(backend=...)`:

| Backend | Class | Notes |
|---|---|---|
| `"minimax"` | `MiniMaxSummarizer` | Calls MiniMax API. Needs `MINIMAX_API_KEY` env var. |
| `"none"` | `NoOpSummarizer` | Returns chunks unchanged — safe default when no API key. |

Adding a new backend: subclass `Summarizer` (ABC in `summarizer_base.py`), implement `summarize()`, `summarize_batch()`, `estimate_tokens()`, register in `summarizer.py`'s factory.

---

## Logging

All activity goes to JSONL files under `log_path/`:

| File | What |
|---|---|
| `compaction.jsonl` | Each compaction: tokens before/after, chunks dropped/summarized |
| `drops.jsonl` | Each individual drop: chunk_id, score, reason, cold storage path |
| `health.jsonl` | Per-message and per-check health snapshots |
| `recovery.jsonl` | Cold storage recoveries |
| `curation.jsonl` | memfas curation runs (when integrated) |

Quick triage:
```bash
# What's been dropped and why?
cat logs/context/drops.jsonl | python3 -m json.tool --compact

# Token savings from compactions in last 24h
# (use logger.get_compaction_stats(hours=24))
```

---

## Dependencies

```toml
[project.optional-dependencies]
context = [
    "pyyaml>=6.0",      # config load/save
    "httpx>=0.25.0",    # MiniMax API calls
]
```

numpy is needed only if embedding-based scoring is active (already pulled in by `agent-memfas[embeddings]`).

---

## What's implemented vs pending

| Feature | Status |
|---|---|
| Config + YAML load | ✅ |
| Token tracking + health checks | ✅ |
| Relevance scoring (embeddings + keyword fallback) | ✅ |
| Three-way classify (keep/summarize/drop) | ✅ |
| min_chunks_to_keep enforcement | ✅ |
| Cold storage write + keyword recovery | ✅ |
| Expiry cleanup | ✅ |
| JSONL logging (all events) | ✅ |
| MiniMax summarizer | ✅ (needs API key) |
| Pluggable summarizer interface | ✅ |
| Auto-recovery on relevance shift | ⬜ planned |
| memfas curation trigger integration | ⬜ planned |
| CLI (`context-manager status/compact/recover`) | ⬜ planned |
| Unit tests | ⬜ needed |
