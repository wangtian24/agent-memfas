# v3 Design: Dynamic Context Engineering

> **Goal:** Proactively curate memory context each turn, not reactively recover after compaction.

---

## Problem Statement

### Current State (v2)
```
User message → Clawdbot → Claude (with full history)
                              ↓
                         Context grows
                              ↓
                         Compaction (reactive)
                              ↓
                         Context lost
                              ↓
                         Agent confused
                              ↓
                         User: "memfas recall" (manual recovery)
```

### Problems
1. **Context rot** — More tokens = worse recall (Anthropic research)
2. **Attention waste** — 80% of context may be irrelevant to current query
3. **Reactive** — Only acts after damage (compaction) happens
4. **Manual** — Agent/user must remember to call memfas

### Desired State (v3)
```
User message → memfas curates → Clawdbot → Claude (with focused context)
                   ↓
              Score memories by relevance
                   ↓
              Fill token budget with top-K
                   ↓
              Inject curated context
                   ↓
              Claude sees SMALL, HIGH-SIGNAL context
```

---

## Design Principles

1. **Smallest viable context** — Only include what's relevant to current turn
2. **Automatic** — No manual `recall` needed; happens every turn
3. **Budget-constrained** — Fixed token allocation, not unbounded growth
4. **Zero new services** — Runs locally, reuses existing embeddings
5. **Clawdbot-agnostic** — Works via MCP, can plug into any agent

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLAWDBOT                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Message                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐    MCP call: get_context(query)              │
│   │   Gateway   │ ──────────────────────────────────┐          │
│   └─────────────┘                                   │          │
│        │                                            ▼          │
│        │                                   ┌───────────────┐   │
│        │                                   │  MEMFAS MCP   │   │
│        │                                   │    SERVER     │   │
│        │                                   ├───────────────┤   │
│        │                                   │ 1. Embed query│   │
│        │         curated context           │ 2. Score all  │   │
│        │ ◄─────────────────────────────────│ 3. Budget fit │   │
│        │                                   │ 4. Return top │   │
│        ▼                                   └───────────────┘   │
│   ┌─────────────┐                                              │
│   │   Claude    │  ← Sees focused context, not full history   │
│   └─────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Topic Detector

Detects what the current conversation is about. Used to boost relevant memories.

```python
class TopicDetector:
    """
    Lightweight topic detection using embeddings.
    No new model needed — reuses existing embedder.
    """
    
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.topic_history: list[Topic] = []
    
    def detect(self, message: str, recent_history: list[str]) -> Topic:
        """
        Detect current topic from message + recent context.
        
        Returns:
            Topic with:
            - name: str (e.g., "memfas project")
            - embedding: list[float]
            - confidence: float
        """
        # Combine message with recent context for better signal
        context = "\n".join(recent_history[-3:] + [message])
        embedding = self.embedder.embed(context)
        
        # Check if topic shifted from previous
        if self.topic_history:
            prev = self.topic_history[-1]
            similarity = cosine_similarity(embedding, prev.embedding)
            if similarity < 0.7:
                # Topic shift detected
                pass
        
        topic = Topic(
            name=self._extract_topic_name(message),
            embedding=embedding,
            confidence=1.0
        )
        self.topic_history.append(topic)
        return topic
    
    def _extract_topic_name(self, message: str) -> str:
        """Extract key entities/topics from message."""
        # Simple: use most distinctive noun phrases
        # Advanced: small classifier or LLM call
        pass
```

### 2. Relevance Scorer

Scores each memory chunk by relevance to current query.

```python
class RelevanceScorer:
    """
    Multi-factor relevance scoring.
    """
    
    def __init__(
        self,
        embedder: Embedder,
        semantic_weight: float = 0.6,
        recency_weight: float = 0.2,
        access_weight: float = 0.1,
        topic_weight: float = 0.1
    ):
        self.embedder = embedder
        self.weights = {
            "semantic": semantic_weight,
            "recency": recency_weight,
            "access": access_weight,
            "topic": topic_weight
        }
    
    def score(
        self,
        query: str,
        memories: list[Memory],
        current_topic: Topic = None
    ) -> list[ScoredMemory]:
        """
        Score all memories by relevance to query.
        
        Factors:
        1. Semantic similarity (embedding distance)
        2. Recency (when was it created/accessed)
        3. Access frequency (hot memories)
        4. Topic coherence (matches current topic)
        """
        query_embedding = self.embedder.embed(query)
        scored = []
        
        for mem in memories:
            # Semantic similarity
            semantic = cosine_similarity(query_embedding, mem.embedding)
            
            # Recency decay (half-life of 7 days)
            days_old = (now() - mem.last_accessed).days
            recency = 0.5 ** (days_old / 7)
            
            # Access frequency (log scale)
            access = math.log1p(mem.access_count) / 10
            
            # Topic coherence
            topic_score = 0.5
            if current_topic:
                topic_score = cosine_similarity(
                    current_topic.embedding, 
                    mem.embedding
                )
            
            # Weighted combination
            final_score = (
                self.weights["semantic"] * semantic +
                self.weights["recency"] * recency +
                self.weights["access"] * access +
                self.weights["topic"] * topic_score
            )
            
            scored.append(ScoredMemory(
                memory=mem,
                score=final_score,
                breakdown={
                    "semantic": semantic,
                    "recency": recency,
                    "access": access,
                    "topic": topic_score
                }
            ))
        
        return sorted(scored, key=lambda x: x.score, reverse=True)
```

### 3. Token Budget Manager

Fills a fixed token budget with highest-value memories.

```python
class TokenBudget:
    """
    Manages token allocation for memory context.
    """
    
    def __init__(
        self,
        total_budget: int = 8000,  # Default 8K tokens for memory
        min_chunk_tokens: int = 50,
        max_chunk_tokens: int = 2000
    ):
        self.total_budget = total_budget
        self.min_chunk = min_chunk_tokens
        self.max_chunk = max_chunk_tokens
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))
    
    def allocate(
        self,
        scored_memories: list[ScoredMemory],
        reserved: int = 0  # For triggers, always-load, etc.
    ) -> CuratedContext:
        """
        Fill budget with highest-scoring memories.
        
        Returns:
            CuratedContext with:
            - memories: list[Memory]
            - total_tokens: int
            - dropped: list[Memory] (what didn't fit)
        """
        available = self.total_budget - reserved
        selected = []
        total_tokens = 0
        dropped = []
        
        for sm in scored_memories:
            tokens = self.count_tokens(sm.memory.text)
            
            if tokens > self.max_chunk:
                # Chunk is too big — truncate or skip
                continue
            
            if total_tokens + tokens <= available:
                selected.append(sm.memory)
                total_tokens += tokens
            else:
                dropped.append(sm.memory)
        
        return CuratedContext(
            memories=selected,
            total_tokens=total_tokens,
            budget=self.total_budget,
            utilization=total_tokens / self.total_budget,
            dropped=dropped
        )
```

### 4. Session State

Tracks conversation state across turns.

```python
class SessionState:
    """
    Maintains state across a conversation session.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_topic: Topic = None
        self.topic_history: list[Topic] = []
        self.active_entities: set[str] = set()  # People, projects mentioned
        self.accessed_memories: dict[str, int] = {}  # memory_id -> access count
        self.turn_count: int = 0
    
    def on_turn(self, query: str, topic: Topic, used_memories: list[str]):
        """Update state after each turn."""
        self.turn_count += 1
        self.current_topic = topic
        self.topic_history.append(topic)
        
        # Track memory access patterns
        for mem_id in used_memories:
            self.accessed_memories[mem_id] = \
                self.accessed_memories.get(mem_id, 0) + 1
        
        # Extract entities (simple version)
        # Could use NER for better extraction
        self.active_entities.update(self._extract_entities(query))
    
    def get_boost_ids(self) -> set[str]:
        """
        Get memory IDs that should be boosted.
        Based on: recently accessed, active entities
        """
        # Boost recently accessed memories
        hot = {
            mid for mid, count in self.accessed_memories.items()
            if count >= 2
        }
        return hot
    
    def detect_topic_shift(self) -> bool:
        """Detect if topic changed significantly."""
        if len(self.topic_history) < 2:
            return False
        
        prev = self.topic_history[-2]
        curr = self.topic_history[-1]
        similarity = cosine_similarity(prev.embedding, curr.embedding)
        return similarity < 0.6
```

### 5. MCP Server

Exposes memfas v3 as an MCP (Model Context Protocol) server.

```python
# agent_memfas/mcp_server.py

from mcp import Server, Resource, Tool

class MemfasMCPServer:
    """
    MCP server for dynamic context curation.
    
    Clawdbot connects to this and calls get_context() each turn.
    """
    
    def __init__(self, config_path: str):
        self.memory = Memory(config_path)
        self.topic_detector = TopicDetector(self.memory._embedder)
        self.scorer = RelevanceScorer(self.memory._embedder)
        self.budget = TokenBudget(total_budget=8000)
        self.sessions: dict[str, SessionState] = {}
    
    # ═══════════════════════════════════════════════════════════
    # MCP Tools (called by Clawdbot)
    # ═══════════════════════════════════════════════════════════
    
    @tool("get_context")
    def get_context(
        self,
        query: str,
        session_id: str = "default",
        token_budget: int = 8000,
        include_triggers: bool = True
    ) -> ContextResponse:
        """
        Get curated context for the current query.
        
        Called by Clawdbot BEFORE each Claude inference.
        Returns formatted context to inject into prompt.
        """
        # Get or create session
        session = self.sessions.setdefault(
            session_id, 
            SessionState(session_id)
        )
        
        # 1. Detect topic
        topic = self.topic_detector.detect(query, [])
        
        # 2. Get all memory chunks
        memories = self._get_all_memories()
        
        # 3. Score by relevance
        scored = self.scorer.score(query, memories, topic)
        
        # 4. Boost hot memories from session
        boost_ids = session.get_boost_ids()
        for sm in scored:
            if sm.memory.id in boost_ids:
                sm.score *= 1.2  # 20% boost
        
        # 5. Re-sort after boosting
        scored.sort(key=lambda x: x.score, reverse=True)
        
        # 6. Allocate budget
        # Reserve space for triggers
        trigger_reserve = 500 if include_triggers else 0
        curated = self.budget.allocate(scored, reserved=trigger_reserve)
        
        # 7. Get triggers (Type 1 fast path)
        triggers = []
        if include_triggers:
            triggers = self.memory._check_triggers(query)
        
        # 8. Format response
        context = self._format_context(triggers, curated.memories)
        
        # 9. Update session state
        session.on_turn(
            query, 
            topic, 
            [m.id for m in curated.memories]
        )
        
        return ContextResponse(
            context=context,
            tokens_used=curated.total_tokens + trigger_reserve,
            budget=token_budget,
            memories_included=len(curated.memories),
            memories_dropped=len(curated.dropped),
            topic=topic.name,
            topic_shifted=session.detect_topic_shift()
        )
    
    @tool("report_usage")
    def report_usage(
        self,
        session_id: str,
        memory_ids: list[str],
        was_helpful: bool = True
    ):
        """
        Report which memories were actually useful.
        Used to improve scoring over time.
        """
        session = self.sessions.get(session_id)
        if session and was_helpful:
            for mid in memory_ids:
                session.accessed_memories[mid] = \
                    session.accessed_memories.get(mid, 0) + 1
    
    @tool("set_budget")
    def set_budget(self, token_budget: int):
        """Adjust token budget for memory context."""
        self.budget.total_budget = token_budget
    
    # ═══════════════════════════════════════════════════════════
    # MCP Resources (passive data)
    # ═══════════════════════════════════════════════════════════
    
    @resource("stats")
    def get_stats(self) -> dict:
        """Get memory system statistics."""
        return {
            "total_memories": self.memory.stats()["memories"],
            "total_triggers": self.memory.stats()["triggers"],
            "active_sessions": len(self.sessions),
            "backend": type(self.memory.backend).__name__
        }
    
    def _format_context(
        self,
        triggers: list,
        memories: list[Memory]
    ) -> str:
        """Format curated context for injection."""
        parts = []
        
        if triggers:
            parts.append("<triggered_context>")
            for t in triggers:
                parts.append(f"[{t.trigger}] {t.hint}")
            parts.append("</triggered_context>")
        
        if memories:
            parts.append("<relevant_memory>")
            for m in memories:
                parts.append(f"[{m.source}]\n{m.text[:500]}")
            parts.append("</relevant_memory>")
        
        return "\n".join(parts)
```

---

## Clawdbot Integration

### Option A: MCP Client (Recommended)

Clawdbot adds memfas as an MCP server:

```yaml
# clawdbot config
mcp_servers:
  - name: memfas
    command: memfas-mcp
    args: ["--config", "~/clawd/memfas.yaml"]
```

Then in gateway, before each Claude call:

```python
# In Clawdbot gateway
async def prepare_context(self, user_message: str, session_id: str):
    # Call memfas MCP to get curated context
    context_response = await self.mcp.call(
        "memfas",
        "get_context",
        query=user_message,
        session_id=session_id,
        token_budget=8000
    )
    
    # Inject into system prompt or message
    return f"""
{context_response.context}

<user_message>
{user_message}
</user_message>
"""
```

### Option B: Direct Integration (Simpler)

If MCP is too complex, direct Python integration:

```python
# In Clawdbot gateway
from agent_memfas.v3 import ContextCurator

curator = ContextCurator("~/clawd/memfas.yaml")

async def handle_message(self, user_msg: str, session_id: str):
    # Get curated context
    curated = curator.get_context(user_msg, session_id)
    
    # Build prompt with focused context
    prompt = f"""
<memory_context tokens="{curated.tokens_used}">
{curated.context}
</memory_context>

{user_msg}
"""
    
    return await self.claude.complete(prompt)
```

---

## Costs & Requirements

### No New Models
- Reuses existing embedder (FastEmbed or Ollama)
- Topic detection via embeddings, not separate model

### No New Services
- Runs locally as CLI or MCP server
- SQLite for storage (same as v2)

### New Dependencies
```toml
[project.optional-dependencies]
v3 = [
    "tiktoken>=0.5.0",  # Token counting
    "mcp>=0.1.0",       # MCP server (optional)
]
```

### Compute Cost Per Turn
| Operation | Cost | Time |
|-----------|------|------|
| Embed query | 1 embedding | ~10ms |
| Score 100 memories | 100 cosine sims | ~1ms |
| Token counting | N texts | ~5ms |
| **Total** | **Minimal** | **<20ms** |

---

## Migration Path

### v2 → v3

1. **v2 users**: No changes needed, v2 API preserved
2. **v3 opt-in**: Start MCP server or use ContextCurator
3. **Gradual**: Can use v2 recall + v3 curation together

```python
# v2 still works
mem = Memory("./memfas.yaml")
context = mem.recall("what project?")

# v3 adds automatic curation
curator = ContextCurator("./memfas.yaml")
curated = curator.get_context("what project?", session_id="abc")
```

---

## Open Questions

1. **How much budget?** 
   - Default 8K tokens for memory seems reasonable
   - Should be configurable per-session

2. **Trigger priority?**
   - Type 1 triggers should always be included (fast path)
   - Reserve ~500 tokens for triggers

3. **Topic detection quality?**
   - Simple embedding similarity may be enough
   - Could add small classifier if needed

4. **Feedback loop?**
   - Track which memories were actually useful
   - Boost frequently-accessed memories

5. **Multi-agent?**
   - Each agent gets own session
   - Shared memory, separate topic tracking

---

## Implementation Plan

### Phase 1: Core Components (1 week)
- [ ] TopicDetector with embedding similarity
- [ ] RelevanceScorer with multi-factor scoring
- [ ] TokenBudget with tiktoken
- [ ] SessionState for topic tracking

### Phase 2: MCP Server (1 week)
- [ ] MemfasMCPServer with get_context tool
- [ ] CLI: `memfas-mcp` command
- [ ] Integration tests

### Phase 3: Clawdbot Integration (1 week)
- [ ] Add MCP client to Clawdbot gateway
- [ ] Hook into message handler
- [ ] End-to-end testing

### Phase 4: Polish (ongoing)
- [ ] Tune scoring weights
- [ ] Add feedback loop
- [ ] Performance optimization
- [ ] Documentation

---

## Observability & Telemetry

### Why Track?
- Prove value: "memfas saved X tokens this session"
- Debug: "why did it include this memory?"
- Tune: "semantic weight too high, recency too low"
- Report: Agent can summarize performance for human

### Telemetry Schema

```python
@dataclass
class TurnMetrics:
    """Metrics for a single turn."""
    timestamp: str
    session_id: str
    turn_number: int
    
    # Query info
    query: str
    query_tokens: int
    detected_topic: str
    topic_shifted: bool
    
    # What would have been used WITHOUT memfas
    baseline_context_tokens: int  # Full history size
    
    # What memfas provided
    curated_context_tokens: int
    memories_scored: int
    memories_included: int
    memories_dropped: int
    triggers_matched: int
    
    # Savings
    tokens_saved: int  # baseline - curated
    compression_ratio: float  # curated / baseline
    
    # Performance
    latency_ms: float
    
    # Top memories (for debugging)
    top_memories: list[dict]  # [{id, score, source, snippet}]
    
    # Scoring breakdown
    score_weights: dict  # {semantic: 0.6, recency: 0.2, ...}


@dataclass 
class SessionMetrics:
    """Aggregated metrics for a session."""
    session_id: str
    start_time: str
    end_time: str
    total_turns: int
    
    # Totals
    total_baseline_tokens: int
    total_curated_tokens: int
    total_tokens_saved: int
    avg_compression_ratio: float
    
    # Topic tracking
    topics_detected: list[str]
    topic_shifts: int
    
    # Memory usage
    unique_memories_used: int
    most_accessed_memories: list[dict]  # [{id, access_count, source}]
    
    # Performance
    avg_latency_ms: float
    p95_latency_ms: float
```

### Telemetry Logger

```python
class TelemetryLogger:
    """
    Logs structured telemetry for analysis.
    Writes JSONL for easy parsing by agents or scripts.
    """
    
    def __init__(self, log_path: str = "./memfas-telemetry.jsonl"):
        self.log_path = Path(log_path)
        self.session_metrics: dict[str, SessionMetrics] = {}
    
    def log_turn(self, metrics: TurnMetrics):
        """Log a single turn's metrics."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "type": "turn",
                **asdict(metrics)
            }) + "\n")
        
        # Update session aggregates
        self._update_session(metrics)
    
    def log_session_end(self, session_id: str):
        """Log session summary when session ends."""
        if session_id in self.session_metrics:
            metrics = self.session_metrics[session_id]
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "type": "session_summary",
                    **asdict(metrics)
                }) + "\n")
    
    def get_summary(
        self, 
        since: str = None,  # ISO timestamp
        session_id: str = None
    ) -> dict:
        """
        Get summary statistics for agent to report.
        
        Returns dict suitable for agent to read and summarize.
        """
        entries = self._read_entries(since, session_id)
        
        turns = [e for e in entries if e["type"] == "turn"]
        
        if not turns:
            return {"message": "No telemetry data found"}
        
        return {
            "period": {
                "start": turns[0]["timestamp"],
                "end": turns[-1]["timestamp"],
                "total_turns": len(turns)
            },
            "compression": {
                "total_baseline_tokens": sum(t["baseline_context_tokens"] for t in turns),
                "total_curated_tokens": sum(t["curated_context_tokens"] for t in turns),
                "total_tokens_saved": sum(t["tokens_saved"] for t in turns),
                "avg_compression_ratio": mean(t["compression_ratio"] for t in turns),
                "best_compression": min(t["compression_ratio"] for t in turns),
                "worst_compression": max(t["compression_ratio"] for t in turns)
            },
            "memory_usage": {
                "total_memories_scored": sum(t["memories_scored"] for t in turns),
                "total_memories_included": sum(t["memories_included"] for t in turns),
                "total_memories_dropped": sum(t["memories_dropped"] for t in turns),
                "avg_memories_per_turn": mean(t["memories_included"] for t in turns)
            },
            "topics": {
                "unique_topics": list(set(t["detected_topic"] for t in turns)),
                "topic_shifts": sum(1 for t in turns if t["topic_shifted"])
            },
            "performance": {
                "avg_latency_ms": mean(t["latency_ms"] for t in turns),
                "p95_latency_ms": percentile(t["latency_ms"] for t in turns, 95),
                "max_latency_ms": max(t["latency_ms"] for t in turns)
            }
        }
```

### CLI Commands

```bash
# View recent telemetry
memfas telemetry show --last 24h

# Get summary stats
memfas telemetry summary --session abc123

# Export for analysis
memfas telemetry export --format csv --output metrics.csv

# Live monitoring
memfas telemetry tail
```

### Example Log Entry (JSONL)

```json
{
  "type": "turn",
  "timestamp": "2025-02-02T15:30:00Z",
  "session_id": "main-session",
  "turn_number": 42,
  "query": "what's the status of the memfas project?",
  "query_tokens": 12,
  "detected_topic": "memfas",
  "topic_shifted": false,
  "baseline_context_tokens": 85000,
  "curated_context_tokens": 6500,
  "memories_scored": 85,
  "memories_included": 8,
  "memories_dropped": 77,
  "triggers_matched": 2,
  "tokens_saved": 78500,
  "compression_ratio": 0.076,
  "latency_ms": 18.5,
  "top_memories": [
    {"id": "abc123", "score": 0.92, "source": "memory/memfas-build.md", "snippet": "Building v2..."},
    {"id": "def456", "score": 0.87, "source": "MEMORY.md", "snippet": "memfas project..."}
  ],
  "score_weights": {"semantic": 0.6, "recency": 0.2, "access": 0.1, "topic": 0.1}
}
```

### Agent-Readable Summary

The agent can call `memfas telemetry summary` and get:

```
📊 Memfas Performance Summary (last 24h)
════════════════════════════════════════

Compression:
  Total baseline tokens:  1,250,000
  Total curated tokens:      95,000
  Tokens saved:           1,155,000 (92.4% reduction!)
  Avg compression ratio:     0.076 (13x smaller)

Memory Usage:
  Memories scored:           8,500 (100/turn avg)
  Memories included:           680 (8/turn avg)
  Hit rate:                    8.0%

Topics:
  Unique topics: memfas, family, running, yupp
  Topic shifts: 12

Performance:
  Avg latency: 18ms
  P95 latency: 45ms
```

### Integration with v3

```python
class ContextCurator:
    def __init__(self, config_path: str, telemetry_path: str = None):
        # ... existing init ...
        self.telemetry = TelemetryLogger(
            telemetry_path or "./memfas-telemetry.jsonl"
        )
    
    def get_context(
        self,
        query: str,
        session_id: str,
        baseline_tokens: int = None  # Pass current context size
    ) -> ContextResponse:
        start = time.perf_counter()
        
        # ... existing curation logic ...
        
        latency = (time.perf_counter() - start) * 1000
        
        # Log metrics
        self.telemetry.log_turn(TurnMetrics(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            turn_number=session.turn_count,
            query=query,
            query_tokens=self.budget.count_tokens(query),
            detected_topic=topic.name,
            topic_shifted=session.detect_topic_shift(),
            baseline_context_tokens=baseline_tokens or 0,
            curated_context_tokens=curated.total_tokens,
            memories_scored=len(scored),
            memories_included=len(curated.memories),
            memories_dropped=len(curated.dropped),
            triggers_matched=len(triggers),
            tokens_saved=(baseline_tokens or 0) - curated.total_tokens,
            compression_ratio=curated.total_tokens / max(baseline_tokens or 1, 1),
            latency_ms=latency,
            top_memories=[
                {
                    "id": m.id,
                    "score": sm.score,
                    "source": m.source,
                    "snippet": m.text[:100]
                }
                for sm, m in zip(scored[:5], curated.memories[:5])
            ],
            score_weights=self.scorer.weights
        ))
        
        return response
```

---

## Success Metrics

1. **Context size reduction**: 50%+ less tokens with same recall quality
2. **Recovery speed**: Agent recovers context in 0 turns (automatic) vs 1+ turns (manual)
3. **Relevance precision**: >80% of injected memories are actually used
4. **Latency**: <50ms overhead per turn

---

## References

- [Anthropic: Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Context Rot Research](https://research.trychroma.com/context-rot)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Karpathy on Context Engineering](https://x.com/karpathy/status/1937902205765607626)
