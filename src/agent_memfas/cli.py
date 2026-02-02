#!/usr/bin/env python3
"""
Command-line interface for agent-memfas.

Usage:
    memfas recall "How's the family?"
    memfas search "preference learning"
    memfas remember tahoe --hint "Family ski trips"
    memfas forget tahoe
    memfas triggers
    memfas index ./memory/
    memfas stats
    memfas init
"""

import argparse
import sys
from pathlib import Path

from .memory import Memory
from .config import Config


def cmd_recall(args):
    """Recall memories for given context."""
    mem = Memory(args.config)
    result = mem.recall(" ".join(args.context))
    if result:
        print(result)
    else:
        print("No memories found.")
    mem.close()


def cmd_search(args):
    """Search memories."""
    mem = Memory(args.config)
    results = mem.search(" ".join(args.query), limit=args.limit)
    if results:
        for r in results:
            print(f"[{r.source}] (score: {r.score:.2f})")
            print(f"  {r.text[:200]}...")
            print()
    else:
        print("No results found.")
    mem.close()


def cmd_remember(args):
    """Add a keyword trigger."""
    mem = Memory(args.config)
    mem.add_trigger(args.keyword, args.hint)
    print(f"✓ Added trigger: [{args.keyword}] → {args.hint}")
    mem.close()


def cmd_forget(args):
    """Remove a keyword trigger."""
    mem = Memory(args.config)
    mem.remove_trigger(args.keyword)
    print(f"✓ Removed trigger: [{args.keyword}]")
    mem.close()


def cmd_triggers(args):
    """List all triggers."""
    mem = Memory(args.config)
    triggers = mem.list_triggers()
    if triggers:
        print(f"{'Keyword':<20} {'Hint':<50}")
        print("-" * 70)
        for t in triggers:
            print(f"{t['keyword']:<20} {t['hint']:<50}")
    else:
        print("No triggers defined.")
    mem.close()


def cmd_index(args):
    """Index files or directories."""
    mem = Memory(args.config)
    
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_dir():
            # Index all markdown files in directory
            for f in path.glob("**/*.md"):
                try:
                    mem.index_file(str(f), "markdown")
                    print(f"✓ Indexed: {f}")
                except Exception as e:
                    print(f"✗ Failed: {f} ({e})")
        elif path.is_file():
            try:
                file_type = "markdown" if path.suffix == ".md" else "text"
                mem.index_file(str(path), file_type)
                print(f"✓ Indexed: {path}")
            except Exception as e:
                print(f"✗ Failed: {path} ({e})")
        else:
            print(f"✗ Not found: {path}")
    
    mem.close()


def cmd_stats(args):
    """Show memory statistics."""
    mem = Memory(args.config)
    stats = mem.stats()
    print(f"Database: {stats['db_path']}")
    print(f"Backend: {stats['backend']}")
    print(f"Memories: {stats['memories']}")
    print(f"Triggers: {stats['triggers']}")
    mem.close()


def cmd_suggest(args):
    """Suggest potential triggers based on indexed content."""
    mem = Memory(args.config)
    
    suggestions = mem.suggest_triggers(min_occurrences=args.min, limit=args.limit)
    
    if not suggestions:
        print("No suggestions found. Index some content first: memfas index <path>")
        mem.close()
        return
    
    print(f"💡 Suggested triggers (min {args.min} occurrences):\n")
    
    # Group by type
    entities = [s for s in suggestions if s["type"] == "entity"]
    phrases = [s for s in suggestions if s["type"] == "phrase"]
    terms = [s for s in suggestions if s["type"] == "term"]
    
    if entities:
        print("Entities (proper nouns):")
        for s in entities[:7]:
            print(f"  memfas remember {s['term']} --hint \"...\"  # {s['count']}x")
    
    if phrases:
        print("\nPhrases:")
        for s in phrases[:5]:
            print(f"  memfas remember \"{s['term']}\" --hint \"...\"  # {s['count']}x")
    
    if terms:
        print("\nFrequent terms:")
        for s in terms[:8]:
            print(f"  memfas remember {s['term']} --hint \"...\"  # {s['count']}x")
    
    mem.close()


def cmd_init(args):
    """Initialize memfas in current directory."""
    config_path = Path(args.output or "memfas.yaml")
    
    if config_path.exists() and not args.force:
        print(f"Config already exists: {config_path}")
        print("Use --force to overwrite.")
        return
    
    config = Config.default(".")
    config.save(str(config_path))
    print(f"✓ Created config: {config_path}")
    
    # Also create the database
    mem = Memory(config)
    mem.index_sources()
    stats = mem.stats()
    print(f"✓ Indexed {stats['memories']} memories from {len(config.sources)} sources")
    mem.close()


def cmd_clear(args):
    """Clear all indexed memories."""
    if not args.yes:
        confirm = input("This will delete all indexed memories. Continue? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    mem = Memory(args.config)
    mem.clear()
    print("✓ Cleared all memories.")
    mem.close()


def cmd_reindex(args):
    """Re-index all memories with a different backend."""
    from .config import Config
    
    # Load config
    if Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config.default(".")
    
    # Determine target backend
    target_backend = args.backend or config.search.backend
    
    # Create embedder if needed
    embedder = None
    if target_backend == "embedding":
        embedder_type = args.embedder or config.search.embedder_type
        embedder_model = args.model or config.search.embedder_model
        
        if not embedder_type:
            print("Embedding backend requires --embedder (fastembed or ollama)")
            sys.exit(1)
        
        if embedder_type == "fastembed":
            try:
                from .embedders.fastembed import FastEmbedEmbedder
                print(f"Loading FastEmbed embedder...")
                embedder = FastEmbedEmbedder(model=embedder_model) if embedder_model else FastEmbedEmbedder()
                print(f"✓ Loaded: {embedder.model_name} ({embedder.dimensions} dims)")
            except ImportError:
                print("FastEmbed not installed. Run: pip install fastembed")
                sys.exit(1)
        elif embedder_type == "ollama":
            try:
                from .embedders.ollama import OllamaEmbedder
                embedder = OllamaEmbedder(model=embedder_model) if embedder_model else OllamaEmbedder()
                print(f"✓ Using Ollama: {embedder.model} ({embedder.dimensions} dims)")
            except ImportError:
                print("requests not installed. Run: pip install requests")
                sys.exit(1)
        else:
            print(f"Unknown embedder type: {embedder_type}")
            sys.exit(1)
    
    # Open memory with current backend
    mem = Memory(config)
    current_backend = type(mem.backend).__name__
    current_count = mem.stats()["memories"]
    
    print(f"Current: {current_backend} with {current_count} memories")
    print(f"Target: {target_backend}")
    
    if current_count == 0:
        print("No memories to reindex.")
        mem.close()
        return
    
    if not args.yes:
        confirm = input(f"Re-index {current_count} memories to {target_backend}? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            mem.close()
            return
    
    print("Re-indexing...")
    mem.reindex(new_backend=target_backend, embedder=embedder)
    
    new_count = mem.stats()["memories"]
    new_backend = type(mem.backend).__name__
    print(f"✓ Migrated to {new_backend}: {new_count} memories")
    
    # Optionally update config
    if args.save_config and Path(args.config).exists():
        config.search.backend = target_backend
        if embedder_type:
            config.search.embedder_type = embedder_type
        if embedder_model:
            config.search.embedder_model = embedder_model
        config.save(args.config)
        print(f"✓ Updated config: {args.config}")
    
    mem.close()


def cmd_curate(args):
    """[v3] Get curated context for a query."""
    try:
        from .v3 import ContextCurator
    except ImportError as e:
        print(f"v3 dependencies not available: {e}")
        print("Install with: pip install agent-memfas[v3]")
        sys.exit(1)
    
    import json as json_lib
    
    query = " ".join(args.query)
    
    curator = ContextCurator(
        args.config,
        token_budget=args.budget
    )
    
    result = curator.get_context(
        query=query,
        session_id=args.session,
        baseline_tokens=args.baseline
    )
    
    if args.json:
        print(json_lib.dumps({
            "context": result.context,
            "tokens_used": result.tokens_used,
            "budget": result.budget,
            "memories_included": result.memories_included,
            "memories_dropped": result.memories_dropped,
            "triggers_matched": result.triggers_matched,
            "topic": result.topic,
            "topic_shifted": result.topic_shifted,
            "baseline_tokens": result.baseline_tokens,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
            "latency_ms": result.latency_ms
        }, indent=2))
    else:
        print(f"📚 Curated Context ({result.tokens_used} tokens, {result.latency_ms:.1f}ms)")
        print("=" * 60)
        print(result.context)
        print("=" * 60)
        print(f"\nTopic: {result.topic}" + (" [SHIFTED]" if result.topic_shifted else ""))
        print(f"Memories: {result.memories_included} included, {result.memories_dropped} dropped")
        print(f"Triggers: {result.triggers_matched} matched")
        if args.baseline > 0:
            print(f"\nSavings: {result.tokens_saved:,} tokens ({(1-result.compression_ratio)*100:.1f}% reduction)")
    
    curator.close()


def cmd_telemetry_summary(args):
    """Show telemetry summary."""
    from .v3 import TelemetryLogger
    
    telem = TelemetryLogger()
    summary = telem.get_summary(
        session_id=getattr(args, 'session', None),
        last_n_turns=getattr(args, 'last', None)
    )
    
    print(telem.format_summary(summary))


def cmd_telemetry_show(args):
    """Show recent telemetry entries."""
    from .v3 import TelemetryLogger
    import json as json_lib
    
    telem = TelemetryLogger()
    entries = telem._read_entries(session_id=getattr(args, 'session', None))
    
    if not entries:
        print("No telemetry data found.")
        return
    
    # Show last N entries
    for entry in entries[-args.limit:]:
        if entry.get("type") == "turn":
            print(f"[{entry['timestamp'][:19]}] {entry['session_id']} turn #{entry['turn_number']}")
            print(f"  Query: {entry['query'][:60]}...")
            print(f"  Topic: {entry['detected_topic']}")
            print(f"  Tokens: {entry['curated_context_tokens']} (saved {entry['tokens_saved']})")
            print(f"  Memories: {entry['memories_included']} included")
            print(f"  Latency: {entry['latency_ms']:.1f}ms")
            print()


def cmd_telemetry_clear(args):
    """Clear telemetry log."""
    from .v3 import TelemetryLogger
    
    if not args.yes:
        confirm = input("Clear all telemetry data? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    telem = TelemetryLogger()
    telem.clear()
    print("✓ Telemetry cleared.")


def main():
    parser = argparse.ArgumentParser(
        description="Memory Fast and Slow for AI Agents",
        prog="memfas"
    )
    parser.add_argument(
        "-c", "--config",
        default="memfas.yaml",
        help="Path to config file (default: memfas.yaml)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # recall
    p_recall = subparsers.add_parser("recall", help="Recall memories for context")
    p_recall.add_argument("context", nargs="+", help="Context to recall for")
    p_recall.set_defaults(func=cmd_recall)
    
    # search
    p_search = subparsers.add_parser("search", help="Search memories")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-n", "--limit", type=int, default=5, help="Max results")
    p_search.set_defaults(func=cmd_search)
    
    # remember
    p_remember = subparsers.add_parser("remember", help="Add a keyword trigger")
    p_remember.add_argument("keyword", help="Keyword to trigger on")
    p_remember.add_argument("--hint", "-H", required=True, help="Description/hint")
    p_remember.set_defaults(func=cmd_remember)
    
    # forget
    p_forget = subparsers.add_parser("forget", help="Remove a keyword trigger")
    p_forget.add_argument("keyword", help="Keyword to remove")
    p_forget.set_defaults(func=cmd_forget)
    
    # triggers
    p_triggers = subparsers.add_parser("triggers", help="List all triggers")
    p_triggers.set_defaults(func=cmd_triggers)
    
    # index
    p_index = subparsers.add_parser("index", help="Index files or directories")
    p_index.add_argument("paths", nargs="+", help="Files or directories to index")
    p_index.set_defaults(func=cmd_index)
    
    # stats
    p_stats = subparsers.add_parser("stats", help="Show statistics")
    p_stats.set_defaults(func=cmd_stats)
    
    # init
    p_init = subparsers.add_parser("init", help="Initialize memfas")
    p_init.add_argument("-o", "--output", help="Config file path")
    p_init.add_argument("-f", "--force", action="store_true", help="Overwrite existing")
    p_init.set_defaults(func=cmd_init)
    
    # clear
    p_clear = subparsers.add_parser("clear", help="Clear all indexed memories")
    p_clear.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_clear.set_defaults(func=cmd_clear)
    
    # suggest
    p_suggest = subparsers.add_parser("suggest", help="Suggest potential triggers from indexed content")
    p_suggest.add_argument("-n", "--limit", type=int, default=15, help="Max suggestions")
    p_suggest.add_argument("-m", "--min", type=int, default=3, help="Min occurrences")
    p_suggest.set_defaults(func=cmd_suggest)
    
    # reindex
    p_reindex = subparsers.add_parser("reindex", help="Re-index memories with different backend")
    p_reindex.add_argument("-b", "--backend", choices=["fts5", "embedding"], help="Target backend")
    p_reindex.add_argument("-e", "--embedder", choices=["fastembed", "ollama"], help="Embedder type")
    p_reindex.add_argument("-m", "--model", help="Embedder model name")
    p_reindex.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_reindex.add_argument("--save-config", action="store_true", help="Update config file")
    p_reindex.set_defaults(func=cmd_reindex)
    
    # curate (v3)
    p_curate = subparsers.add_parser("curate", help="[v3] Get curated context for a query")
    p_curate.add_argument("query", nargs="+", help="Query to curate context for")
    p_curate.add_argument("-b", "--budget", type=int, default=8000, help="Token budget")
    p_curate.add_argument("--baseline", type=int, default=0, help="Baseline context tokens (for comparison)")
    p_curate.add_argument("-s", "--session", default="cli", help="Session ID")
    p_curate.add_argument("--json", action="store_true", help="Output as JSON")
    p_curate.set_defaults(func=cmd_curate)
    
    # telemetry
    p_telem = subparsers.add_parser("telemetry", help="[v3] View telemetry data")
    p_telem_sub = p_telem.add_subparsers(dest="telem_cmd")
    
    p_telem_summary = p_telem_sub.add_parser("summary", help="Show summary statistics")
    p_telem_summary.add_argument("-s", "--session", help="Filter by session ID")
    p_telem_summary.add_argument("-n", "--last", type=int, help="Last N turns")
    p_telem_summary.set_defaults(func=cmd_telemetry_summary)
    
    p_telem_show = p_telem_sub.add_parser("show", help="Show recent telemetry entries")
    p_telem_show.add_argument("-n", "--limit", type=int, default=10, help="Number of entries")
    p_telem_show.add_argument("-s", "--session", help="Filter by session ID")
    p_telem_show.set_defaults(func=cmd_telemetry_show)
    
    p_telem_clear = p_telem_sub.add_parser("clear", help="Clear telemetry log")
    p_telem_clear.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_telem_clear.set_defaults(func=cmd_telemetry_clear)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except FileNotFoundError as e:
        # Config not found, try to auto-init
        if "memfas.yaml" in str(e) and args.command not in ("init",):
            print("No memfas.yaml found. Run 'memfas init' first, or specify --config.")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
