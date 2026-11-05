"""wolfmem command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import WolfmemConfig
from .embeddings import EmbeddingClient
from .journal import Journal
from .mcp_server import WolfmemMcpServer
from .observer import Observer
from .store import FactStore


def _build(config: WolfmemConfig | None = None):
    config = config or WolfmemConfig()
    config.ensure_dirs()
    journal = Journal(config.journal_dir)
    embeddings = EmbeddingClient(config)
    store = FactStore(config, embeddings)
    observer = Observer(config, journal, store)
    return config, journal, embeddings, store, observer


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wolfmem",
        description="Persistent memory for AI agents: journal + observer + vector fact store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the data directory and database")
    sub.add_parser("serve", help="run the MCP server on stdio (for IDEs and agents)")

    p_remember = sub.add_parser("remember", help="store one fact")
    p_remember.add_argument("text")
    p_remember.add_argument("--importance", type=int, default=3)

    p_recall = sub.add_parser("recall", help="search facts")
    p_recall.add_argument("query")
    p_recall.add_argument("--top", type=int, default=4)

    p_list = sub.add_parser("list", help="list recent facts")
    p_list.add_argument("--limit", type=int, default=20)

    p_forget = sub.add_parser("forget", help="delete by id / tag / keyword")
    p_forget.add_argument("target")

    p_journal = sub.add_parser("journal", help="append a turn to the daily journal")
    p_journal.add_argument("who")
    p_journal.add_argument("text")
    p_journal.add_argument("--session", default="default")

    p_observe = sub.add_parser("observe", help="run the observer")
    p_observe.add_argument("--once", action="store_true", help="single scan and exit")

    sub.add_parser("stats", help="memory statistics")

    sub.add_parser("journal-tail", help="show recent journal turns")

    args = parser.parse_args(argv)

    if args.command == "serve":
        WolfmemMcpServer().serve()
        return 0

    config, journal, embeddings, store, observer = _build()

    if args.command == "init":
        _print({"home": str(config.home), "embedding": embeddings.available})
        return 0
    if args.command == "remember":
        _print(asyncio.run(store.remember(args.text, args.importance)))
        return 0
    if args.command == "recall":
        _print(
            [
                {"id": f.id, "text": f.text, "score": f.score}
                for f in asyncio.run(store.recall(args.query, args.top))
            ]
        )
        return 0
    if args.command == "list":
        _print(
            [
                {"id": f.id, "text": f.text, "importance": f.importance}
                for f in store.list(args.limit)
            ]
        )
        return 0
    if args.command == "forget":
        _print(store.forget(args.target))
        return 0
    if args.command == "journal":
        rec = journal.append(args.who, args.text, session=args.session)
        _print({"ok": True, "record": rec})
        return 0
    if args.command == "journal-tail":
        _print(journal.tail())
        return 0
    if args.command == "observe":
        if args.once:
            _print(asyncio.run(observer.observe_once()))
            return 0
        asyncio.run(observer.run_forever())
        return 0
    if args.command == "stats":
        _print(store.stats())
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
