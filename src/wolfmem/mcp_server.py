"""MCP server (Model Context Protocol) over stdio.

Implements the small, stable subset of the MCP spec that memory tools need —
``initialize``, ``tools/list``, ``tools/call``, ``ping`` — as newline-delimited
JSON-RPC 2.0 on stdin/stdout. Zero heavyweight SDK dependencies, so the server
runs anywhere Python 3.10+ runs (Claude Code, Cursor, Cline, Zed, your own
agent harness: all speak this).

Logging goes to stderr ONLY; stdout is the protocol channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from .config import WolfmemConfig
from .embeddings import EmbeddingClient
from .journal import Journal
from .store import FactStore
from .observer import Observer

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "wolfmem", "version": "0.1.0"}

TOOLS = [
    {
        "name": "remember",
        "description": (
            "Store one durable fact. Prefix it with a needle tag so future "
            "recall can filter: [决定] decisions, [坑] pitfalls, [隐私] privacy "
            "boundaries, [安全] security, [偏好] preferences, [事实] plain facts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The fact to remember, e.g. '[决定] use SQLite for persistence'."},
                "importance": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
            "required": ["text"],
        },
    },
    {
        "name": "recall",
        "description": "Search stored facts by semantic similarity (or keyword fallback). Returns the most relevant old facts for the current context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top": {"type": "integer", "minimum": 1, "maximum": 20, "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_memories",
        "description": "List recently stored facts, newest first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
            },
        },
    },
    {
        "name": "forget",
        "description": "Delete facts by numeric id, needle tag (e.g. '[坑]') or keyword substring.",
        "inputSchema": {
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
    },
    {
        "name": "journal_write",
        "description": "Append one conversation turn to the daily journal (the append-only ledger the observer reads).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "who": {"type": "string", "description": "Speaker label, e.g. 'user' or 'assistant'."},
                "text": {"type": "string"},
                "session": {"type": "string", "default": "default"},
            },
            "required": ["who", "text"],
        },
    },
    {
        "name": "observe_once",
        "description": "Run one observer scan immediately: read new journal turns and distill durable facts via the configured LLM.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "stats",
        "description": "Memory statistics: fact counts by needle tag and embedding availability.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class WolfmemMcpServer:
    def __init__(self, config: WolfmemConfig | None = None):
        self.config = config or WolfmemConfig()
        self.config.ensure_dirs()
        self.journal = Journal(self.config.journal_dir)
        self.embeddings = EmbeddingClient(self.config)
        self.store = FactStore(self.config, self.embeddings)
        self.observer = Observer(self.config, self.journal, self.store)

    # ---------------- dispatch ----------------

    def call_tool(self, name: str, args: dict) -> str:
        """Synchronous tool dispatch; async endpoints are driven by asyncio.run."""
        if name == "remember":
            result = asyncio.run(self.store.remember(args["text"], int(args.get("importance", 3))))
            return json.dumps(result, ensure_ascii=False)
        if name == "recall":
            facts = asyncio.run(self.store.recall(args["query"], args.get("top", 4)))
            return json.dumps(
                [{"id": f.id, "text": f.text, "score": f.score} for f in facts],
                ensure_ascii=False,
            )
        if name == "list_memories":
            facts = self.store.list(args.get("limit", 20))
            return json.dumps(
                [{"id": f.id, "text": f.text, "importance": f.importance} for f in facts],
                ensure_ascii=False,
            )
        if name == "forget":
            return json.dumps(self.store.forget(args["target"]), ensure_ascii=False)
        if name == "journal_write":
            rec = self.journal.append(
                args["who"], args["text"], session=args.get("session", "default")
            )
            return json.dumps({"ok": True, "record": rec}, ensure_ascii=False)
        if name == "observe_once":
            result = asyncio.run(self.observer.observe_once())
            return json.dumps(result, ensure_ascii=False)
        if name == "stats":
            return json.dumps(self.store.stats(), ensure_ascii=False)
        raise KeyError(f"unknown tool: {name}")

    # ---------------- protocol ----------------

    def _handle(self, msg: dict) -> dict | None:
        method = msg.get("method", "")
        msg_id = msg.get("id")
        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": SERVER_INFO,
                    },
                }
            if method == "ping":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
            if method == "tools/call":
                params = msg.get("params") or {}
                name = params.get("name", "")
                args = params.get("arguments") or {}
                try:
                    text = self.call_tool(name, args)
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"content": [{"type": "text", "text": text}]},
                    }
                except KeyError as exc:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32602, "message": str(exc)},
                    }
                except Exception as exc:  # noqa: BLE001 - tool failure is a result, not a transport error
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": f"error: {exc!r}"}],
                            "isError": True,
                        },
                    }
            if msg_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            return None  # notification: nothing to answer
        except Exception as exc:  # noqa: BLE001
            if msg_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32603, "message": f"internal error: {exc!r}"},
                }
            return None

    def serve(self) -> None:
        """Blocking stdio loop."""
        logging.basicConfig(
            stream=sys.stderr, level=logging.INFO,
            format="[wolfmem] %(levelname)s %(message)s",
        )
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            reply = self._handle(msg)
            if reply is not None:
                sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main() -> None:
    WolfmemMcpServer().serve()


if __name__ == "__main__":
    main()
