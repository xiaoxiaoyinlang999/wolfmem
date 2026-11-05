"""Offline regression tests: write → observe → persist → restart → recall.

Run:  python tests/test_closed_loop.py
No network needed; the distiller and embedder are monkeypatched.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wolfmem import EmbeddingClient, FactStore, Journal, Observer, WolfmemConfig  # noqa: E402
from wolfmem.mcp_server import WolfmemMcpServer  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f" FAIL {name} {detail}")


def fresh(tmp: str) -> tuple[WolfmemConfig, Journal, FactStore, Observer]:
    config = WolfmemConfig(home=Path(tmp))
    config.ensure_dirs()
    journal = Journal(config.journal_dir)
    store = FactStore(config, EmbeddingClient(config))
    observer = Observer(config, journal, store)
    return config, journal, store, observer


def test_api_surface() -> None:
    print("[1] API surface")
    check("Observer.run_forever is a class method", callable(getattr(Observer, "run_forever", None)))
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "wolfmem.cli", "init"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
        env={**dict(__import__("os").environ), "PYTHONPATH": "src", "WOLFMEM_HOME": tempfile.mkdtemp()},
    )
    check("cli init exits 0", out.returncode == 0, out.stderr[-200:])
    server = WolfmemMcpServer(WolfmemConfig(home=Path(tempfile.mkdtemp())))
    reply = server._handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    check("mcp tools/list works", "tools" in (reply or {}).get("result", {}))


def test_offset_semantics(tmp: str) -> None:
    print("[2] offset commit semantics")
    config, journal, store, observer = fresh(tmp)

    # First sighting commits the tail (history is never re-read) — start from
    # a seed turn, then everything appended after is subject to the rules.
    journal.append("user", "seed turn so the file exists with some content")
    asyncio.run(observer.observe_once())
    baseline = max(json.loads(config.observe_state_file.read_text())["offsets"].values(), default=0)

    # Batch under min_lines: nothing consumed.
    journal.append("user", "under threshold line 1 with real content here")
    journal.append("user", "under threshold line 2 with real content here")
    r1 = asyncio.run(observer.observe_once())
    check("under min_lines -> skipped", r1.get("skipped") is not None, str(r1))
    now = max(json.loads(config.observe_state_file.read_text())["offsets"].values(), default=0)
    check("offsets NOT committed on skip", now == baseline, f"{now} != {baseline}")

    # Distiller failure: nothing consumed, retry next tick.
    journal.append("user", "line three crossing the threshold with real content")

    async def boom(_t):
        raise RuntimeError("llm down")

    observer._distill = boom
    r2 = asyncio.run(observer.observe_once())
    check("llm failure reported", "error" in r2, str(r2))
    now = max(json.loads(config.observe_state_file.read_text())["offsets"].values(), default=0)
    check("offsets NOT committed on failure", now == baseline, f"{now} != {baseline}")

    # Success: offsets commit to end of scanned region.
    async def distill_ok(_t):
        return ["[决定] 测试决定条目"]

    observer._distill = distill_ok
    r3 = asyncio.run(observer.observe_once())
    check("success stores facts", r3.get("facts") == 1, str(r3))
    state = json.loads(config.observe_state_file.read_text())["offsets"]
    check("offsets committed after success", max(state.values(), default=0) > 0, str(state))


def test_restart_recall(tmp: str) -> None:
    print("[3] write -> restart -> recall")
    config, journal, store, observer = fresh(tmp)
    asyncio.run(store.remember("[决定] 闭环测试用 SQLite"))
    journal.append("user", "restart comes later, fact is already stored here")
    # Simulate restart: rebuild everything from disk.
    _, _, store2, observer2 = fresh(tmp)
    hits = asyncio.run(store2.recall("闭环 SQLite"))
    check("fact survives restart", any("SQLite" in f.text for f in hits), str(hits))
    # Restarted observer must not re-read history (offset state persisted).
    asyncio.run(observer2.observe_once())
    r = asyncio.run(observer2.observe_once())
    check("restart keeps offsets", r.get("new_lines", 0) == 0 or r.get("skipped"), str(r))


def test_fused_recall(tmp: str) -> None:
    print("[4] fused recall after offline->online migration")
    config, journal, store, observer = fresh(tmp)
    # Keyword-mode row (empty emb), written while embeddings were offline.
    async def offline(t):
        return None
    store._embed_safe = offline
    asyncio.run(store.remember("[偏好] 夜间回复要短句"))
    # Now embeddings come online: fake a client that returns a vector.
    async def online(t):
        return _fake_vec(t)
    store.embeddings.embed = online
    hits = asyncio.run(store.recall("夜间 短句"))
    check("keyword-mode row recallable in vector mode", len(hits) == 1, str(hits))


_DIM = 8

def _fake_vec(text: str):
    return [1.0 if "夜" in text or "短" in text else 0.0] * _DIM


def test_dedupe_updates_vector(tmp: str) -> None:
    print("[5] dedupe updates the stored vector")
    config, journal, store, observer = fresh(tmp)
    async def online(t):
        return _fake_vec(t)
    store.embeddings.embed = online
    asyncio.run(store.remember("[事实] 原始表述内容 夜间样例一", importance=2))
    r = asyncio.run(store.remember("[事实] 原始表述内容 夜间样例二", importance=4))
    check("dedupe hits -> updated", r.get("action") == "updated", str(r))
    import sqlite3
    conn = sqlite3.connect(store.db_file)
    row = conn.execute("SELECT text, emb FROM facts").fetchone()
    conn.close()
    check("text updated", "样例二" in row[0], row[0])
    check("vector refreshed with new text", json.loads(row[1]) == _fake_vec(row[0]))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_api_surface()
        test_offset_semantics(tempfile.mkdtemp())
        test_restart_recall(tempfile.mkdtemp())
        test_fused_recall(tempfile.mkdtemp())
        test_dedupe_updates_vector(tempfile.mkdtemp())
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
