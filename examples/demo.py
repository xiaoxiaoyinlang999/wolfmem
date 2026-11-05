"""Minimal demo: journal → observe → recall, no framework required.

Run (offline mode — keyword recall, no distiller):

    WOLFMEM_HOME=$(mktemp -d) python examples/demo.py

Add WOLFMEM_EMB_KEY / WOLFMEM_LLM_* to switch embeddings and the observer on.
"""

import asyncio
import os

from wolfmem import EmbeddingClient, FactStore, Journal, Observer, WolfmemConfig


async def main() -> None:
    config = WolfmemConfig()
    config.ensure_dirs()
    print(f"data dir: {config.home}")

    journal = Journal(config.journal_dir)
    store = FactStore(config, EmbeddingClient(config))

    # 1) Talk: every turn lands in today's ledger.
    journal.append("user", "我们决定用 SQLite 存事实，够用了")
    journal.append("assistant", "好，SQLite + 余弦检索，简单可靠")
    journal.append("user", "另外记住，私聊底账不能发境外模型，隐私红线")
    journal.append("assistant", "明白，境内端点，数据不出境")

    # 2) Observe: the LLM distills durable facts (needs WOLFMEM_LLM_*; offline = skipped).
    observer = Observer(config, journal, store)
    print("observer:", await observer.observe_once())

    # 3) Recall: semantic search over what the system knows.
    for fact in await store.recall("存储方案选型"):
        print(f"  [{fact.score:.2f}] {fact.text}")

    # 4) Direct remember also works — the observer is just automation on top.
    await store.remember("[偏好] 深夜回复喜欢短句", importance=4)
    print("stats:", store.stats())


if __name__ == "__main__":
    asyncio.run(main())
