"""Fact store: SQLite-backed, vector-first with keyword fallback."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import WolfmemConfig
from .embeddings import EmbeddingClient

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 1,
    created REAL NOT NULL,
    last_hit REAL NOT NULL DEFAULT 0,
    hits INTEGER NOT NULL DEFAULT 0,
    emb TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created DESC);
"""


@dataclass
class Fact:
    id: int
    text: str
    importance: int
    created: float
    hits: int
    score: float = 0.0


def _tokenize(text: str) -> set[str]:
    """Very small CJK-aware tokenizer for the keyword fallback.

    Latin/digit runs become lowercase tokens; each CJK character is its own
    token (bigrams would be better, but unigrams already beat nothing, and the
    fallback exists so the project works without any embedding key at all).
    """
    tokens: set[str] = set()
    for chunk in re.findall(r"[A-Za-z0-9]+", text.lower()):
        tokens.add(chunk)
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.add(ch)
    return tokens


class FactStore:
    def __init__(self, config: WolfmemConfig, embeddings: EmbeddingClient):
        self.config = config
        self.embeddings = embeddings
        self.db_file: Path = config.db_file
        conn = sqlite3.connect(self.db_file)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    # ---------------- internals ----------------

    def _db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _keyword_score(self, query: str, text: str) -> float:
        q, t = _tokenize(query), _tokenize(text)
        if not q:
            return 0.0
        overlap = len(q & t) / len(q)
        return round(min(1.0, overlap * 2.0), 3)  # half-overlap ~= strong hit

    async def _embed_safe(self, text: str) -> Optional[list[float]]:
        """Embed with degradation: endpoint errors fall back to keyword mode.

        Returns ``None`` when embeddings are unavailable or fail; callers then
        write keyword-mode rows (empty ``emb``) instead of losing the fact.
        Rows without vectors still participate in recall via keyword scoring.
        """
        try:
            return await self.embeddings.embed(text)
        except Exception:  # noqa: BLE001 - degraded write beats failed write
            return None

    # ---------------- write ----------------

    async def remember(self, text: str, importance: int = 3) -> dict:
        """Store one fact. Dedupes by embedding when available.

        Returns ``{"ok": True, "action": "inserted"|"updated"|"keyword", "id": ...}``.
        """
        text = (text or "").strip()
        if not text:
            return {"ok": False, "reason": "empty text"}
        vec = await self._embed_safe(text)
        conn = self._db()
        try:
            if vec is not None:
                for row in conn.execute("SELECT id, text, emb FROM facts").fetchall():
                    try:
                        old = json.loads(row["emb"])
                    except (ValueError, TypeError):
                        continue
                    if EmbeddingClient.cosine(vec, old) >= self.config.dedupe_score:
                        # Update the vector together with the text: keeping a
                        # stale embedding would make recall match the old
                        # wording while the UI shows the new one.
                        conn.execute(
                            "UPDATE facts SET text = ?, emb = ?, last_hit = ?, importance = MAX(importance, ?) WHERE id = ?",
                            (text, json.dumps(vec), time.time(), importance, row["id"]),
                        )
                        conn.commit()
                        return {"ok": True, "action": "updated", "id": row["id"]}

            count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            if count >= self.config.max_facts:
                # Evict the least valuable half-slice: low importance, never
                # recalled, oldest hit wins the boot.
                conn.execute(
                    "DELETE FROM facts WHERE id IN ("
                    "  SELECT id FROM facts ORDER BY importance ASC, hits ASC, last_hit ASC LIMIT 50"
                    ")"
                )
            cur = conn.execute(
                "INSERT INTO facts (text, importance, created, emb) VALUES (?, ?, ?, ?)",
                (text, importance, time.time(), json.dumps(vec) if vec else ""),
            )
            conn.commit()
            return {"ok": True, "action": "inserted" if vec is not None else "keyword", "id": cur.lastrowid}
        finally:
            conn.close()

    # ---------------- read ----------------

    async def recall(self, query: str, top: Optional[int] = None) -> list[Fact]:
        """Rank facts against ``query``; embeddings when available, keywords otherwise."""
        query = (query or "").strip()
        if not query:
            return []
        top = top or self.config.recall_top
        vec = await self._embed_safe(query)
        conn = self._db()
        try:
            rows = conn.execute("SELECT * FROM facts").fetchall()
        finally:
            conn.close()

        if vec is not None:
            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                kw = self._keyword_score(query, row["text"])
                if not row["emb"]:
                    # Facts written in keyword mode stay recallable after the
                    # embedding service is switched on — fused scoring keeps
                    # the offline→online migration path lossless.
                    if kw > 0:
                        scored.append((kw, row))
                    continue
                try:
                    old = json.loads(row["emb"])
                except (ValueError, TypeError):
                    if kw > 0:
                        scored.append((kw, row))
                    continue
                score = EmbeddingClient.cosine(vec, old)
                if score >= self.config.min_score:
                    scored.append((score, row))
        else:
            scored = [(self._keyword_score(query, row["text"]), row) for row in rows]
            scored = [(s, r) for s, r in scored if s > 0]

        scored.sort(key=lambda x: -x[0])
        picked = scored[: max(1, top)]
        if not picked:
            return []

        conn = self._db()
        try:
            for score, row in picked:
                conn.execute(
                    "UPDATE facts SET last_hit = ?, hits = hits + 1 WHERE id = ?",
                    (time.time(), row["id"]),
                )
            conn.commit()
        finally:
            conn.close()

        return [
            Fact(
                id=row["id"],
                text=row["text"],
                importance=row["importance"],
                created=row["created"],
                hits=row["hits"],
                score=round(score, 3),
            )
            for score, row in picked
        ]

    def list(self, limit: int = 20) -> list[Fact]:
        conn = self._db()
        try:
            rows = conn.execute(
                "SELECT * FROM facts ORDER BY created DESC LIMIT ?", (max(1, limit),)
            ).fetchall()
        finally:
            conn.close()
        return [
            Fact(
                id=r["id"],
                text=r["text"],
                importance=r["importance"],
                created=r["created"],
                hits=r["hits"],
            )
            for r in rows
        ]

    def forget(self, target: str) -> dict:
        """Forget by numeric id, needle tag, or keyword substring."""
        target = (target or "").strip()
        if not target:
            return {"ok": False, "reason": "empty target"}
        conn = self._db()
        try:
            if target.isdigit():
                cur = conn.execute("DELETE FROM facts WHERE id = ?", (int(target),))
            else:
                cur = conn.execute("DELETE FROM facts WHERE text LIKE ?", (f"%{target}%",))
            conn.commit()
            return {"ok": True, "deleted": cur.rowcount}
        finally:
            conn.close()

    def stats(self) -> dict:
        conn = self._db()
        try:
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            by_tag: dict[str, int] = {}
            for tag in ("决定", "坑", "隐私", "安全", "偏好", "事实"):
                n = conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE text LIKE ?", (f"[{tag}]%",)
                ).fetchone()[0]
                by_tag[tag] = n
            return {"facts": total, "by_tag": by_tag, "embedding": self.embeddings.available}
        finally:
            conn.close()
