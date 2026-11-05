"""Append-only daily journal.

One JSONL file per day (YYYY-MM-DD.jsonl). Every record is a conversation
turn:

    {"who": "user", "session": "main", "text": "...", "t": "2026-11-05T18:08:00"}

The journal is the *ground truth* ledger. Nothing is ever rewritten; the
observer only reads new bytes past its saved offset, so crashes and restarts
never lose or double-count a turn.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Journal:
    def __init__(self, journal_dir: Path):
        self.dir = Path(journal_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---------------- write ----------------

    def append(self, who: str, text: str, *, session: str = "default") -> dict[str, Any]:
        """Append one turn to today's file and return the stored record."""
        rec = {
            "who": str(who),
            "session": str(session),
            "text": str(text),
            "t": _now_iso(),
        }
        path = self._today_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ---------------- read ----------------

    def files(self, last_n: int = 3) -> list[Path]:
        names = sorted(p.name for p in self.dir.glob("*.jsonl"))
        return [self.dir / n for n in names[-last_n:]]

    def tail(self, last_n: int = 3, limit: int = 100) -> list[dict[str, Any]]:
        """Read the last ``limit`` records across the newest files."""
        out: list[dict[str, Any]] = []
        for path in reversed(self.files(last_n)):
            if len(out) >= limit:
                break
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError:
                continue
            for line in reversed(lines):
                if len(out) >= limit:
                    break
                try:
                    out.append(json.loads(line))
                except (ValueError, UnicodeDecodeError):
                    continue
        out.reverse()
        return out

    def transcript(self, records: Iterable[dict[str, Any]], max_chars: int = 8000) -> str:
        """Render records as the plain transcript the distiller sees."""
        lines = [
            f"{r.get('who', '?')}({str(r.get('t', ''))[:16]}): {r.get('text', '')}"
            for r in records
        ]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    # ---------------- helpers ----------------

    def _today_path(self) -> Path:
        return self.dir / (time.strftime("%Y-%m-%d") + ".jsonl")
