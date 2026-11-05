"""The observer: polls the journal, distills durable facts via an LLM.

This is the "claude-mem style transcript watcher" from the reference
deployment, generalized:

1. Track a per-file byte offset so restarts never re-read or skip turns.
2. First sighting of a journal file starts from its *tail* — history is never
   retroactively distilled, which keeps costs predictable.
3. Only complete lines are consumed; a partially flushed line waits for the
   next tick.
4. Fact extraction runs on the caller's loop with a hard HTTP timeout; a hung
   endpoint wastes one tick at most, never the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import aiohttp

from .config import OBSERVE_PROMPT, WolfmemConfig
from .journal import Journal
from .store import FactStore

logger = logging.getLogger("wolfmem.observer")

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _observation_worthy(text: str) -> bool:
    text = (text or "").strip()
    if not (12 <= len(text) <= 300):
        return False
    if text.startswith(("/", "!", "#")):
        return False  # commands / markdown are not conversation
    if re.fullmatch(r"[\W_0-9]+", text):
        return False  # punctuation-only / emoji lines
    return True


class Observer:
    def __init__(self, config: WolfmemConfig, journal: Journal, store: FactStore):
        self.config = config
        self.journal = journal
        self.store = store
        self._offsets: dict[str, int] = {}
        self._load_state()

    # ---------------- state ----------------

    def _load_state(self) -> None:
        try:
            with open(self.config.observe_state_file, encoding="utf-8") as f:
                self._offsets = dict(json.load(f).get("offsets") or {})
        except (OSError, ValueError):
            self._offsets = {}

    def _save_state(self) -> None:
        tmp = str(self.config.observe_state_file) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"offsets": self._offsets}, f, ensure_ascii=False)
        os.replace(tmp, self.config.observe_state_file)

    # ---------------- scan ----------------

    async def observe_once(self) -> dict:
        """Scan for new journal turns; distill and store when enough gathered."""
        new_records: list[dict] = []
        changed = False
        for path in self.journal.files(last_n=3):
            name = path.name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            offset = self._offsets.get(name)
            if offset is None:
                # First sighting: start from the tail, never re-read history.
                self._offsets[name] = size
                changed = True
                continue
            if size <= offset:
                continue
            with open(path, "rb") as f:
                f.seek(offset)
                blob = f.read()
            cut = blob.rfind(b"\n")
            if cut < 0:
                continue  # incomplete line; wait for the next tick
            self._offsets[name] = offset + cut + 1
            changed = True
            for raw in blob[: cut + 1].splitlines():
                try:
                    rec = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(rec, dict) and _observation_worthy(str(rec.get("text", ""))):
                    new_records.append(rec)
        if changed:
            self._save_state()

        if len(new_records) < self.config.observe_min_lines:
            return {"new_lines": len(new_records), "facts": 0, "skipped": "not enough lines"}

        recent = new_records[-self.config.observe_max_lines:]
        transcript = self.journal.transcript(recent)
        try:
            facts = await self._distill(transcript)
        except Exception as exc:  # noqa: BLE001 - one bad LLM reply must not kill the loop
            return {"new_lines": len(new_records), "facts": 0, "error": repr(exc)}

        saved = 0
        for fact in facts:
            text = str(fact).strip()
            if not (6 <= len(text) <= 160):
                continue
            try:
                await self.store.remember(text, importance=2)
                saved += 1
            except Exception as exc:  # noqa: BLE001 - one bad fact must not stop the batch
                logger.debug("store failed for fact %r: %s", text[:40], exc)
        return {"new_lines": len(new_records), "facts": saved}

    # ---------------- distill ----------------

    async def _distill(self, transcript: str) -> list[str]:
        if not self.config.llm_key:
            logger.info("observer: no LLM key configured, skipping distill")
            return []
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "user", "content": f"{OBSERVE_PROMPT}\n\n对话记录：\n{transcript}"}
            ],
            "temperature": 0.3,
        }
        url = self.config.llm_api_base.rstrip("/") + "/chat/completions"
        timeout = aiohttp.ClientTimeout(total=self.config.http_timeout)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(
                url,
                headers={"Authorization": f"Bearer {self.config.llm_key}"},
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        raw = str(
            ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        ).strip()
        if not raw:
            return []
        raw = _CODE_FENCE.sub("", raw).strip()
        parsed = _parse_facts_array(raw)
        if not isinstance(parsed, list):
            raise ValueError("distiller returned non-list JSON")
        return [x for x in parsed if isinstance(x, str)]

    @staticmethod
    def _parse_facts(raw: str) -> list:  # pragma: no cover - kept for API symmetry
        return _parse_facts_array(raw)


def _parse_facts_array(raw: str) -> list:
    """Tolerant JSON-array extraction.

    Chat models love decorating their JSON with prose ("好的，以下是提炼结果：").
    ``json.loads`` on the whole blob fails with "Extra data"; instead, locate
    the first ``[`` and let ``JSONDecoder.raw_decode`` stop at the array's end.
    """
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass
    start = raw.find("[")
    if start < 0:
        raise ValueError(f"no JSON array in distiller reply: {raw[:120]!r}")
    parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
    return parsed

    # ---------------- loop ----------------

    async def run_forever(self) -> None:
        logger.info("observer started, interval %ss", self.config.observe_interval)
        while True:
            try:
                result = await self.observe_once()
                if result.get("facts"):
                    logger.info("observer: %s", result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("observer tick failed: %r", exc)
            await asyncio.sleep(self.config.observe_interval)


def start_observer_background(
    config: WolfmemConfig, journal: Journal, store: FactStore
) -> Optional[asyncio.Task]:
    """Best-effort background task for hosts that already run an event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    observer = Observer(config, journal, store)
    return loop.create_task(observer.run_forever())
