"""Embedding client with an offline keyword fallback."""

from __future__ import annotations

import hashlib
from typing import Optional

import aiohttp

from .config import WolfmemConfig


class EmbeddingClient:
    """OpenAI-compatible embeddings.

    When ``config.emb_key`` is empty the client reports
    ``available=False`` and the store switches to keyword-overlap recall:
    no network, no key, still useful.
    """

    def __init__(self, config: WolfmemConfig):
        self.config = config
        self._cache: dict[str, list[float]] = {}

    @property
    def available(self) -> bool:
        return bool(self.config.emb_key)

    async def embed(self, text: str) -> Optional[list[float]]:
        """Return one embedding vector, or ``None`` when embeddings are unavailable."""
        text = (text or "").strip()
        if not text:
            return None
        if not self.available:
            return None
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        hit = self._cache.get(digest)
        if hit is not None:
            return hit
        import aiohttp  # local import keeps module import cheap for MCP use

        timeout = aiohttp.ClientTimeout(total=self.config.http_timeout)
        payload = {"model": self.config.emb_model, "input": text}
        headers = {
            "Authorization": f"Bearer {self.config.emb_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.config.emb_api, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"embedding endpoint HTTP {resp.status}")
                data = await resp.json()
        vec = data["data"][0]["embedding"]
        if len(self._cache) > 500:
            self._cache.clear()
        self._cache[digest] = vec
        return vec

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
