"""wolfmem — persistent memory for AI agents and IDEs.

Layers:
    Journal   append-only daily JSONL ledger (ground truth)
    Observer  background watcher that distills durable facts via an LLM
    FactStore SQLite + vector (or keyword) recall, needle-tagged

Expose it over MCP (``wolfmem serve``) or use the library directly.
"""

from .config import OBSERVE_PROMPT, NEEDLE_TAGS, WolfmemConfig
from .embeddings import EmbeddingClient
from .journal import Journal
from .observer import Observer
from .store import FactStore

__version__ = "0.1.0"

__all__ = [
    "WolfmemConfig",
    "Journal",
    "FactStore",
    "Observer",
    "EmbeddingClient",
    "NEEDLE_TAGS",
    "OBSERVE_PROMPT",
    "__version__",
]
