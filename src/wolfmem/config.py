"""wolfmem configuration.

Everything is environment-first so the library works in CLIs, scripts, IDEs and
long-running agents without code changes. No secret ever ships with the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Needle tags: every stored fact starts with one of these, mirroring the
# "needle classification" idea (decisions, pitfalls, privacy, security...).
NEEDLE_TAGS = ("决定", "坑", "隐私", "安全", "偏好", "事实")

# The distiller prompt. Kept in Chinese by design: the reference deployment is
# a Chinese-first assistant, and the tags are Chinese literals. Translating the
# prompt without translating the tags breaks recall filtering.
OBSERVE_PROMPT = """你是记忆管理员。下面是一段对话记录。
提炼值得长期记住的事实：拍板的决定、bug 和教训、隐私边界、安全相关、偏好、人物关系。
每条格式：[标签]一句话。标签只能用：决定/坑/隐私/安全/偏好/事实。
寒暄、闲聊、正在做的临时任务不要记。没有值得记的就返回 []。
只返回 JSON 字符串数组，不要解释，不要 markdown 代码块。"""

EMBED_DIM_FALLBACK = 1024  # BAAI/bge-m3 dimension; keyword mode does not use it


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class WolfmemConfig:
    """Runtime configuration, all overridable via environment variables."""

    # Storage root. Journal, sqlite database and observer state live here.
    home: Path = field(default_factory=lambda: Path(_env("WOLFMEM_HOME", "~/.wolfmem")).expanduser())

    # Embedding endpoint (OpenAI-compatible /v1/embeddings). When the key is
    # empty, wolfmem falls back to keyword-overlap recall and remember() skips
    # dedupe-by-embedding — the store still works fully offline.
    emb_api: str = field(default_factory=lambda: _env("WOLFMEM_EMB_API", "https://api.siliconflow.cn/v1/embeddings"))
    emb_model: str = field(default_factory=lambda: _env("WOLFMEM_EMB_MODEL", "BAAI/bge-m3"))
    emb_key: str = field(default_factory=lambda: _env("WOLFMEM_EMB_KEY"))

    # Chat-completion endpoint (OpenAI-compatible /v1/chat/completions), used
    # by the observer to distill facts. Privacy note: the observer sends raw
    # journal text to this endpoint, so point it at a provider you trust.
    llm_api_base: str = field(default_factory=lambda: _env("WOLFMEM_LLM_API_BASE"))
    llm_model: str = field(default_factory=lambda: _env("WOLFMEM_LLM_MODEL", "glm-5.3-flash"))
    llm_key: str = field(default_factory=lambda: _env("WOLFMEM_LLM_KEY"))

    # Recall tuning (same defaults as the reference deployment).
    recall_top: int = field(default_factory=lambda: _env_int("WOLFMEM_RECALL_TOP", 4))
    min_score: float = field(default_factory=lambda: _env_float("WOLFMEM_MIN_SCORE", 0.42))
    dedupe_score: float = field(default_factory=lambda: _env_float("WOLFMEM_DEDUPE_SCORE", 0.93))
    max_facts: int = field(default_factory=lambda: _env_int("WOLFMEM_MAX_FACTS", 800))

    # Observer tuning.
    observe_interval: int = field(default_factory=lambda: _env_int("WOLFMEM_OBSERVE_INTERVAL", 300))
    observe_min_lines: int = field(default_factory=lambda: _env_int("WOLFMEM_OBSERVE_MIN_LINES", 3))
    observe_max_lines: int = field(default_factory=lambda: _env_int("WOLFMEM_OBSERVE_MAX_LINES", 60))

    # HTTP timeouts (seconds).
    http_timeout: int = field(default_factory=lambda: _env_int("WOLFMEM_HTTP_TIMEOUT", 30))

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)

    @property
    def journal_dir(self) -> Path:
        return self.home / "journal"

    @property
    def db_file(self) -> Path:
        return self.home / "mem.db"

    @property
    def observe_state_file(self) -> Path:
        return self.home / "observe_state.json"
