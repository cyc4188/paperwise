from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

APP_NAME = "research-helper"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / APP_NAME
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / APP_NAME
DEFAULT_CACHE_DIR = Path.home() / ".cache" / APP_NAME


def _expand_path(raw: str | None, default: Path, *, base: Path | None = None) -> Path:
    if not raw:
        return default.expanduser()
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = (base or default.parent) / path
    return path


def _read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}


def _nested_get(data: dict, dotted_key: str):
    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_scalar(value):
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


CONFIG_DIR = _expand_path(os.getenv("RH_CONFIG_DIR"), DEFAULT_CONFIG_DIR)
CONFIG_FILE = _expand_path(
    os.getenv("RH_CONFIG_FILE"),
    CONFIG_DIR / "config.toml",
    base=CONFIG_DIR,
)
_RAW_CONFIG = _read_toml(CONFIG_FILE) if CONFIG_FILE.exists() else {}
CONFIG_BASE_DIR = CONFIG_FILE.parent if CONFIG_FILE.exists() else CONFIG_DIR


def _setting(env_name: str, dotted_key: str, default=None):
    if env_name in os.environ:
        return os.environ[env_name]
    value = _nested_get(_RAW_CONFIG, dotted_key)
    value = _coerce_scalar(value)
    return default if value is None else value


def _path_setting(env_name: str, dotted_key: str, default: Path) -> Path:
    raw = _setting(env_name, dotted_key, None)
    return _expand_path(str(raw) if raw is not None else None, default, base=CONFIG_BASE_DIR)


CONFIG_DIR = _path_setting("RH_CONFIG_DIR", "paths.config_dir", CONFIG_DIR)
DATA_DIR = _path_setting("RH_DATA_DIR", "paths.data_dir", DEFAULT_DATA_DIR)
CACHE_DIR = _path_setting("RH_CACHE_DIR", "paths.cache_dir", DEFAULT_CACHE_DIR)

OUTPUTS_DIR = _path_setting("RH_OUTPUTS_DIR", "paths.outputs_dir", DATA_DIR / "outputs")
KB_DIR = _path_setting("RH_KB_DIR", "paths.kb_dir", DATA_DIR / "kb")
COST_LOG_PATH = _path_setting("RH_COST_LOG_PATH", "paths.cost_log_path", DATA_DIR / "cost_log.jsonl")


def ensure_app_dirs() -> None:
    for path in (CONFIG_DIR, DATA_DIR, CACHE_DIR, OUTPUTS_DIR, KB_DIR):
        path.mkdir(parents=True, exist_ok=True)


def cache_path(*parts: str) -> Path:
    return CACHE_DIR.joinpath(*parts)


# ── LLM ──────────────────────────────────────────────────────────────────────
# provider: anthropic | openai | deepseek | qwen
LLM_PROVIDER = str(_setting("LLM_PROVIDER", "llm.provider", "anthropic"))
LLM_MODEL = str(_setting("LLM_MODEL", "llm.model", "claude-sonnet-4-6"))

ANTHROPIC_API_KEY = str(_setting("ANTHROPIC_API_KEY", "api_keys.anthropic", ""))
OPENAI_API_KEY = str(_setting("OPENAI_API_KEY", "api_keys.openai", ""))
DEEPSEEK_API_KEY = str(_setting("DEEPSEEK_API_KEY", "api_keys.deepseek", ""))
QWEN_API_KEY = str(_setting("QWEN_API_KEY", "api_keys.qwen", ""))

ANTHROPIC_BASE_URL = str(_setting("ANTHROPIC_BASE_URL", "base_urls.anthropic", ""))
OPENAI_BASE_URL = str(_setting("OPENAI_BASE_URL", "base_urls.openai", ""))
DEEPSEEK_BASE_URL = str(_setting("DEEPSEEK_BASE_URL", "base_urls.deepseek", ""))
QWEN_BASE_URL = str(_setting("QWEN_BASE_URL", "base_urls.qwen", ""))


# ── Embedding ────────────────────────────────────────────────────────────────
# provider: openai | qwen | local
def _default_embedding_provider() -> str:
    configured = _setting("EMBEDDING_PROVIDER", "embedding.provider", "")
    if configured:
        return str(configured)
    if QWEN_API_KEY:
        return "qwen"
    if OPENAI_API_KEY:
        return "openai"
    return "local"


EMBEDDING_PROVIDER = _default_embedding_provider()
EMBEDDING_MODEL = str(_setting("EMBEDDING_MODEL", "embedding.model", ""))


# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(_setting("CHUNK_SIZE", "runtime.chunk_size", 12_000))
CHUNK_OVERLAP = int(_setting("CHUNK_OVERLAP", "runtime.chunk_overlap", 500))
CHUNK_SUMMARY_CONCURRENCY = max(
    1,
    int(_setting("CHUNK_SUMMARY_CONCURRENCY", "runtime.chunk_summary_concurrency", 2)),
)
REPORT_SECTION_CONCURRENCY = max(
    1,
    int(_setting("REPORT_SECTION_CONCURRENCY", "runtime.report_section_concurrency", 3)),
)


# ── Survey ───────────────────────────────────────────────────────────────────
SURVEY_MAX_PAPERS = int(_setting("SURVEY_MAX_PAPERS", "survey.max_papers", 20))
SURVEY_MAX_ABSTRACT_CHARS = int(
    _setting("SURVEY_MAX_ABSTRACT_CHARS", "survey.max_abstract_chars", 1_500)
)


# ── Zotero ───────────────────────────────────────────────────────────────────
ZOTERO_DATA_DIR = _path_setting("ZOTERO_DATA_DIR", "zotero.data_dir", Path.home() / "Zotero")


# ── Pricing（USD per 1M tokens）──────────────────────────────────────────────
PRICES: dict[str, dict] = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87},
    "deepseek-reasoner": {"input": 0.14, "output": 0.28},
    "deepseek-v3-0324": {"input": 0.14, "output": 0.28},
    "qwen-max": {"input": 1.60, "output": 6.40},
    "qwen-plus": {"input": 0.40, "output": 1.20},
    "qwen-turbo": {"input": 0.05, "output": 0.20},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.0},
    "text-embedding-v3": {"input": 0.069, "output": 0},
    "text-embedding-3-small": {"input": 0.02, "output": 0},
}
