import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUTPUTS_DIR = Path("outputs")

# ── LLM ──────────────────────────────────────────────────────────────────────
# LLM_PROVIDER: anthropic | openai | deepseek | qwen
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL    = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
QWEN_API_KEY      = os.getenv("QWEN_API_KEY", "")   # DashScope key

# ── Embedding ─────────────────────────────────────────────────────────────────
# EMBEDDING_PROVIDER: openai | qwen | local
# Defaults: qwen if QWEN_API_KEY set, openai if OPENAI_API_KEY set, else local
def _default_embedding_provider() -> str:
    ep = os.getenv("EMBEDDING_PROVIDER", "")
    if ep:
        return ep
    if os.getenv("QWEN_API_KEY"):
        return "qwen"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "local"

EMBEDDING_PROVIDER = _default_embedding_provider()
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "")   # override if needed

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 12_000
CHUNK_OVERLAP = 500

# ── Survey ────────────────────────────────────────────────────────────────────
SURVEY_MAX_PAPERS         = 20
SURVEY_MAX_ABSTRACT_CHARS = 1_500

# ── Pricing（USD per 1M tokens，可在 .env 中用 PRICE_* 覆盖）────────────────────
# 格式：input, output（embedding 只有 input）
PRICES: dict[str, dict] = {
    # DeepSeek（来源：api-docs.deepseek.com/quick_start/pricing，2026-05）
    # deepseek-chat / deepseek-reasoner 即将弃用，分别对应 v4-flash 非思考/思考模式
    "deepseek-v4-flash":   {"input": 0.14,  "output": 0.28},   # cache hit: $0.0028/M
    "deepseek-chat":       {"input": 0.14,  "output": 0.28},   # alias → v4-flash
    "deepseek-v4-pro":     {"input": 0.435, "output": 0.87},   # 促销价（75% off，至 2026-05-31）
    "deepseek-reasoner":   {"input": 0.14,  "output": 0.28},   # alias → v4-flash thinking
    "deepseek-v3-0324":    {"input": 0.14,  "output": 0.28},   # 已并入 v4-flash 体系
    # Qwen（来源：alibabacloud.com/help/en/model-studio/model-pricing，国际区，2026-05）
    "qwen-max":            {"input": 1.60,  "output": 6.40},
    "qwen-plus":           {"input": 0.40,  "output": 1.20},
    "qwen-turbo":          {"input": 0.05,  "output": 0.20},
    # OpenAI
    "gpt-4o":              {"input": 2.50,  "output": 10.0},
    "gpt-4o-mini":         {"input": 0.15,  "output": 0.60},
    # Anthropic
    "claude-sonnet-4-6":   {"input": 3.00,  "output": 15.0},
    # Embedding（只有 input）
    "text-embedding-v3":       {"input": 0.069, "output": 0},  # Qwen，¥0.5/M → $0.069
    "text-embedding-3-small":  {"input": 0.02,  "output": 0},
}
