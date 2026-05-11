"""Session-level cost tracking + persistent cost log."""
from __future__ import annotations
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from research_helper import config

_lock = threading.Lock()
_LOG_PATH = config.OUTPUTS_DIR / ".cost_log.jsonl"

# ── Session accumulator (reset each process) ─────────────────────────────────

@dataclass
class _Session:
    input_tokens:  int   = 0
    output_tokens: int   = 0
    embed_tokens:  int   = 0
    cost_usd:      float = 0.0
    calls:         int   = 0

_session = _Session()


def record_llm(model: str, input_tokens: int, output_tokens: int) -> float:
    price = _price(model)
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    with _lock:
        _session.input_tokens  += input_tokens
        _session.output_tokens += output_tokens
        _session.cost_usd      += cost
        _session.calls         += 1
    return cost


def record_embedding(model: str, tokens: int) -> float:
    price = _price(model)
    cost = tokens * price["input"] / 1_000_000
    with _lock:
        _session.embed_tokens += tokens
        _session.cost_usd     += cost
    return cost


def session_summary() -> dict:
    with _lock:
        return {
            "input_tokens":  _session.input_tokens,
            "output_tokens": _session.output_tokens,
            "embed_tokens":  _session.embed_tokens,
            "cost_usd":      _session.cost_usd,
            "calls":         _session.calls,
        }


def flush_to_log(label: str) -> None:
    """Append current session totals to the persistent JSONL log."""
    summary = session_summary()
    if summary["cost_usd"] == 0:
        return
    entry = {
        "ts":    datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "model": config.LLM_MODEL,
        **summary,
    }
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Persistent log reader ─────────────────────────────────────────────────────

def read_log() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    entries = []
    for line in _LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def total_cost() -> float:
    return sum(e["cost_usd"] for e in read_log())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _price(model: str) -> dict:
    # Exact match first, then prefix match
    prices = config.PRICES
    if model in prices:
        return prices[model]
    for key in prices:
        if model.startswith(key) or key.startswith(model):
            return prices[key]
    return {"input": 0.0, "output": 0.0}   # unknown model → free (warn silently)
