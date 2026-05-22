"""Embedding generation: qwen → openai → local sentence-transformers."""
from __future__ import annotations
from research_helper import config

_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "qwen":   "text-embedding-v3",
}

_st_model = None


def embed(texts: list[str]) -> list[list[float]]:
    p = config.EMBEDDING_PROVIDER
    if p == "qwen":
        return _embed_openai_compat(texts, "qwen")
    if p == "openai":
        return _embed_openai_compat(texts, "openai")
    return _embed_local(texts)


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def _embed_openai_compat(texts: list[str], provider: str) -> list[list[float]]:
    from openai import OpenAI
    from research_helper.llm.client import _DEFAULT_HEADERS, _api_key, _base_url
    from research_helper.utils import cost_tracker

    model = config.EMBEDDING_MODEL or _DEFAULT_MODELS[provider]
    client = OpenAI(
        api_key=_api_key(provider),
        base_url=_base_url(provider),
        default_headers=_DEFAULT_HEADERS,
    )
    results: list[list[float]] = []
    total_tokens = 0
    for batch in _batched(texts, 10 if provider == "qwen" else 25):
        resp = client.embeddings.create(model=model, input=batch)
        results.extend(d.embedding for d in resp.data)
        if resp.usage:
            total_tokens += resp.usage.total_tokens
    if total_tokens:
        cost_tracker.record_embedding(model, total_tokens)
    return results


def _embed_local(texts: list[str]) -> list[list[float]]:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vecs = _st_model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
