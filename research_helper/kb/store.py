"""ChromaDB-backed knowledge base store."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from research_helper.kb import embedder

_KB_DIR = Path("outputs/.kb")
_COLLECTION = "papers"

_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _KB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(_KB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


@dataclass
class KBEntry:
    doc_id: str
    text: str
    title: str
    arxiv_id: str
    published: str
    source: str          # "abstract" | "report"
    distance: float = 0.0


def add(
    arxiv_id: str,
    title: str,
    published: str,
    abstract: str,
    report_text: str,
) -> None:
    col = _get_collection()
    docs, ids, metas = [], [], []

    def _stage(suffix: str, text: str, source: str):
        doc_id = f"{arxiv_id}::{suffix}"
        # Skip if already indexed
        existing = col.get(ids=[doc_id])
        if existing["ids"]:
            return
        docs.append(text[:4000])   # cap per chunk to stay within embedding limits
        ids.append(doc_id)
        metas.append({"arxiv_id": arxiv_id, "title": title,
                       "published": published, "source": source})

    _stage("abstract", abstract, "abstract")
    # Index report in 1500-char chunks to improve retrieval granularity
    for i, chunk in enumerate(_chunk(report_text, 1500)):
        _stage(f"report_{i}", chunk, "report")

    if not docs:
        return  # all already indexed

    vectors = embedder.embed(docs)
    col.add(documents=docs, embeddings=vectors, ids=ids, metadatas=metas)


def query(text: str, top_k: int = 5) -> list[KBEntry]:
    col = _get_collection()
    if col.count() == 0:
        return []
    vec = embedder.embed_one(text[:2000])
    results = col.query(
        query_embeddings=[vec],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    entries = []
    seen_papers: set[str] = set()
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        arxiv_id = meta["arxiv_id"]
        # Deduplicate: keep only the most relevant chunk per paper
        if arxiv_id in seen_papers:
            continue
        seen_papers.add(arxiv_id)
        entries.append(KBEntry(
            doc_id=f'{arxiv_id}::{meta["source"]}',
            text=doc,
            title=meta["title"],
            arxiv_id=arxiv_id,
            published=meta["published"],
            source=meta["source"],
            distance=dist,
        ))
    return entries


def list_papers() -> list[dict]:
    col = _get_collection()
    if col.count() == 0:
        return []
    results = col.get(where={"source": "abstract"}, include=["metadatas"])
    seen, papers = set(), []
    for meta in results["metadatas"]:
        aid = meta["arxiv_id"]
        if aid not in seen:
            seen.add(aid)
            papers.append({"arxiv_id": aid, "title": meta["title"],
                           "published": meta["published"]})
    papers.sort(key=lambda p: p["published"], reverse=True)
    return papers


def count() -> int:
    return len(list_papers())


def _chunk(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]
