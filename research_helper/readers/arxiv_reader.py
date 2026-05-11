from __future__ import annotations
import re
import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str        # ISO date string
    pdf_url: str
    categories: list[str] = field(default_factory=list)


def normalize_id(arxiv_id: str) -> str:
    """Strip URL prefix if user pastes a full arxiv URL."""
    arxiv_id = arxiv_id.strip().rstrip("/")
    m = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", arxiv_id)
    return m.group(1) if m else arxiv_id


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_meta(arxiv_id: str) -> PaperMeta:
    arxiv_id = normalize_id(arxiv_id)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        raise ValueError(f"Arxiv ID not found: {arxiv_id}")
    r = results[0]
    return PaperMeta(
        arxiv_id=arxiv_id,
        title=r.title,
        authors=[a.name for a in r.authors],
        abstract=r.summary.replace("\n", " "),
        published=r.published.strftime("%Y-%m-%d"),
        pdf_url=r.pdf_url,
        categories=r.categories,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_pdf(meta: PaperMeta, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-]", "_", meta.arxiv_id)
    dest = dest_dir / f"{safe_name}.pdf"
    if dest.exists():
        return dest
    urllib.request.urlretrieve(meta.pdf_url, dest)
    time.sleep(1)  # be polite to arxiv
    return dest


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_papers(query: str, max_results: int = 20) -> list[PaperMeta]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for r in client.results(search):
        results.append(PaperMeta(
            arxiv_id=r.entry_id.split("/")[-1],
            title=r.title,
            authors=[a.name for a in r.authors],
            abstract=r.summary.replace("\n", " "),
            published=r.published.strftime("%Y-%m-%d"),
            pdf_url=r.pdf_url,
            categories=r.categories,
        ))
    return results
