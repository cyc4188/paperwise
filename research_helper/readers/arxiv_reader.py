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


def _fetch_meta_html(arxiv_id: str) -> PaperMeta:
    """Fallback: scrape the arxiv abstract page when the API is rate-limited."""
    import urllib.request as _req
    url = f"https://arxiv.org/abs/{arxiv_id}"
    req = _req.Request(url, headers={"User-Agent": "research-helper/0.1 (mailto:user@example.com)"})
    with _req.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    def _tag(pattern: str) -> str:
        m = re.search(pattern, html, re.S)
        return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""

    title = _tag(r'<h1[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</h1>')
    title = re.sub(r"^Title:\s*", "", title)

    authors_block = _tag(r'<div[^>]*class="[^"]*authors[^"]*"[^>]*>(.*?)</div>')
    authors = [a.strip() for a in re.split(r",\s*", authors_block) if a.strip()] or ["Unknown"]

    abstract = _tag(r'<blockquote[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</blockquote>')
    abstract = re.sub(r"^Abstract:\s*", "", abstract).replace("\n", " ")

    # "Submitted 3 February, 2024" or "Submitted on 3 February, 2024"
    date_raw = _tag(r'<div[^>]*class="[^"]*submission-history[^"]*"[^>]*>.*?Submitted\s+(?:on\s+)?([^;(]+?)[\s;(]')
    published = ""
    if date_raw:
        try:
            from datetime import datetime
            for fmt in ("%d %B, %Y", "%d %B %Y", "%B %d, %Y"):
                try:
                    published = datetime.strptime(date_raw.strip(), fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass
        except Exception:
            pass

    if not title:
        raise ValueError(f"Could not parse arxiv page for {arxiv_id}")

    return PaperMeta(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        published=published,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        categories=[],
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=30))
def fetch_meta(arxiv_id: str) -> PaperMeta:
    arxiv_id = normalize_id(arxiv_id)
    try:
        client = arxiv.Client(delay_seconds=5, num_retries=3)
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
    except Exception as api_err:
        import sys
        print(f"[arxiv] API failed ({api_err}), falling back to HTML scrape…", file=sys.stderr)
        return _fetch_meta_html(arxiv_id)


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
