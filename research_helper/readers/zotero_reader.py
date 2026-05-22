from __future__ import annotations

import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from research_helper import config
from research_helper.readers.arxiv_reader import PaperMeta

_DEFAULT_ZOTERO_DIRS = [
    Path.home() / "Zotero",
    Path.home() / "Library/Application Support/Zotero",
    Path.home() / ".zotero/zotero",
]


@dataclass
class ZoteroMatch:
    item_id: int
    item_key: str
    attachment_key: str
    title: str
    published: str
    abstract: str
    authors: list[str]
    pdf_path: Path

    def to_meta(self) -> PaperMeta:
        return PaperMeta(
            arxiv_id=f"zotero:{self.item_key}",
            title=self.title,
            authors=self.authors or ["Unknown"],
            abstract=self.abstract,
            published=self.published,
            pdf_url=str(self.pdf_path),
            categories=["zotero"],
        )


def resolve_data_dir() -> Path:
    configured = config.ZOTERO_DATA_DIR
    candidates = [configured, *_DEFAULT_ZOTERO_DIRS]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if (path / "zotero.sqlite").exists() and (path / "storage").exists():
            return path
    raise FileNotFoundError(
        "Zotero data directory not found. Set zotero.data_dir in config.toml or ZOTERO_DATA_DIR in the environment."
    )


def find_paper(query: str) -> ZoteroMatch:
    query = query.strip()
    if not query:
        raise ValueError("Empty Zotero query")

    data_dir = resolve_data_dir()
    matches = _search(data_dir, query)
    if not matches:
        raise LookupError(f'No Zotero paper with a local PDF matched "{query}".')
    if len(matches) > 1:
        preview = "\n".join(
            f"  - {m.item_key} [{m.published or 'unknown'}] {m.title}" for m in matches[:5]
        )
        raise LookupError(
            f'Multiple Zotero papers matched "{query}". Please use a more specific title or item key:\n'
            f"{preview}"
        )
    return matches[0]


def search_papers(query: str, limit: int = 10) -> list[ZoteroMatch]:
    return _search(resolve_data_dir(), query.strip(), limit=limit)


def _search(data_dir: Path, query: str, limit: int = 10) -> list[ZoteroMatch]:
    with _snapshot_db(data_dir / "zotero.sqlite") as db_path:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = _candidate_rows(conn, query, limit=max(limit, 5))
            matches: list[ZoteroMatch] = []
            for row in rows:
                pdf_path = _resolve_pdf_path(data_dir, row["attachmentKey"], row["attachmentPath"])
                if not pdf_path.exists():
                    continue
                matches.append(
                    ZoteroMatch(
                        item_id=row["itemID"],
                        item_key=row["itemKey"],
                        attachment_key=row["attachmentKey"],
                        title=row["title"] or row["attachmentTitle"] or row["itemKey"],
                        published=_normalize_date(row["dateValue"] or ""),
                        abstract=row["abstractNote"] or "",
                        authors=_authors_for_item(conn, row["itemID"]),
                        pdf_path=pdf_path,
                    )
                )
                if len(matches) >= limit:
                    break
            return matches
        finally:
            conn.close()


def _candidate_rows(conn: sqlite3.Connection, query: str, limit: int) -> list[sqlite3.Row]:
    key_like = query.upper()
    title_like = f"%{query.lower()}%"
    sql = """
    WITH title_data AS (
      SELECT d.itemID, v.value AS title
      FROM itemData d
      JOIN fields f ON f.fieldID = d.fieldID
      JOIN itemDataValues v ON v.valueID = d.valueID
      WHERE f.fieldName = 'title'
    ),
    abstract_data AS (
      SELECT d.itemID, v.value AS abstractNote
      FROM itemData d
      JOIN fields f ON f.fieldID = d.fieldID
      JOIN itemDataValues v ON v.valueID = d.valueID
      WHERE f.fieldName = 'abstractNote'
    ),
    date_data AS (
      SELECT d.itemID, v.value AS dateValue
      FROM itemData d
      JOIN fields f ON f.fieldID = d.fieldID
      JOIN itemDataValues v ON v.valueID = d.valueID
      WHERE f.fieldName = 'date'
    )
    SELECT
      parent.itemID AS itemID,
      parent.key AS itemKey,
      title_data.title AS title,
      abstract_data.abstractNote AS abstractNote,
      date_data.dateValue AS dateValue,
      attachment.key AS attachmentKey,
      attachment_title.title AS attachmentTitle,
      ia.path AS attachmentPath
    FROM itemAttachments ia
    JOIN items attachment ON attachment.itemID = ia.itemID
    LEFT JOIN items parent ON parent.itemID = ia.parentItemID
    LEFT JOIN title_data ON title_data.itemID = COALESCE(parent.itemID, attachment.itemID)
    LEFT JOIN abstract_data ON abstract_data.itemID = COALESCE(parent.itemID, attachment.itemID)
    LEFT JOIN date_data ON date_data.itemID = COALESCE(parent.itemID, attachment.itemID)
    LEFT JOIN title_data AS attachment_title ON attachment_title.itemID = attachment.itemID
    WHERE ia.contentType = 'application/pdf'
      AND (
        parent.key = ? OR attachment.key = ?
        OR lower(COALESCE(title_data.title, '')) LIKE ?
        OR lower(COALESCE(attachment_title.title, '')) LIKE ?
      )
    ORDER BY
      CASE
        WHEN parent.key = ? OR attachment.key = ? THEN 0
        WHEN lower(COALESCE(title_data.title, '')) = lower(?) THEN 1
        ELSE 2
      END,
      COALESCE(date_data.dateValue, '') DESC,
      COALESCE(title_data.title, attachment_title.title, parent.key, attachment.key) ASC
    LIMIT ?
    """
    return conn.execute(
        sql,
        (key_like, key_like, title_like, title_like, key_like, key_like, query, limit),
    ).fetchall()


def _authors_for_item(conn: sqlite3.Connection, item_id: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT c.firstName, c.lastName, c.fieldMode
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        WHERE ic.itemID = ?
        ORDER BY ic.orderIndex
        """,
        (item_id,),
    ).fetchall()
    authors: list[str] = []
    for row in rows:
        if row["fieldMode"] == 1:
            name = row["lastName"] or row["firstName"] or ""
        else:
            first = (row["firstName"] or "").strip()
            last = (row["lastName"] or "").strip()
            name = " ".join(part for part in (first, last) if part)
        if name:
            authors.append(name)
    return authors


def _resolve_pdf_path(data_dir: Path, attachment_key: str, attachment_path: str | None) -> Path:
    if attachment_path and attachment_path.startswith("storage:"):
        filename = attachment_path.split("storage:", 1)[1]
        return data_dir / "storage" / attachment_key / filename
    if attachment_path:
        path = Path(attachment_path).expanduser()
        if path.is_absolute():
            return path
    return data_dir / "storage" / attachment_key


def _normalize_date(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    if len(raw) >= 7 and raw[4] == "-":
        return f"{raw[:7]}-01"
    if len(raw) >= 4 and raw[:4].isdigit():
        return f"{raw[:4]}-01-01"
    return raw


class _snapshot_db:
    def __init__(self, src: Path):
        self.src = src
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="research-helper-zotero-")
        self.path = Path(self._tmpdir.name) / self.src.name
        shutil.copy2(self.src, self.path)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
