import json
import hashlib
from pathlib import Path


def _cache_path(paper_dir: Path, key: str) -> Path:
    return paper_dir / f".cache_{key}.json"


def load_cache(paper_dir: Path, key: str):
    p = _cache_path(paper_dir, key)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_cache(paper_dir: Path, key: str, data) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(paper_dir, key).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]
