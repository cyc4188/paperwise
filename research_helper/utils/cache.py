import json
import hashlib
from pathlib import Path

from research_helper import config


def _paper_cache_dir(paper_dir: Path) -> Path:
    digest = hashlib.md5(str(paper_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return config.cache_path("papers", f"{paper_dir.name}-{digest}")


def _cache_path(paper_dir: Path, key: str) -> Path:
    return _paper_cache_dir(paper_dir) / f"{key}.json"


def load_cache(paper_dir: Path, key: str):
    p = _cache_path(paper_dir, key)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_cache(paper_dir: Path, key: str, data) -> None:
    _paper_cache_dir(paper_dir).mkdir(parents=True, exist_ok=True)
    _cache_path(paper_dir, key).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]
