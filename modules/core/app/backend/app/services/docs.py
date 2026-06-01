"""Loads workflow documentation markdown files at import time. The files
ship next to the backend in `backend/documentation/`."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "documentation"


@dataclass(frozen=True)
class DocEntry:
    file: str
    title: str
    content: str


@lru_cache(maxsize=1)
def list_documents() -> tuple[DocEntry, ...]:
    if not DOCS_DIR.exists():
        return ()
    entries: list[DocEntry] = []
    for md_path in sorted(DOCS_DIR.glob("*.md")):
        if md_path.name == "index.md":
            continue
        content = md_path.read_text()
        title = md_path.stem.replace("_", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
        entries.append(DocEntry(file=md_path.name, title=title, content=content))
    return tuple(entries)
