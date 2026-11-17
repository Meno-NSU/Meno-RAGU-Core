"""One authority scale for the stand corpus: the quality of the Menon KB.

The KB v3.8 documents carry the KB's quality (posts about .55, NSU news .45-.6, NSU pages .8-.9). The graph corpus of
NSU v1 carries its own, squeezed into .70-.80 (its VK posts .77, its nsu.ru pages .73, its Pure profiles .75), so under
the source prior its posts outranked official KB pages. ``QualityScale`` puts a document on the KB's scale: the KB's
quality of the same page; else the KB's median quality of the nearest section of the site (URL directory) that holds
at least ``min_documents`` KB documents; else the document keeps its own.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit


def _parts(url: str) -> tuple[str, list[str]]:
    split = urlsplit((url or "").strip().lower())
    return split.netloc.removeprefix("www."), [part for part in split.path.split("/") if part]


def page_key(url: str) -> str:
    host, segments = _parts(url)
    return "/".join([host, *segments]) if host else ""


def section_keys(url: str) -> list[str]:
    """The sections (URL directories) a page lies in, the nearest first, the site last."""
    host, segments = _parts(url)
    if not host:
        return []
    return ["/".join([host, *segments[:depth]]) for depth in range(len(segments) - 1, -1, -1)]


@dataclass(frozen=True)
class QualityScale:
    pages: dict[str, float]
    sections: dict[str, float]

    @classmethod
    def from_kb(cls, rows: Iterable[tuple[str, float]], min_documents: int = 30) -> QualityScale:
        """``rows``: (url, quality) of the KB documents, one row per URL of a document."""
        pages: dict[str, float] = {}
        members: dict[str, list[float]] = {}
        for url, quality in rows:
            key = page_key(url)
            if not key:
                continue
            pages[key] = float(quality)
            for section in section_keys(url):
                members.setdefault(section, []).append(float(quality))
        sections = {key: float(statistics.median(values)) for key, values in members.items()
                    if len(values) >= min_documents}
        return cls(pages, sections)

    def quality_of(self, url: str, own: float) -> float:
        key = page_key(url)
        if key in self.pages:
            return self.pages[key]
        for section in section_keys(url):
            if section in self.sections:
                return self.sections[section]
        return own
