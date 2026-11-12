"""Post-hoc source attribution, as in Menon 1.1: the sources of an answer are the context pieces its sentences rest on.

Every sentence of the answer is scored against every evidence unit of the context with the cross-encoder
(bge-reranker-v2-m3, the stand's reranker service); a unit counts when some sentence scores at least the threshold
against it. The shown sources are the links of those units, best first, one row per page.

A unit is a chunk of a context document with the document's link. A document compiled from several pages (the
handbook summaries list many URLs) is split into lines instead, and each line carries the link written in it: the
compilation names each page next to what it took from it, while the document's first URL is usually another page. Such a
line is titled by the text of its link when that names the page ("Режим занятий обучающихся в НГУ", not
"table.nsu.ru"), since the compilation's own title describes all its pages at once. A line without a link of its own
gives no source rather than a wrong one.

A document whose sections come from different pages lists them in its metadata ("sections": heading, url, title; a
teacher's record joins the staff page, the timetable and the publications portal under one timetable link). It is
split into its sections, and each carries the page and title listed for its heading ("Расписание (часть 1/2)" is
listed as "Расписание"); a section listed without a page gives no source.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from razdel import sentenize

from meno_rag.stand.context import normalize_urls

MIN_WORDS = 4
# a Markdown link target may escape its brackets ("persons/--\\(0a94…\\).html"); they belong to the address
_LINK = re.compile(r"\[([^\]]*)\]\((https?://(?:\\[()]|[^)\s\\])+)\)|(https?://[^\s)\]>\"«»]+)")
Scorer = Callable[[str, list[str]], Awaitable[list[float]]]


@dataclass(frozen=True)
class EvidenceUnit:
    text: str
    title: str
    url: str


def _words(text: str) -> int:
    # the text of a Markdown link is part of the sentence, its address is not
    return len(re.findall(r"\w+", _LINK.sub(lambda match: f" {match.group(1) or ''} ", text)))


def split_sentences(answer: str) -> list[str]:
    out: list[str] = []
    for line in answer.splitlines():
        for sentence in sentenize(line.strip()):
            text = sentence.text.strip()
            if text.startswith("_") or text.lower().startswith("источник"):
                continue  # markers appended to the answer, not claims
            if _words(text) >= MIN_WORDS:
                out.append(text)
    return out


def norm_url(url: str) -> str:
    """The page of a link: scheme, "www." and a trailing slash do not matter; a query and an anchor do (the staff
    page of nsu.ru is one document per person, "…/sveden/employees/#Фамилия_Имя_Отчество")."""
    parts = urlsplit((url or "").strip().lower())
    return (f"{parts.netloc.removeprefix('www.')}{parts.path.rstrip('/')}" + (f"?{parts.query}" if parts.query else "")
            + (f"#{parts.fragment}" if parts.fragment else ""))


def _links(text: str) -> list[tuple[str, str]]:
    """The links of a line with the text they are written under ("" for a bare URL)."""
    return [((match.group(2) or match.group(3)).replace("\\(", "(").replace("\\)", ")").rstrip(".,;:!?"),
             " ".join((match.group(1) or "").split()))
            for match in _LINK.finditer(text)]


def _names_a_page(anchor: str) -> bool:
    # "Режим занятий обучающихся в НГУ" names the page; "table.nsu.ru" only repeats its address
    return bool(re.search(r"[А-Яа-яЁё]", anchor)) or " " in anchor


_SECTION = re.compile(r"(?m)^## (.+)$")


def _section_units(document: dict[str, Any], chunk_texts: Sequence[str], title: str, url: str) -> list[EvidenceUnit]:
    sections = [section for section in document["sections"] if section.get("heading")]
    full_text = str(document.get("doc_full_text") or document.get("text") or "")

    def listed(heading: str) -> dict[str, Any] | None:
        fitting = [section for section in sections if heading.strip().startswith(section["heading"])]
        return max(fitting, key=lambda section: len(section["heading"]), default=None)

    units = []
    for text in chunk_texts:
        start = full_text.find(text)
        earlier = _SECTION.findall(full_text[:start]) if start > 0 else []
        current = listed(earlier[-1]) if earlier else None  # a chunk may start inside a section
        blocks: list[tuple[dict[str, Any] | None, list[str]]] = [(current, [])]
        for line in text.splitlines():
            match = _SECTION.match(line)
            if match:
                blocks.append((listed(match.group(1)), []))
            blocks[-1][1].append(line)
        for section, lines in blocks:
            block = "\n".join(lines).strip()
            if _words(block) < MIN_WORDS:
                continue
            if section is None:  # not listed: the document's own page
                units.append(EvidenceUnit(block, title, url))
            else:
                units.append(EvidenceUnit(block, " ".join(str(section.get("title") or title).split()),
                                          str(section.get("url") or "")))
    return units


def document_units(document: dict[str, Any], chunk_texts: Sequence[str]) -> list[EvidenceUnit]:
    title = " ".join(str(document.get("doc_title") or "").split())
    urls = normalize_urls(document.get("url"))
    if document.get("sections"):
        return _section_units(document, chunk_texts, title, urls[0] if len(urls) == 1 else "")
    if len(urls) <= 1:
        return [EvidenceUnit(text, title, urls[0] if urls else "") for text in chunk_texts]
    units = []
    for text in chunk_texts:
        for line in text.splitlines():
            line = line.strip()
            if _words(line) >= MIN_WORDS:
                links = _links(line)
                url, anchor = links[0] if links else ("", "")
                units.append(EvidenceUnit(line, anchor if _names_a_page(anchor) else title, url))
    return units


def _heading(document: dict[str, Any]) -> str:
    first = str(document.get("doc_full_text") or document.get("text") or "").lstrip().split("\n", 1)[0].strip()
    return first[2:].strip() if first.startswith("# ") else ""


def page_titles(documents: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Titles of the pages that are split into several documents, by normalised link.

    A page cut into parts (a teacher's record is one link and over a hundred documents titled after their own
    parts: a publication, the intellectual property) is shown once, so it needs the page's title, not a part's: the
    first-level heading all its parts share ("# Иванов Иван Иванович"), else the title of its first document.
    Compilations of several pages are left out: their lines carry their own links and titles.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        urls = normalize_urls(document.get("url"))
        if len(urls) == 1:
            groups.setdefault(norm_url(urls[0]), []).append(document)
    titles = {}
    for key, parts in groups.items():
        if len(parts) < 2:
            continue
        headings = {_heading(part) for part in parts}
        heading = headings.pop() if len(headings) == 1 else ""
        titles[key] = heading or " ".join(str(parts[0].get("doc_title") or "").split())
    return titles


async def attribute(answer: str, units: Sequence[EvidenceUnit], scorer: Scorer,
                    threshold: float) -> list[tuple[EvidenceUnit, float]]:
    sentences = split_sentences(answer)
    if not sentences or not units:
        return []
    texts = [unit.text for unit in units]
    best = [0.0] * len(units)
    for scores in await asyncio.gather(*(scorer(sentence, texts) for sentence in sentences)):
        for index, score in enumerate(scores):
            best[index] = max(best[index], float(score))
    used = [(unit, score) for unit, score in zip(units, best, strict=True) if score >= threshold]
    used.sort(key=lambda pair: -pair[1])
    return used


def to_sources(scored: Sequence[tuple[EvidenceUnit, float]], max_sources: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for unit, _score in scored:
        key = norm_url(unit.url)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append({"document_title": unit.title, "source_url": unit.url})
        if len(rows) >= max_sources:
            break
    return rows


class RerankerScorer:
    """Scores a sentence against texts with the vLLM ``/score`` endpoint of the stand's reranker."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, model: str, api_key: str, timeout: float = 15.0):
        self.client = client
        self.url = base_url.rstrip("/") + "/score"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    async def __call__(self, sentence: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        response = await self.client.post(self.url, headers=self.headers, timeout=self.timeout,
                                          json={"model": self.model, "text_1": sentence, "text_2": texts})
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return [float(row["score"]) for row in rows]
