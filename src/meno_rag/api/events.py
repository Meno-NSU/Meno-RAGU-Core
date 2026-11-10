from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


class StageName:
    ABBREVIATION_EXPANSION = "abbreviation_expansion"
    QUERY_REWRITE = "query_rewrite"
    FEWSHOT_SELECTION = "fewshot_selection"
    RETRIEVAL = "retrieval"
    FUSION = "fusion"
    RERANK = "rerank"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION = "generation"


class StageStatus:
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


MAX_SOURCES = 20
MAX_TITLE_CHARS = 500
MAX_URL_CHARS = 2000
MAX_DETAIL_LINES = 50
MAX_DETAIL_LINE_CHARS = 500


def clean_sources(raw: Any) -> list[dict[str, str]]:
    """Sources as meno-backend/1 asks: at most 20, each with a title and an http(s) URL or "".

    Entries without a title are dropped, a non-http(s) URL becomes "", the order is kept.
    """
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, str]] = []
    for item in raw:
        if len(rows) >= MAX_SOURCES:
            break
        if not isinstance(item, dict):
            continue
        title = item.get("document_title")
        title = "" if title is None else str(title).strip()
        if not title:
            continue
        url = item.get("source_url")
        url = "" if url is None else str(url).strip()
        if not url.lower().startswith(("http://", "https://")) or len(url) > MAX_URL_CHARS:
            url = ""
        row = {"document_title": title[:MAX_TITLE_CHARS], "source_url": url}
        date = item.get("date")
        if date not in (None, ""):
            row["date"] = str(date)
        rows.append(row)
    return rows


def _num(detail: dict[str, Any], key: str) -> int | None:
    value = detail.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def stage_lines(stage: str, detail: dict[str, Any] | None) -> list[str]:
    """Russian display lines for a stage, built from its existing detail fields."""
    detail = detail or {}
    lines: list[str] = []
    if stage == StageName.ABBREVIATION_EXPANSION:
        expanded = detail.get("expanded")
        if (
            isinstance(expanded, str)
            and expanded.strip()
            and expanded.strip() != str(detail.get("original") or "").strip()
        ):
            parts = [part.strip() for part in expanded.strip().splitlines() if part.strip()]
            lines.extend(f"Расшифровано: {part}" if i == 0 else part for i, part in enumerate(parts))
    elif stage == StageName.QUERY_REWRITE:
        queries = detail.get("search_queries")
        if isinstance(queries, list):
            lines.append(f"Поисковых запросов: {len(queries)}")
    elif stage == StageName.RETRIEVAL:
        found = _num(detail, "chunks_found")
        if found is not None:
            lines.append(f"Найдено фрагментов: {found}")
        entities, relations = _num(detail, "entities"), _num(detail, "relations")
        if entities is not None and relations is not None:
            lines.append(f"сущностей: {entities}, связей: {relations}")
    elif stage == StageName.FUSION:
        candidates = _num(detail, "candidates")
        if candidates is not None:
            lines.append(f"Кандидатов: {candidates}")
    elif stage == StageName.RERANK:
        kept, total = _num(detail, "kept"), _num(detail, "from")
        if kept is not None and total is not None:
            lines.append(f"Оставлено {kept} из {total}")
    elif stage == StageName.CONTEXT_ASSEMBLY:
        sources, tokens = _num(detail, "sources"), _num(detail, "context_tokens")
        if sources is not None and tokens is not None:
            lines.append(f"Источников: {sources}, токенов контекста: {tokens}")
    return [line[:MAX_DETAIL_LINE_CHARS] for line in lines[:MAX_DETAIL_LINES]]


@dataclass
class StageEvent:
    stage: str
    status: str
    ts: float = field(default_factory=time.time)
    duration_ms: float | None = None
    detail: dict[str, Any] | None = None
    model_id: str | None = None

    def to_sse(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        payload["detail"] = {**(payload.get("detail") or {}), "lines": stage_lines(self.stage, self.detail)}
        return sse_event("stage", payload)


@dataclass
class StageSummary:
    total_ms: float
    stages: dict[str, float]

    def to_sse(self) -> str:
        return sse_event("summary", asdict(self))


def sse_event(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_data(payload: Any) -> str:
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def openai_chunk(
    *, completion_id: str, created: int, model: str, delta: dict[str, Any], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
