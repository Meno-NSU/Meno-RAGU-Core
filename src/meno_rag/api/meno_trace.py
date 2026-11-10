"""X-Meno-Trace: 1 (meno-backend/1 section 3): the `contexts` and `pipeline_trace` payloads."""

from __future__ import annotations

import re
from typing import Any

from fastapi import Request

from meno_rag.api.events import clean_sources

_DOC_BLOCK = re.compile(r"={10}\nDOCUMENT \d+\n={10}\n\n")


def wants_meno_trace(request: Request) -> bool:
    return request.headers.get("x-meno-trace", "").strip() == "1"


def _first_valid_source(evidence: dict[str, Any]) -> dict[str, str]:
    rows = clean_sources(evidence.get("sources") or [])
    return rows[0] if rows else {"document_title": "", "source_url": ""}


def build_contexts(outcome: Any) -> list[dict[str, Any]]:
    """The context blocks the answer was built on, in the order they were used."""
    trace = getattr(outcome, "trace", None) or {}
    contexts: list[dict[str, Any]] = []
    for evidence in trace.get("final_evidence") or []:
        text = str(evidence.get("text") or "")
        text = _DOC_BLOCK.sub("", text, count=1).strip()
        if not text:
            continue
        source = _first_valid_source(evidence)
        rank = len(contexts) + 1
        doc_index = evidence.get("document_index")
        contexts.append(
            {
                "id": str(doc_index) if doc_index is not None else f"ctx-{rank}",
                "url": source["source_url"],
                "title": source["document_title"],
                "text": text,
                "rank": rank,
            }
        )
    if contexts:
        return contexts
    pieces = [p.strip() for p in _DOC_BLOCK.split(getattr(outcome, "context", "") or "")]
    return [
        {"id": f"ctx-{i}", "url": "", "title": "", "text": piece, "rank": i}
        for i, piece in enumerate((p for p in pieces if p), start=1)
    ]


def build_pipeline_trace(outcome: Any) -> dict[str, Any] | None:
    """RAGU's own trace, with the keys the gateway understands added on top."""
    own = getattr(outcome, "trace", None)
    if own is None:
        return None
    prompt = own.get("prompt") or {}
    runtime = own.get("runtime") or {}
    messages = getattr(outcome, "qa_messages", None) or []
    system = prompt.get("system") or next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = prompt.get("user") or next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    return {
        **own,
        "search_queries": list(getattr(outcome, "search_queries", None) or own.get("search_queries") or []),
        "system_prompt": system,
        "user_prompt": user,
        "dialogue_history": getattr(outcome, "prepared_dialogue_history", "") or "",
        "retrieved": getattr(outcome, "retrieved", None) or [],
        "fewshots": getattr(outcome, "fewshots", None) or [],
        "stage_details": getattr(outcome, "stage_details", None) or own.get("stage_details") or {},
        "generation_model": (runtime.get("generation") or {}).get("model_id", ""),
        "core_model": (runtime.get("core") or {}).get("model_id", ""),
    }
