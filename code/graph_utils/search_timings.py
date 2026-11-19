"""Where the time of one graph search goes: the embedder, Qdrant, the graph code and the cross-encoder.

Recording only. Every wrapper calls the wrapped method with the same arguments and returns its result unchanged; the
numbers go to the dict of the search that is running. A search is one task on the engines' event loop
(GraphSearchEngines.asearch) and the tasks it starts (asyncio.gather) inherit its context, so a ContextVar holding the
search's dict collects the calls of that search only, while other searches run on the same loop. Outside a search
(engine loading, scripts) nothing is recorded.

Sums of call durations, not wall time: the local and naive channels of a query call the services concurrently.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Any

_CURRENT: ContextVar[dict[str, Any] | None] = ContextVar('graph_search_timings', default=None)
_WRAPPED = '_meno_search_timing'


def new_timings() -> dict[str, Any]:
    return {'search_ms': 0.0, 'embed_ms': 0.0, 'embed_calls': 0, 'embed_texts': 0, 'qdrant_ms': 0.0,
            'qdrant_calls': 0, 'graph_cpu_ms': 0.0, 'ce_ms': 0.0, 'ce_calls': 0, 'ce_pairs': 0,
            'ce_pairs_by_kind': {}}


@contextlib.contextmanager
def collecting(timings: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Collect the calls made in this context (and in the tasks it starts) into ``timings``."""
    token = _CURRENT.set(timings)
    try:
        yield timings
    finally:
        _CURRENT.reset(token)


def record(ms_key: str, started: float, calls_key: str | None = None, **counts: int) -> None:
    """Add the time since ``started`` (``time.perf_counter()``) and the counts to the running search, if any."""
    timings = _CURRENT.get()
    if timings is None:
        return
    timings[ms_key] = timings.get(ms_key, 0.0) + (time.perf_counter() - started) * 1000
    if calls_key is not None:
        timings[calls_key] = timings.get(calls_key, 0) + 1
    for key, value in counts.items():
        timings[key] = timings.get(key, 0) + value


def record_pairs(kind: str, pairs: int) -> None:
    """Cross-encoder pairs of one kind of candidate (entity, relation, summary, chunk) for the running search."""
    timings = _CURRENT.get()
    if timings is None:
        return
    by_kind = timings.setdefault('ce_pairs_by_kind', {})
    by_kind[kind] = by_kind.get(kind, 0) + pairs


def instrument(obj: Any, name: str, ms_key: str, calls_key: str,
               counts: Callable[[tuple, dict], dict[str, int]] | None = None) -> None:
    """Time every call of the async method ``obj.<name>`` (wrapped on the instance, once)."""
    original = getattr(obj, name, None)
    if original is None or getattr(original, _WRAPPED, False):
        return

    async def timed(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original(*args, **kwargs)
        finally:
            record(ms_key, started, calls_key, **(counts(args, kwargs) if counts is not None else {}))

    setattr(timed, _WRAPPED, True)
    setattr(obj, name, timed)


def _length_of(position: int, name: str, key: str) -> Callable[[tuple, dict], dict[str, int]]:
    """Counts the items of the call argument ``name`` (or the positional one at ``position``) under ``key``."""
    def counts(args: tuple, kwargs: dict) -> dict[str, int]:
        items = kwargs[name] if name in kwargs else (args[position] if len(args) > position else None)
        return {key: len(items) if items is not None else 0}
    return counts


def instrument_embedder(embedder: Any) -> None:
    """The dense query embeddings (the retrievers call ``batch_embed_text``; ``embed_text`` covers the probe)."""
    instrument(embedder, 'batch_embed_text', 'embed_ms', 'embed_calls', _length_of(0, 'texts', 'embed_texts'))
    instrument(embedder, 'embed_text', 'embed_ms', 'embed_calls', lambda args, kwargs: {'embed_texts': 1})


def instrument_storage(storage: Any) -> None:
    """The Qdrant queries of a vector storage."""
    instrument(storage, 'query', 'qdrant_ms', 'qdrant_calls')


def instrument_scorer(scorer: Any) -> None:
    """The cross-encoder: ``score(text_1, text_2)`` is one call of ``len(text_2)`` pairs."""
    instrument(scorer, 'score', 'ce_ms', 'ce_calls', _length_of(1, 'text_2', 'ce_pairs'))


def finished(timings: dict[str, Any], started: float) -> dict[str, Any]:
    """The search's dict as reported: its wall time, the naive channel's pairs (the pairs the local engine did not
    tag), milliseconds rounded."""
    timings['search_ms'] = (time.perf_counter() - started) * 1000
    by_kind = timings.setdefault('ce_pairs_by_kind', {})
    naive = timings.get('ce_pairs', 0) - sum(by_kind.values())
    if naive > 0:
        by_kind['chunk_naive'] = naive
    return {key: round(value, 2) if isinstance(value, float) else value for key, value in timings.items()}
