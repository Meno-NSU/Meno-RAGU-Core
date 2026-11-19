"""Gated, in-process ABBA timing experiment for the candidate RAGU implementation.

Loads the graph once. The four phases are A(X), B(X), B(Y), A(Y), with eight
standalone questions in each half and four concurrent turns. Output contains
public case IDs, hashes, counts and timings. Model requests begin only after
the explicitly supplied start-gate file exists.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import signal
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from meno_rag.api.benchmark_config import runtime_identity
from meno_rag.config import Settings
from meno_rag.llm.client import VLLMClient, endpoint_api_key
from meno_rag.llm.router import LLMRouter
from meno_rag.logging_config import configure_logging
from meno_rag.schemas import ChatMessage
from meno_rag.stand import attribution as attribution_module
from meno_rag.stand import pipeline as pipeline_module
from meno_rag.stand import ragu_pipeline as ragu_module
from meno_rag.stand.attribution import RerankerScorer, page_titles
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.ragu_pipeline import RaguRagPipeline
from meno_rag.stand.resources import load_stand_resources

MODEL = "gemma-4-12B-it"
CONCURRENCY = 4
SELECTION_SEED = 1120
REQUEST_COUNTS: ContextVar[Counter | None] = ContextVar("abba_request_counts", default=None)


class PhaseStopped(RuntimeError):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def progress(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "utc": now(), **fields}, ensure_ascii=False), flush=True)


def select_questions(items_path: Path, ids_path: Path) -> list[dict]:
    rows = {}
    with items_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError("Duplicate item ID")
            rows[row["id"]] = row
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if isinstance(ids, dict):
        ids = ids["ids"]
    if not isinstance(ids, list) or any(not isinstance(case_id, str) for case_id in ids):
        raise ValueError("The IDs input must contain a JSON array of strings")
    eligible = []
    seen = set()
    for case_id in ids:
        if case_id in seen:
            continue
        seen.add(case_id)
        row = rows[case_id]
        history = row.get("history")
        if history not in (None, [], "", "[]"):
            continue
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("An eligible item has an empty question")
        eligible.append({"id": case_id, "question": question})
    if len(eligible) < 16:
        raise ValueError("At least 16 standalone items are required")
    return random.Random(SELECTION_SEED).sample(eligible, 16)


def numbers(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    if isinstance(value, dict):
        return {key: result for key, item in value.items() if (result := numbers(item)) is not None}
    if isinstance(value, list):
        return [result for item in value if (result := numbers(item)) is not None]
    return None


def aggregates(records: list[dict]) -> dict:
    result = {}
    for key in ("prepare_ms", "ttft_ms", "generation_ms", "source_tail_ms", "total_ms"):
        values = sorted(row[key] for row in records if row.get("status") == "ok" and row.get(key) is not None)
        if not values:
            continue
        position = (len(values) - 1) * 0.95
        lower, upper = math.floor(position), math.ceil(position)
        result[key] = {"n": len(values), "median": round(statistics.median(values), 3),
                       "p95": round(values[lower] + (values[upper] - values[lower]) * (position - lower), 3)}
    return result


async def count_request(request: httpx.Request) -> None:
    counts = REQUEST_COUNTS.get()
    if counts is None:
        return
    path = request.url.path.rstrip("/")
    category = path.rsplit("/", 1)[-1]
    if path.endswith("/chat/completions"):
        payload = json.loads(request.content)
        if payload.get("stream"):
            category = "generation"
        elif payload.get("max_tokens") == 1 and payload.get("logprobs"):
            category = "rerank"
        elif payload.get("max_tokens") == 20 and payload.get("response_format"):
            category = "rerank_json_fallback"
        else:
            category = "analysis_or_rewrite"
    counts[category] += 1


async def require_gemma(http: httpx.AsyncClient, base_url: str, settings: Settings) -> dict:
    key = endpoint_api_key(base_url, settings.openai_api_key, settings.vllm_endpoint_api_keys)
    for attempt in range(3):
        try:
            response = await http.get(base_url.rstrip("/") + "/models",
                                      headers={"Authorization": f"Bearer {key}"}, timeout=10)
            response.raise_for_status()
            data = response.json()["data"]
            if not isinstance(data, list) or any(not isinstance(entry, dict) for entry in data):
                raise ValueError("Malformed model registry")
            model = next((entry for entry in data if entry.get("id") == MODEL), None)
            if model is not None:
                return {"checks": attempt + 1, "context_length": numbers(model.get("max_model_len"))}
            progress("model_check_failed", attempt=attempt + 1, error_type="RequiredModelAbsent")
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            progress("model_check_failed", attempt=attempt + 1, error_type=type(exc).__name__)
        if attempt < 2:
            await asyncio.sleep(2)
    raise PhaseStopped("The required Gemma model is unavailable")


async def measure_case(pipeline, runtime, case, optimized, max_tokens) -> dict:
    started = time.perf_counter()
    counts = Counter()
    token = REQUEST_COUNTS.set(counts)
    attribution = None
    stream = None
    record = {"id": case["id"], "question_sha256": digest(case["question"]), "status": "error"}
    try:
        outcome = await pipeline.prepare(messages=[ChatMessage(role="user", content=case["question"])],
                                         runtime=runtime, capture_trace=False, generation_max_tokens=max_tokens)
        prepared = time.perf_counter()
        attribution = pipeline.start_source_attribution(outcome) if optimized else None
        pieces = []
        answer = ""
        first = None
        stream = pipeline.stream_text(outcome=outcome, runtime=runtime, max_tokens=max_tokens, temperature=None)
        async for piece in stream:
            if piece and first is None:
                first = time.perf_counter()
            pieces.append(piece)
            if attribution is not None:
                answer += piece
                attribution.feed(answer)
        generated = time.perf_counter()
        answer = "".join(pieces)
        if not answer.strip():
            raise ValueError("The model returned an empty answer")
        if attribution is None:
            await pipeline.finalize_sources(outcome, answer)
        else:
            await pipeline.finalize_sources(outcome, answer, streaming=attribution)
        finished = time.perf_counter()
        record.update(status="ok", prepare_ms=round((prepared - started) * 1000, 3),
                      ttft_ms=round((first - started) * 1000, 3) if first is not None else None,
                      generation_ms=round((generated - prepared) * 1000, 3),
                      source_tail_ms=round((finished - generated) * 1000, 3),
                      total_ms=round((finished - started) * 1000, 3),
                      answer_chars=len(answer), context_chars=len(outcome.context), sources_count=len(outcome.sources),
                      evidence_units=len(outcome.evidence), retrieved_count=len(outcome.retrieved),
                      answer_sha256=digest(answer), context_sha256=digest(outcome.context),
                      sources_sha256=digest(outcome.sources), evidence_sha256=digest(outcome.evidence),
                      search_queries_sha256=digest(outcome.search_queries),
                      stages_ms=numbers(outcome.stage_durations_ms),
                      stage_timings={stage: numbers(detail["timings"]) for stage, detail in outcome.stage_details.items()
                                     if "timings" in detail})
        attribution_failed = outcome.stage_details.get("source_attribution", {}).get("timings", {}).get("failed")
        if attribution_failed:
            record.update(status="error", error_type="SourceAttributionFailed")
    except Exception as exc:
        record.update(error_type=type(exc).__name__, total_ms=round((time.perf_counter() - started) * 1000, 3))
    finally:
        if stream is not None:
            await stream.aclose()
        if attribution is not None:
            await attribution.aclose()
        record["http_requests"] = dict(counts)
        REQUEST_COUNTS.reset(token)
    return record


async def run_phase(name, optimized, cases, settings, resources, router, scorer, runtime,
                    hub_index, titles, document, output):
    resources.graph.engines.local_engine.hub_index = hub_index if optimized else None
    phase_settings = settings.model_copy(update={"early_first_query": optimized, "rerank_priority": int(optimized),
                                                 "graph_hub_min_edges": 5000 if optimized else 0})
    executor = ThreadPoolExecutor(max_workers=settings.retrieval_executor_max_workers or 4)
    pipeline = RaguRagPipeline(
        settings=phase_settings, resources=resources, llm_router=router,
        rewrite_semaphore=asyncio.Semaphore(settings.rewrite_concurrency),
        rerank_semaphore=asyncio.Semaphore(settings.rerank_concurrency),
        generation_semaphore=asyncio.Semaphore(settings.generation_concurrency),
        embed_semaphore=asyncio.Semaphore(settings.embed_concurrency),
        bm25_semaphore=asyncio.Semaphore(settings.bm25_concurrency),
        retrieval_executor=executor, source_scorer=scorer)
    pipeline.page_titles = titles
    phase = {"name": name, "arm": "B" if optimized else "A", "status": "running", "started_utc": now(),
             "requested": len(cases), "early_first_query": optimized, "rerank_priority": int(optimized),
             "streamed_attribution": optimized, "hub_nodes": len(hub_index) if optimized and hub_index else 0,
             "cases": []}
    document["phases"].append(phase)
    atomic_write(output, document)
    progress("phase_started", phase=name, items=len(cases))
    admission = asyncio.Semaphore(CONCURRENCY)
    bad = 0
    started = time.perf_counter()

    async def one(case):
        nonlocal bad
        async with admission:
            if bad >= 3:
                raise PhaseStopped("The phase failure limit has already been reached")
            record = await measure_case(pipeline, runtime, case, optimized,
                                        max(settings.max_output_tokens, settings.min_output_tokens))
            phase["cases"].append(record)
            phase["aggregates"] = aggregates(phase["cases"])
            atomic_write(output, document)
            progress("case_finished", phase=name, id=case["id"], status=record["status"], total_ms=record["total_ms"])
            if record["status"] != "ok":
                bad += 1
                if bad >= 3:
                    raise PhaseStopped("Three turns failed in one phase")

    stopped = False
    completed_normally = False
    try:
        try:
            async with asyncio.TaskGroup() as group:
                for case in cases:
                    group.create_task(one(case))
        except* PhaseStopped:
            stopped = True
        completed_normally = not stopped
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        phase.update(status="complete" if completed_normally else "stopped", elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                     completed=len(phase["cases"]), errors=bad, aggregates=aggregates(phase["cases"]))
        atomic_write(output, document)
    progress("phase_finished", phase=name, status=phase["status"], completed=phase["completed"], errors=bad)
    if stopped:
        raise PhaseStopped("A phase reached the failure limit")


def paired_results(phases: list[dict]) -> list[dict]:
    arms = {"A": {}, "B": {}}
    for phase in phases:
        arms[phase["arm"]].update({row["id"]: row for row in phase["cases"] if row["status"] == "ok"})
    pairs = []
    for case_id in sorted(arms["A"].keys() & arms["B"].keys()):
        a, b = arms["A"][case_id], arms["B"][case_id]
        pairs.append({"id": case_id, "context_equal": a["context_sha256"] == b["context_sha256"],
                      "sources_equal": a["sources_sha256"] == b["sources_sha256"],
                      "answer_equal": a["answer_sha256"] == b["answer_sha256"],
                      "queries_equal": a["search_queries_sha256"] == b["search_queries_sha256"],
                      "total_delta_ms": round(b["total_ms"] - a["total_ms"], 3),
                      "ttft_delta_ms": round(b["ttft_ms"] - a["ttft_ms"], 3)
                      if a["ttft_ms"] is not None and b["ttft_ms"] is not None else None})
    return pairs


async def main(args) -> int:
    task = asyncio.current_task()
    if task is not None:
        with contextlib.suppress(NotImplementedError):
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    configure_logging("CRITICAL", content_previews=False)
    logging.disable(logging.CRITICAL)
    with contextlib.suppress(ImportError):
        from loguru import logger
        logger.remove()
    os.environ["TQDM_DISABLE"] = "1"
    cases = select_questions(args.items, args.ids)
    settings = Settings(_env_file=None).model_copy(update={
        "graph_hub_min_edges": 5000, "max_context_chunks": 16, "context_style": "sources",
        "rerank_prompt_order": "legacy", "qa_fewshots_enabled": False, "capture_pipeline_trace": False,
        "log_content_previews": False})
    if settings.stand_resources_dir.name != "stand-v6k3b":
        raise ValueError("This experiment requires stand-v6k3b")
    base_url = settings.vllm_endpoint_list[0].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    document = {"schema_version": 1, "status": "loading", "created_utc": now(),
                "model": MODEL, "concurrency": CONCURRENCY, "max_context_chunks": 16,
                "selection_seed": SELECTION_SEED,
                "rerank_prompt_order": "legacy", "halves": {"X": [row["id"] for row in cases[:8]],
                                                             "Y": [row["id"] for row in cases[8:]]},
                "items_sha256": file_digest(args.items), "ids_sha256": file_digest(args.ids),
                "driver_sha256": file_digest(Path(__file__)), "settings_sha256": digest(runtime_identity(settings)),
                "manifest_sha256": file_digest(settings.stand_resources_dir / "build_manifest.json"),
                "code_sha256": digest({module.__name__: file_digest(Path(module.__file__))
                                      for module in (pipeline_module, ragu_module, attribution_module)}),
                "phases": []}
    atomic_write(args.output, document)
    resources = None
    try:
        progress("loading_resources")
        resources = await asyncio.to_thread(load_stand_resources, settings)
        hub_index = resources.graph.engines.local_engine.hub_index
        titles = await asyncio.to_thread(page_titles, resources.documents)
        loaded = ("meno_rag.config", "meno_rag.stand.resources", "graph_utils.graph_utils", "graph_utils.local_search")
        document.update(status="waiting_gate", hub_nodes=len(hub_index) if hub_index else 0,
                        loaded_code_sha256=digest({name: file_digest(Path(sys.modules[name].__file__))
                                                  for name in loaded if name in sys.modules}))
        atomic_write(args.output, document)
        progress("ready", hub_nodes=document["hub_nodes"], items=16, waiting_for_gate=not args.start_gate.exists())
        while not args.start_gate.is_file():  # noqa: ASYNC110 - another process controls this file
            await asyncio.sleep(2)
        document["status"] = "running"
        atomic_write(args.output, document)
        runtime = PipelineRuntime.uniform(ModelRuntime(MODEL, base_url))
        limits = httpx.Limits(max_connections=settings.httpx_max_connections,
                              max_keepalive_connections=settings.httpx_max_keepalive)
        async with httpx.AsyncClient(timeout=600, limits=limits, event_hooks={"request": [count_request]}) as http:
            router = LLMRouter(vllm=VLLMClient(http_client=http, api_key=settings.openai_api_key,
                                            endpoint_api_keys=settings.vllm_endpoint_api_keys), openrouter=None)
            scorer = RerankerScorer(http, settings.engine_reranker_url, settings.engine_reranker_model,
                                   settings.engine_reranker_api_key)
            for name, optimized, half in [("A_X", False, cases[:8]), ("B_X", True, cases[:8]),
                                          ("B_Y", True, cases[8:]), ("A_Y", False, cases[8:])]:
                check = await require_gemma(http, base_url, settings)
                await run_phase(name, optimized, half, settings, resources, router, scorer, runtime,
                                hub_index, titles, document, args.output)
                document["phases"][-1]["model_check"] = check
                atomic_write(args.output, document)
        document.update(status="complete", finished_utc=now(), pairs=paired_results(document["phases"]))
        atomic_write(args.output, document)
        progress("complete", pairs=len(document["pairs"]))
        return 0
    except BaseException as exc:
        document.update(status="stopped", error_type=type(exc).__name__, finished_utc=now(),
                        pairs=paired_results(document["phases"]))
        atomic_write(args.output, document)
        progress("stopped", error_type=type(exc).__name__)
        raise
    finally:
        if resources is not None:
            await resources.graph.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-gate", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(main(arguments)))
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as error:
        progress("exit", error_type=type(error).__name__)
        sys.exit(1)
