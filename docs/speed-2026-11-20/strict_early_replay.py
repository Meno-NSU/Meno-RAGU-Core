"""Closed recorded-LLM replay of EARLY_FIRST_QUERY on identical retrieved batches.

Loads one graph. Each baseline query runs retrieval once, and its exact batch is
deep-copied for the early arm. No missing LLM request is forwarded to a model.
Fixed time, previous dialogue answers and attribution scores isolate scheduling.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import httpx

import meno_rag.stand.pipeline as pipeline_module
from meno_rag.config import Settings
from meno_rag.llm.client import VLLMClient
from meno_rag.llm.router import LLMRouter
from meno_rag.schemas import ChatMessage
from meno_rag.stand.attribution import RerankerScorer
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.ragu_pipeline import RaguRagPipeline
from meno_rag.stand.resources import load_stand_resources


class ReplayMissing(BaseException):
    """A missing fixture must escape the pipeline's normal model fallback."""


class FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 11, 19)


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 19, 23, 0, 0, tzinfo=tz)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def key(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def log(*values):
    print(datetime.utcnow().isoformat(timespec="seconds"), *values, flush=True)


def read_rows(path):
    return {r.get("id", r.get("item_id")): r for r in map(json.loads, path.open())}


def messages_of(item, recorded):
    history = item.get("history") or []
    if isinstance(history, str):
        import ast
        history = ast.literal_eval(history)
    if history and all(t["role"] == "user" for t in history):
        answers = recorded.get("inproc", {}).get("replay_answers") or []
        if len(history) != len(answers):
            raise ValueError("Recorded assistant history does not cover every prior turn")
        result = []
        for turn, answer in zip(history, answers, strict=True):
            result.extend([{"role": "user", "content": turn["content"]},
                           {"role": "assistant", "content": answer}])
    else:
        result = copy.deepcopy(history)
    result.append({"role": "user", "content": item["question"]})
    return [ChatMessage(**m) for m in result]


class ReplayTransport(httpx.AsyncBaseTransport):
    def __init__(self, fixtures):
        self.fixtures = fixtures
        self.calls = Counter()
        self.used = Counter()
        self.misses = []

    def reset(self):
        self.calls.clear()
        self.used.clear()
        self.misses.clear()

    async def handle_async_request(self, request):
        if request.url.path.endswith("/tokenize"):
            raise ReplayMissing("Tokenization fixture unavailable for oversized prompt")
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "gemma-4-12B-it", "max_model_len": 60000}]})
        body = json.loads(request.content)
        signature = key(body)
        self.calls[signature] += 1
        fixtures = self.fixtures.get(signature)
        if not fixtures:
            self.misses.append({"sha256": signature, "max_tokens": body.get("max_tokens"),
                                "messages_sha256": key(body.get("messages"))})
            raise ReplayMissing(f"Missing recorded LLM request {signature}")
        index = min(self.used[signature], len(fixtures) - 1)
        self.used[signature] += 1
        await asyncio.sleep(0)
        return httpx.Response(200, json=copy.deepcopy(fixtures[index]))


class PairedGraph:
    def __init__(self, graph):
        self.graph = graph
        self.phase = "baseline"
        self.cache = {}
        self.calls = []

    def reset(self):
        self.cache.clear()
        self.calls.clear()

    async def search(self, query):
        self.calls.append(query)
        if self.phase == "baseline":
            if query not in self.cache:
                self.cache[query] = await self.graph.search(query)
        elif query not in self.cache:
            raise ReplayMissing(f"Early arm requested an unrecorded search query {query!r}")
        await asyncio.sleep(0)
        return copy.deepcopy(self.cache[query])


class PairedScorer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.phase = "baseline"
        self.cache = {}

    async def __call__(self, sentence, texts):
        signature = key([sentence, texts])
        if signature not in self.cache:
            if self.phase != "baseline":
                raise ReplayMissing("Early arm changed an attribution scoring input")
            self.cache[signature] = await self.scorer(sentence, texts)
        return copy.deepcopy(self.cache[signature])


def outcome_signature(outcome):
    trace = outcome.trace or {}
    return {
        "search_queries": outcome.search_queries,
        "context": outcome.context,
        "qa_messages": outcome.qa_messages,
        "prepared_dialogue_history": outcome.prepared_dialogue_history,
        "retrieved": outcome.retrieved,
        "sources": outcome.sources,
        "evidence": outcome.evidence,
        "final_evidence": trace.get("final_evidence"),
        "graph_selected": (trace.get("graph") or {}).get("selected_elements"),
        "rerank": trace.get("rerank"),
    }


async def main(args):
    root = Path(args.root)
    ids = json.loads((root / "p1/ids.json").read_text())[:args.limit]
    items = read_rows(root / "union1101.jsonl")
    recorded = read_rows(root / "answers/v6k3b16.records.jsonl")
    wanted = {items[i]["question"] for i in ids}
    wanted.update(recorded[i].get("v2", {}).get("standalone_question", "") for i in ids)
    needles = [json.dumps(q, ensure_ascii=False)[1:-1] for q in wanted if q]
    fixtures = defaultdict(list)
    captured = kept = 0
    started = time.perf_counter()
    with (root / "capture_v6k3b.jsonl").open() as handle:
        for line in handle:
            captured += 1
            if '"max_tokens": 1,' in line and not any(q in line for q in needles):
                continue
            row = json.loads(line)
            request = row.get("request") or {}
            if row.get("status") != 200 or "response" not in row:
                continue
            if request.get("max_tokens") == 1:
                text = "\n".join(m.get("content", "") for m in request.get("messages", []))
                if not any(q in text for q in wanted if q):
                    continue
            elif int(request.get("max_tokens") or 0) > 1024:
                continue
            fixtures[key(request)].append(row["response"])
            kept += 1
    log("fixtures", captured, "scanned", kept, "kept", len(fixtures), "unique", "seconds", round(time.perf_counter()-started, 2))
    settings = Settings(_env_file=None)
    settings.early_first_query = False
    pipeline_module.date = FrozenDate
    pipeline_module.datetime = FrozenDatetime
    resources = load_stand_resources(settings)
    graph = PairedGraph(resources.graph)
    resources = replace(resources, graph=graph)
    log("resources loaded")
    transport = ReplayTransport(fixtures)
    replay_http = httpx.AsyncClient(transport=transport, timeout=120)
    live_http = httpx.AsyncClient(timeout=60)
    router = LLMRouter(vllm=VLLMClient(http_client=replay_http), openrouter=None)
    scorer = PairedScorer(RerankerScorer(live_http, settings.engine_reranker_url,
                                       settings.engine_reranker_model, settings.engine_reranker_api_key))
    runtime = PipelineRuntime.uniform(ModelRuntime("gemma-4-12B-it", "http://closed-replay/v1"))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for item_id in ids:
        item_started = time.perf_counter()
        graph.reset()
        scorer.cache.clear()
        signatures = []
        request_sets = []
        retrieval_sets = []
        result = {"id": item_id}
        pipelines = []
        try:
            messages = messages_of(items[item_id], recorded[item_id])
            for enabled in (False, True):
                phase = "early" if enabled else "baseline"
                graph.phase = scorer.phase = phase
                graph.calls.clear()
                transport.reset()
                arm_settings = settings.model_copy(update={"early_first_query": enabled})
                p = RaguRagPipeline(
                    settings=arm_settings, resources=resources, llm_router=router,
                    rewrite_semaphore=asyncio.Semaphore(settings.rewrite_concurrency),
                    rerank_semaphore=asyncio.Semaphore(settings.rerank_concurrency),
                    generation_semaphore=asyncio.Semaphore(settings.generation_concurrency),
                    embed_semaphore=asyncio.Semaphore(settings.embed_concurrency),
                    bm25_semaphore=asyncio.Semaphore(settings.bm25_concurrency),
                    retrieval_executor=ThreadPoolExecutor(max_workers=4), source_scorer=scorer)
                pipelines.append(p)
                p._context_lengths = {(runtime.core.base_url, runtime.core.model_id): 60000}
                outcome = await p.prepare(messages=copy.deepcopy(messages), runtime=runtime,
                                          capture_trace=True, generation_max_tokens=max(settings.max_output_tokens, settings.min_output_tokens))
                await p.finalize_sources(outcome, recorded[item_id]["response"])
                signatures.append(outcome_signature(outcome))
                request_sets.append(transport.calls.copy())
                retrieval_sets.append(Counter(graph.calls))
                result[phase + "_llm_calls"] = sum(transport.calls.values())
            differences = [k for k in signatures[0] if signatures[0][k] != signatures[1][k]]
            if request_sets[0] != request_sets[1]:
                differences.append("llm_request_multiset")
                result["request_multiset_extra"] = dict(request_sets[1] - request_sets[0])
                result["request_multiset_missing"] = dict(request_sets[0] - request_sets[1])
            if retrieval_sets[0] != retrieval_sets[1]:
                differences.append("retrieval_query_multiset")
            result.update(status="different" if differences else "identical", differences=differences)
            result["signature_sha256"] = [key(s) for s in signatures]
        except ReplayMissing as error:
            result.update(status="missing_fixture", reason=str(error), misses=copy.deepcopy(transport.misses))
        except Exception as error:
            result.update(status="error", reason=type(error).__name__ + ": " + str(error))
        finally:
            for p in pipelines:
                p.retrieval_executor.shutdown(wait=False, cancel_futures=True)
        result["elapsed_s"] = round(time.perf_counter() - item_started, 2)
        results.append(result)
        with output.open("a") as handle:  # noqa: ASYNC230 - small receipt append outside measured turn execution
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        log(item_id, result["status"], result.get("differences", []), result["elapsed_s"])
    summary = {"items": len(ids), "counts": dict(Counter(r["status"] for r in results)),
               "live_llm_requests": 0, "fixtures_scanned": captured, "fixtures_retained": kept,
               "fixed_time": "2026-11-19T23:00:00", "source_answer": "recorded v6k3b16",
               "comparison": "full literal contexts, QA prompts, evidence, sources, rerank trace and request multisets"}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log("SUMMARY", json.dumps(summary, ensure_ascii=False))
    await replay_http.aclose()
    await live_http.aclose()
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/m.kulakov/meno-v2/rel1119b")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", required=True)
    asyncio.run(main(parser.parse_args()))
