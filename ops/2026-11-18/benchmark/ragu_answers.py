"""In-process RAGU answers in bench format for the 18.11 benchmark (RAGU stand-v6k vs Menon 1.1, Meno Eval v1).

A copy of ~/meno-v2/final1116/inproc/ragu_inproc.py (the 16.11 in-process runner) with the same pipeline build, turn
path, history handling and record format, extended for one unattended run:
  * several jobs in one process (one resource load), answered in priority order (the job list order: core stratum,
    set B, synthetic_kb, the items without a reference), each into its own records file;
  * an optional smoke job first (--smoke): every smoke question must give a non-empty answer with sources, else exit 3;
  * an optional graph check (--graph-check NODES,EDGES,LO,HI): the loaded graph must have LO..HI times the given
    node and edge counts (v5's), else exit 4; the graph info goes to --resources-json;
  * an adjustable gate instead of a fixed concurrency (--adaptive): starts at --start, goes up by 2 while the live
    stand probe (~/meno-v2/ragu-probe/v6/probe.py, bench mode through the gateway) stays under --fast seconds and the
    H100 gemma has room (kv_cache_usage_perc < .8, no new preemptions), down by 2 when the probe takes --fast..--slow
    seconds, and pauses new items while it takes over --slow s (or fails on time); the H100 metrics (every minute)
    cut it by 2 at KV >= .9 or new preemptions, by half at KV >= .95 or 5+ preemptions in a minute. Every change goes
    to the log ("CTL") with its reason;
  * a deadline (--deadline UTC ISO): no new items after it;
  * a one-line progress file (--progress-file).
History, as the shim runner (mqb.runner) and ragu_inproc.py: a history of user turns only is a real dialog, so its
turns are sent one by one and the stand's own answers become the assistant turns (replay); any other history is sent
as it is. Resume: items with an ok record are skipped; pass 2 answers the failed items again; at the end every OUT is
rewritten in input order, one record per item (ok preferred, then the latest), superseded attempts in
OUT.superseded.jsonl.

usage: ragu_answers.py JOBS.json --model gemma-4-12B-it --base-url http://everest.nsu.ru:9022/v1
       [--smoke SMOKE.json] [--graph-check N,E,LO,HI] [--resources-json PATH] [--adaptive --start 8 --min 2 --max 24]
       [--concurrency 6] [--timeout 300] [--passes 2] [--min-free-gb 15.5] [--deadline 2026-11-18T14:00:00Z]
       [--probe-out PATH] [--progress-file PATH]
JOBS.json: {"jobs": [{"name": "core", "items": ITEMS.jsonl, "ids": IDS.json, "out": OUT.records.jsonl}, ...]}
SMOKE.json: {"name": "smoke", "items": ITEMS.jsonl, "ids": IDS.json, "out": OUT.records.jsonl, "report": OUT.md}
Exit: 0 all answered, 1 some failed, 3 smoke failed, 4 graph check failed.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from meno_rag.config import Settings
from meno_rag.llm.client import VLLMClient
from meno_rag.llm.router import LLMRouter
from meno_rag.logging_config import configure_logging
from meno_rag.schemas import ChatMessage
from meno_rag.stand.attribution import RerankerScorer
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.ragu_pipeline import RaguRagPipeline
from meno_rag.stand.resources import load_stand_resources

PROBE = os.path.expanduser("~/meno-v2/ragu-probe/v6/probe.py")
H100_METRICS = "http://everest.nsu.ru:9022/metrics"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(*parts) -> None:
    print(now(), "INPROC", *parts, flush=True)


def load_items(path: str, ids: list[str]) -> list[dict]:
    want = set(ids)
    by_id = {}
    for line in Path(path).read_text(encoding="utf-8").split("\n"):  # not splitlines(): U+2028 in texts
        if not line.strip():
            continue
        item = json.loads(line)
        if item["id"] not in want:
            continue
        history = item.get("history") or []
        if isinstance(history, str):
            history = ast.literal_eval(history) if history.strip() else []
        by_id[item["id"]] = {"id": item["id"], "question": item["question"],
                             "history": [{"role": h["role"], "content": h.get("content") or ""} for h in history]}
    missing = [i for i in ids if i not in by_id]
    if missing:
        sys.exit(f"{len(missing)} ids not in {path}: {missing[:5]}")
    return [by_id[i] for i in ids]


def is_replay(item: dict) -> bool:
    """As mqb.runner.is_replay: a history of user turns only is a real dialog the system answers itself."""
    return bool(item["history"]) and all(t["role"] == "user" for t in item["history"])


def ragu_contexts(trace: dict) -> tuple[list[dict], str]:
    """run_bench_final._ragu_contexts, verbatim in effect (as ragu_inproc.py)."""
    evidence = trace.get("final_evidence") or []
    if evidence:
        out = []
        for rank, ev in enumerate(evidence, 1):
            srcs = ev.get("sources") or []
            urls = [s.get("source_url") for s in srcs if s.get("source_url")]
            out.append({
                "text": ev.get("text") or "",
                "kind": ev.get("kind"),
                "url": urls[0] if urls else None,
                "title": srcs[0].get("document_title") if srcs else None,
                "urls": list(dict.fromkeys(urls)),
                "chunk_id": ",".join(str(c) for c in ev.get("chunk_ids") or []),
                "chunk_ids": list(ev.get("chunk_ids") or []),
                "rank": rank,
                "context_start": ev.get("context_start"),
                "context_end": ev.get("context_end"),
            })
        return out, "final_evidence"
    if trace.get("retrieved_contexts"):
        return [{"text": t, "rank": k} for k, t in enumerate(trace["retrieved_contexts"], 1)], "retrieved_contexts"
    if trace.get("final_context"):
        return [{"text": trace["final_context"], "rank": 1}], "final_context"
    return [], "none"


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a line cut by a crash; that item is answered again
    return out


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def rewrite_in_order(out: Path, items: list[dict]) -> tuple[int, int, int]:
    """One record per item in input order (ok preferred, then the latest); the rest to OUT.superseded.jsonl."""
    records = read_records(out)
    best: dict[str, dict] = {}
    for r in records:
        cur = best.get(r["item_id"])
        if cur is None or r.get("ok") or not cur.get("ok"):
            best[r["item_id"]] = r
    keep = [best[i["id"]] for i in items if i["id"] in best]
    kept = {id(r) for r in keep}
    superseded = [r for r in records if id(r) not in kept]
    if superseded:
        with open(str(out) + ".superseded.jsonl", "a", encoding="utf-8") as f:
            for r in superseded:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp = Path(str(out) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return len(keep), sum(bool(r.get("ok")) for r in keep), len(superseded)


# ------------------------------------------------------------------ the adjustable gate and its controller
class Gate:
    """A semaphore whose limit can change; limit 0 pauses new admissions (running items finish)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.paused = False
        self.active = 0
        self.cond = asyncio.Condition()

    def effective(self) -> int:
        return 0 if self.paused else self.limit

    async def acquire(self) -> None:
        async with self.cond:
            await self.cond.wait_for(lambda: self.active < self.effective())
            self.active += 1

    async def release(self) -> None:
        async with self.cond:
            self.active -= 1
            self.cond.notify_all()

    async def set(self, limit: int | None = None, paused: bool | None = None) -> None:
        async with self.cond:
            if limit is not None:
                self.limit = limit
            if paused is not None:
                self.paused = paused
            self.cond.notify_all()


def h100_metrics() -> dict:
    try:
        text = urllib.request.urlopen(H100_METRICS, timeout=10).read().decode()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}
    got = {}
    for line in text.splitlines():
        for name in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total"):
            if line.startswith(f"vllm:{name}{{") or line.startswith(f"vllm:{name} "):
                got[name] = got.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return got


def run_probe(out: str) -> dict:
    try:
        p = subprocess.run(["/usr/bin/python3", "-I", PROBE, out], capture_output=True, text=True, timeout=240)
        lines = [line for line in p.stdout.splitlines() if line.startswith("{")]
        return json.loads(lines[-1]) if lines else {"ok": False, "error": f"probe exit {p.returncode}: "
                                                    f"{p.stderr.strip()[-200:]}", "latency_s": None}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "probe timeout 240 s", "latency_s": 240.0}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200], "latency_s": None}


async def controller(gate: Gate, args, stop: dict) -> None:
    """Probe the live stand every 5 min (2 min while paused or slow), the H100 metrics every minute; adjust the gate."""
    last_probe = 0.0
    probe_every = args.probe_every
    prev = h100_metrics()
    pre_hist: list[tuple[float, float]] = []  # (time, preemptions in that minute)
    error_streak = 0
    while not stop["done"]:
        await asyncio.sleep(60)
        if stop["done"]:
            break
        m = await asyncio.to_thread(h100_metrics)
        kv = m.get("kv_cache_usage_perc")
        dpre = None
        if "num_preemptions_total" in m and "num_preemptions_total" in prev:
            dpre = max(0.0, m["num_preemptions_total"] - prev["num_preemptions_total"])
            pre_hist.append((time.time(), dpre))
        if "error" not in m:
            prev = m
        pre_hist[:] = [(t, d) for t, d in pre_hist if t > time.time() - 300]
        if kv is not None and not gate.paused:
            if kv >= 0.95 or (dpre or 0) >= 5:
                new = max(args.min, gate.limit // 2)
                if new != gate.limit:
                    log(f"CTL limit {gate.limit} -> {new}: H100 kv {kv:.2f}, preemptions +{dpre:.0f}/min")
                    await gate.set(limit=new)
            elif kv >= 0.90 or (dpre or 0) > 0:
                new = max(args.min, gate.limit - 2)
                if new != gate.limit:
                    log(f"CTL limit {gate.limit} -> {new}: H100 kv {kv:.2f}, preemptions +{dpre:.0f}/min")
                    await gate.set(limit=new)
        if time.time() - last_probe < probe_every:
            continue
        last_probe = time.time()
        p = await asyncio.to_thread(run_probe, args.probe_out)
        lat = p.get("latency_s")
        ok = bool(p.get("ok"))
        slow = (not ok and lat is not None and lat >= args.slow) or (ok and lat > args.slow)
        info = (f"probe {'ok' if ok else 'FAILED'} {lat} s{' ' + p['error'] if p.get('error') else ''}; H100 kv "
                f"{kv if kv is None else round(kv, 2)}, running {m.get('num_requests_running')}, waiting "
                f"{m.get('num_requests_waiting')}, preemptions 5 min {sum(d for _, d in pre_hist):.0f}; "
                f"gate {gate.effective()} (limit {gate.limit}{', paused' if gate.paused else ''}), active {gate.active}")
        if slow:
            error_streak = 0
            if not gate.paused:
                log(f"CTL PAUSE (limit {gate.limit} -> {max(args.min, gate.limit // 2)} on resume): {info}")
                await gate.set(limit=max(args.min, gate.limit // 2), paused=True)
            else:
                log(f"CTL still paused: {info}")
            probe_every = 120
        elif not ok:  # an error, not slowness (gateway or stand problem): do not stall forever on it
            error_streak += 1
            healthy = kv is not None and kv < 0.8 and not m.get("num_requests_waiting")
            if error_streak >= 3 and healthy:
                log(f"CTL probe failing {error_streak}x but the H100 is healthy: continue at {args.min}: {info}")
                await gate.set(limit=args.min, paused=False)
            else:
                log(f"CTL probe error {error_streak}x, paused: {info}")
                await gate.set(paused=True)
            probe_every = 120
        elif lat > args.fast:
            error_streak = 0
            new = max(args.min, gate.limit - 2)
            log(f"CTL limit {gate.limit} -> {new}{', resumed' if gate.paused else ''}: {info}")
            await gate.set(limit=new, paused=False)
            probe_every = 120
        else:
            error_streak = 0
            room = (kv is None or kv < 0.8) and sum(d for _, d in pre_hist) == 0
            new = gate.limit
            if room and gate.active >= gate.limit and not gate.paused:
                new = min(args.max, gate.limit + 2)
            if new != gate.limit or gate.paused:
                log(f"CTL limit {gate.limit} -> {new}{', resumed' if gate.paused else ''}: {info}")
                await gate.set(limit=new, paused=False)
            else:
                log(f"CTL keep {gate.limit}: {info}")
            probe_every = args.probe_every


# ------------------------------------------------------------------ main
async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--graph-check", default=None)
    ap.add_argument("--resources-json", default=None)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--start", type=int, default=8)
    ap.add_argument("--min", type=int, default=2)
    ap.add_argument("--max", type=int, default=24)
    ap.add_argument("--fast", type=float, default=40.0)
    ap.add_argument("--slow", type=float, default=50.0)
    ap.add_argument("--probe-every", type=float, default=300.0)
    ap.add_argument("--probe-out", default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=15.5)
    ap.add_argument("--deadline", default=None, help="UTC ISO time; no new items after it")
    ap.add_argument("--progress-file", default=None)
    ap.add_argument("--label", default="v6k")
    args = ap.parse_args()
    if args.adaptive and not args.probe_out:
        sys.exit("--adaptive needs --probe-out")
    deadline = (datetime.datetime.strptime(args.deadline, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc).timestamp() if args.deadline else None)

    spec = json.load(open(args.jobs, encoding="utf-8"))
    jobs = []
    for j in spec["jobs"]:
        jobs.append({**j, "items_list": load_items(j["items"], json.load(open(j["ids"], encoding="utf-8"))),
                     "out_path": Path(j["out"])})
    smoke = None
    if args.smoke:
        s = json.load(open(args.smoke, encoding="utf-8"))
        smoke = {**s, "items_list": load_items(s["items"], json.load(open(s["ids"], encoding="utf-8"))),
                 "out_path": Path(s["out"])}

    settings = Settings(_env_file=None)
    configure_logging(settings.log_level, content_previews=settings.log_content_previews)
    graph_dir = settings.graph_dir
    log(f"label={args.label} jobs " + ", ".join(f"{j['name']} {len(j['items_list'])} (replay "
                                                  f"{sum(is_replay(i) for i in j['items_list'])})" for j in jobs)
        + f" model={args.model} base_url={args.base_url} "
        + (f"adaptive start {args.start} min {args.min} max {args.max}" if args.adaptive else f"conc={args.concurrency}")
        + f" deadline={args.deadline} pid={os.getpid()}")
    log(f"stand={settings.stand_resources_dir} graph_dir={graph_dir} qdrant={settings.qdrant_url} "
        f"embedder={settings.embedder_url} engine_reranker={settings.engine_reranker_url} "
        f"embed_conc={settings.embed_concurrency} llm_rerank={settings.llm_rerank} turn_routing={settings.turn_routing} "
        f"dialogue_memory={settings.dialogue_memory} qa_fewshots={settings.qa_fewshots_enabled} "
        f"source_attribution={settings.source_attribution} max_out={settings.max_output_tokens} "
        f"min_out={settings.min_output_tokens} rerank_conc={settings.rerank_concurrency} TZ={os.environ.get('TZ')}")
    manifest = Path(settings.stand_resources_dir) / "build_manifest.json"
    if manifest.is_file():
        log("build_manifest", json.dumps(json.load(open(manifest, encoding="utf-8")), ensure_ascii=False))

    t0 = time.time()
    resources = load_stand_resources(settings)
    index = resources.graph.engines.local_engine.knowledge_graph.index
    nxg = getattr(index.graph_backend, "_graph", None)
    graph_info = {
        "nodes": nxg.number_of_nodes() if nxg is not None else None,
        "edges": nxg.number_of_edges() if nxg is not None else None,
        "vdb_entity": index.nodes_vector_db.collection_name,
        "vdb_relation": index.edges_vector_db.collection_name,
        "vdb_chunk": index.chunks_vector_db.collection_name,
        "stand": str(settings.stand_resources_dir), "graph_dir": str(graph_dir),
        "load_s": round(time.time() - t0),
    }
    log(f"resources loaded in {time.time() - t0:.0f} s; graph", json.dumps(graph_info, ensure_ascii=False))
    if args.graph_check:
        n0, e0, lo, hi = (float(x) for x in args.graph_check.split(","))
        n, e = graph_info["nodes"] or 0, graph_info["edges"] or 0
        graph_info["check"] = {"ref_nodes": n0, "ref_edges": e0, "band": [lo, hi],
                               "node_ratio": round(n / n0, 4), "edge_ratio": round(e / e0, 4),
                               "ok": lo <= n / n0 <= hi and lo <= e / e0 <= hi}
        log("graph check", json.dumps(graph_info["check"]))
    if args.resources_json:
        Path(args.resources_json).write_text(json.dumps(graph_info, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.graph_check and not graph_info["check"]["ok"]:
        log("GRAPH CHECK FAILED")
        os._exit(4)  # the engine's non-daemon loop thread would keep the process alive

    http = httpx.AsyncClient(timeout=600)
    router = LLMRouter(vllm=VLLMClient(http_client=http, api_key=settings.openai_api_key,
                                       endpoint_api_keys=settings.vllm_endpoint_api_keys), openrouter=None)
    runtime = PipelineRuntime.uniform(ModelRuntime(args.model, args.base_url))
    workers = settings.retrieval_executor_max_workers or (settings.embed_concurrency + settings.bm25_concurrency)
    scorer = None
    if settings.source_attribution and settings.engine_reranker_model.strip() and settings.engine_reranker_url.strip():
        scorer = RerankerScorer(http, settings.engine_reranker_url, settings.engine_reranker_model,
                                settings.engine_reranker_api_key)
    pipeline = RaguRagPipeline(
        settings=settings, resources=resources, llm_router=router,
        rewrite_semaphore=asyncio.Semaphore(settings.rewrite_concurrency),
        rerank_semaphore=asyncio.Semaphore(settings.rerank_concurrency),
        generation_semaphore=asyncio.Semaphore(settings.generation_concurrency),
        embed_semaphore=asyncio.Semaphore(settings.embed_concurrency),
        bm25_semaphore=asyncio.Semaphore(settings.bm25_concurrency),
        retrieval_executor=ThreadPoolExecutor(max_workers=workers, thread_name_prefix="retrieval"),
        source_scorer=scorer,
    )
    max_tokens = max(settings.max_output_tokens, settings.min_output_tokens)  # the API with no max_tokens in the request
    model_tag = f"{args.model}@ragu-inproc-{args.label}"

    async def turn(messages: list[dict]):
        """One API turn (non-stream bench path): the outcome, the answer and the generation time."""
        outcome = await pipeline.prepare(messages=[ChatMessage(**m) for m in messages], runtime=runtime,
                                         capture_trace=True, generation_max_tokens=max_tokens)
        prep = time.perf_counter()
        answer = await pipeline.generate_text(outcome=outcome, runtime=runtime, max_tokens=max_tokens, temperature=None)
        gen_ms = round((time.perf_counter() - prep) * 1000, 2)
        if getattr(pipeline, "late_sources", False):
            await pipeline.finalize_sources(outcome, answer)
        return outcome, answer, gen_ms

    async def answer_item(item: dict) -> dict:
        t0 = time.perf_counter()
        replay_answers: list[str] = []
        if is_replay(item):
            messages: list[dict] = []
            for t in item["history"]:
                messages.append({"role": "user", "content": t["content"]})
                try:
                    _, prior, _ = await turn(messages)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"replay: {type(exc).__name__}: {exc}") from exc
                replay_answers.append(prior)
                messages.append({"role": "assistant", "content": prior})
        else:
            messages = [{"role": t["role"], "content": t["content"]} for t in item["history"]]
        messages.append({"role": "user", "content": item["question"]})
        t_q = time.perf_counter()
        outcome, answer, gen_ms = await turn(messages)
        trace = outcome.trace or {}
        ctx, how = ragu_contexts(trace)
        details = (outcome.stage_details or {}).get("query_rewrite") or {}
        return {
            "item_id": item["id"], "ok": bool(answer.strip()), "error": None if answer.strip() else "empty response",
            "http_status": None, "response": answer, "ttft_s": None, "total_s": time.perf_counter() - t0,
            "stages_ms": {**(outcome.stage_durations_ms or {}), "generation": gen_ms},
            "sources": list(outcome.sources or []), "contexts": ctx,
            "search_queries": list(trace.get("search_queries") or outcome.search_queries or []),
            "model": model_tag, "knowledge_base_id": None,
            "v2": {"contexts_from": how, "final_context_chars": len(trace.get("final_context") or ""),
                   "context_kinds": dict(Counter(c.get("kind") or "?" for c in ctx)),
                   **{k: details[k] for k in ("route", "standalone_question", "subqueries") if k in details}},
            "inproc": {"graph": args.label, "graph_dir": str(graph_dir), "replayed_turns": len(replay_answers),
                       "replay_answers": replay_answers, "question_s": round(time.perf_counter() - t_q, 2)},
        }

    lock = asyncio.Lock()
    stop = {"disk": False, "deadline": False, "done": False}
    handles: dict[str, object] = {}
    stats = Counter()
    t_run = time.time()

    def progress(note: str = "") -> None:
        if not args.progress_file:
            return
        done = stats["ok"] + stats["err"]
        rate = done / max(time.time() - t_run, 1) * 60
        parts = [f"{j['name']} {stats['ok:' + j['name']]}/{len(j['items_list'])}" for j in jobs]
        line = (f"{now()} answers: {', '.join(parts)} ok; errors {stats['err']}; {rate:.1f}/min; gate "
                f"{gate.effective() if gate else '-'} (active {gate.active if gate else '-'}){' ' + note if note else ''}")
        try:
            Path(args.progress_file).write_text(line + "\n", encoding="utf-8")
        except OSError:
            pass

    gate: Gate | None = None

    async def one(job: dict, item: dict, counters: Counter) -> None:
        await gate.acquire()
        try:
            if stop["disk"] or stop["deadline"]:
                return
            if deadline and time.time() > deadline:
                if not stop["deadline"]:
                    log("DEADLINE reached: no new items")
                stop["deadline"] = True
                return
            free = free_gb(job["out_path"].parent)
            if free < args.min_free_gb:
                stop["disk"] = True
                log(f"STOP: free disk {free:.1f} GB < {args.min_free_gb} GB")
                return
            n_requests = (len(item["history"]) if is_replay(item) else 0) + 1
            limit_s = args.timeout * n_requests
            t0 = time.perf_counter()
            try:
                rec = await asyncio.wait_for(answer_item(item), limit_s)
            except asyncio.TimeoutError:
                rec = {"item_id": item["id"], "ok": False, "error": f"timeout: no answer within {limit_s:.0f} s "
                       f"({n_requests} request(s))"}
            except Exception as exc:  # noqa: BLE001 - one bad item must not stop the run
                rec = {"item_id": item["id"], "ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
            if not rec["ok"]:
                rec = {"item_id": item["id"], "ok": False, "error": rec.get("error"), "http_status": None,
                       "response": rec.get("response") or "", "ttft_s": None, "total_s": time.perf_counter() - t0,
                       "stages_ms": {}, "sources": [], "contexts": [], "search_queries": [], "model": model_tag,
                       "knowledge_base_id": None, "v2": {}, "inproc": {"graph": args.label}}
            async with lock:
                fh = handles.get(job["name"]) or handles.setdefault(job["name"], job["out_path"].open("a", encoding="utf-8"))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                counters["ok" if rec["ok"] else "err"] += 1
                stats["ok" if rec["ok"] else "err"] += 1
                if rec["ok"]:
                    stats["ok:" + job["name"]] += 1
                done = counters["ok"] + counters["err"]
                if not rec["ok"]:
                    log(f"ERROR {job['name']} {item['id']}: {rec['error'][:300]}")
                if done % 10 == 0 or done == counters["todo"]:
                    rate = done / max(time.time() - counters["t0"], 1) * 60
                    eta = (counters["todo"] - done) / max(rate, 1e-9)
                    log(f"pass {counters['pass']} done {done}/{counters['todo']} errors {counters['err']} "
                        f"{rate:.1f}/min eta {eta:.0f} min gate {gate.effective()} active {gate.active} "
                        f"free {free_gb(job['out_path'].parent):.1f} GB")
                    progress()
        finally:
            await gate.release()

    async def run_jobs(job_list: list[dict], fixed: int | None) -> None:
        nonlocal gate
        gate = Gate(fixed if fixed is not None else (args.start if args.adaptive else args.concurrency))
        ctl = None
        if fixed is None and args.adaptive:
            ctl = asyncio.create_task(controller(gate, args, stop))
        for j in job_list:
            stats["ok:" + j["name"]] = sum(1 for r in read_records(j["out_path"]) if r.get("ok"))
        for p in range(1, args.passes + 1):
            todo = []
            for j in job_list:
                ok_ids = {r["item_id"] for r in read_records(j["out_path"]) if r.get("ok")}
                todo += [(j, i) for i in j["items_list"] if i["id"] not in ok_ids]
            log(f"pass {p}: {len(todo)} to answer (" + ", ".join(
                f"{j['name']} {sum(1 for jj, _ in todo if jj is j)}" for j in job_list) + ")")
            if not todo or stop["disk"] or stop["deadline"]:
                break
            counters = Counter(todo=len(todo), t0=time.time(), **{"pass": p})
            await asyncio.gather(*(one(j, i, counters) for j, i in todo))
        stop["done"] = True
        if ctl:
            ctl.cancel()
        for fh in handles.values():
            fh.close()
        handles.clear()

    rc = 0
    if smoke:
        await run_jobs([smoke], fixed=len(smoke["items_list"]))
        stop["done"] = False
        _, n_ok, _ = rewrite_in_order(smoke["out_path"], smoke["items_list"])
        recs = {r["item_id"]: r for r in read_records(smoke["out_path"])}
        lines = [f"# Smoke run, stand {settings.stand_resources_dir}, {now()}", "",
                 f"graph: {graph_info['nodes']} nodes, {graph_info['edges']} edges; collections "
                 f"{graph_info['vdb_entity']}, {graph_info['vdb_relation']}, {graph_info['vdb_chunk']}", ""]
        bad = []
        for it in smoke["items_list"]:
            r = recs.get(it["id"]) or {}
            good = bool(r.get("ok") and (r.get("response") or "").strip() and r.get("sources"))
            if not good:
                bad.append(it["id"])
            kinds = json.dumps((r.get("v2") or {}).get("context_kinds"), ensure_ascii=False)
            srcs = [f"- {s.get('document_title')} — {s.get('source_url')}" for s in r.get("sources") or []] or ["- (none)"]
            lines += [f"## {it['question']}", "",
                      f"ok {r.get('ok')}, error {r.get('error')}, {round(r.get('total_s') or 0, 1)} s, "
                      f"context kinds {kinds}, route {(r.get('v2') or {}).get('route')}", "",
                      (r.get("response") or "(no answer)").strip(), "", "Sources:", *srcs, ""]
        lines.append(f"VERDICT: {'OK' if not bad else 'FAILED: ' + ', '.join(bad)}")
        if smoke.get("report"):
            Path(smoke["report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"SMOKE {'OK' if not bad else 'FAILED'}: {len(smoke['items_list']) - len(bad)}/{len(smoke['items_list'])} "
            f"with an answer and sources")
        if bad:
            await http.aclose()
            os._exit(3)

    if jobs:
        await run_jobs(jobs, fixed=None)
        progress("finished")
        for j in jobs:
            n, n_ok, n_sup = rewrite_in_order(j["out_path"], j["items_list"])
            log(f"DONE {j['name']}: {n} records, ok {n_ok}, failed {n - n_ok}, never answered "
                f"{len(j['items_list']) - n}, superseded {n_sup}")
            if n_ok != len(j["items_list"]):
                rc = 1
    await http.aclose()
    try:
        await resources.graph.aclose()
    except Exception:  # noqa: BLE001
        pass
    return rc


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
    except SystemExit as exc:  # argparse / load_items exits
        code = exc.code if isinstance(exc.code, int) else 2
        if not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
    except BaseException:  # noqa: BLE001 - a crash: exit 2 (the chain resumes the run), never hang
        import traceback
        traceback.print_exc()
        code = 2
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # the engine's non-daemon loop thread would keep a finished or crashed run alive
