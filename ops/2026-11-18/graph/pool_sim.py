"""Balanced graph candidate pool (GRAPH_BALANCED_POOL) against the entity-order pool, judge-free, on the gold items of
the 558 questions. Each question is rewritten once (gemma); retrieval then runs with the pool off and on, with the same
queries and engines (the switch is the local engine's attribute, flipped under a lock). Per arm and question: the pages
of the chunk candidates the LLM rerank would see (the per-query candidate pool), the pages of all retrieved chunks,
retrieval time. Summary: gold page among the reranker's candidates / among all retrieved chunks, won and lost against
"off", distinct pages per question.

usage: pool_sim.py ITEMS.jsonl OUT.jsonl MODEL BASE_URL   (balanced-pool code on PYTHONPATH, stand env)
       pool_sim.py --summary ITEMS.jsonl OUT.jsonl
"""
import asyncio, json, os, sys, time
from urllib.parse import urlsplit

ARMS = ("off", "on")


def norm(url):
    p = urlsplit((url or "").strip().lower())
    return f"{p.netloc.removeprefix('www.')}{p.path.rstrip('/')}"


def gold_items(path):
    return [i for i in map(json.loads, open(path, encoding="utf-8")) if i.get("answerable") and i.get("gold_urls")]


def summary(items_path, out_path):
    items = {i["id"]: i for i in gold_items(items_path)}
    rows = [r for r in map(json.loads, open(out_path, encoding="utf-8")) if r.get("arms")]
    print(f"{len(rows)} gold items")
    print(f"{'arm':4} | {'gold in rerank pool':>19} | {'won':>4} | {'lost':>4} | {'gold in retrieval':>17} | "
          f"{'pages/q':>7} | {'s/q':>5}")
    for arm in ARMS:
        hit = hit_all = won = lost = 0
        pages = secs = 0.0
        for r in rows:
            gold = {norm(u) for u in items[r["item_id"]]["gold_urls"]}
            a, base = r["arms"][arm], r["arms"]["off"]
            h, h0 = bool(gold & set(a["pool_pages"])), bool(gold & set(base["pool_pages"]))
            hit += h; won += h and not h0; lost += h0 and not h
            hit_all += bool(gold & set(a["all_pages"]))
            pages += len(set(a["pool_pages"])); secs += a["seconds"]
        n = max(len(rows), 1)
        print(f"{arm:4} | {hit:9d}/{len(rows):<9d} | {won:4d} | {lost:4d} | {hit_all:7d}/{len(rows):<9d} | "
              f"{pages / n:7.1f} | {secs / n:5.1f}")


async def main(items_path, out_path, model, base_url):
    from meno_rag.config import Settings
    from meno_rag.llm.client import VLLMClient
    from meno_rag.llm.router import LLMRouter
    from meno_rag.stand.context import normalize_urls
    from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
    from meno_rag.stand.ragu_pipeline import RaguRagPipeline
    from meno_rag.stand.resources import load_stand_resources
    import httpx

    items = gold_items(items_path)
    done = {json.loads(l)["item_id"] for l in open(out_path, encoding="utf-8")} if os.path.exists(out_path) else set()
    settings = Settings(_env_file=None)
    resources = load_stand_resources(settings)
    engine = resources.graph.engines.local_engine
    http = httpx.AsyncClient(timeout=600)
    router = LLMRouter(vllm=VLLMClient(http_client=http), openrouter=None)
    runtime = PipelineRuntime.uniform(ModelRuntime(model, base_url))
    pipeline = RaguRagPipeline(settings=settings, resources=resources, llm_router=router,
                               rewrite_semaphore=asyncio.Semaphore(4), rerank_semaphore=asyncio.Semaphore(4),
                               generation_semaphore=asyncio.Semaphore(2), embed_semaphore=asyncio.Semaphore(4))
    switch = asyncio.Lock()
    gate = asyncio.Semaphore(int(os.environ.get("SIM_CONCURRENCY", "2")))
    out = open(out_path, "a", encoding="utf-8")

    def pages(chunk_ids):
        found = []
        for chunk_id in chunk_ids:
            mapping = resources.chunk_mapping.get(str(chunk_id))
            if mapping is not None:
                found.extend(norm(u) for u in normalize_urls(resources.documents[mapping["doc_index"]].get("url")))
        return found

    async def one(item):
        async with gate:
            queries = await pipeline._rewrite_question(item["question"], "", runtime.core)
            arms = {}
            for arm in ARMS:
                async with switch:
                    engine.balanced_pool = arm == "on"
                    started = time.monotonic()
                    fused = pipeline._fuse(await pipeline._retrieve(queries))
                    seconds = time.monotonic() - started
                pool = [c.chunk_id for batch in fused for c in pipeline._candidate_pool(batch["pool"])
                        if c.chunk_id is not None]
                every = [c.chunk_id for batch in fused for c in batch["pool"] if c.chunk_id is not None]
                arms[arm] = {"pool_pages": sorted(set(pages(pool))), "all_pages": sorted(set(pages(every))),
                             "seconds": round(seconds, 2)}
            out.write(json.dumps({"item_id": item["id"], "queries": queries, "arms": arms}, ensure_ascii=False) + "\n")
            out.flush()

    started = time.monotonic()
    todo = [i for i in items if i["id"] not in done]
    for index in range(0, len(todo), 16):
        await asyncio.gather(*(one(item) for item in todo[index:index + 16]))
        print(f"{len(done) + index + min(16, len(todo) - index)}/{len(items)} in {time.monotonic() - started:.0f} s",
              flush=True)
    out.close()
    await http.aclose()
    await resources.graph.aclose()


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2], sys.argv[3])
    else:
        asyncio.run(main(*sys.argv[1:5]))
    os._exit(0)  # the engine's loop thread is not a daemon
