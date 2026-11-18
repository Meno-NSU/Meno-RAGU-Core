"""Hybrid fusion variants of the vector search (HYBRID_FUSION: rrf = Qdrant's default RRF, the live behaviour;
rrf60 = RRF with k = 60; dbsf = distribution-based score fusion), judge-free, on the gold items of the 558 questions.
Each question is rewritten once (gemma); retrieval then runs twice under every variant, with the same queries and
engines (the variant is an attribute the storages read per query, flipped under a lock). Per arm and question: the
pages of the chunk candidates the LLM rerank would see, the pages of all retrieved chunks, and the ordered chunk ids
of the reranker pool (to measure how much a repeated run differs).

usage: fusion_sim.py ITEMS.jsonl OUT.jsonl MODEL BASE_URL   (fusion-switch code on PYTHONPATH, stand env)
       fusion_sim.py --summary ITEMS.jsonl OUT.jsonl
"""
import asyncio, json, os, sys, time
from urllib.parse import urlsplit

VARIANTS = ("rrf", "rrf60", "dbsf")
# FUSION_ARMS="rrf#1,rrfw#1,rrfw#2" replays only these arms (rrf#1 stays the baseline of the summary)
ARMS = tuple(os.environ["FUSION_ARMS"].split(",")) if os.environ.get("FUSION_ARMS") else \
    tuple(f"{v}#{run}" for v in VARIANTS for run in (1, 2))


def norm(url):
    p = urlsplit((url or "").strip().lower())
    return f"{p.netloc.removeprefix('www.')}{p.path.rstrip('/')}"


def gold_items(path):
    return [i for i in map(json.loads, open(path, encoding="utf-8")) if i.get("answerable") and i.get("gold_urls")]


def summary(items_path, out_path):
    items = {i["id"]: i for i in gold_items(items_path)}
    rows = [r for r in map(json.loads, open(out_path, encoding="utf-8")) if r.get("arms")]
    print(f"{len(rows)} gold items")
    print(f"{'arm':8} | {'gold in rerank pool':>19} | {'won':>4} | {'lost':>4} | {'gold in retrieval':>17} | "
          f"{'pool = run #1':>13}")
    for arm in ARMS:
        hit = hit_all = won = lost = same = 0
        base = "rrf#1"
        twin = arm[:-1] + "1"
        for r in rows:
            gold = {norm(u) for u in items[r["item_id"]]["gold_urls"]}
            a, b = r["arms"][arm], r["arms"][base]
            h, h0 = bool(gold & set(a["pool_pages"])), bool(gold & set(b["pool_pages"]))
            hit += h; won += h and not h0; lost += h0 and not h
            hit_all += bool(gold & set(a["all_pages"]))
            same += a["pool_ids"] == r["arms"][twin]["pool_ids"]
        print(f"{arm:8} | {hit:9d}/{len(rows):<9d} | {won:4d} | {lost:4d} | {hit_all:7d}/{len(rows):<9d} | "
              f"{same:6d}/{len(rows):<6d}")


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
    settings = Settings(_env_file=None)
    resources = load_stand_resources(settings)
    index = resources.graph.engines.local_engine.knowledge_graph.index
    storages = [index.nodes_vector_db, index.edges_vector_db, index.chunks_vector_db]
    http = httpx.AsyncClient(timeout=600)
    router = LLMRouter(vllm=VLLMClient(http_client=http), openrouter=None)
    runtime = PipelineRuntime.uniform(ModelRuntime(model, base_url))
    pipeline = RaguRagPipeline(settings=settings, resources=resources, llm_router=router,
                               rewrite_semaphore=asyncio.Semaphore(4), rerank_semaphore=asyncio.Semaphore(4),
                               generation_semaphore=asyncio.Semaphore(2), embed_semaphore=asyncio.Semaphore(4))
    gate = asyncio.Semaphore(int(os.environ.get("SIM_CONCURRENCY", "4")))

    def pages(chunk_ids):
        found = []
        for chunk_id in chunk_ids:
            mapping = resources.chunk_mapping.get(str(chunk_id))
            if mapping is not None:
                found.extend(norm(u) for u in normalize_urls(resources.documents[mapping["doc_index"]].get("url")))
        return found

    # every question is rewritten once; the variants run in phases, one fusion for all questions at a time
    qpath = out_path + ".queries.json"
    queries = json.load(open(qpath, encoding="utf-8")) if os.path.exists(qpath) else {}

    async def rewrite(item):
        async with gate:
            queries[item["id"]] = await pipeline._rewrite_question(item["question"], "", runtime.core)

    todo = [i for i in items if i["id"] not in queries]
    for k in range(0, len(todo), 32):
        await asyncio.gather(*(rewrite(i) for i in todo[k:k + 32]))
        json.dump(queries, open(qpath, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"rewrites: {len(queries)}", flush=True)
    results = {}
    started = time.monotonic()
    for arm in ARMS:
        for storage in storages:
            storage.meno_hybrid_fusion = arm.split("#")[0]

        async def retrieve(item, arm=arm):
            async with gate:
                t0 = time.monotonic()
                for attempt in range(3):  # a transient embedder or reranker error must not end the replay
                    try:
                        fused = pipeline._fuse(await pipeline._retrieve(queries[item["id"]]))
                        break
                    except Exception as exc:  # noqa: BLE001
                        print(f"retry {item['id']} {arm}: {type(exc).__name__}: {exc}"[:300], flush=True)
                        await asyncio.sleep(5)
                else:
                    results.setdefault(item["id"], {})[arm] = {"pool_pages": [], "all_pages": [], "pool_ids": [],
                                                               "seconds": None, "error": True}
                    return
                seconds = time.monotonic() - t0
            pool = [c.chunk_id for batch in fused for c in pipeline._candidate_pool(batch["pool"])
                    if c.chunk_id is not None]
            every = [c.chunk_id for batch in fused for c in batch["pool"] if c.chunk_id is not None]
            results.setdefault(item["id"], {})[arm] = {
                "pool_pages": sorted(set(pages(pool))), "all_pages": sorted(set(pages(every))),
                "pool_ids": [str(c) for c in pool], "seconds": round(seconds, 2)}

        await asyncio.gather(*(retrieve(item) for item in items))
        print(f"arm {arm} done in {time.monotonic() - started:.0f} s", flush=True)
    out = open(out_path, "w", encoding="utf-8")
    for item in items:
        out.write(json.dumps({"item_id": item["id"], "queries": queries[item["id"]], "arms": results[item["id"]]},
                             ensure_ascii=False) + "\n")
    out.close()
    await http.aclose()
    await resources.graph.aclose()


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2], sys.argv[3])
        sys.stdout.flush()
    else:
        asyncio.run(main(*sys.argv[1:5]))
    os._exit(0)  # the engine's loop thread is not a daemon
