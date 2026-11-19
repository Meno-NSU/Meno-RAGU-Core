"""Why fragments that are among the LLM-rerank candidates do not reach the context (stand-v6k3, baseline search):
for the miss facts of nav_sim3.jsonl whose fragment is a candidate, the LLM label of the fragment chunk, its merged
score and rank, and what was kept instead. usage: nav_diag3.py SIM.jsonl OUT.jsonl MODEL BASE_URL"""
import asyncio, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_sim3 as S


async def main(sim_path, out_path, model, base_url):
    import httpx
    from meno_rag.config import Settings
    from meno_rag.llm.client import VLLMClient
    from meno_rag.llm.router import LLMRouter
    from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
    from meno_rag.stand.ragu_pipeline import RaguRagPipeline
    from meno_rag.stand.resources import load_stand_resources
    from meno_rag.stand.context import normalize_urls
    want = {}
    for r in map(json.loads, open(sim_path, encoding="utf-8")):
        if r["set"] == "miss":
            fr = {c for k, f in enumerate(r["facts"]) if f["fragments"] and r["arms"]["base"]["cand"][k]["fragment"] for c in f["fragments"]}
            if fr: want[r["item_id"]] = fr
    items = [i for i in S.load_items() if i["id"] in want]
    settings = Settings(_env_file=None); resources = load_stand_resources(settings)
    http = httpx.AsyncClient(timeout=600)
    pipeline = RaguRagPipeline(settings=settings, resources=resources, llm_router=LLMRouter(vllm=VLLMClient(http_client=http), openrouter=None),
                               rewrite_semaphore=asyncio.Semaphore(4), rerank_semaphore=asyncio.Semaphore(8),
                               generation_semaphore=asyncio.Semaphore(2), embed_semaphore=asyncio.Semaphore(4))
    runtime = PipelineRuntime.uniform(ModelRuntime(model, base_url))
    gate = asyncio.Semaphore(2); out = open(out_path, "w", encoding="utf-8")

    async def one(item):
        async with gate:
            fused = pipeline._fuse(await pipeline._retrieve(item["queries"]))
            sel = await pipeline._rerank(fused, item["question"], "", runtime.core)
        rows = sel.graph_trace["rerank"]
        chunks = [x for x in rows if x.get("chunk_id") is not None]
        kept = [x for x in chunks if x["kept"]]
        from meno_rag.stand.rerank import rerank_merge_score
        from meno_rag.stand.ragu_pipeline import Candidate
        def prio(y):
            return pipeline._priority(Candidate(y["key"], y["score"], chunk_id=y["chunk_id"]), rerank_merge_score(y["score"], y["rerank_score"], settings.rerank_weight))
        for x in chunks:
            if int(x["chunk_id"]) in want[item["id"]]:
                page = resources.chunk_mapping[str(x["chunk_id"])]["doc_index"]
                out.write(json.dumps({"item": item["id"], "chunk": x["chunk_id"], "llm_score": x["rerank_score"], "engine_score": x.get("score"), "kept": x["kept"],
                    "candidates": len(chunks), "llm_ge": sum(1 for y in chunks if y["rerank_score"] >= x["rerank_score"]), "llm_gt": sum(1 for y in chunks if y["rerank_score"] > x["rerank_score"]),
                    "kept_n": len(kept), "kept_min_llm": min((y["rerank_score"] for y in kept), default=None),
                    "kept_same_page": sum(1 for y in kept if resources.chunk_mapping.get(str(y["chunk_id"]), {}).get("doc_index") == page),
                    "quality": resources.documents[page].get("quality_score"), "source": resources.documents[page].get("content_source"),
                    "kept_quality": sorted(round(float(resources.documents[resources.chunk_mapping[str(y["chunk_id"])]["doc_index"]].get("quality_score") or 1), 2) for y in kept),
                    "kept_same_url": sum(1 for y in kept if set(normalize_urls(resources.documents[resources.chunk_mapping[str(y["chunk_id"])]["doc_index"]].get("url"))) & set(normalize_urls(resources.documents[page].get("url")))),
                    "priority_rank": sum(1 for y in chunks if prio(y) > prio(x)), "kept_min_llm_rank_cut": sorted(prio(y) for y in kept)[0] if kept else None, "priority": prio(x),
                    "same_page_candidates_ahead": sum(1 for y in chunks if prio(y) > prio(x) and resources.chunk_mapping.get(str(y["chunk_id"]), {}).get("doc_index") == page),
                    "llm_full_score_candidates": sum(1 for y in chunks if y["rerank_score"] >= 1.0), "keys": sorted(x.keys())}, ensure_ascii=False) + "\n"); out.flush()

    await asyncio.gather(*(one(i) for i in items))
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    best = {}
    for r in rows:  # the best-scored fragment chunk of a task
        if r["item"] not in best or r["llm_score"] > best[r["item"]]["llm_score"]: best[r["item"]] = r
    b = list(best.values())
    nk = [r for r in best.values() if not r["kept"] and r["llm_score"] >= 0.75]
    print("relevant by the LLM, not kept:", len(nk))
    for r in nk:
        print({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ("item", "llm_score", "engine_score", "quality", "source", "priority_rank", "same_page_candidates_ahead", "kept_same_url", "kept_same_page", "kept_quality")})
    await http.aclose(); await resources.graph.aclose()

asyncio.run(main(*sys.argv[1:5])); sys.stdout.flush(); os._exit(0)
