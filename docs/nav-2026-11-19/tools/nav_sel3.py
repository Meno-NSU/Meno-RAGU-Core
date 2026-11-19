"""Selection after the LLM rerank on stand-v6k3 (build 53c94ca, today's search): the same candidates and LLM labels
(one gemma call per question and candidate, at most 8 in flight) are selected under several settings. Arms: base
(MAX_CONTEXT_CHUNKS=12, SOURCE_PRIOR_WEIGHT=1), mc14, mc16, sp0.5, sp0, div (the best candidate of every distinct
document first, then by priority), div+mc14. Units: the retrieval-miss facts and the facts of the first 40 control
tasks of nav_sim3. usage: nav_sel3.py OUT.jsonl MODEL BASE_URL | nav_sel3.py --summary OUT.jsonl"""
import asyncio, collections, hashlib, json, os, sys
from urllib.parse import urlsplit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_sim3 as S

ARMS = {"base": {}, "mc14": {"max_context_chunks": 14}, "mc16": {"max_context_chunks": 16}, "sp0.5": {"source_prior_weight": 0.5},
        "sp0": {"source_prior_weight": 0.0}, "div": {"_div": True}, "div+mc14": {"_div": True, "max_context_chunks": 14}}


async def main(out_path, model, base_url):
    import httpx
    from meno_rag.config import Settings
    from meno_rag.llm.client import VLLMClient
    from meno_rag.llm.router import LLMRouter
    from meno_rag.stand.context import global_chunk_index_to_text, normalize_urls
    from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
    from meno_rag.stand.ragu_pipeline import RaguRagPipeline
    from meno_rag.stand.resources import load_stand_resources
    items = S.load_items()
    control = [i for i in items if i["set"] == "control"][:40]
    items = [i for i in items if i["set"] == "miss"] + control
    done = {json.loads(l)["item_id"] for l in open(out_path, encoding="utf-8")} if os.path.exists(out_path) else set()
    settings = Settings(_env_file=None); resources = load_stand_resources(settings)
    docs, cmap = resources.documents, resources.chunk_mapping
    by_url, by_doc = collections.defaultdict(list), collections.defaultdict(list)
    for di, d in enumerate(docs):
        for u in normalize_urls(d.get("url")): by_url[S.nurl(u)].append(di)
    for cid, place in cmap.items(): by_doc[place["doc_index"]].append(int(cid))
    for item in items:
        for f in item["facts"]:
            f["pages"] = sorted({S.nurl(u) for u, _ in f["sources"]}); frag = set()
            for u, q in f["sources"]:
                qn = S.norm(q)
                for size in (120, 60):
                    for di in by_url.get(S.nurl(u), []):
                        for cid in by_doc.get(di, []):
                            if qn and qn[:size] in S.norm(global_chunk_index_to_text(cid, docs, cmap)): frag.add(cid)
                    if frag: break
            f["fragments"] = sorted(frag); f["page_in_base_context"] = bool(set(f["pages"]) & item["ctx"])
    http = httpx.AsyncClient(timeout=600)
    pipeline = RaguRagPipeline(settings=settings, resources=resources, llm_router=LLMRouter(vllm=VLLMClient(http_client=http), openrouter=None),
                               rewrite_semaphore=asyncio.Semaphore(4), rerank_semaphore=asyncio.Semaphore(8),
                               generation_semaphore=asyncio.Semaphore(2), embed_semaphore=asyncio.Semaphore(4))
    runtime = PipelineRuntime.uniform(ModelRuntime(model, base_url))
    llm_gate, cache, inner = asyncio.Semaphore(8), {}, pipeline._score_document_with_llm

    async def cached(q, h, doc, rt, cid):
        key = hashlib.md5((q + "\x00" + doc).encode()).digest()
        if key not in cache:
            async with llm_gate:
                if key not in cache: cache[key] = await inner(q, h, doc, rt, cid)
        return cache[key]
    pipeline._score_document_with_llm = cached
    plain = type(pipeline)._capped_chunks

    def diverse(self, ranked):
        """The best candidate of every distinct document first (in priority order), the remaining slots by priority."""
        limit, per_page = self.settings.max_context_chunks, self.settings.rerank_candidates_per_page
        first, seen = [], set()
        for pair in ranked:
            page = self._page_of(pair[0].chunk_id)
            if page not in seen and len(first) < limit:
                seen.add(page); first.append(pair)
        taken = collections.Counter(self._page_of(p[0].chunk_id) for p in first); chosen = {id(p) for p in first}
        for pair in ranked:
            if len(chosen) >= limit: break
            page = self._page_of(pair[0].chunk_id)
            if id(pair) in chosen or (per_page > 0 and taken[page] >= per_page): continue
            taken[page] += 1; chosen.add(id(pair))
        return [p for p in ranked if id(p) in chosen]

    def kind(cid):
        d = docs[cmap[str(cid)]["doc_index"]]
        urls = [u.lower() for u in normalize_urls(d.get("url"))]
        host = urlsplit(urls[0]).netloc if urls else ""
        return "post" if host.endswith(("vk.com", "t.me")) else "news" if (d.get("content_source") == "news" or any("/news" in u for u in urls)) else "other"
    gate = asyncio.Semaphore(3); out = open(out_path, "a", encoding="utf-8"); base_settings = pipeline.settings; lock = asyncio.Lock()

    async def one(item):
        async with gate:
            fused = pipeline._fuse(await pipeline._retrieve(item["queries"]))
            await pipeline._rerank(fused, item["question"], "", runtime.core)  # fills the label cache
        async with lock:  # the settings are swapped on the shared pipeline: one task at a time, no new LLM calls
            arms = {}
            for arm, change in ARMS.items():
                pipeline.settings = base_settings.model_copy(update={k: v for k, v in change.items() if not k.startswith("_")})
                pipeline._capped_chunks = (lambda ranked: diverse(pipeline, ranked)) if change.get("_div") else (lambda ranked: plain(pipeline, ranked))
                sel = await pipeline._rerank(fused, item["question"], "", runtime.core)
                final = [int(cid) for cid, _ in sel]
                pages = set()
                for c in final: pages.update(S.nurl(u) for u in normalize_urls(docs[cmap[str(c)]["doc_index"]].get("url")))
                arms[arm] = {"facts": [{"page": bool(set(f["pages"]) & pages), "fragment": bool(set(f["fragments"]) & set(final))} for f in item["facts"]],
                             "n": len(final), "documents": len({cmap[str(c)]["doc_index"] for c in final}), "kinds": dict(collections.Counter(kind(c) for c in final)),
                             "chars": sum(len(global_chunk_index_to_text(c, docs, cmap)) for c in final), "elements": len(sel.elements)}
            pipeline.settings = base_settings
        out.write(json.dumps({"item_id": item["id"], "set": item["set"], "facts": [{k: f[k] for k in ("n", "fragments", "page_in_base_context")} for f in item["facts"]], "arms": arms}, ensure_ascii=False) + "\n"); out.flush()

    todo = [i for i in items if i["id"] not in done]
    for k in range(0, len(todo), 12):
        await asyncio.gather(*(one(i) for i in todo[k:k + 12]))
        print(k + 12, len(todo), "llm calls", len(cache), flush=True)
    await http.aclose(); await resources.graph.aclose()


def summary(out_path):
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    groups = {"miss (92 with a located fragment)": [(r, k) for r in rows if r["set"] == "miss" for k, f in enumerate(r["facts"]) if f["fragments"]],
              "  of them: page was in context": [(r, k) for r in rows if r["set"] == "miss" for k, f in enumerate(r["facts"]) if f["fragments"] and f["page_in_base_context"]],
              "control": [(r, k) for r in rows if r["set"] == "control" for k, f in enumerate(r["facts"]) if f["fragments"]]}
    res = {}
    for name, us in groups.items():
        print(f"\n== {name}: {len(us)} facts")
        for arm in ARMS:
            got = sum(r["arms"][arm]["facts"][k]["fragment"] for r, k in us); page = sum(r["arms"][arm]["facts"][k]["page"] for r, k in us)
            won = sum(r["arms"][arm]["facts"][k]["fragment"] and not r["arms"]["base"]["facts"][k]["fragment"] for r, k in us)
            lost = sum(r["arms"]["base"]["facts"][k]["fragment"] and not r["arms"][arm]["facts"][k]["fragment"] for r, k in us)
            res.setdefault(name, {})[arm] = {"fragment": got, "won": won, "lost": lost, "page": page, "n": len(us)}
            print(f"  {arm:9} fragment in final {got:3d}/{len(us)} (+{won} -{lost}) | page {page}")
    print("\nkept chunks per task (all tasks of the run)")
    for s in ("miss", "control"):
        rs = [r for r in rows if r["set"] == s]
        for arm in ARMS:
            n = sum(r["arms"][arm]["n"] for r in rs); kinds = collections.Counter()
            for r in rs: kinds.update(r["arms"][arm]["kinds"])
            row = {"tasks": len(rs), "chunks": round(n / len(rs), 1), "documents": round(sum(r["arms"][arm]["documents"] for r in rs) / len(rs), 1),
                   "posts": round(kinds["post"] / n, 3), "news": round(kinds["news"] / n, 3), "chars": round(sum(r["arms"][arm]["chars"] for r in rs) / len(rs))}
            res.setdefault("kept:" + s, {})[arm] = row; print(f"  {s:8} {arm:9} {row}")
    json.dump(res, open(out_path + ".summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--summary": summary(sys.argv[2])
    else:
        asyncio.run(main(*sys.argv[1:4])); sys.stdout.flush(); os._exit(0)
