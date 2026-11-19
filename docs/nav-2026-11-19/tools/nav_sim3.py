"""Judge-free replay of three retrieval options on stand-v6k3 (release/v6k3 code + feat/nav-expansion, build 53c94ca),
in-process. Units are required facts: the 108 retrieval misses of the release judging (failures.json) and the facts of
200 random other tasks (control). Arms: base, exp (GRAPH_EXPANSION_CANDIDATES=100), link (GRAPH_ALIAS_TABLE),
sib (SIBLING_FRAGMENT_CANDIDATES=12), all. Every arm gets the search queries the stand used for the task (answer
records of 19.11). Per fact and arm: evidence page / evidence fragment (a chunk holding the first 120 characters of
the quote) among the LLM-rerank candidates and in the chunks kept after the LLM rerank (the final context before its
character budget). The LLM rerank (gemma, at most 8 calls in flight, one score per question and candidate for all
arms) runs for the miss tasks and the first NAV_LLM_CONTROL control tasks.

usage: nav_sim3.py OUT.jsonl ALIAS_TABLE.json MODEL BASE_URL | nav_sim3.py --summary OUT.jsonl
"""
import asyncio, collections, hashlib, json, os, random, re, sys, time
from pathlib import Path
from urllib.parse import urlsplit

H = Path.home()
ARMS = ("base", "exp", "link", "sib", "all")
EXPANSION, SIBLINGS = int(os.environ.get("NAV_EXPANSION", "100")), int(os.environ.get("NAV_SIBLINGS", "12"))
LLM_CONTROL = int(os.environ.get("NAV_LLM_CONTROL", "40"))


def nurl(u):
    p = urlsplit((u or "").strip().lower()); return p.netloc.removeprefix("www.") + p.path.rstrip("/")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower().replace("ё", "е")).strip()


def load_items():
    items = {json.loads(l)["id"]: json.loads(l) for l in open(H / "meno-v2/judge1119/judge/data/items_d54d794.jsonl", encoding="utf-8") if l.strip()}
    recs = {}
    for name in ("v6k3", "r353_v6k3"):
        for l in open(H / f"meno-v2/rel1119b/answers/{name}.records.jsonl", encoding="utf-8"):
            r = json.loads(l)
            ctx = set()
            for c in r.get("contexts") or []:
                for u in [c.get("url"), *(c.get("urls") or [])]:
                    if u: ctx.add(nurl(u))
            for s in r.get("sources") or []:
                if isinstance(s, dict): ctx.add(nurl(s.get("source_url") or s.get("url")))
            recs[r["item_id"]] = {"queries": r.get("search_queries") or [], "question": (r.get("v2") or {}).get("standalone_question"), "ctx": ctx}
    fails = collections.defaultdict(set)
    for r in json.load(open(H / "meno-v2/judge1119/judge/failures/failures.json")):
        if r["class"] == "retrieval": fails[r["qid"]].add(int(r["fact"][1:]) - 1)
    def facts(it):
        groups = {g["id"]: g for g in it.get("evidence_groups") or []}
        out = []
        for n, f in enumerate([k for k in it["key_facts"] if (k.get("fact") or "").strip()]):
            src = [(s.get("url"), s.get("quote")) for gid in f.get("evidence_groups") or [] for s in (groups.get(gid) or {}).get("sources") or [] if s.get("url")]
            if src: out.append({"n": n, "required": f.get("required", True), "sources": src})
        return out
    chosen = []
    for qid in sorted(fails):
        if qid in recs and recs[qid]["queries"]:
            chosen.append({"id": qid, "set": "miss", "facts": [f for f in facts(items[qid]) if f["n"] in fails[qid]], **recs[qid]})
    rest = sorted(q for q in items if q not in fails and q in recs and recs[q]["queries"] and facts(items[q]))
    random.Random(1119).shuffle(rest)
    for qid in rest[:200]:
        chosen.append({"id": qid, "set": "control", "facts": [f for f in facts(items[qid]) if f["required"]], **recs[qid]})
    for c in chosen:
        c["question"] = c["question"] or items[c["id"]]["question"]
    return chosen


async def main(out_path, table_path, model, base_url):
    import httpx
    from graph_utils.alias_table import AliasTable
    from meno_rag.config import Settings
    from meno_rag.llm.client import VLLMClient
    from meno_rag.llm.router import LLMRouter
    from meno_rag.stand.context import global_chunk_index_to_text, normalize_urls
    from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
    from meno_rag.stand.ragu_pipeline import RaguRagPipeline
    from meno_rag.stand.resources import load_stand_resources

    items = load_items()
    if os.environ.get("NAV_LIMIT"):
        k = int(os.environ["NAV_LIMIT"]); items = [i for i in items if i["set"] == "miss"][:k] + [i for i in items if i["set"] == "control"][:k]
    done = {json.loads(l)["item_id"] for l in open(out_path, encoding="utf-8")} if os.path.exists(out_path) else set()
    settings = Settings(_env_file=None)
    resources = load_stand_resources(settings)
    docs, cmap = resources.documents, resources.chunk_mapping
    addresses = getattr(resources, "chunk_addresses", None) or {}
    by_url, by_doc = collections.defaultdict(list), collections.defaultdict(list)
    for di, d in enumerate(docs):
        for u in normalize_urls(d.get("url")): by_url[nurl(u)].append(di)
    for cid, place in cmap.items(): by_doc[place["doc_index"]].append(int(cid))
    for cid, places in addresses.items():
        for place in places: by_doc[place["doc_index"]].append(int(cid))

    def pages_of(cid):
        found = set()
        places = [cmap[str(cid)]] if str(cid) in cmap else []
        for place in [*places, *addresses.get(str(cid), ())]:
            found.update(nurl(u) for u in normalize_urls(docs[place["doc_index"]].get("url")))
        return found

    for item in items:  # where the evidence is in the stand corpus
        for f in item["facts"]:
            f["pages"] = sorted({nurl(u) for u, _ in f["sources"]})
            f["page_in_corpus"] = any(p in by_url for p in f["pages"])
            frag = set()
            for u, q in f["sources"]:
                qn = norm(q)
                for size in (120, 60):
                    for di in by_url.get(nurl(u), []):
                        for cid in by_doc.get(di, []):
                            if str(cid) in cmap and qn and qn[:size] in norm(global_chunk_index_to_text(cid, docs, cmap)): frag.add(cid)
                    if frag: break
            f["fragments"] = sorted(frag)
            f["page_in_base_context"] = bool(set(f["pages"]) & item["ctx"])
    print(f"items {collections.Counter(i['set'] for i in items)} facts {collections.Counter(i['set'] for i in items for f in i['facts'])} "
          f"with fragment {collections.Counter(i['set'] for i in items for f in i['facts'] if f['fragments'])} done {len(done)}", flush=True)
    local = resources.graph.engines.local_engine
    table = AliasTable.load(table_path)
    http = httpx.AsyncClient(timeout=600)
    router = LLMRouter(vllm=VLLMClient(http_client=http), openrouter=None)
    runtime = PipelineRuntime.uniform(ModelRuntime(model, base_url))
    pipeline = RaguRagPipeline(settings=settings, resources=resources, llm_router=router,
                               rewrite_semaphore=asyncio.Semaphore(4), rerank_semaphore=asyncio.Semaphore(8),
                               generation_semaphore=asyncio.Semaphore(2), embed_semaphore=asyncio.Semaphore(4))
    llm_gate, llm_cache, llm_calls = asyncio.Semaphore(8), {}, collections.Counter()
    inner = pipeline._score_document_with_llm
    arm_now = {"arm": "base"}

    async def cached_score(user_question, dialogue_history, cur_doc, runtime_, chunk_id):
        key = hashlib.md5((user_question + "\x00" + cur_doc).encode()).digest()
        llm_calls[arm_now["arm"]] += 1
        if key not in llm_cache:
            llm_calls[arm_now["arm"] + ":new"] += 1
            async with llm_gate:
                if key not in llm_cache:
                    llm_cache[key] = await inner(user_question, dialogue_history, cur_doc, runtime_, chunk_id)
        return llm_cache[key]

    pipeline._score_document_with_llm = cached_score
    base_settings = pipeline.settings
    gate = asyncio.Semaphore(int(os.environ.get("SIM_CONCURRENCY", "3")))
    out = open(out_path, "a", encoding="utf-8")
    todo = [i for i in items if i["id"] not in done]
    llm_ids = {i["id"] for i in items if i["set"] == "miss"} | {i["id"] for i in [x for x in items if x["set"] == "control"][:LLM_CONTROL]}
    started = time.monotonic()
    for start in range(0, len(todo), 12):
        block = todo[start:start + 12]
        results = {i["id"]: {} for i in block}
        for arm in ARMS:
            local.expansion_candidates = EXPANSION if arm in ("exp", "all") else 0
            local.alias_table = table if arm in ("link", "all") else None
            pipeline.settings = base_settings.model_copy(update={"sibling_fragment_candidates": SIBLINGS if arm in ("sib", "all") else 0})
            arm_now["arm"] = arm

            async def run(item, arm=arm):
                async with gate:
                    for attempt in range(3):
                        try:
                            fused = pipeline._fuse(await pipeline._retrieve(item["queries"]))
                            break
                        except Exception as exc:  # noqa: BLE001
                            print(f"retry {item['id']} {arm}: {type(exc).__name__}: {exc}"[:300], flush=True); await asyncio.sleep(5)
                    else:
                        results[item["id"]][arm] = {"error": True}; return
                    cands = [c for batch in fused for c in pipeline._candidate_pool(batch["pool"])]
                    cand_chunks = {c.chunk_id for c in cands if c.chunk_id is not None}
                    every = {c.chunk_id for batch in fused for c in batch["pool"] if c.chunk_id is not None}
                    final = None
                    if item["id"] in llm_ids:
                        try:
                            selection = await pipeline._rerank(fused, item["question"], "", runtime.core)
                            final = {int(cid) for cid, _ in selection}
                        except Exception as exc:  # noqa: BLE001
                            print(f"rerank failed {item['id']} {arm}: {type(exc).__name__}: {exc}"[:300], flush=True)
                def where(chunks):
                    pages = set().union(*(pages_of(c) for c in chunks)) if chunks else set()
                    return [{"page": bool(set(f["pages"]) & pages), "fragment": bool(set(f["fragments"]) & chunks)} for f in item["facts"]]
                results[item["id"]][arm] = {
                    "pool": where(every), "cand": where(cand_chunks), "final": where(final) if final is not None else None,
                    "n_pool": len(every), "n_cand_chunks": len(cand_chunks), "n_cand": len({c.key for c in cands}),
                    "n_final": len(final) if final is not None else None}

            await asyncio.gather(*(run(item) for item in block))
        for item in block:
            out.write(json.dumps({"item_id": item["id"], "set": item["set"], "n_queries": len(item["queries"]),
                                  "facts": [{k: f[k] for k in ("n", "pages", "page_in_corpus", "fragments", "page_in_base_context")} for f in item["facts"]],
                                  "arms": results[item["id"]]}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} {start + len(block)}/{len(todo)} items, {time.monotonic() - started:.0f} s, llm {dict(llm_calls)}", flush=True)
    json.dump(dict(llm_calls), open(out_path + ".llm.json", "w"))
    await http.aclose()
    await resources.graph.aclose()


def summary(out_path):
    rows = [r for r in map(json.loads, open(out_path, encoding="utf-8")) if all(a in r["arms"] and not r["arms"][a].get("error") for a in ARMS)]
    units = []  # (group, row, fact index)
    for r in rows:
        for k, f in enumerate(r["facts"]):
            if not f["page_in_corpus"] or not f["fragments"]:
                group = "miss: corpus/edition gap" if r["set"] == "miss" else None
            elif r["set"] == "miss":
                group = "miss: page in context, fragment not" if f["page_in_base_context"] else "miss: page not in context"
            else:
                group = "control"
            if group: units.append((group, r, k))
    res = {}
    for group in ["miss: page not in context", "miss: page in context, fragment not", "miss: corpus/edition gap", "control"]:
        us = [(r, k) for g, r, k in units if g == group]
        if not us: continue
        fin = [(r, k) for r, k in us if r["arms"]["base"]["final"] is not None]
        print(f"\n== {group}: {len(us)} facts in {len({r['item_id'] for r, _ in us})} tasks; with LLM rerank: {len(fin)} facts")
        print(f"{'arm':5} | cand page | cand fragment (won/lost) | final page | final fragment (won/lost) | pool fragment")
        for arm in ARMS:
            def cnt(stage, what, src):
                return sum(r["arms"][arm][stage][k][what] for r, k in src)
            def wl(stage, what, src):
                return (sum(r["arms"][arm][stage][k][what] and not r["arms"]["base"][stage][k][what] for r, k in src),
                        sum(r["arms"]["base"][stage][k][what] and not r["arms"][arm][stage][k][what] for r, k in src))
            row = {"cand_page": cnt("cand", "page", us), "cand_fragment": cnt("cand", "fragment", us), "cand_fragment_won_lost": wl("cand", "fragment", us),
                   "cand_page_won_lost": wl("cand", "page", us), "pool_fragment": cnt("pool", "fragment", us), "pool_page": cnt("pool", "page", us),
                   "final_page": cnt("final", "page", fin), "final_fragment": cnt("final", "fragment", fin),
                   "final_fragment_won_lost": wl("final", "fragment", fin), "final_page_won_lost": wl("final", "page", fin), "n": len(us), "n_final": len(fin)}
            res.setdefault(group, {})[arm] = row
            print(f"{arm:5} | {row['cand_page']:4d}/{len(us):<4d} (+{row['cand_page_won_lost'][0]} -{row['cand_page_won_lost'][1]}) | {row['cand_fragment']:4d}/{len(us):<4d} (+{row['cand_fragment_won_lost'][0]} -{row['cand_fragment_won_lost'][1]}) | "
                  f"{row['final_page']:4d}/{len(fin):<4d} (+{row['final_page_won_lost'][0]} -{row['final_page_won_lost'][1]}) | {row['final_fragment']:4d}/{len(fin):<4d} (+{row['final_fragment_won_lost'][0]} -{row['final_fragment_won_lost'][1]}) | {row['pool_fragment']:4d} (page {row['pool_page']})")
    print("\ncost per task (mean): cross-encoder pool chunks | LLM-rerank candidates (chunks + elements) | chunks kept")
    for s in ("miss", "control"):
        rs = [r for r in rows if r["set"] == s]
        for arm in ARMS:
            fin = [r["arms"][arm]["n_final"] for r in rs if r["arms"][arm]["n_final"] is not None]
            row = {"tasks": len(rs), "pool": round(sum(r["arms"][arm]["n_pool"] for r in rs) / len(rs), 1), "cand": round(sum(r["arms"][arm]["n_cand"] for r in rs) / len(rs), 1),
                   "final": round(sum(fin) / len(fin), 1) if fin else None}
            res.setdefault("cost:" + s, {})[arm] = row
            print(f"  {s:8} {arm:5} {row}")
    if os.path.exists(out_path + ".llm.json"):
        res["llm_calls"] = json.load(open(out_path + ".llm.json")); print("LLM rerank calls:", res["llm_calls"])
    json.dump(res, open(out_path + ".summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2])
    else:
        asyncio.run(main(*sys.argv[1:5]))
        sys.stdout.flush()
        os._exit(0)
