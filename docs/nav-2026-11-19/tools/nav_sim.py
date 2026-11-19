"""Judge-free replay of graph navigation (feat/nav-expansion b709314) on stand-v6k2, in-process as fusion_sim.py.
Arms: base (today's search), exp (GRAPH_EXPANSION_CANDIDATES=100), link (GRAPH_ALIAS_TABLE), exp+link. Every arm gets
the same queries: cached rewrites of the 324 gold questions (fusion_rrfw.jsonl.queries.json), the search queries of
the set B answers of 18.11 (ctx1118), the question itself for the unified tasks. No LLM. The cross-encoder is called
through a cache (one score per query and text for all arms) with at most 4 calls in flight; while it is down the
replay waits. Per arm and item: pages of the cross-encoder pool (every chunk the search returns), pages of the
LLM-rerank candidates, sizes, texts sent to the cross-encoder.

usage: nav_sim.py OUT.jsonl ALIAS_TABLE.json      (stand env, code on PYTHONPATH)
       nav_sim.py --summary OUT.jsonl
"""
import asyncio, collections, hashlib, json, os, random, sys, time
from pathlib import Path
from urllib.parse import urlsplit

H = Path.home()
ARMS = ("base", "exp", "link", "exp+link")
EXPANSION = int(os.environ.get("NAV_EXPANSION", "100"))


def norm(url):
    p = urlsplit((url or "").strip().lower())
    return f"{p.netloc.removeprefix('www.')}{p.path.rstrip('/')}"


def load_items():
    items = []
    queries = json.load(open(H / "meno-v2/ragu-probe/v61/fusion_rrfw.jsonl.queries.json", encoding="utf-8"))
    for i in map(json.loads, open(H / "meno-v2/bench/items558.jsonl", encoding="utf-8")):
        if i.get("answerable") and i.get("gold_urls") and i["id"] in queries:
            items.append({"id": i["id"], "set": "gold324", "queries": queries[i["id"]], "gold": [[u] for u in i["gold_urls"]], "any": True})
    recs = {}
    for l in open(H / "meno-v2/ctx1118/answers/setB.sources.records.jsonl", encoding="utf-8"):
        r = json.loads(l); recs[r["item_id"]] = r.get("search_queries") or []
    for i in map(json.loads, open(H / "meno-v2/final1118/items/setB_items.jsonl", encoding="utf-8")):
        if str(i.get("category", "")).startswith("G") and i.get("gold_urls") and recs.get(i["id"]):
            seen_b = locals().get("seen_b", 0) + 1
            if seen_b % int(os.environ.get("NAV_SETB_STEP", "2")):
                continue  # every second graph question of set B: the replay shares one search thread
            items.append({"id": i["id"], "set": "setB-G", "queries": recs[i["id"]], "gold": [[u] for u in i["gold_urls"]], "any": False})
    needs = [json.loads(l) for l in open(H / "meno-v2/audit1119/nav/needs.jsonl", encoding="utf-8")]
    uni = [r for r in needs if r["src"] == "bench" and r["classified"] and r.get("gold")]
    gate = [r for r in uni if set(r["profiles"]) & {"graph", "release_gate"}]
    rest = [r for r in uni if r not in gate]
    random.Random(1119).shuffle(rest)
    for r in gate + rest[:156]:
        groups = collections.defaultdict(list)
        for g in r["gold"]:
            groups[g["group"]].append(g["url"])
        items.append({"id": r["qid"], "set": "unified", "queries": [json.loads(l)["question"] if False else r["question"]],
                      "gold": list(groups.values()), "any": False, "ladder": r.get("ladder_worst"),
                      "profile": "graph+gate" if r in gate else "general"})
    # needs.jsonl keeps 400 characters of a question; take the full text
    full = {}
    for l in open(H / "meno-v2/audit1119/nav/demand.jsonl", encoding="utf-8"):
        d = json.loads(l); full[d["qid"]] = d["question"]
    for i in items:
        if i["set"] == "unified":
            i["queries"] = [full[i["id"]]]
    by_set = collections.defaultdict(list)
    for i in items:
        by_set[i["set"]].append(i)
    order = []
    for k in range(max(len(v) for v in by_set.values())):  # interleaved: a prefix of the run is balanced
        for v in by_set.values():
            if k < len(v):
                order.append(v[k])
    return order


class CachedScorer:
    def __init__(self, inner):
        self.inner, self.cache, self.gate = inner, {}, None
        self.sent = collections.Counter(); self.arm = "base"

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def score(self, text_1, text_2, **kwargs):
        if self.gate is None:
            self.gate = asyncio.Semaphore(4)
        self.sent[self.arm] += len(text_2)
        keys = [hashlib.md5((text_1 + "\x00" + t).encode()).digest() for t in text_2]
        miss = [n for n, k in enumerate(keys) if k not in self.cache]
        self.sent[self.arm + ":new"] += len(miss)
        if miss:
            waited = 0
            while True:
                try:
                    async with self.gate:
                        ranked = await self.inner.score(text_1, [text_2[n] for n in miss], **kwargs)
                    break
                except Exception as exc:  # noqa: BLE001 - the shared cross-encoder goes down for the judges
                    if waited % 600 == 0:
                        print(f"{time.strftime('%H:%M:%S', time.gmtime())} scorer {type(exc).__name__}: {str(exc)[:120]}; waiting", flush=True)
                    waited += 30
                    if waited > 3 * 3600:
                        raise
                    await asyncio.sleep(30)
            for index, value in ranked:
                self.cache[keys[miss[index]]] = float(value)
        return sorted(((n, self.cache[k]) for n, k in enumerate(keys)), key=lambda p: (-p[1], p[0]))


async def main(out_path, table_path):
    from graph_utils.alias_table import AliasTable
    from meno_rag.config import Settings
    from meno_rag.stand.context import normalize_urls
    from meno_rag.stand.ragu_pipeline import RaguRagPipeline
    from meno_rag.stand.resources import load_stand_resources

    items = load_items()
    if os.environ.get('NAV_LIMIT'):
        items = items[:int(os.environ['NAV_LIMIT'])]
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["item_id"] for l in open(out_path, encoding="utf-8")}
    print(f"items {len(items)} {collections.Counter(i['set'] for i in items)} done {len(done)}", flush=True)
    settings = Settings(_env_file=None)
    resources = load_stand_resources(settings)
    engines = resources.graph.engines
    local = engines.local_engine
    scorer = CachedScorer(local.reranker)
    local.reranker = scorer
    engines.naive_engine.reranker = scorer
    table = AliasTable.load(table_path)
    pipeline = RaguRagPipeline(settings=settings, resources=resources, llm_router=None,
                               rewrite_semaphore=asyncio.Semaphore(4), rerank_semaphore=asyncio.Semaphore(4),
                               generation_semaphore=asyncio.Semaphore(2), embed_semaphore=asyncio.Semaphore(4))
    gate = asyncio.Semaphore(int(os.environ.get("SIM_CONCURRENCY", "3")))

    def pages(chunk_ids):
        found = []
        for chunk_id in chunk_ids:
            mapping = resources.chunk_mapping.get(str(chunk_id))
            if mapping is not None:
                found.extend(norm(u) for u in normalize_urls(resources.documents[mapping["doc_index"]].get("url")))
        return found

    out = open(out_path, "a", encoding="utf-8")
    todo = [i for i in items if i["id"] not in done]
    started = time.monotonic()
    for start in range(0, len(todo), 24):
        block = todo[start:start + 24]
        results = {i["id"]: {} for i in block}
        for arm in ARMS:
            local.expansion_candidates = EXPANSION if "exp" in arm else 0
            local.alias_table = table if "link" in arm else None
            scorer.arm = arm

            async def retrieve(item, arm=arm):
                async with gate:
                    t0 = time.monotonic()
                    for attempt in range(3):
                        try:
                            fused = pipeline._fuse(await pipeline._retrieve(item["queries"]))
                            break
                        except Exception as exc:  # noqa: BLE001
                            print(f"retry {item['id']} {arm}: {type(exc).__name__}: {exc}"[:300], flush=True)
                            await asyncio.sleep(5)
                    else:
                        results[item["id"]][arm] = {"error": True}
                        return
                pool = [c.chunk_id for batch in fused for c in pipeline._candidate_pool(batch["pool"]) if c.chunk_id is not None]
                every = [c.chunk_id for batch in fused for c in batch["pool"] if c.chunk_id is not None]
                elements = sum(1 for batch in fused for c in batch["pool"] if c.chunk_id is None)
                results[item["id"]][arm] = {
                    "pool_pages": sorted(set(pages(pool))), "all_pages": sorted(set(pages(every))),
                    "n_candidates": len(set(pool)), "n_pool": len(set(every)), "n_elements": elements,
                    "seconds": round(time.monotonic() - t0, 2),
                    "linked": [len(table.node_ids(q)[:local.max_linked_entities]) for q in item["queries"]] if "link" in arm else None}

            await asyncio.gather(*(retrieve(item) for item in block))
        for item in block:
            out.write(json.dumps({"item_id": item["id"], "set": item["set"], "gold": item["gold"], "any": item["any"],
                                  "ladder": item.get("ladder"), "profile": item.get("profile"), "n_queries": len(item["queries"]),
                                  "arms": results[item["id"]]}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} {start + len(block)}/{len(todo)} items, {time.monotonic() - started:.0f} s, "
              f"cross-encoder texts {dict(scorer.sent)}", flush=True)
    json.dump(dict(scorer.sent), open(out_path + ".scorer.json", "w"))
    await resources.graph.aclose()


def summary(out_path):
    rows = [r for r in map(json.loads, open(out_path, encoding="utf-8")) if all(a in r["arms"] and not r["arms"][a].get("error") for a in ARMS)]
    def hit(r, arm, key, full):
        got = set(r["arms"][arm][key])
        flags = [any(norm(u) in got for u in group) for group in r["gold"]]
        return all(flags) if full else any(flags)
    def groups(r, arm, key):
        got = set(r["arms"][arm][key]); return sum(any(norm(u) in got for u in group) for group in r["gold"])
    sets = ["ALL"] + sorted({r["set"] for r in rows}) + ["unified:graph+gate"]
    res = {}
    for s in sets:
        rs = [r for r in rows if s == "ALL" or r["set"] == s or (s == "unified:graph+gate" and r.get("profile") == "graph+gate")]
        if not rs: continue
        print(f"\n== {s}: {len(rs)} items, {sum(len(r['gold']) for r in rs)} gold pages/groups, {sum(r['n_queries'] for r in rs)} queries")
        print(f"{'arm':9} | {'CE pool: any':>12} | {'full':>5} | {'pages':>6} | {'won/lost full':>13} | {'LLM cand: any':>13} | {'full':>5} | {'pages':>6} | {'won/lost full':>13} | {'pool/q':>7} | {'cand/q':>7}")
        for arm in ARMS:
            row = {}
            for key, name in (("all_pages", "pool"), ("pool_pages", "cand")):
                row[name + "_any"] = sum(hit(r, arm, key, False) for r in rs); row[name + "_full"] = sum(hit(r, arm, key, True) for r in rs)
                row[name + "_pages"] = sum(groups(r, arm, key) for r in rs)
                row[name + "_won"] = sum(hit(r, arm, key, True) and not hit(r, "base", key, True) for r in rs)
                row[name + "_lost"] = sum(hit(r, "base", key, True) and not hit(r, arm, key, True) for r in rs)
            def gset(r, a, key):
                got = set(r["arms"][a][key]); return {n for n, group in enumerate(r["gold"]) if any(norm(u) in got for u in group)}
            row["pool_pages_new"] = sum(len(gset(r, arm, "all_pages") - gset(r, "base", "all_pages")) for r in rs)
            row["pool_pages_new_in_cand"] = sum(len((gset(r, arm, "all_pages") - gset(r, "base", "all_pages")) & gset(r, arm, "pool_pages")) for r in rs)
            row["pool_pages_gone"] = sum(len(gset(r, "base", "all_pages") - gset(r, arm, "all_pages")) for r in rs)
            row["cand_pages_new"] = sum(len(gset(r, arm, "pool_pages") - gset(r, "base", "pool_pages")) for r in rs)
            row["cand_pages_gone"] = sum(len(gset(r, "base", "pool_pages") - gset(r, arm, "pool_pages")) for r in rs)
            q = sum(r["n_queries"] for r in rs)
            row["pool_per_query"] = round(sum(r["arms"][arm]["n_pool"] for r in rs) / q, 1); row["cand_per_query"] = round(sum(r["arms"][arm]["n_candidates"] for r in rs) / q, 1)
            res.setdefault(s, {"items": len(rs), "gold": sum(len(r["gold"]) for r in rs), "queries": q})[arm] = row
            print(f"{arm:9} | {row['pool_any']:12d} | {row['pool_full']:5d} | {row['pool_pages']:6d} | {row['pool_won']:5d}/{row['pool_lost']:<7d} | {row['cand_any']:13d} | {row['cand_full']:5d} | {row['cand_pages']:6d} | {row['cand_won']:5d}/{row['cand_lost']:<7d} | {row['pool_per_query']:7.1f} | {row['cand_per_query']:7.1f} | pool pages +{row['pool_pages_new']} (in cand {row['pool_pages_new_in_cand']}) -{row['pool_pages_gone']}; cand pages +{row['cand_pages_new']} -{row['cand_pages_gone']}")
    lad = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r["set"] == "unified":
            for arm in ARMS:
                lad[r["ladder"]]["n:" + arm] += 1; lad[r["ladder"]]["pool:" + arm] += hit(r, arm, "all_pages", True); lad[r["ladder"]]["cand:" + arm] += hit(r, arm, "pool_pages", True)
    print("\nunified by audit ladder (full gold in CE pool / in LLM candidates):")
    for k in sorted(lad, key=str):
        print(f"  {str(k):32} n={lad[k]['n:base']:3d} | " + " | ".join(f"{arm} {lad[k]['pool:' + arm]}/{lad[k]['cand:' + arm]}" for arm in ARMS))
    res["ladder"] = {str(k): dict(v) for k, v in lad.items()}
    if os.path.exists(out_path + ".scorer.json"):
        res["cross_encoder_texts"] = json.load(open(out_path + ".scorer.json")); print("\ncross-encoder texts:", res["cross_encoder_texts"])
    json.dump(res, open(out_path + ".summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2])
    else:
        asyncio.run(main(sys.argv[1], sys.argv[2]))
        sys.stdout.flush()
        os._exit(0)
