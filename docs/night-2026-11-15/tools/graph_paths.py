"""Graph as navigator, judge-free: for every question, the chunks each RAGU channel finds for the question itself —
local (graph: entities -> neighbours -> relations -> their chunks) and naive (chunk vectors) — after the engine's
cross-encoder, mapped to pages, against the question's gold pages. No LLM involved.

usage: python -u graph_paths.py ITEMS.jsonl OUT.jsonl
"""
import asyncio, json, os, re, sys, time

from meno_rag.config import Settings
from meno_rag.stand.resources import _load_chunk_mapping, _load_documents
from graph_utils.graph_utils import load_graph_engines

items_path, out_path = sys.argv[1:3]
norm = lambda u: re.sub(r"^https?://(www\.)?", "", (u or "").split("#")[0]).rstrip("/").lower()
s = Settings(_env_file=None)
documents, _ = _load_documents(s.corpus_path)
mapping = _load_chunk_mapping(s.chunk_mapping_path)


def doc_urls(doc):
    u = doc.get("url") or doc.get("urls") or []
    return {norm(x) for x in ([u] if isinstance(u, str) else u) if x}


def pages(chunks):
    out = []
    for gid, score in chunks:
        m = mapping.get(str(gid))
        if m is None:
            continue
        out.append(sorted(doc_urls(documents[m["doc_index"]])))
    return out


eng = load_graph_engines(graph_dir=str(s.graph_dir), embedder_name=s.embedder_model, embedder_url=s.embedder_url,
                         embedder_key=s.embedder_api_key, qdrant_location=s.qdrant_url,
                         engine_reranker_name=s.engine_reranker_model, engine_reranker_url=s.engine_reranker_url,
                         engine_reranker_key=s.engine_reranker_api_key, max_related_candidates=s.graph_related_candidates)
loop = eng.event_loop
asyncio.set_event_loop(loop)
from graph_utils.graph_utils import LocalParams as LP, NaiveSearchParams as NP, _metric  # noqa: E402

lp = LP(top_k=s.graph_top_k, use_summary=False, use_chunks=True, rerank_top_k=s.engine_rerank_top_k)
npar = NP(top_k=s.top_k, rerank_top_k=s.engine_rerank_top_k)
done = set()
if os.path.exists(out_path):
    done = {json.loads(line)["item_id"] for line in open(out_path, encoding="utf-8")}
out = open(out_path, "a", encoding="utf-8")
for line in open(items_path, encoding="utf-8"):
    it = json.loads(line)
    if it["id"] in done or not it.get("gold_urls"):
        continue
    q = it["question"]
    t0 = time.perf_counter()
    loc, nai = loop.run_until_complete(asyncio.gather(eng.local_engine.search(q, params=lp), eng.naive_engine.search(q, params=npar)))
    ls = _metric(loc, "chunk_scores")
    ns = list(getattr(nai.result, "scores", None) or [])
    lch = eng._map_chunks(loc.result.chunks, ls if ls is not None and len(ls) == len(loc.result.chunks) else [1.0] * len(loc.result.chunks))
    nch = eng._map_chunks(nai.result.chunks, ns if len(ns) == len(nai.result.chunks) else [1.0] * len(nai.result.chunks))
    rec = {"item_id": it["id"], "category": it.get("category"), "gold": sorted({norm(u) for u in it["gold_urls"]}),
           "local_chunks": [g for g, _ in lch], "naive_chunks": [g for g, _ in nch],
           "local_pages": pages(lch), "naive_pages": pages(nch), "n_entities": len(loc.result.entities),
           "n_relations": len(loc.result.relations), "s": round(time.perf_counter() - t0, 2)}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
