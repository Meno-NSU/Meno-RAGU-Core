"""Read-only checks of a RAGU stand dir before the benchmark (18.11, stand-v6k): files, Qdrant collections (graph v6's
three and the stand's chunk collection non-empty and green), every chunk id of the stand's menon_chunk_mapping.json a
point of the chunk collection (scroll of the payload key __ragu_id__, as v6_stand_chunks.py checks), and
document_quality.json present. Writes VERIFY_JSON; exit 1 on any failed check.
usage: python3 -I verify_stand.py STAND_DIR GRAPH_DIR VERIFY_JSON [EXCLUDE_IDS_JSON]
  GRAPH_DIR  the graph's own dir (bundle/knowledge/ragu_graph_v6) with the build's qdrant_collections.json
"""
import json, os, sys, time, urllib.request
from pathlib import Path

Q = "http://127.0.0.1:6433"
TAG = os.environ.get("GRAPH_TAG", "ragu_graph_v6")  # the graph build dir name inside its collection names
stand, graph, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
excluded = set(json.loads(Path(sys.argv[4]).read_text())) if len(sys.argv) > 4 and sys.argv[4] else set()
R = stand / "knowledge" / "ragu_graph"
res = {"stand": str(stand), "graph": str(graph), "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "checks": {}, "failed": []}


def check(name, ok, detail):
    res["checks"][name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        res["failed"].append(name)
    print(("OK  " if ok else "FAIL"), name, json.dumps(detail, ensure_ascii=False)[:600], flush=True)


def get(path):
    with urllib.request.urlopen(Q + path, timeout=60) as r:
        return json.load(r)["result"]


def post(path, body):
    req = urllib.request.Request(Q + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["result"]


need = [stand / "chunked_texts_about_nsu_with_metadata_and_scores.jsonl", stand / "chunk_mapping_to_texts.json",
        stand / "build_manifest.json", R / "knowledge_graph.gml", R / "menon_chunk_mapping.json", R / "qdrant_collections.json"]
missing = [str(p) for p in need if not p.is_file()]
check("files", not missing, {"missing": missing, "sizes_mb": {p.name: round(p.stat().st_size / 1e6, 1) for p in need if p.is_file()}})
try:
    res["build_manifest"] = json.loads((stand / "build_manifest.json").read_text())
except Exception as exc:  # noqa: BLE001
    res["build_manifest"] = f"unreadable: {exc}"

stand_cols = json.loads((R / "qdrant_collections.json").read_text()) if (R / "qdrant_collections.json").is_file() else {}
graph_cols = json.loads((graph / "qdrant_collections.json").read_text()) if (graph / "qdrant_collections.json").is_file() else {}
res["collections"] = {"stand": stand_cols, "graph": graph_cols}
check(f"graph collections named {TAG}", set(graph_cols) >= {"vdb_entity", "vdb_relation", "vdb_chunk"}
      and all(f"{TAG}_vdb_" in graph_cols.get(k, "") for k in ("vdb_entity", "vdb_relation", "vdb_chunk")), graph_cols)
check("stand uses the graph's entity/relation collections", stand_cols.get("vdb_entity") == graph_cols.get("vdb_entity")
      and stand_cols.get("vdb_relation") == graph_cols.get("vdb_relation"), stand_cols)
infos = {}
for name in sorted(set(graph_cols.values()) | set(stand_cols.values())):
    try:
        info = get(f"/collections/{name}")
        infos[name] = {"status": info["status"], "points": info["points_count"],
                       "indexed_vectors": info.get("indexed_vectors_count"),
                       "indexing_threshold": (info.get("config") or {}).get("optimizer_config", {}).get("indexing_threshold")}
    except Exception as exc:  # noqa: BLE001
        infos[name] = {"status": f"error {type(exc).__name__}: {exc}", "points": 0}
    check(f"collection {name}", infos[name]["status"] == "green" and (infos[name]["points"] or 0) > 0, infos[name])
res["collection_info"] = infos

chunk_col = stand_cols.get("vdb_chunk")
if chunk_col and (R / "menon_chunk_mapping.json").is_file():
    t0 = time.time()
    ids, offset, pages = set(), None, 0
    while True:
        body = {"limit": 2000, "with_vector": False, "with_payload": {"include": ["__ragu_id__"]}}
        if offset is not None:
            body["offset"] = offset
        page = post(f"/collections/{chunk_col}/points/scroll", body)
        ids.update(p["payload"].get("__ragu_id__") for p in page["points"])
        pages += 1
        offset = page.get("next_page_offset")
        if offset is None:
            break
    ids.discard(None)
    count = post(f"/collections/{chunk_col}/points/count", {"exact": True})["count"]
    mapping = set(json.loads((R / "menon_chunk_mapping.json").read_text()))
    miss = mapping - excluded - ids
    detail = {"collection": chunk_col, "points": count, "distinct_ragu_ids": len(ids), "mapping_ids": len(mapping),
              "excluded": len(excluded), "mapping_ids_missing": len(miss), "points_not_in_mapping": len(ids - mapping),
              "missing_sample": sorted(miss)[:5], "scroll_pages": pages, "seconds": round(time.time() - t0)}
    check("every menon_chunk_mapping id is a point", not miss and len(mapping) > 0, detail)
else:
    check("every menon_chunk_mapping id is a point", False, {"collection": chunk_col})

dq = stand / "document_quality.json"
try:
    n_dq = len(json.loads(dq.read_text()))
except Exception as exc:  # noqa: BLE001
    n_dq = None
check("document_quality.json", n_dq is not None and n_dq > 0, {"path": str(dq), "documents_changed": n_dq})
side = R / "chunk_collection_check.json"
if side.is_file():
    res["v6_chain_chunk_check"] = json.loads(side.read_text())
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("VERIFY", "OK" if not res["failed"] else "FAILED: " + ", ".join(res["failed"]))
sys.exit(1 if res["failed"] else 0)
