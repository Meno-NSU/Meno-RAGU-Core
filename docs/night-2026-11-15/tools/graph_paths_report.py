"""Graph as navigator: per question type, how often the gold page is reached by RAGU's local (graph) channel, by its
naive (chunk vector) channel, by the graph only, by naive only; and whether graph-only gold pages reach the final context.

usage: python3 graph_paths_report.py PATHS.jsonl RAGU_RECORDS[#variant] [ITEMS_FOR_TYPES.jsonl]
"""
import collections, json, re, sys
from statistics import mean
norm = lambda u: re.sub(r"^https?://(www\.)?", "", (u or "").split("#")[0]).rstrip("/").lower()
paths_p, rec_p = sys.argv[1:3]
variant = None
if "#" in rec_p:
    rec_p, variant = rec_p.split("#", 1)
recs = {r["item_id"]: r for r in map(json.loads, open(rec_p)) if r.get("ok") and (not variant or r.get("variant") == variant)}
rows = [json.loads(l) for l in open(paths_p)]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    gold = set(r["gold"])
    if not gold:
        continue
    lp = {u for ps in r["local_pages"] for u in ps}; np_ = {u for ps in r["naive_pages"] for u in ps}
    g_l, g_n = gold & lp, gold & np_
    final = {norm(u) for u in (recs.get(r["item_id"], {}).get("context_pages") or [])}
    t = r.get("category") or "?"
    for key in ("ALL", ("G" if t.startswith("G") else "L" if t.startswith("L") else t), t):
        a = agg[key]
        a["n"].append(1)
        a["local_any"].append(bool(g_l)); a["naive_any"].append(bool(g_n))
        a["local_only"].append(bool(g_l - g_n)); a["naive_only"].append(bool(g_n - g_l))
        a["union_any"].append(bool(g_l | g_n))
        a["local_recall"].append(len(g_l) / len(gold)); a["naive_recall"].append(len(g_n) / len(gold))
        a["union_recall"].append(len(g_l | g_n) / len(gold))
        if r["item_id"] in recs:
            a["final_recall"].append(len(gold & final) / len(gold))
            a["graph_only_in_final"].append(bool((g_l - g_n) & final))
        a["n_local_chunks"].append(len(r["local_chunks"])); a["n_naive_chunks"].append(len(r["naive_chunks"]))
print(f"{'group':10s} {'n':>4s} | any gold: local naive union | graph-only naive-only | page recall: local naive union final | graph-only gold in final | chunks local/naive")
for k in sorted(agg, key=lambda x: (x != "ALL", len(x) > 1, x)):
    a = agg[k]; m = lambda f: mean(a[f]) if a[f] else float("nan")
    print(f"{k:10s} {len(a['n']):4d} | {m('local_any'):.2f} {m('naive_any'):.2f} {m('union_any'):.2f} | {m('local_only'):.2f} {m('naive_only'):.2f} | "
          f"{m('local_recall'):.2f} {m('naive_recall'):.2f} {m('union_recall'):.2f} {m('final_recall'):.2f} | {m('graph_only_in_final'):.2f} | {m('n_local_chunks'):.0f}/{m('n_naive_chunks'):.0f}")
