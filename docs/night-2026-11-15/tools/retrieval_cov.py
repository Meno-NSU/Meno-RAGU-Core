"""Judge-free retrieval features per question: share of gold documents in each system's context, whether all gold
documents are there, the number of graph elements in RAGU's prompt, and latency per system.

usage: python3 retrieval_cov.py SET ITEMS RAGU_RECORDS[#variant] MENON_RECORDS OUT.json
"""
import json, re, sys
from statistics import mean, median
norm = lambda u: re.sub(r"^https?://(www\.)?", "", (u or "").split("#")[0]).rstrip("/").lower()
s, items_p, ragu_p, menon_p, out_p = sys.argv[1:6]
items = {json.loads(l)["id"]: json.loads(l) for l in open(items_p, encoding="utf-8")}
variant = None
if "#" in ragu_p:
    ragu_p, variant = ragu_p.split("#", 1)
ragu = {r["item_id"]: r for r in map(json.loads, open(ragu_p, encoding="utf-8")) if r.get("ok") and (not variant or r.get("variant") == variant)}
menon = {r["item_id"]: r for r in map(json.loads, open(menon_p, encoding="utf-8")) if r.get("ok")}
rows = {}
for qid, it in items.items():
    gold = {norm(u) for u in it.get("gold_urls") or []}
    row = {"type": it.get("category"), "n_gold": len(gold)}
    if qid in ragu:
        r = ragu[qid]
        pages = {norm(u) for u in r.get("context_pages") or []}
        elems = sum(1 for c in r.get("contexts") or [] if "Элемент графа" in (c.get("text") or "") or "Связь" in (c.get("text") or "")[:40])
        row.update(ragu_cov=len(gold & pages) / len(gold) if gold else None, ragu_elements=elems, ragu_total_s=r.get("total_s"))
    if qid in menon:
        m = menon[qid]
        pages = {norm(c.get("url")) for c in m.get("contexts") or [] if isinstance(c, dict)}
        row.update(menon_cov=len(gold & pages) / len(gold) if gold else None, menon_total_s=m.get("total_s"), menon_ttft_s=m.get("ttft_s"))
    rows[qid] = row
json.dump(rows, open(out_p, "w"), ensure_ascii=False, indent=0)
both = [r for r in rows.values() if r.get("ragu_cov") is not None and r.get("menon_cov") is not None]
def summ(sub, label):
    if not sub:
        return
    print(f"{label:28s} n={len(sub):4d} gold share ragu {mean(r['ragu_cov'] for r in sub):.3f} menon {mean(r['menon_cov'] for r in sub):.3f}"
          f" | all gold ragu {mean(r['ragu_cov'] == 1 for r in sub):.3f} menon {mean(r['menon_cov'] == 1 for r in sub):.3f}")
summ(both, f"{s} ALL")
types = sorted({r["type"] for r in both})
if s == "B":
    summ([r for r in both if r["type"].startswith("G")], "G types"); summ([r for r in both if r["type"].startswith("L")], "L types")
    summ([r for r in both if r["type"].startswith("G") and r["n_gold"] >= 2], "G with >=2 gold docs")
for t in types:
    summ([r for r in both if r["type"] == t], t)
lat_r = [r["ragu_total_s"] for r in rows.values() if r.get("ragu_total_s")]
lat_m = [r["menon_total_s"] for r in rows.values() if r.get("menon_total_s")]
if lat_r: print(f"latency ragu median {median(lat_r):.1f} s (n={len(lat_r)})")
if lat_m: print(f"latency menon median {median(lat_m):.1f} s (n={len(lat_m)})")
