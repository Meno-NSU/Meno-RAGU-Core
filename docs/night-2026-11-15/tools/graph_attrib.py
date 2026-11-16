"""Judge-free attribution: for every vital nugget the judge marked as supported in RAGU's answer, is its content found in
RAGU's graph elements, its document chunks, both or neither? Salient tokens = numbers and word stems (first 6 letters of
words of 5+ letters). A source "has" the nugget when it holds >= 60% of the nugget's salient tokens.

usage: python3 graph_attrib.py SET RAGU_RECORDS[#variant] TASK_GLOB OUT_GLOB
"""
import collections, glob, json, re, sys
s, rec_p, task_glob, out_glob = sys.argv[1:5]
variant = None
if "#" in rec_p:
    rec_p, variant = rec_p.split("#", 1)
pk = json.load(open("/home/devuser/meno-night-1115/judge/keys/point_keys.json"))
recs = {r["item_id"]: r for r in map(json.loads, open(rec_p)) if r.get("ok") and (not variant or r.get("variant") == variant)}
tasks = {it["id"]: it for p in glob.glob(task_glob) for it in json.load(open(p))}
outs = {r["id"]: r for p in glob.glob(out_glob) for r in map(json.loads, open(p))}
WORD = re.compile(r"\d+(?:[.,]\d+)?|[a-zа-яё]{5,}", re.I)
def toks(t):
    return {w if w[0].isdigit() else w.lower()[:6] for w in WORD.findall(t or "")}
def cov(nt, src):
    return len(nt & src) / len(nt) if nt else 0.0
cat = {}
for path in ("/home/devuser/meno-night-1115/setA/items_v3new_ref.jsonl", "/home/devuser/meno-night-1115/setB/setB_items.jsonl"):
    for line in open(path, encoding="utf-8"):
        x = json.loads(line); cat[x["id"]] = x["category"]
agg = collections.defaultdict(collections.Counter); n_items = collections.Counter(); share = collections.defaultdict(list)
for pid, it in tasks.items():
    k = pk.get(pid)
    if not k or k["system"] != "ragu" or pid not in outs or k["qid"] not in recs:
        continue
    r = recs[k["qid"]]
    gp, dp, in_graph = [], [], False
    for c in r.get("contexts") or []:
        t = c["text"]
        if t.startswith("**Элемент графа"):
            in_graph = True
        elif t.startswith("**Название документа:**") or t.startswith("====="):
            in_graph = False
        (gp if in_graph else dp).append(t)
    g, d = " ".join(gp), " ".join(dp)
    gt, dt = toks(g), toks(d)
    labels = outs[pid].get("labels") or {}
    typ = cat.get(k["qid"], "?")
    n_items[typ] += 1
    share[typ].append(len(g) / max(1, len(g) + len(d)))
    for n in it["nuggets"]:
        if n.get("importance") != "vital" or labels.get(n["id"]) != "support":
            continue
        nt = toks(n.get("text") or n.get("fact"))
        if len(nt) < 2:
            continue
        cg, cd = cov(nt, gt) >= 0.6, cov(nt, dt) >= 0.6
        cls = "both" if cg and cd else "graph_only" if cg else "chunks_only" if cd else "neither"
        agg[typ][cls] += 1; agg["ALL"][cls] += 1
for typ in sorted(agg):
    c = agg[typ]; tot = sum(c.values())
    sh = share.get(typ) or [x for v in share.values() for x in v]
    print(f"{s} {typ:14s} supported vital nuggets {tot:4d} | graph_only {c['graph_only']/tot:.2f} both {c['both']/tot:.2f} "
          f"chunks_only {c['chunks_only']/tot:.2f} neither {c['neither']/tot:.2f} | graph share of context chars {sum(sh)/len(sh):.2f}")
