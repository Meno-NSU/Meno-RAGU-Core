"""The same answers judged twice by qwen: the 18.11 benchmark (context cut to 12,000 characters) and judge1118b (up to
40,000). Per system and set, over the items judged both times: share with unsupported claims, with a contradiction
of the reference, and mean share of supported nuggets; flips of 'has unsupported claims' between the two runs.
usage: cut_vs_full.py OLD_JUDGE_DIR NEW_JUDGE_DIR SYSTEM..."""
import ast, glob, json, sys

def lst(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except Exception:
        try:
            return ast.literal_eval(v)
        except Exception:
            return []

def load(d):
    keys = json.load(open(f"{d}/keys/point_keys.json"))
    out = {}
    for p in glob.glob(f"{d}/out/*.qwen.jsonl"):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); k = keys.get(r["id"])
            if k:
                lab = r.get("labels") or {}
                out[(k["set"], k["system"], k["qid"])] = {
                    "uns": bool(lst(r.get("unsupported"))), "con": bool(lst(r.get("contradicts_reference"))),
                    "sup": (sum(v == "support" for v in lab.values()) / len(lab)) if lab else None,
                    "type": r.get("response_type")}
    return out

old, new = load(sys.argv[1]), load(sys.argv[2])
print(f"{'set':3} {'system':8} {'n':>5} | {'unsupported 12k':>15} {'full':>6} | {'contradicts 12k':>15} {'full':>6} | "
      f"{'nuggets 12k':>11} {'full':>6} | flips uns: 1->0 / 0->1")
for sys_ in sys.argv[3:]:
    for s in ("V3", "B"):
        ks = [k for k in old if k[0] == s and k[1] == sys_ and k in new]
        if not ks:
            continue
        n = len(ks); f = lambda d, m: sum(d[k][m] for k in ks) / n
        sup = lambda d: sum(d[k]["sup"] for k in ks if d[k]["sup"] is not None) / max(1, sum(d[k]["sup"] is not None for k in ks))
        f10 = sum(old[k]["uns"] and not new[k]["uns"] for k in ks); f01 = sum(new[k]["uns"] and not old[k]["uns"] for k in ks)
        print(f"{s:3} {sys_:8} {n:5d} | {f(old, 'uns'):15.3f} {f(new, 'uns'):6.3f} | {f(old, 'con'):15.3f} {f(new, 'con'):6.3f} | "
              f"{sup(old):11.3f} {sup(new):6.3f} | {f10} / {f01}")
