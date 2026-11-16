"""Scores of the night-1115 judged comparison: per set, question type and system; paired differences with clustered
bootstrap CIs; pairwise outcomes over both orders; judge agreement; complementarity. Writes results.json + prints tables.

usage: python3 analyze.py OUT_DIR
"""
import collections, glob, json, os, random, sys
from statistics import mean

ROOT = "/home/devuser/meno-night-1115"
out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
pkeys = json.load(open(f"{ROOT}/judge/keys/point_keys.json")); qkeys = json.load(open(f"{ROOT}/judge/keys/pair_keys.json"))
items = {}
for s, path in (("A", f"{ROOT}/setA/items_v3new_ref.jsonl"), ("B", f"{ROOT}/setB/setB_items.jsonl")):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); items[(s, r["id"])] = r
tasks = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{ROOT}/judge/tasks/*.json")}
nug_by_pid = {it["id"]: it["nuggets"] for name, t in tasks.items() if name.startswith("point") for it in t}


def load_out(prefix):
    res = {}
    for p in glob.glob(f"{ROOT}/judge/out/{prefix}_*.jsonl"):
        judge = p.rsplit(".", 2)[-2]
        for line in open(p, encoding="utf-8"):
            if line.strip():
                d = json.loads(line); res[(d["id"], judge)] = d
    return res


point = {**load_out("point"), **load_out("point2")}
pair = {**load_out("pair"), **load_out("pair2")}


def score(pid, d):
    nugs = nug_by_pid.get(pid) or []
    labels = d.get("labels") or {}
    vital = [n for n in nugs if n.get("importance") == "vital"] or nugs
    if not vital:
        return None
    lab = lambda n: labels.get(n["id"], "not_support")
    strict = mean(lab(n) == "support" for n in vital)
    soft = mean({"support": 1, "partial": .5}.get(lab(n), 0) for n in nugs)
    contra_n = any(lab(n) == "contradicts" for n in nugs)
    contra_c = len(d.get("contradicts_reference") or [])
    unsup = len(d.get("unsupported") or [])
    nc = max(d.get("n_claims") or 0, 1)
    return {"vital_recall": strict, "recall": soft, "contradiction": float(contra_n or contra_c > 0),
            "contra_rate": min(contra_c / nc, 1), "unsupported_rate": min(unsup / nc, 1), "any_unsupported": float(unsup > 0),
            "refusal": float(d.get("response_type") == "refusal"), "solved": float(strict == 1 and not (contra_n or contra_c))}


rows = []  # one per (set, system, qid, judge)
for (pid, judge), d in point.items():
    k = pkeys.get(pid)
    if not k:
        continue
    sc = score(pid, d)
    if sc is None:
        continue
    it = items[(k["set"], k["qid"])]
    rows.append({**k, "judge": judge, "type": it.get("category"), "cluster": it.get("topic") if k["set"] == "B" else k["qid"], **sc})
RERUN = {"menon60": "menon"}  # a rerun system replaces the original one for the questions it covers
rerun_q = {(r["set"], r["qid"], RERUN[r["system"]]) for r in rows if r["system"] in RERUN}
rows = [r for r in rows if (r["set"], r["qid"], r["system"]) not in rerun_q]
for r in rows:
    if r["system"] in RERUN:
        r["system"] = RERUN[r["system"]]
J1 = "sonnet"
j1 = [r for r in rows if r["judge"] == J1]
by = collections.defaultdict(dict)
for r in j1:
    by[(r["set"], r["qid"])][r["system"]] = r
METRICS = ["vital_recall", "recall", "contradiction", "unsupported_rate", "any_unsupported", "refusal", "solved"]
rng = random.Random(1115)


def boot(diffs_by_cluster, n=4000):
    clusters = list(diffs_by_cluster)
    if not clusters:
        return (None, None)
    est = []
    for _ in range(n):
        sample = [rng.choice(clusters) for _ in clusters]
        vals = [v for c in sample for v in diffs_by_cluster[c]]
        est.append(mean(vals))
    est.sort()
    return (round(est[int(.025 * n)], 3), round(est[int(.975 * n)], 3))


def group_key(s, t):
    if s == "B":
        return "G (graph types)" if t.startswith("G") else "L (single-doc controls)"
    return t


results = {"point": [], "pair": [], "agreement": {}, "complementarity": []}
sets = sorted({r["set"] for r in j1})
for s in sets:
    systems = sorted({r["system"] for r in j1 if r["set"] == s})
    types = sorted({r["type"] for r in j1 if r["set"] == s})
    groups = [("ALL", None)] + ([(g, g) for g in ("G (graph types)", "L (single-doc controls)")] if s == "B" else []) + [(t, t) for t in types]
    for gname, gval in groups:
        for x in systems:
            for y in systems:
                if x >= y:
                    continue
                keys = [q for (ss, q), v in by.items() if ss == s and x in v and y in v and
                        (gval is None or v[x]["type"] == gval or group_key(s, v[x]["type"]) == gval)]
                if not keys:
                    continue
                row = {"set": s, "group": gname, "a": x, "b": y, "n": len(keys)}
                for m in METRICS:
                    va = [by[(s, q)][x][m] for q in keys]; vb = [by[(s, q)][y][m] for q in keys]
                    dc = collections.defaultdict(list)
                    for q in keys:
                        dc[by[(s, q)][x]["cluster"]].append(by[(s, q)][x][m] - by[(s, q)][y][m])
                    row[m] = {"a": round(mean(va), 3), "b": round(mean(vb), 3), "diff": round(mean(va) - mean(vb), 3), "ci": boot(dc)}
                results["point"].append(row)
    # complementarity (solved = all vital nuggets supported, no contradiction)
    for x in systems:
        for y in systems:
            if x >= y:
                continue
            keys = [q for (ss, q), v in by.items() if ss == s and x in v and y in v]
            c = collections.Counter((by[(s, q)][x]["solved"], by[(s, q)][y]["solved"]) for q in keys)
            results["complementarity"].append({"set": s, "a": x, "b": y, "n": len(keys), "both": c[(1.0, 1.0)],
                                               "only_a": c[(1.0, 0.0)], "only_b": c[(0.0, 1.0)], "neither": c[(0.0, 0.0)]})
# pairwise: per question and judge, combine the two orders
pq = collections.defaultdict(dict)
for (pid, judge), d in pair.items():
    k = qkeys.get(pid)
    if not k:
        continue
    a1, a2 = RERUN.get(k["answer_1"], k["answer_1"]), RERUN.get(k["answer_2"], k["answer_2"])
    is_rerun = a1 != k["answer_1"] or a2 != k["answer_2"]
    if not is_rerun and any((k["set"], k["qid"], sys_) in rerun_q for sys_ in (a1, a2)):
        continue  # superseded by the rerun's pairs
    win = {"1": a1, "2": a2}.get(str(d.get("overall")), "tie")
    pq[(k["set"], k["qid"], judge)][(a1, a2)] = win
outcomes = collections.defaultdict(list)
flips = collections.Counter()
for (s, q, judge), orders in pq.items():
    if len(orders) < 2:
        continue
    (o1, w1), (o2, w2) = list(orders.items())[:2]
    systems_pair = tuple(sorted(o1))
    final = w1 if w1 == w2 else "tie"
    flips[(judge, "flip" if w1 != w2 and "tie" not in (w1, w2) else "same")] += 1
    t = items[(s, q)].get("category")
    for g in dict.fromkeys(("ALL", group_key(s, t) if s == "B" else t, t)):
        outcomes[(s, judge, g, systems_pair)].append(final)
for (s, judge, g, sp), finals in sorted(outcomes.items()):
    a, b = sp
    wa, wb = finals.count(a), finals.count(b)
    results["pair"].append({"set": s, "judge": judge, "group": g, "a": a, "b": b, "n": len(finals), "a_wins": wa, "b_wins": wb,
                            "ties": len(finals) - wa - wb, "net_a": round((wa - wb) / len(finals), 3)})
results["pair_flips"] = {f"{k[0]}:{k[1]}": v for k, v in flips.items()}
# judge agreement on nugget labels (support vs not) and pairwise overall
lab_pairs = []
for (pid, judge), d in point.items():
    if judge == J1:
        continue
    o = point.get((pid, J1))
    if not o:
        continue
    for nid, lab in (d.get("labels") or {}).items():
        if nid in (o.get("labels") or {}):
            lab_pairs.append((o["labels"][nid] == "support", lab == "support"))
if lab_pairs:
    po = mean(a == b for a, b in lab_pairs); pa = mean(a for a, _ in lab_pairs); pb = mean(b for _, b in lab_pairs)
    pe = pa * pb + (1 - pa) * (1 - pb)
    results["agreement"]["nugget_support_kappa"] = round((po - pe) / (1 - pe), 3) if pe < 1 else None
    results["agreement"]["nugget_pairs"] = len(lab_pairs)
pw = [(d.get("overall"), pair[(pid, J1)].get("overall")) for (pid, judge), d in pair.items() if judge != J1 and (pid, J1) in pair]
if pw:
    results["agreement"]["pair_overall_agreement"] = round(mean(a == b for a, b in pw), 3); results["agreement"]["pair_items"] = len(pw)
json.dump(results, open(f"{out_dir}/results.json", "w"), ensure_ascii=False, indent=1)
for r in results["point"]:
    if r["group"] in ("ALL", "G (graph types)", "L (single-doc controls)") or r["set"] == "B":
        m = r["vital_recall"]; c = r["contradiction"]; u = r["any_unsupported"]
        print(f'{r["set"]} {r["group"][:24]:24s} {r["a"]}-{r["b"]} n={r["n"]:4d} vital {m["a"]:.3f}/{m["b"]:.3f} d={m["diff"]:+.3f} {m["ci"]} | contra {c["a"]:.2f}/{c["b"]:.2f} | unsup {u["a"]:.2f}/{u["b"]:.2f}')
for r in results["pair"]:
    if r["group"] in ("ALL", "G (graph types)", "L (single-doc controls)"):
        print("PAIR", r)
print("complementarity", results["complementarity"]); print("agreement", results["agreement"], results["pair_flips"])
