"""RAGAS side by side for the 18.11 benchmark: RAGU v6k (this run) against Menon 1.1 (16.11 run, reused), same judge
(qwen3.8-27b) and embedder (bge-m3). Per set and metric: the mean over all scored samples of each system (as
summary.json of ragas_eval_v3.py: NaN skipped), and the paired difference over the items both have a score for, with
a 95% bootstrap CI over items (10,000 resamples, seed 1115). Reads batch_*.csv of each run dir.
usage: python ragas_side.py OUT.json OUT.md NAME=RAGU_DIR,MENON_DIR [...] [--labels A,B]
"""
import csv, glob, json, math, os, random, sys

METRICS = ("answer_correctness", "faithfulness")
args = [a for a in sys.argv[1:] if not a.startswith("--labels")]
labels = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--labels=")), "RAGU v6k,Менон 1.1").split(",")
out_json, out_md, specs = args[0], args[1], args[2:]


def load(d):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "batch_*.csv"))):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["item_id"]] = r
    return rows


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def boot(d, n=10000, seed=1115):
    rng = random.Random(seed)
    k = len(d)
    est = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return round(est[int(.025 * n)], 4), round(est[int(.975 * n)], 4)


res = {"labels": labels, "sets": {}}
for spec in specs:
    name, dirs = spec.split("=", 1)
    a_dir, b_dir = dirs.split(",")
    A, Bm = load(a_dir), load(b_dir)
    summ = {}
    for side, d in (("a", a_dir), ("b", b_dir)):
        try:
            summ[side] = json.load(open(os.path.join(d, "summary.json")))
        except OSError:
            summ[side] = None
    s = {"dirs": [a_dir, b_dir], "items": [len(A), len(Bm)], "metrics": {}}
    for m in METRICS:
        av = [num(r.get(m)) for r in A.values()]
        bv = [num(r.get(m)) for r in Bm.values()]
        av, bv = [x for x in av if x is not None], [x for x in bv if x is not None]
        if not av and not bv:
            continue
        common = [q for q in A if q in Bm and num(A[q].get(m)) is not None and num(Bm[q].get(m)) is not None]
        diffs = [num(A[q][m]) - num(Bm[q][m]) for q in common]
        s["metrics"][m] = {"a": round(sum(av) / len(av), 4) if av else None, "n_a": len(av),
                           "b": round(sum(bv) / len(bv), 4) if bv else None, "n_b": len(bv),
                           "paired_n": len(common),
                           "paired_a": round(sum(num(A[q][m]) for q in common) / len(common), 4) if common else None,
                           "paired_b": round(sum(num(Bm[q][m]) for q in common) / len(common), 4) if common else None,
                           "diff": round(sum(diffs) / len(diffs), 4) if diffs else None,
                           "ci": boot(diffs) if len(diffs) > 1 else (None, None)}
    s["summary_json"] = {k: (v or {}).get("scores") for k, v in summ.items()}
    res["sets"][name] = s
json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
A_, B_ = labels
L = [f"# RAGAS: {A_} против {B_}", "", "Судья qwen3.8-27b, эмбеддер bge-m3. Среднее — по всем оценённым ответам системы "
     "(как summary.json); разность — по общим вопросам, 95% бутстреп-ДИ по вопросам.", "",
     f"| набор | метрика | {A_} (n) | {B_} (n) | общих | {A_} − {B_}, 95% ДИ |", "|---|---|---|---|---:|---|"]
for name, s in res["sets"].items():
    for m, x in s["metrics"].items():
        ci = f" [{x['ci'][0]:+.3f}; {x['ci'][1]:+.3f}]" if x["ci"][0] is not None else ""
        a = f"{x['a']:.3f} ({x['n_a']})" if x["a"] is not None else "—"
        b = f"{x['b']:.3f} ({x['n_b']})" if x["b"] is not None else "—"
        d = f"{x['diff']:+.3f}{ci}" if x["diff"] is not None else "—"
        L.append(f"| {name} | {m} | {a} | {b} | {x['paired_n']} | {d} |")
open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
