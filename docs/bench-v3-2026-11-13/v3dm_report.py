"""RAGU fixes on the 538 targeted questions of benchmark v3 (items_dm.jsonl: every multi-turn, meta-dialogue,
time-sensitive, unanswerable and long-dialogue question, plus the questions where RAGU v5k missed the gold page):
means with bootstrap CI per run, paired differences for the agreed comparisons, by category.

Runs: m11 (Menon 1.1), ragu-v5k (before today), ragu-v5k-dm (dialogue memory + turn analysis + date rule),
ragu-v5k-fix (dm + one cross-encoder scale + document dates), ragu-v5k-hyb (fix + Menon's 8 context chunks),
ragu-v5k-mg (Menon's chunks + RAGU's graph elements, no RAGU documents), ragu-v5k-mo (Menon's chunks only, through
RAGU's prompt and dialogue code).

usage: python v3dm_report.py OUT_DIR > v3dm/report.json   (OUT_DIR holds <run>-ref/ and <run>-noref/ batch CSVs)
"""
import csv, glob, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
RUNS = ["m11", "ragu-v5k", "ragu-v5k-dm", "ragu-v5k-fix", "ragu-v5k-hyb", "ragu-v5k-mg", "ragu-v5k-mo"]
PAIRS = [("ragu-v5k-dm", "ragu-v5k", "диалоговые правки"),
         ("ragu-v5k-fix", "ragu-v5k-dm", "единая шкала поиска + даты"),
         ("ragu-v5k-fix", "m11", "RAGU со всеми исправлениями против Менона 1.1"),
         ("ragu-v5k-mg", "ragu-v5k-mo", "вклад графа поверх документов Менона"),
         ("ragu-v5k-mo", "m11", "промпт и диалог RAGU против Менона на тех же документах"),
         ("ragu-v5k-hyb", "ragu-v5k-mg", "собственные документы RAGU поверх документов Менона + графа"),
         ("ragu-v5k-hyb", "ragu-v5k-fix", "документы Менона поверх RAGU"),
         ("ragu-v5k-hyb", "m11", "гибрид против Менона 1.1"),
         ("ragu-v5k-mg", "m11", "документы Менона + граф против Менона 1.1")]
REF = {"answer_correctness": "answer_correctness", "factual_correctness": "factual_correctness(mode=f1)",
       "faithfulness": "faithfulness", "context_recall": "context_recall", "answer_relevancy": "answer_relevancy"}
NOREF = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy"}
rng = np.random.default_rng(0)

items = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "v3new" / "items_v3new.jsonl", encoding="utf-8")}
target = [json.loads(l)["id"] for l in open(HERE / "v3dm" / "items_dm.jsonl", encoding="utf-8")]


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def load(run, part):
    rows = {}
    for p in sorted(glob.glob(str(OUT / f"{run}-{part}" / "batch_*.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rows[r["item_id"]] = r
    return rows


def boot(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if len(v) == 0:
        return None
    b = rng.choice(v, (2000, len(v))).mean(axis=1)
    return [round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4),
            int(len(v))]


scores = {}  # run -> metric key ("ref:answer_correctness") -> item -> value
for run in RUNS:
    s = defaultdict(dict)
    for part, metrics in (("ref", REF), ("noref", NOREF)):
        rows = load(run, part)
        for iid in target:
            if iid in rows:
                for key, col in metrics.items():
                    s[f"{part}:{key}"][iid] = num(rows[iid].get(col))
    scores[run] = s
report = {"questions": len(target), "with_reference": sum(bool((items[i].get("ground_truth") or "").strip())
                                                           for i in target if i in items), "runs": {}, "pairs": [],
          "by_category": {}}
for run in RUNS:
    report["runs"][run] = {k: boot(list(v.values())) for k, v in sorted(scores[run].items())}
for a, b, label in PAIRS:
    row = {"a": a, "b": b, "label": label}
    for key in ("ref:answer_correctness", "ref:factual_correctness", "ref:faithfulness", "ref:context_recall",
                "noref:faithfulness"):
        sa, sb = scores[a].get(key, {}), scores[b].get(key, {})
        common = [i for i in target if sa.get(i) is not None and sb.get(i) is not None]
        row[key] = boot([sa[i] - sb[i] for i in common])
    report["pairs"].append(row)
cats = defaultdict(list)
for i in target:
    cats[items[i]["category"]].append(i)
for cat, ids in sorted(cats.items(), key=lambda kv: -len(kv[1])):
    entry = {"n": len(ids)}
    for run in RUNS:
        ac = [scores[run].get("ref:answer_correctness", {}).get(i) for i in ids]
        ac = [x for x in ac if x is not None]
        entry[run] = round(sum(ac) / len(ac), 4) if ac else None
        entry[run + ":n"] = len(ac)
    report["by_category"][cat] = entry
print(json.dumps(report, ensure_ascii=False, indent=1))
