"""Paired leak counts of the two context styles (ctx_answers.py records): per item, does the legacy answer and does the
sources answer repeat an internal label of the context (leaks.py patterns)? Strong = a numbered section or fragment,
the graph or its elements and types, or a section heading of either style; meta = «в предоставленных документах» etc.
usage: leaks_pair.py ANSWERS_DIR [JOB ...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaks import PATTERNS, answer_of

def flags(a):
    f = {k: bool(p.search(a)) for k, p in PATTERNS.items()}
    f["strong"] = f["numbered"] or f["graph"] or f["sources_style"]
    return f

d = sys.argv[1]
for job in sys.argv[2:] or ["core", "setB", "synth", "dislikes"]:
    A = {r["item_id"]: r for r in map(json.loads, open(f"{d}/{job}.legacy.records.jsonl", encoding="utf-8")) if r.get("ok")}
    B = {r["item_id"]: r for r in map(json.loads, open(f"{d}/{job}.sources.records.jsonl", encoding="utf-8")) if r.get("ok")}
    ids = sorted(set(A) & set(B)); n = len(ids)
    row = {k: [0, 0] for k in ("strong", "numbered", "graph", "sources_style", "meta")}
    only = {"legacy": 0, "sources": 0, "both": 0}
    for i in ids:
        fa, fb = flags(answer_of(A[i]) or ""), flags(answer_of(B[i]) or "")
        for k in row:
            row[k][0] += fa[k]; row[k][1] += fb[k]
        if fa["strong"] and fb["strong"]: only["both"] += 1
        elif fa["strong"]: only["legacy"] += 1
        elif fb["strong"]: only["sources"] += 1
    pct = lambda x: f"{100 * x / max(n, 1):.1f}%"
    print(f"{job}: {n} pairs | strong labels legacy {row['strong'][0]} ({pct(row['strong'][0])}) -> sources "
          f"{row['strong'][1]} ({pct(row['strong'][1])}); only legacy {only['legacy']}, only sources {only['sources']}, "
          f"both {only['both']} | numbered {row['numbered'][0]}->{row['numbered'][1]}, graph {row['graph'][0]}->"
          f"{row['graph'][1]}, headings {row['sources_style'][0]}->{row['sources_style'][1]} | meta "
          f"{pct(row['meta'][0])} -> {pct(row['meta'][1])}")
