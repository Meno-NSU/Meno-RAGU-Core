import json, random, re, sys
sys.path.insert(0, "/home/m.kulakov/meno-v2/ground1118")
from signals import stems, fragments
recs = {r["item_id"]: r for r in map(json.loads, open("core_with_ctx.jsonl", encoding="utf-8"))}
lab = {l["item_id"]: l for l in map(json.loads, open("labels_core.jsonl", encoding="utf-8"))}
rows = list(map(json.loads, open("core.signals.jsonl", encoding="utf-8")))
cases = [(r["item_id"], s) for r in rows for s in r["sentences"] if "unsupported" in s["bad"]]
random.seed(7)
for item_id, s in random.sample(cases, 5):
    frags = fragments([c.get("text") or "" for c in recs[item_id]["contexts"]])
    st = stems(s["sentence"])
    best = sorted(frags, key=lambda f: -len(st & stems(f)))[:2]
    claim = max(lab[item_id]["unsupported"], key=lambda c: len(stems(str(c)) & st))
    print(f"### {item_id}\nSENTENCE: {s['sentence'][:300]}\nJUDGE: {str(claim)[:200]}")
    for f in best:
        print("  CTX:", re.sub(r"\s+", " ", f)[:420])
    print()
