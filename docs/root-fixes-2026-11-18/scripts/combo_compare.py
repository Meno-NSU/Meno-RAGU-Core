"""Judge-free check of main 00233d3's defaults (rrfw + sources, combo1118) against the sources arm of ctx1118 (rrf +
sources) on the same questions: errors, label leaks, answer length, 'information missing' answers, sources and context
sizes, and how much of the context's chunk set the two share (rrfw breaks the rank ties differently)."""
import json, re, statistics as st, sys
sys.path.insert(0, "/home/m.kulakov/meno-v2/ctx1118")
from leaks import PATTERNS
NO = re.compile(r"нет (?:информации|сведений|данных)|не (?:содержится|найден|удалось найти|указан)|отсутству", re.I)

def load(path):
    return {r["item_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}

def chunks(r):
    return {str(c) for x in r.get("contexts") or [] for c in (x.get("chunk_ids") or [])}

for job in ("core", "dislikes"):
    new = load(f"/home/m.kulakov/meno-v2/combo1118/answers/{job}.records.jsonl")
    old = load(f"/home/m.kulakov/meno-v2/ctx1118/answers/{job}.sources.records.jsonl")
    ids = sorted(set(new) & set(old))
    ok_new = [i for i in ids if new[i].get("ok")]; err = len(ids) - len(ok_new)
    def rate(d, f):
        return sum(bool(f(d[i].get("response") or "")) for i in ok_new) / max(len(ok_new), 1)
    strong = lambda a: PATTERNS["numbered"].search(a) or PATTERNS["graph"].search(a) or PATTERNS["sources_style"].search(a)
    jac = [len(chunks(new[i]) & chunks(old[i])) / max(len(chunks(new[i]) | chunks(old[i])), 1) for i in ok_new
           if chunks(new[i]) or chunks(old[i])]
    same = sum(chunks(new[i]) == chunks(old[i]) for i in ok_new)
    print(f"{job}: {len(ids)} questions, errors {err} | strong labels {rate(old, strong):.1%} -> {rate(new, strong):.1%} | "
          f"meta {rate(old, PATTERNS['meta'].search):.1%} -> {rate(new, PATTERNS['meta'].search):.1%} | "
          f"info missing {rate(old, NO.search):.1%} -> {rate(new, NO.search):.1%}")
    print(f"   median chars {st.median(len(old[i]['response']) for i in ok_new):.0f} -> "
          f"{st.median(len(new[i]['response']) for i in ok_new):.0f} | sources/answer "
          f"{st.mean(len(old[i].get('sources') or []) for i in ok_new):.2f} -> {st.mean(len(new[i].get('sources') or []) for i in ok_new):.2f}"
          f" | context blocks {st.mean(len(old[i].get('contexts') or []) for i in ok_new):.1f} -> "
          f"{st.mean(len(new[i].get('contexts') or []) for i in ok_new):.1f} | same chunk set {same}/{len(ok_new)}, "
          f"mean Jaccard {st.mean(jac) if jac else 0:.2f} | total s median {st.median(old[i]['total_s'] for i in ok_new):.1f} -> "
          f"{st.median(new[i]['total_s'] for i in ok_new):.1f}")
