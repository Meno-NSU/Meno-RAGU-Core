"""Does the nugget judge's 12,000-character context cut explain its 'unsupported' flags? For every judged sentence of the
core v6k2 answers: the context fragment sharing the most word stems with it (share of the sentence's stems found in it)
and where that fragment starts in the text the judge was given (contexts joined with '\n---\n', cut at 12,000)."""
import json, sys
sys.path.insert(0, "/home/m.kulakov/meno-v2/ground1118")
from signals import stems, fragments
CUT = 12000
recs = {r["item_id"]: r for r in map(json.loads, open("core_with_ctx.jsonl", encoding="utf-8"))}
rows = list(map(json.loads, open("core.signals.jsonl", encoding="utf-8")))
stat = {"bad": [], "other": []}
for r in rows:
    blocks = [c.get("text") or "" for c in recs[r["item_id"]]["contexts"]]
    joined, offsets = "", []
    for b in blocks:
        if joined:
            joined += "\n---\n"
        offsets.append(len(joined)); joined += b
    frags = []  # (start offset in the joined text, fragment)
    for b, off in zip(blocks, offsets):
        pos = 0
        for f in fragments([b]):
            k = b.find(f[:60], pos); k = pos if k < 0 else k
            frags.append((off + k, f)); pos = k
    for s in r["sentences"]:
        st = stems(s["sentence"])
        if len(st) < 4:
            continue
        cover, start = max(((len(st & stems(f)) / len(st), off) for off, f in frags), default=(0.0, 0))
        stat["bad" if "unsupported" in s["bad"] else "other"].append((cover, start))
for k, v in stat.items():
    strong = [x for x in v if x[0] >= 0.7]
    beyond = [x for x in strong if x[1] >= CUT]
    print(f"{k}: {len(v)} sentences; best fragment holds >=70% of the sentence's stems for {len(strong)} "
          f"({len(strong) / max(len(v), 1):.0%}); of those, the fragment starts past {CUT} chars for {len(beyond)} "
          f"({len(beyond) / max(len(strong), 1):.0%})")
lens = [len("\n---\n".join(c.get("text") or "" for c in r["contexts"])) for r in recs.values()]
lens.sort()
print(f"v6k2 core context chars: median {lens[len(lens) // 2]}, p90 {lens[int(len(lens) * .9)]}, "
      f"share past {CUT}: {sum(max(0, l - CUT) for l in lens) / max(sum(lens), 1):.0%} of all context text; "
      f"answers whose context is cut: {sum(l > CUT for l in lens)}/{len(lens)}")
