"""Judge task files (blind) for the night-1115 comparison.

usage: python3 make_tasks.py CONFIG.json
CONFIG: {"sets": {"A": {"items": path, "nuggets_dir": path|null}, "B": {"items": path}},
         "answers": {"A": {"ragu": [paths], "menon": [paths]}, "B": {...}},
         "pairs": [["A", "ragu", "menon"], ["B", "ragu", "menon"]], "point_size": 15, "pair_size": 25, "j2_share": 0.2}
Records: bench records (item_id, ok, response, contexts) or RAGU records (variant, item_id, ...); a path may end with
"#variant" to take one variant. Later paths override earlier ones per item (e.g. a rerun of some items).
"""
import glob, hashlib, json, os, random, sys

ROOT = "/home/devuser/meno-night-1115/judge"
cfg = json.load(open(sys.argv[1]))
rng = random.Random(1115)


def load_items(spec):
    items = {}
    for line in open(spec["items"], encoding="utf-8"):
        r = json.loads(line)
        hist = r.get("history") or []
        if isinstance(hist, str):
            hist = []
        nug = [{"id": f"n{i + 1}", "text": k.get("fact"), "importance": k.get("importance") or "vital"}
               for i, k in enumerate(r.get("key_facts") or []) if k.get("fact")]
        items[r["id"]] = {"question": r["question"], "category": r.get("category"), "reference": r["ground_truth"],
                          "history": [{"role": h.get("role"), "content": (h.get("content") or "")[:500]} for h in hist][-6:],
                          "nuggets": nug, "kind": "facts", "gold_urls": r.get("gold_urls") or []}
    if spec.get("nuggets_dir"):
        for f in glob.glob(os.path.join(spec["nuggets_dir"], "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    n = json.loads(line)
                    if n["id"] in items:
                        items[n["id"]]["nuggets"] = n["nuggets"]
                        items[n["id"]]["kind"] = n.get("kind") or "facts"
                        items[n["id"]]["reference_problem"] = n.get("reference_problem")
    return items


def ctx_text(rec):
    blocks = []
    for c in rec.get("contexts") or []:
        blocks.append(c.get("text") if isinstance(c, dict) else str(c))
    text = "\n---\n".join(b for b in blocks if b)
    return text[:3000] + (" […]" if len(text) > 3000 else "")


def load_answers(paths):
    out = {}
    for p in paths:
        variant = None
        if "#" in p:
            p, variant = p.split("#", 1)
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if variant and r.get("variant") != variant:
                continue
            if r.get("ok") and (r.get("response") or "").strip():
                out[r["item_id"]] = {"answer": r["response"], "context": ctx_text(r)}
    return out


def chunked(rows, size, key):
    rows = rows[:]
    rng.shuffle(rows)
    files = []
    while rows:
        cur, seen, rest = [], set(), []
        for r in rows:
            if len(cur) < size and key(r) not in seen:
                cur.append(r); seen.add(key(r))
            else:
                rest.append(r)
        files.append(cur); rows = rest
    return files


def in_j2(qid):
    return int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 1000 < cfg.get("j2_share", 0.2) * 1000


items = {s: load_items(spec) for s, spec in cfg["sets"].items()}
answers = {s: {sys_: load_answers(paths) for sys_, paths in d.items()} for s, d in cfg["answers"].items()}
point, keys = [], {}
for s, by_sys in answers.items():
    for sys_, ans in by_sys.items():
        if cfg.get("point_systems") and sys_ not in cfg["point_systems"]:
            continue
        for qid, a in ans.items():
            if qid not in items[s]:
                continue
            it = items[s][qid]
            pid = "p" + hashlib.sha1(f"{s}|{sys_}|{qid}".encode()).hexdigest()[:10]
            keys[pid] = {"set": s, "system": sys_, "qid": qid}
            point.append({"id": pid, "question": it["question"], "history": it["history"], "reference": it["reference"],
                          "nuggets": it["nuggets"], "kind": it["kind"], "answer": a["answer"], "context": a["context"],
                          "_q": f"{s}|{qid}"})
pairs, pkeys = [], {}
for s, x, y in cfg.get("pairs", []):
    for qid in sorted(set(answers[s][x]) & set(answers[s][y]) & set(items[s])):
        it = items[s][qid]
        for order, (a1, a2) in (("xy", (x, y)), ("yx", (y, x))):
            pid = "q" + hashlib.sha1(f"{s}|{qid}|{x}|{y}|{order}".encode()).hexdigest()[:10]
            pkeys[pid] = {"set": s, "qid": qid, "answer_1": a1, "answer_2": a2}
            pairs.append({"id": pid, "question": it["question"], "history": it["history"], "reference": it["reference"],
                          "answer_1": answers[s][a1][qid]["answer"], "answer_2": answers[s][a2][qid]["answer"],
                          "_q": f"{s}|{qid}", "_order": order})
strip = lambda rows: [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
os.makedirs(f"{ROOT}/tasks", exist_ok=True); os.makedirs(f"{ROOT}/keys", exist_ok=True)
PFX = cfg.get("prefix", "")
manifest = {"point_j1": [], "point_j2": [], "pair_j1": [], "pair_j2": []}
for i, f in enumerate(chunked(point, cfg["point_size"], lambda r: r["_q"])):
    name = f"point_{PFX}{i:03d}"; json.dump(strip(f), open(f"{ROOT}/tasks/{name}.json", "w"), ensure_ascii=False)
    manifest["point_j1"].append(name)
for i, f in enumerate(chunked([r for r in point if in_j2(r["_q"])], cfg["point_size"], lambda r: r["_q"])):
    name = f"point2_{PFX}{i:03d}"; json.dump(strip(f), open(f"{ROOT}/tasks/{name}.json", "w"), ensure_ascii=False)
    manifest["point_j2"].append(name)
for tag, rows in (("pair", pairs), ("pair2", [r for r in pairs if in_j2(r["_q"])])):
    for order in ("xy", "yx"):
        for i, f in enumerate(chunked([r for r in rows if r["_order"] == order], cfg["pair_size"], lambda r: r["_q"])):
            name = f"{tag}_{order}_{PFX}{i:03d}"; json.dump(strip(f), open(f"{ROOT}/tasks/{name}.json", "w"), ensure_ascii=False)
            manifest["pair_j1" if tag == "pair" else "pair_j2"].append(name)
def merge(path, new):
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(new); json.dump(old, open(path, "w"), ensure_ascii=False)
merge(f"{ROOT}/keys/point_keys.json", keys); merge(f"{ROOT}/keys/pair_keys.json", pkeys)
json.dump(manifest, open(f"{ROOT}/keys/manifest_{PFX or 'main'}.json", "w"), ensure_ascii=False, indent=1)
print({k: len(v) for k, v in manifest.items()}, "point items", len(point), "pair items", len(pairs))
