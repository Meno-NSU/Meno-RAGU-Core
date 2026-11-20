"""Read-only W5 case-level causal diagnostics. Outputs aggregate data and selected public-case excerpts."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", type=Path, required=True)
B = parser.parse_args().base_dir
J = B / "judge1119/judge"
items = {r["id"]: r for r in map(json.loads, (J / "data/items_d54d794.jsonl").open())}
keys = json.loads((J / "keys/point_keys.json").read_text())
systems = ["ragu_v6k3b16", "ragu_v6k3bq"]
pid_of = {(r["system"], r["qid"]): p for p, r in keys.items() if r["system"] in systems}
wanted = set(pid_of.values())
tasks = {}
for p in (J / "tasks").glob("point_*.json"):
    for r in json.loads(p.read_text()):
        if r["id"] in wanted:
            tasks[r["id"]] = r
judges = {}
for p in (J / "out").glob("point_*.jsonl"):
    judge = p.name.rsplit(".", 2)[-2]
    for line in p.open():
        r = json.loads(line)
        if r["id"] in wanted:
            judges[r["id"], judge] = r
answers = {}
for sys, names in zip(systems, [["v6k3b16", "r353_v6k3b16"], ["v6k3bq", "r353_v6k3bq"]], strict=False):
    answers[sys] = {
        r["item_id"]: r
        for name in names
        for r in map(json.loads, (B / f"rel1119b/answers/{name}.records.jsonl").open())
    }


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def score(t, j):
    core = [n for n in t["nuggets"] if not n["id"].startswith("f")]
    vital = [n for n in core if n.get("importance") == "vital"] or core
    return sum(j.get("labels", {}).get(n["id"]) == "support" for n in vital) / len(vital) if vital else 0


def cids(r):
    return {str(i) for c in r.get("contexts", []) for i in c.get("chunk_ids", [])}


def diagnostics(i):
    a, b = [answers[s][i] for s in systems]
    ca, cb = cids(a), cids(b)
    result = {
        "id": i,
        "profiles": items[i]["evaluation_profiles"],
        "history_turns": len(items[i].get("history") or []),
        "same_answer": a["response"] == b["response"],
        "same_contexts": digest(a.get("contexts")) == digest(b.get("contexts")),
        "same_queries": a.get("search_queries") == b.get("search_queries"),
        "same_context_texts": [(x["kind"], x["text"]) for x in a.get("contexts", [])]
        == [(x["kind"], x["text"]) for x in b.get("contexts", [])],
        "same_replayed_history": a.get("inproc", {}).get("replay_answers") == b.get("inproc", {}).get("replay_answers"),
        "context_counts": [len(a.get("contexts", [])), len(b.get("contexts", []))],
        "document_jaccard": len(ca & cb) / max(len(ca | cb), 1),
        "routes": [r.get("v2", {}).get("route") for r in [a, b]],
        "judges": {},
    }
    for judge in ["qwen", "gptoss"]:
        pids = [pid_of.get((s, i)) for s in systems]
        if all((p, judge) in judges and p in tasks for p in pids):
            ds = [judges[p, judge] for p in pids]
            ts = [tasks[p] for p in pids]
            scores = [score(t, j) for t, j in zip(ts, ds, strict=False)]
            flip = []
            for n in ts[0]["nuggets"]:
                old = ds[0].get("labels", {}).get(n["id"])
                new = ds[1].get("labels", {}).get(n["id"])
                if old != new:
                    flip.append({"nugget_id": n["id"], "legacy": old, "question_first": new})
            result["judges"][judge] = {
                "legacy": scores[0],
                "question_first": scores[1],
                "delta": scores[1] - scores[0],
                "label_flips": flip,
            }
    return result


rows = [diagnostics(i) for i in items if all(i in answers[s] for s in systems)]
robust = sorted(
    [r for r in rows if "robustness" in r["profiles"]], key=lambda r: r["judges"].get("qwen", {}).get("delta", 0)
)
p1 = set(json.loads((B / "rel1119b/p1/ids.json").read_text()))
result = {
    "items": len(rows),
    "all_counts": {
        k: sum(r[k] for r in rows)
        for k in ["same_answer", "same_contexts", "same_context_texts", "same_queries", "same_replayed_history"]
    },
    "robustness": robust,
    "p1_robust_overlap": sorted(p1 & {r["id"] for r in robust}),
    "both_judges_degraded": sorted(
        [r for r in rows if all(r["judges"].get(j, {}).get("delta", 0) < 0 for j in ["qwen", "gptoss"])],
        key=lambda r: sum(x["delta"] for x in r["judges"].values()),
    )[:12],
    "both_judges_improved": sorted(
        [r for r in rows if all(r["judges"].get(j, {}).get("delta", 0) > 0 for j in ["qwen", "gptoss"])],
        key=lambda r: -sum(x["delta"] for x in r["judges"].values()),
    )[:12],
    "public_examples": [],
}
public_ids = ["dialogue-certificate-referent-missing-order", "doc-b3-005", "legacy-003", "prod-0443"]
by_id = {r["id"]: r for r in rows}
public = [by_id[i] for i in public_ids if i in by_id]
for r in public:
    i = r["id"]
    a, b = [answers[s][i] for s in systems]
    t = tasks[pid_of[systems[0], i]]

    def excerpt(s):
        return s if len(s) <= 800 else s[:350] + " […] " + s[-400:]

    def quote_groups(record, item_id=i):
        text = " ".join("\n".join(c.get("text", "") for c in record.get("contexts", [])).split())
        required = {g for f in items[item_id]["key_facts"] if f["required"] for g in f["evidence_groups"]}
        return {
            g["id"]: any(" ".join(s["quote"].split()) in text for s in g["sources"])
            for g in items[item_id]["evidence_groups"]
            if g["id"] in required
        }

    result["public_examples"].append(
        {
            "id": i,
            "question_excerpt": items[i]["question"][:220],
            "legacy_excerpt": excerpt(a["response"]),
            "question_first_excerpt": excerpt(b["response"]),
            "p1_captured": i in p1,
            "diagnostics": r,
            "required_quote_groups": [quote_groups(a), quote_groups(b)],
            "nuggets": [
                {"id": n["id"], "text": n.get("text", n.get("fact", n.get("nugget", "")))[:200]} for n in t["nuggets"]
            ],
            "contexts_lost": [
                {"kind": c.get("kind"), "title": c.get("title"), "cid": c.get("chunk_ids")}
                for c in a.get("contexts", [])
                if not (set(map(str, c.get("chunk_ids", []))) & cids(b))
            ][:6],
            "contexts_gained": [
                {"kind": c.get("kind"), "title": c.get("title"), "cid": c.get("chunk_ids")}
                for c in b.get("contexts", [])
                if not (set(map(str, c.get("chunk_ids", []))) & cids(a))
            ][:6],
        }
    )
print(json.dumps(result, ensure_ascii=False))
