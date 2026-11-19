"""CPU-only extraction of four public diagnostics; raw payloads stay on prod."""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
args = parser.parse_args()
B = args.base_dir
OUT = args.out_dir
OUT.mkdir(parents=True, exist_ok=True)
IDS = ["dialogue-certificate-referent-missing-order", "doc-b3-005", "legacy-003", "prod-0443"]


def h(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


items = {r["id"]: r for r in map(json.loads, (B / "judge1119/judge/data/items_d54d794.jsonl").open()) if r["id"] in IDS}
records = {}
for arm in ["v6k3b16", "v6k3bq"]:
    records[arm] = {
        r["item_id"]: r
        for r in map(json.loads, (B / f"rel1119b/answers/{arm}.records.jsonl").open())
        if r["item_id"] in IDS
    }
HEADER = re.compile(
    r"==========\n(ABBREVIATION DICTIONARY|DOCUMENT 1|DIALOGUE HISTORY|CURRENT QUESTION|INSTRUCTION)\n==========\n\n"
)


def sections(s):
    hits = list(HEADER.finditer(s))
    assert hits and hits[0].start() == 0
    d = {}
    for j, m in enumerate(hits):
        end = hits[j + 1].start() - 2 if j + 1 < len(hits) else len(s)
        d[m[1]] = s[m.start() : end]
    assert "\n\n".join(d.values()) == s
    return d


def value(d, name):
    return d.get(name, "").split("\n==========\n\n", 1)[-1]


targets = {r["v2"]["standalone_question"]: i for i, r in records["v6k3b16"].items()}
needles = [q.encode() for q in targets]
pools = {i: {} for i in IDS}
stats = Counter()
exact_qa = {i: [] for i in IDS}
for line in (B / "rel1119b/capture_v6k3b.jsonl").open("rb"):
    stats["capture_rows"] += 1
    if not any(n in line for n in needles):
        continue
    r = json.loads(line)
    req = r.get("request", {})
    if req.get("max_tokens") != 1 or not r.get("response", {}).get("choices"):
        continue
    try:
        d = sections(req["messages"][-1]["content"])
    except (AssertionError, KeyError):
        continue
    q = value(d, "CURRENT QUESTION")
    if q not in targets:
        continue
    i = targets[q]
    sig = h(req)
    if sig in pools[i]:
        pools[i][sig]["multiplicity"] += 1
        continue
    assert list(d) == [
        k
        for k in ["ABBREVIATION DICTIONARY", "DOCUMENT 1", "DIALOGUE HISTORY", "CURRENT QUESTION", "INSTRUCTION"]
        if k in d
    ]
    orders = {
        "L": ["ABBREVIATION DICTIONARY", "DOCUMENT 1", "DIALOGUE HISTORY", "CURRENT QUESTION", "INSTRUCTION"],
        "Q": ["DIALOGUE HISTORY", "CURRENT QUESTION", "INSTRUCTION", "ABBREVIATION DICTIONARY", "DOCUMENT 1"],
        "QI": ["DIALOGUE HISTORY", "CURRENT QUESTION", "ABBREVIATION DICTIONARY", "DOCUMENT 1", "INSTRUCTION"],
    }
    requests = {}
    for order, names in orders.items():
        requests[order] = copy.deepcopy(req)
        requests[order]["messages"][-1]["content"] = "\n\n".join(d[k] for k in names if k in d)
        assert {k: h(v) for k, v in sections(requests[order]["messages"][-1]["content"]).items()} == {
            k: h(v) for k, v in d.items()
        }
    assert requests["L"] == req
    pools[i][sig] = {
        "request": requests,
        "original_response": r["response"],
        "multiplicity": 1,
        "document_sha256": h(value(d, "DOCUMENT 1")),
        "document": value(d, "DOCUMENT 1"),
        "section_hashes": {k: h(v) for k, v in d.items()},
    }
original_to_id = {items[i]["question"]: i for i in IDS}
for line in (B / "rel1119b/misses_v6k3b16.jsonl").open():
    r = json.loads(line)
    req = r.get("request", {})
    if req.get("max_tokens") == 1:
        continue
    text = req.get("messages", [{}])[-1].get("content", "")
    marker = "==========\nCURRENT QUESTION\n==========\n\n"
    if marker not in text:
        continue
    q = text.rsplit(marker, 1)[1].split("\n\n==========\nINSTRUCTION\n==========", 1)[0]
    if q in original_to_id:
        exact_qa[original_to_id[q]].append(req)


def context(record):
    docs = [c["text"] for c in record["contexts"] if c["kind"] == "document"]
    graph = [c["text"] for c in record["contexts"] if c["kind"] != "document"]
    result = "\n\n".join(docs + (["**СПРАВКИ ПО МАТЕРИАЛАМ**", *graph] if graph else []))
    assert len(result) == record["v2"]["final_context_chars"], (len(result), record["v2"]["final_context_chars"])
    for c in record["contexts"]:
        assert result[c["context_start"] : c["context_end"]] == c["text"]
    return result


manifest = {}
payload = {"cases": {}}
for i in IDS:
    a, b = [records[arm][i] for arm in ["v6k3b16", "v6k3bq"]]
    ca, cb = context(a), context(b)
    qa = list({h(r): r for r in exact_qa[i]}.values())
    matched = [r for r in qa if ca in r["messages"][-1]["content"]]
    assert len(matched) == 1, (i, len(qa), len(matched))
    req = matched[0]
    before, after = req["messages"][-1]["content"].split(ca, 1)
    # Saved full native QA messages include final rendered sources and prepared history.
    payload["cases"][i] = {
        "item": items[i],
        "records": {"L": a, "Q": b},
        "pool": list(pools[i].values()),
        "legacy_qa_request": req,
        "contexts": {"L": ca, "Q": cb},
    }
    manifest[i] = {
        "unique_scoring_requests": len(pools[i]),
        "captured_multiplicity": sum(x["multiplicity"] for x in pools[i].values()),
        "legacy_qa_candidates": len(qa),
        "legacy_qa_exact_match": True,
        "history_turns": len(items[i].get("history") or []),
        "same_replayed_history": a["inproc"]["replay_answers"] == b["inproc"]["replay_answers"],
        "context_chars": {"L": len(ca), "Q": len(cb)},
        "qa_generation": {k: req.get(k) for k in ["model", "temperature", "max_tokens", "seed", "stream"]},
        "pool_sha256": h([x["section_hashes"] for x in pools[i].values()]),
        "item_facts": items[i]["key_facts"],
        "item_evidence_groups": items[i]["evidence_groups"],
    }
(OUT / "focus_payloads.json").write_text(json.dumps(payload, ensure_ascii=False))
(OUT / "preparation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(
    json.dumps(
        {
            "cases": {
                i: {k: v for k, v in r.items() if k not in ["item_facts", "item_evidence_groups"]}
                for i, r in manifest.items()
            },
            "scanned": dict(stats),
            "gold_doc_b3_005": {
                "facts": manifest["doc-b3-005"]["item_facts"],
                "groups": manifest["doc-b3-005"]["item_evidence_groups"],
            },
        },
        ensure_ascii=False,
    )
)
