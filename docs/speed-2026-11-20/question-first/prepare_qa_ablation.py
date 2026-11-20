"""CPU-only, public Karpov example. Creates explicit optional eight-QA plan."""

import argparse
import copy
import json
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer
from run_focus import digest, prepare_qa

from meno_rag.stand.qa import prepare_prompt_for_question_answering
from meno_rag.stand.rewriting import load_abbreviations

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=Path, required=True)
parser.add_argument("--abbr", type=Path, required=True)
args = parser.parse_args()
D = args.dir
abbr_path = args.abbr
p = json.loads((D / "focus_payloads.json").read_text())
prepare_qa(p, abbr_path)
c = p["cases"]["doc-b3-005"]
abbr = load_abbreviations(abbr_path)
stemmer = SnowballStemmer("russian")
docs = [x["text"] for x in c["records"]["Q"]["contexts"] if x["kind"] == "document"]
graphs = {a: [x for x in c["records"][a]["contexts"] if x["kind"] != "document"] for a in ["L", "Q"]}


def key(x):
    return x.get("element_key") or x["text"].split("\n", 1)[0]


bykey = {key(x): x for x in graphs["Q"]}
lkeys = [key(x) for x in graphs["L"]]
assert len(bykey) == len(graphs["Q"])
common = [k for k in lkeys if k in bykey]
ordered = [bykey[k] for k in common] + [x for x in graphs["Q"] if key(x) not in common]


def context(notes):
    return "\n\n".join(docs + (["**СПРАВКИ ПО МАТЕРИАЛАМ**"] + [x["text"] for x in notes] if notes else []))


assert context(graphs["Q"]) == c["contexts"]["Q"]
contexts = {
    "Q_control": c["contexts"]["Q"],
    "Q_without_graph": context([]),
    "Q_graph_L_order": context(ordered),
    "Q_graph_L_set": context(graphs["L"]),
}
reqs = {}
for arm, ctx in contexts.items():
    req = copy.deepcopy(c["qa_requests"]["Q"])
    req["messages"][-1]["content"] = prepare_prompt_for_question_answering(
        c["item"]["question"], "", ctx, abbr, stemmer, route=c["records"]["Q"]["v2"]["route"]
    )
    reqs[arm] = req
assert reqs["Q_control"] == c["qa_requests"]["Q"]
# Only note order changes in this arm. Same note-text multiset and exact docs.
assert sorted(digest(x["text"]) for x in ordered) == sorted(digest(x["text"]) for x in graphs["Q"])
out = {
    "case": "doc-b3-005",
    "requests": reqs,
    "seeds": [42, 43],
    "manifest": {
        "qa_count": 8,
        "document_hashes": list(map(digest, docs)),
        "common_note_count": len(common),
        "L_note_keys": lkeys,
        "Q_note_keys": [key(x) for x in graphs["Q"]],
        "order_only_changes_prompt": reqs["Q_graph_L_order"] != reqs["Q_control"],
        "context_hashes": {a: digest(ctx) for a, ctx in contexts.items()},
        "caveats": [
            "All-graph removal also changes native note instructions and may change abbreviation dictionary.",
            "L-set arm changes graph-note selection/text, preserving Q documents byte-for-byte.",
            "Order arm preserves graph-note text multiset, source docs, model generation fields.",
        ],
    },
}
(D / "qa_ablation_payloads.json").write_text(json.dumps(out, ensure_ascii=False))
print(json.dumps(out["manifest"], ensure_ascii=False))
