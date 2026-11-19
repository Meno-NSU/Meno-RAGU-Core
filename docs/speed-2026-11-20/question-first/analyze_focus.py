"""Post-hoc numeric/limited-public-evidence analysis; no new inference."""

import argparse
import json
import statistics
from pathlib import Path

from run_focus import corr_ranks, old_score

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=Path, required=True)
D = parser.parse_args().dir
p = json.loads((D / "focus_payloads.json").read_text())
rows = [json.loads(s) for s in (D / "run/raw_results.jsonl").open()]
scores = {(r["case"], r["arm"], r["repeat"], r["candidate"]): r for r in rows if r["kind"] == "score" and r["ok"]}
result = {"cases": {}}
for cid, c in p["cases"].items():
    n = len(c["pool"])
    out = {
        "pool_size": n,
        "protocol_comparisons": [],
        "gold_candidate_diagnostics": [],
        "original_question": c["item"]["question"],
        "rerank_question": c["records"]["L"]["v2"]["standalone_question"],
    }
    for a, b in [("L", "SL"), ("Q", "SQ")]:
        for rep in [0, 1]:
            av = [scores[cid, a, rep, i]["score"] for i in range(n)]
            bv = [scores[cid, b, rep, i]["score"] for i in range(n)]
            out["protocol_comparisons"].append(
                {
                    "a": a,
                    "b": b,
                    "repeat": rep,
                    "mean_abs_score_delta": statistics.mean(abs(x - y) for x, y in zip(av, bv, strict=False)),
                    "spearman": corr_ranks(av, bv),
                    "first_token_changes": sum(
                        scores[cid, a, rep, i]["first_token"] != scores[cid, b, rep, i]["first_token"] for i in range(n)
                    ),
                    "top16_overlap": len(
                        set(sorted(range(n), key=lambda i: -av[i])[:16])
                        & set(sorted(range(n), key=lambda i: -bv[i])[:16])
                    ),
                }
            )
    ranks = {
        a: {i: j + 1 for j, i in enumerate(sorted(range(n), key=lambda i: -scores[cid, a, 0, i]["score"]))}
        for a in ["L", "Q", "QI", "SL", "SQ"]
    }
    for i, cand in enumerate(c["pool"]):
        text = cand["document"]
        public_cue = None
        if cid == "legacy-003" and "Вектор" in text and "базовый институт" in text:
            at = text.index("Кафедра молекулярной биологии")
            public_cue = text[max(0, at - 2) : at + 250]
            assert "Вектор" in public_cue
        if cid == "doc-b3-005" and "| Физика | 31.05.01" in text:
            at = text.index("| Физика | 31.05.01")
            public_cue = text[at : at + 200]
        if public_cue:
            out["gold_candidate_diagnostics"].append(
                {
                    "candidate": i,
                    "document_sha256": cand["document_sha256"],
                    "public_cue": public_cue,
                    "score_rank": {
                        a: {
                            "score": scores[cid, a, 0, i]["score"],
                            "rank": ranks[a][i],
                            "label": scores[cid, a, 0, i]["first_token"],
                        }
                        for a in ranks
                    },
                }
            )
    # Recorded reference vs newL: same historical prompt bytes, runtime numerical variance.
    reference = [old_score(x["original_response"]["choices"][0]) for x in c["pool"]]
    new = [scores[cid, "L", 0, i]["score"] for i in range(n)]
    out["saved_vs_new_L"] = {
        "mean_abs_score_delta": statistics.mean(abs(x - y) for x, y in zip(reference, new, strict=False)),
        "spearman": corr_ranks(reference, new),
        "first_token_changes": sum(
            c["pool"][i]["original_response"]["choices"][0]["logprobs"]["content"][0]["token"]
            != scores[cid, "L", 0, i]["first_token"]
            for i in range(n)
        ),
    }
    result["cases"][cid] = out
(D / "focus_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
