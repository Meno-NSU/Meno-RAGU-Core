"""Gated, fixed-pool prompt order experiment. Raw records remain in prod scratch.

Historical L/Q/QI preserve every request field except prompt section order and
add a matched seed. Modern SL/SQ separately use structured_outputs plus corrected
duplicate-token aggregation; their results must not be pooled with historical.
QA compares frozen observed contexts, not contexts selected by this experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def old_score(choice):
    d = {str(x["token"]): float(x["logprob"]) for x in choice["logprobs"]["content"][0]["top_logprobs"] if "token" in x}
    return score_values({i: d.get(str(i), -100.0) for i in range(3)})


def fixed_score(choice):
    d = {i: [] for i in range(3)}
    for x in choice["logprobs"]["content"][0]["top_logprobs"]:
        if x.get("token") in ["0", "1", "2"]:
            d[int(x["token"])].append(float(x["logprob"]))
    if not any(math.isfinite(v) for vs in d.values() for v in vs):
        raise ValueError("No finite digit labels")
    out = {}
    for i, vs in d.items():
        if not vs:
            out[i] = -100.0
            continue
        peak = max(vs)
        out[i] = peak if peak == -math.inf else peak + math.log(sum(math.exp(v - peak) for v in vs))
    return score_values(out)


def score_values(v):
    m = max(v.values())
    p = {i: math.exp(x - m) for i, x in v.items()}
    z = sum(p.values())
    return sum(i * x / z for i, x in p.items()) / 2


def med(xs):
    return statistics.median(xs) if xs else None


def corr_ranks(a, b):
    # Spearman with deterministic average ties; candidates are identical by index.
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: -v[i])
        res = [0.0] * len(v)
        j = 0
        while j < len(v):
            k = j + 1
            while k < len(v) and v[order[k]] == v[order[j]]:
                k += 1
            for q in order[j:k]:
                res[q] = (j + k - 1) / 2
            j = k
        return res

    x, y = ranks(a), ranks(b)
    mx = statistics.mean(x)
    my = statistics.mean(y)
    denom = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
    return sum((u - mx) * (v - my) for u, v in zip(x, y, strict=False)) / denom if denom else None


def prepare_qa(payload, abbr_path):
    from nltk.stem.snowball import SnowballStemmer

    from meno_rag.stand.qa import prepare_prompt_for_question_answering
    from meno_rag.stand.rewriting import load_abbreviations

    abbr = load_abbreviations(Path(abbr_path))
    stemmer = SnowballStemmer("russian")
    manifest = {}
    for cid, c in payload["cases"].items():
        old = c["legacy_qa_request"]
        user = old["messages"][-1]["content"]
        marker = "==========\nDIALOGUE HISTORY\n==========\n\n"
        tail = user.split(c["contexts"]["L"], 1)[1]
        history = tail.split(marker, 1)[1].split("\n\n==========\n", 1)[0] if marker in tail else ""
        c["qa_requests"] = {}
        for arm in ["L", "Q"]:
            req = copy.deepcopy(old)
            rebuilt = prepare_prompt_for_question_answering(
                c["item"]["question"],
                history,
                c["contexts"][arm],
                abbr,
                stemmer,
                route=c["records"][arm]["v2"]["route"],
            )
            if arm == "L" and rebuilt != user:
                i = next(
                    (i for i, (x, y) in enumerate(zip(rebuilt, user, strict=False)) if x != y),
                    min(len(rebuilt), len(user)),
                )
                raise ValueError(f"{cid}: native legacy QA mismatch at {i}; lengths {len(rebuilt)}/{len(user)}")
            req["messages"][-1]["content"] = rebuilt
            req["stream"] = False
            c["qa_requests"][arm] = req
        manifest[cid] = {
            "pool_size": len(c["pool"]),
            "native_L_QA_exact": True,
            "qa_request_hashes": {a: digest(r) for a, r in c["qa_requests"].items()},
            "qa_context_hashes": {a: digest(s) for a, s in c["contexts"].items()},
            "history_chars": len(history),
        }
    return manifest


def summarize(payload, rows):
    scores = {(r["case"], r["arm"], r["repeat"], r["candidate"]): r for r in rows if r["kind"] == "score" and r["ok"]}
    result = {
        "requests": len(rows),
        "errors": sum(not r["ok"] for r in rows),
        "cases": {},
        "notes": [
            "Diagnostic case selection is purposive, not a population quality estimate.",
            "Top16/rank metrics below use rerank scores alone, not production mixed retrieval/source-prior ranking.",
            "QI is scorer-only; QA always uses frozen observed L/Q contexts.",
            "Historical guided_choice is ignored by vLLM0.32; modern structured arms use corrected duplicate-logprob parser.",
            "Two repeats quantify limited within-run sampling/numerical variability, not independent model retraining.",
        ],
    }
    for cid, c in payload["cases"].items():
        n = len(c["pool"])
        cr = {"pool_size": n, "arms": {}, "comparisons": []}
        arms = sorted({a for case, a, r, i in scores if case == cid})
        for a in arms:
            arr = [r for (case, arm, _, _), r in scores.items() if case == cid and arm == a]
            cr["arms"][a] = {
                "n": len(arr),
                "first_tokens": dict(Counter(r["first_token"] for r in arr)),
                "all_labels_present": sum(r["digit_coverage"] == 3 for r in arr),
                "missing_any_label": sum(r["digit_coverage"] < 3 for r in arr),
                "duplicate_decoded_digits": sum(r["duplicate_digits"] for r in arr),
                "median_s": med([r["secs"] for r in arr]),
            }
        pairs = [
            ("L", 0, "L", 1),
            ("Q", 0, "Q", 1),
            ("QI", 0, "QI", 1),
            ("L", 0, "Q", 0),
            ("L", 1, "Q", 1),
            ("Q", 0, "QI", 0),
            ("Q", 1, "QI", 1),
            ("L", 0, "QI", 0),
            ("L", 1, "QI", 1),
            ("SL", 0, "SQ", 0),
            ("SL", 1, "SQ", 1),
            ("SL", 0, "SL", 1),
            ("SQ", 0, "SQ", 1),
        ]
        for a, ra, b, rb in pairs:
            if not all((cid, arm, rep, i) in scores for arm, rep in [(a, ra), (b, rb)] for i in range(n)):
                continue
            av = [scores[cid, a, ra, i]["score"] for i in range(n)]
            bv = [scores[cid, b, rb, i]["score"] for i in range(n)]
            atop = sorted(range(n), key=lambda i: -av[i])[:16]
            btop = sorted(range(n), key=lambda i: -bv[i])[:16]
            cr["comparisons"].append(
                {
                    "a": [a, ra],
                    "b": [b, rb],
                    "mean_abs_score_delta": statistics.mean(abs(x - y) for x, y in zip(av, bv, strict=False)),
                    "spearman": corr_ranks(av, bv),
                    "top16_overlap": len(set(atop) & set(btop)),
                    "first_token_changes": sum(
                        scores[cid, a, ra, i]["first_token"] != scores[cid, b, rb, i]["first_token"] for i in range(n)
                    ),
                }
            )
        result["cases"][cid] = cr
    result["qa"] = [
        {k: r[k] for k in ["case", "arm", "repeat", "seed", "ok", "secs", "answer_sha256"] if k in r}
        for r in rows
        if r["kind"] == "qa"
    ]
    return result


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--payload", required=True)
    p.add_argument("--abbr", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--structured", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    payload = json.loads(Path(args.payload).read_text())  # noqa: ASYNC240
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    manifest = prepare_qa(payload, args.abbr)
    n = sum(len(c["pool"]) for c in payload["cases"].values())
    plan = {
        "manifest": manifest,
        "score_requests": n * (10 if args.structured else 6),
        "qa_requests": len(payload["cases"]) * 4,
        "arms": ["L", "Q", "QI"] + (["SL", "SQ"] if args.structured else []),
        "seed": {"score": [42, 43], "QA": [42, 43]},
        "structured": args.structured,
    }
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(json.dumps(plan), flush=True)
    if args.prepare_only:
        return
    gate = Path(args.gate)
    while not gate.exists():  # noqa: ASYNC110, ASYNC240
        await asyncio.sleep(2)
    rows = []
    sem = asyncio.Semaphore(args.concurrency)
    raw = (out / "raw_results.jsonl").open("a")
    start = time.monotonic()
    abort = asyncio.Event()
    error_count = 0
    async with (
        asyncio.timeout(900),
        httpx.AsyncClient(
            timeout=180,
            headers={"Authorization": "Bearer EMPTY"},
            limits=httpx.Limits(max_connections=args.concurrency),
        ) as client,
    ):
        registry = await client.get(args.base_url.rstrip("/") + "/models")
        registry.raise_for_status()
        registered = {m["id"] for m in registry.json().get("data", [])}
        models = {c["legacy_qa_request"]["model"] for c in payload["cases"].values()}
        models.update(r["request"]["L"]["model"] for c in payload["cases"].values() for r in c["pool"])
        if not models <= registered or not all("gemma" in m.lower() for m in models):
            raise RuntimeError(
                f"Expected Gemma model registry before inference; expected={models}, actual={registered}"
            )

        async def call(meta, request):
            nonlocal error_count
            async with sem:
                if abort.is_set():
                    return
                t = time.monotonic()
                row = {**meta, "request_sha256": digest(request)}
                try:
                    r = await client.post(args.base_url.rstrip("/") + "/chat/completions", json=request)
                    r.raise_for_status()
                    resp = r.json()
                    choice = resp["choices"][0]
                    row.update(ok=True, secs=time.monotonic() - t, response=resp)
                    if meta["kind"] == "score":
                        tops = choice["logprobs"]["content"][0]["top_logprobs"]
                        tokens = [x.get("token") for x in tops if x.get("token") in ["0", "1", "2"]]
                        row.update(
                            score=(fixed_score if meta["arm"].startswith("S") else old_score)(choice),
                            first_token=choice["logprobs"]["content"][0]["token"],
                            digit_coverage=len(set(tokens)),
                            duplicate_digits=len(tokens) > len(set(tokens)),
                        )
                    else:
                        row.update(
                            answer=choice["message"]["content"], answer_sha256=digest(choice["message"]["content"])
                        )
                except Exception as e:
                    row.update(ok=False, secs=time.monotonic() - t, error=f"{type(e).__name__}: {e}")
                    error_count += 1
                    if error_count >= 3:
                        abort.set()
                rows.append(row)
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()

        for rep, seed in enumerate([42, 43]):
            for ci, (cid, c) in enumerate(payload["cases"].items()):
                tasks = []
                arms = ["L", "Q", "QI"] + (["SL", "SQ"] if args.structured else [])
                indices = list(range(len(c["pool"])))
                random.Random(103 + rep).shuffle(indices)
                for k, i in enumerate(indices):
                    order = arms[(k + rep + ci) % len(arms) :] + arms[: (k + rep + ci) % len(arms)]
                    if rep:
                        order = list(reversed(order))
                    for arm in order:
                        req = copy.deepcopy(c["pool"][i]["request"][arm[1:] if arm.startswith("S") else arm])
                        req["seed"] = seed
                        req["stream"] = False
                        if arm.startswith("S"):
                            for field in [
                                "guided_choice",
                                "guided_json",
                                "guided_regex",
                                "guided_grammar",
                                "guided_decoding_backend",
                            ]:
                                req.pop(field, None)
                            req["structured_outputs"] = {"choice": ["0", "1", "2"]}
                        tasks.append(
                            call(
                                {"kind": "score", "case": cid, "arm": arm, "repeat": rep, "seed": seed, "candidate": i},
                                req,
                            )
                        )
                await asyncio.gather(*tasks)
                (out / "aggregate.json").write_text(json.dumps(summarize(payload, rows), ensure_ascii=False, indent=2))
                print(
                    json.dumps(
                        {
                            "progress": "scores",
                            "case": cid,
                            "repeat": rep,
                            "completed": len(rows),
                            "elapsed_s": round(time.monotonic() - start, 2),
                        }
                    ),
                    flush=True,
                )
                if abort.is_set():
                    raise RuntimeError(
                        "Stopped after three request/model/parser errors; no pending requests are forwarded"
                    )
        # QA contexts are observationally fixed; score-arm results cannot change these requests.
        for rep, seed in enumerate([42, 43]):
            tasks = []
            for ci, (cid, c) in enumerate(payload["cases"].items()):
                for arm in ["L", "Q"] if (rep + ci) % 2 == 0 else ["Q", "L"]:
                    req = copy.deepcopy(c["qa_requests"][arm])
                    req["seed"] = seed
                    req["stream"] = False
                    tasks.append(call({"kind": "qa", "case": cid, "arm": arm, "repeat": rep, "seed": seed}, req))
            await asyncio.gather(*tasks)
            if abort.is_set():
                (out / "aggregate.json").write_text(json.dumps(summarize(payload, rows), ensure_ascii=False, indent=2))
                raise RuntimeError("Stopped after three request/model/parser errors")
    raw.close()
    agg = summarize(payload, rows)
    agg["elapsed_s"] = time.monotonic() - start
    (out / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {"complete": True, "requests": len(rows), "errors": agg["errors"], "elapsed_s": round(agg["elapsed_s"], 2)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
