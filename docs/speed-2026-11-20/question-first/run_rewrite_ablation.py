"""CPU preparation + gated32 one-token Vector intent-preservation tests."""

import argparse
import asyncio
import copy
import json
import re
import time
from pathlib import Path

import httpx
from run_focus import digest, old_score

HDR = re.compile(
    r"==========\n(ABBREVIATION DICTIONARY|DOCUMENT 1|DIALOGUE HISTORY|CURRENT QUESTION|INSTRUCTION)\n==========\n\n"
)


def sections(text):
    ms = list(HDR.finditer(text))
    assert ms and ms[0].start() == 0
    d = {m[1]: text[m.start() : ms[j + 1].start() - 2 if j + 1 < len(ms) else len(text)] for j, m in enumerate(ms)}
    assert "\n\n".join(d.values()) == text
    return d


def value(d, k):
    return d.get(k, "").split("\n==========\n\n", 1)[-1]


def prepare(D, indices, abbr_path):
    from nltk.stem.snowball import SnowballStemmer

    from meno_rag.stand.rerank import build_prompt
    from meno_rag.stand.rewriting import load_abbreviations

    p = json.loads((D / "focus_payloads.json").read_text())
    c = p["cases"]["legacy-003"]
    abbr = load_abbreviations(Path(abbr_path))
    stemmer = SnowballStemmer("russian")
    original = c["item"]["question"]
    jobs = []
    manifest = []
    for i in indices:
        cand = c["pool"][i]
        d = sections(cand["request"]["L"]["messages"][-1]["content"])
        oldq = value(d, "CURRENT QUESTION")
        hist = value(d, "DIALOGUE HISTORY")
        doc = value(d, "DOCUMENT 1")
        nativeL = build_prompt(oldq, hist, doc, abbr, stemmer, order="legacy")
        assert nativeL == cand["request"]["L"]["messages"]
        native_new = build_prompt(original, hist, doc, abbr, stemmer, order="legacy")
        nd = sections(native_new[-1]["content"])
        other_changes = [k for k in set(d) | set(nd) if k != "CURRENT QUESTION" and d.get(k) != nd.get(k)]
        assert not other_changes, (i, other_changes)
        manifest.append(
            {
                "candidate": i,
                "document_sha256": cand["document_sha256"],
                "native_changed_sections": ["CURRENT QUESTION"],
                "role": "required-base-institute-evidence" if i in [12, 18] else "negative-control",
                "pre_inference_reason": (
                    "Contains explicit molecular-biology department base institutes, including Vector."
                    if i in [12, 18]
                    else "Public unit/department profile from the same pool; contains neither Vector nor practice/placement claims, so cannot establish either queried association."
                ),
            }
        )
        if i not in [12, 18]:
            assert "Вектор" not in doc and "практик" not in doc.lower()
        for order in ["L", "Q"]:
            for qarm, q in [("lossy", oldq), ("original", original)]:
                req = copy.deepcopy(cand["request"][order])
                messages = build_prompt(
                    q, hist, doc, abbr, stemmer, order="legacy" if order == "L" else "question_first"
                )
                req["messages"] = messages
                req["stream"] = False
                if qarm == "lossy":
                    assert req == cand["request"][order]
                for seed in [42, 43]:
                    jobs.append(
                        {
                            "candidate": i,
                            "order": order,
                            "question_arm": qarm,
                            "seed": seed,
                            "request": dict(req, seed=seed),
                        }
                    )
    out = {"jobs": jobs, "manifest": manifest, "original_question": original, "lossy_question": oldq}
    (D / "rewrite_ablation_payloads.json").write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "jobs"} | {"n": len(jobs)}, ensure_ascii=False), flush=True)
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--abbr", required=True)
    p.add_argument("--base-url", required=True)
    a = p.parse_args()
    D = Path(a.dir)
    payload = prepare(D, [int(i) for i in a.indices.split(",")], a.abbr)
    if a.prepare_only:
        return
    while not Path(a.gate).exists():  # noqa: ASYNC110, ASYNC240
        await asyncio.sleep(2)
    results = []
    sem = asyncio.Semaphore(8)
    errors = 0
    abort = asyncio.Event()
    start = time.monotonic()
    async with asyncio.timeout(180), httpx.AsyncClient(timeout=60, headers={"Authorization": "Bearer EMPTY"}) as client:
        registry = await client.get(a.base_url.rstrip("/") + "/models")
        registry.raise_for_status()
        models = {m["id"] for m in registry.json()["data"]}
        assert all(
            "gemma" in j["request"]["model"].lower() and j["request"]["model"] in models for j in payload["jobs"]
        )

        async def one(job):
            nonlocal errors
            async with sem:
                if abort.is_set():
                    return
                r = {k: v for k, v in job.items() if k != "request"}
                r["request_sha256"] = digest(job["request"])
                t = time.monotonic()
                try:
                    x = await client.post(a.base_url.rstrip("/") + "/chat/completions", json=job["request"])
                    x.raise_for_status()
                    response = x.json()
                    ch = response["choices"][0]
                    r.update(
                        ok=True,
                        secs=time.monotonic() - t,
                        score=old_score(ch),
                        label=ch["logprobs"]["content"][0]["token"],
                        response=response,
                    )
                except Exception as e:
                    errors += 1
                    r.update(ok=False, secs=time.monotonic() - t, error=str(e))
                    if errors >= 3:
                        abort.set()
                results.append(r)
                (D / "rewrite_ablation_raw.json").write_text(json.dumps(results, ensure_ascii=False))

        # Balance label/order/seed groups using alternating candidate order and arms.
        jobs = payload["jobs"]
        jobs = sorted(
            jobs,
            key=lambda j: ((j["candidate"] + j["seed"]) % 2, j["seed"], j["question_arm"], j["order"], j["candidate"]),
        )
        await asyncio.gather(*(one(j) for j in jobs))
    aggregate = {
        "manifest": payload["manifest"],
        "original_question": payload["original_question"],
        "lossy_question": payload["lossy_question"],
        "n": len(results),
        "errors": errors,
        "elapsed_s": time.monotonic() - start,
        "rows": [{k: v for k, v in r.items() if k != "response"} for r in results],
    }
    (D / "rewrite_ablation_aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
