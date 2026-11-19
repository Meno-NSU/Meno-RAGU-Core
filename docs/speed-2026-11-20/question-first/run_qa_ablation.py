"""Root-gated8QA public diagnostic; no production mutations or scorer requests."""

import argparse
import asyncio
import copy
import json
import time
from pathlib import Path

import httpx
from run_focus import digest


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--base-url", required=True)
    args = p.parse_args()
    D = Path(args.dir)
    payload = json.loads((D / "qa_ablation_payloads.json").read_text())
    print(json.dumps({"prepared": True, "requests": 8, "arms": list(payload["requests"])}), flush=True)
    while not Path(args.gate).exists():  # noqa: ASYNC110, ASYNC240
        await asyncio.sleep(2)
    rows = []
    start = time.monotonic()
    sem = asyncio.Semaphore(4)
    errors = 0
    abort = asyncio.Event()
    async with (
        asyncio.timeout(600),
        httpx.AsyncClient(timeout=180, headers={"Authorization": "Bearer EMPTY"}) as client,
    ):
        registry = await client.get(args.base_url.rstrip("/") + "/models")
        registry.raise_for_status()
        models = {x["id"] for x in registry.json()["data"]}
        assert all("gemma" in req["model"].lower() and req["model"] in models for req in payload["requests"].values())

        async def one(arm, seed):
            nonlocal errors
            async with sem:
                if abort.is_set():
                    return
                req = copy.deepcopy(payload["requests"][arm])
                req["seed"] = seed
                req["stream"] = False
                t = time.monotonic()
                r = {"arm": arm, "seed": seed, "request_sha256": digest(req)}
                try:
                    response = await client.post(args.base_url.rstrip("/") + "/chat/completions", json=req)
                    response.raise_for_status()
                    data = response.json()
                    answer = data["choices"][0]["message"]["content"]
                    r.update(
                        ok=True, secs=time.monotonic() - t, answer=answer, answer_sha256=digest(answer), response=data
                    )
                except Exception as e:
                    errors += 1
                    r.update(ok=False, secs=time.monotonic() - t, error=f"{type(e).__name__}: {e}")
                    if errors >= 3:
                        abort.set()
                rows.append(r)
                (D / "qa_ablation_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

        for j, seed in enumerate([42, 43]):
            arms = list(payload["requests"])
            arms = arms[j:] + arms[:j]
            await asyncio.gather(*(one(arm, seed) for arm in arms))
            if abort.is_set():
                raise RuntimeError("Stopped after3request/model errors")
    print(
        json.dumps({"complete": True, "n": len(rows), "errors": errors, "elapsed_s": time.monotonic() - start}),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
