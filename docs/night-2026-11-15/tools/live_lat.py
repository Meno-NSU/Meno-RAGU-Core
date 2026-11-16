"""Latency of the live RAGU stand (:9007, deployed build) on the load test's questions: the same pool and shuffle as
loadtest.py (single-turn factual/complex/general, Random(7)), 3 warm-up questions, then 12 at each level.
Requests carry X-Meno-Request-Id, so the stand does not store them.

usage: python3 live_lat.py ITEMS.jsonl OUT.jsonl LEVELS
"""
import asyncio, json, random, sys, time, uuid
import httpx

items_path, out_path, levels = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
items = [json.loads(line) for line in open(items_path, encoding="utf-8")]
pool = [i["question"] for i in items if not i.get("history") and i["category"] in ("factual", "complex", "general")]
random.Random(7).shuffle(pool)
URL = "http://127.0.0.1:9007/v1/chat/completions"


async def one(client, q, level, out):
    row = {"variant": "live-4bc8a1e", "level": level, "question": q}
    started = time.perf_counter()
    body = {"model": "gemma-4-12B-it", "knowledge_base_id": "nsu-ragu", "stream": True,
            "messages": [{"role": "user", "content": q}]}
    try:
        async with client.stream("POST", URL, json=body, headers={"X-Meno-Request-Id": f"night1115-lat-{uuid.uuid4().hex[:12]}"}) as r:
            r.raise_for_status()
            chars = 0
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                delta = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                if delta and "ttft_s" not in row:
                    row["ttft_s"] = time.perf_counter() - started
                chars += len(delta)
        row.update(ok=True, total_s=time.perf_counter() - started, answer_chars=chars)
    except Exception as exc:  # noqa: BLE001
        row.update(ok=False, total_s=time.perf_counter() - started, error=repr(exc)[:300])
    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
    return row


async def main():
    out = open(out_path, "a", encoding="utf-8")
    async with httpx.AsyncClient(timeout=600) as client:
        await asyncio.gather(*(one(client, q, 0, out) for q in pool[:3]))
        for level in levels:
            gate = asyncio.Semaphore(level)

            async def gated(q):
                async with gate:
                    return await one(client, q, level, out)

            t0 = time.perf_counter()
            rows = await asyncio.gather(*(gated(q) for q in pool[3:3 + max(3 * level, 12)]))
            wall = time.perf_counter() - t0
            ok = sorted(r["total_s"] for r in rows if r.get("ok"))
            tt = sorted(r["ttft_s"] for r in rows if r.get("ok") and "ttft_s" in r)
            print(json.dumps({"variant": "live-4bc8a1e", "level": level, "n": len(rows), "ok": len(ok),
                              "total_p50": round(ok[len(ok) // 2], 2) if ok else None, "total_p95": round(ok[min(len(ok) - 1, int(.95 * len(ok)))], 2) if ok else None,
                              "ttft_p50": round(tt[len(tt) // 2], 2) if tt else None, "answers_per_min": round(60 * len(ok) / wall, 1)}), flush=True)


asyncio.run(main())
