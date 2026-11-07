from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger("benchmarking")


class FatalRunError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_dataset(path: Path) -> tuple[str, list[dict]]:
    raw = path.read_bytes()
    dataset_hash = hashlib.sha256(raw).hexdigest()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Бенчмарк должен быть непустым JSON-массивом.")
    samples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) or not row[k].strip() for k in ("question", "ground_truth")
        ):
            raise ValueError(f"Некорректная запись бенчмарка {index}.")
        samples.append({
            "sample_id": digest([dataset_hash, index, row]),
            "index": index, "question": row["question"], "ground_truth": row["ground_truth"],
        })
    return dataset_hash, samples


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def export_jsonl(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((canonical(row) + "\n").encode())
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)


def safe_name(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in model)[:80] + "-" + digest(model)[:8]


async def run_jobs(
    store: Any,
    jobs: list[tuple[str, dict]],
    operation: Callable[[dict], Awaitable[dict]],
    *,
    concurrency: int,
    attempts: int,
    label: str,
) -> None:
    queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
    for key, value in jobs:
        if not store.is_done(key):
            queue.put_nowait((key, value))
    pending = queue.qsize()
    counters = {"ok": 0, "failed": 0}
    started = time.monotonic()

    def progress() -> None:
        finished = counters["ok"] + counters["failed"]
        elapsed = time.monotonic() - started
        rate = finished / elapsed if elapsed else 0
        eta = (pending - finished) / rate if rate else None
        log.info(
            "%s: готово %d/%d; из чекпоинта %d; ошибок %d; %.3f записей/с; прошло %.0f с; осталось %s",
            label, len(jobs) - pending + finished, len(jobs), len(jobs) - pending,
            counters["failed"], rate, elapsed, f"~{eta:.0f} с" if eta is not None else "неизвестно",
        )

    async def worker() -> None:
        while not queue.empty():
            key, value = queue.get_nowait()
            for attempt in range(attempts):
                store.start(key, value)
                try:
                    result = await operation(value)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    store.fail(key, f"{type(exc).__name__}: {exc}", details=getattr(exc, "details", None))
                    if isinstance(exc, FatalRunError):
                        raise
                    log.warning("%s: %s, попытка %d/%d: %s", label, key, attempt + 1, attempts, exc)
                    if attempt + 1 < attempts:
                        await asyncio.sleep(min(2 ** attempt, 8))
                    else:
                        counters["failed"] += 1
                else:
                    store.complete(key, result)
                    counters["ok"] += 1
                    break
            queue.task_done()
            finished = counters["ok"] + counters["failed"]
            if finished == pending or finished % 10 == 0:
                progress()

    async def monitor() -> None:
        while True:
            await asyncio.sleep(30)
            progress()

    progress()
    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, pending))]
    heartbeat = asyncio.create_task(monitor())
    try:
        await asyncio.gather(*workers)
    finally:
        for task in [*workers, heartbeat]:
            task.cancel()
        await asyncio.gather(*workers, heartbeat, return_exceptions=True)
