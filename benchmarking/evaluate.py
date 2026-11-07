from __future__ import annotations

import logging
import math
import statistics
from collections import Counter, defaultdict

from benchmarking.common import atomic_json, digest, export_jsonl, run_jobs, safe_name
from benchmarking.storage import Store

log = logging.getLogger("benchmarking")


def export_evaluation(store: Store, models: list[str], total: int) -> dict:
    summary = {"schema_version": 1, "config": store.metadata("config"), "models": {}}
    config = store.metadata("config")
    for model in models:
        directory = store.directory / safe_name(model)
        directory.mkdir(exist_ok=True)
        export_jsonl(directory / "scores.jsonl.gz", (r for r in store.records(failures=True) if r["model"] == model))
        scores = defaultdict(list)
        failed = defaultdict(int)
        for row in store.records(failures=True):
            if row["model"] != model:
                continue
            if row.get("status") == "complete":
                scores[row["metric"]].append(row["score"])
            else:
                failed[row["metric"]] += 1
        metrics = {
            metric: {
                "mean": statistics.mean(scores[metric]) if scores[metric] else None,
                "scored": len(scores[metric]), "failed_or_interrupted": failed[metric],
                "dataset_size": total, "coverage": len(scores[metric]) / total,
                "available_answers": config["available_samples"][model],
                "missing_answers": total - config["available_samples"][model],
            }
            for metric in config["metrics"]
        }
        atomic_json(directory / "summary.json", {"model": model, "metrics": metrics})
        summary["models"][model] = metrics
    atomic_json(store.directory / "summary.json", summary)
    return summary


async def evaluate(args) -> int:
    from benchmarking.ragas_adapter import RagasAdapter, RagasConfig

    with Store(args.collection_dir, readonly=True) as collection:
        source = collection.metadata("config")
        if not source or source.get("kind") != "collection":
            raise ValueError("Каталог не содержит чекпоинты первого прохода.")
        total = source["dataset_size"]
        models = source["models"]
        saved_count = collection.counts().get("complete", 0)
        log.info("Читаю сохранённые ответы и контексты: %d записей.", saved_count)
        captures = []
        for count, row in enumerate(collection.records(), start=1):
            captures.append({
                "model": row["model"], "sample_id": row["sample_id"], "index": row["index"],
                "capture_sha256": row["capture_sha256"],
                "user_input": row["question"], "response": row["response"], "reference": row["ground_truth"],
                "retrieved_contexts": row["retrieved_contexts"],
            })
            if count % 100 == 0:
                log.info("Чтение ответов: %d/%d.", count, saved_count)
        if not args.allow_partial and len(captures) != total * len(models):
            raise ValueError(
                f"Первый проход не завершён: {len(captures)}/{total * len(models)} ответов. "
                "Продолжите collect или явно задайте --allow-partial для предварительной оценки."
            )
        log.info("Проверяю контрольную сумму сохранённого сбора.")
        collection_hash = collection.content_digest(
            progress=lambda count: log.info("Проверка сбора: %d/%d.", count, saved_count),
        )
    available_samples = dict(Counter(sample["model"] for sample in captures))
    available_samples = {model: available_samples.get(model, 0) for model in models}
    log.info("Сохранённые ответы для оценки: %s; полный набор — %d на модель.", available_samples, total)
    if len(captures) != total * len(models):
        log.warning("Неполный сбор: отсутствующие ответы будут отражены в покрытии, оценки для них не создаются.")
    judge = RagasConfig(
        judge_url=args.judge_url, judge_model=args.judge_model, judge_revision=args.judge_revision,
        embedder_url=args.embedder_url, embedder_model=args.embedder_model,
        cache_dir=args.output_dir / "request-cache", judge_api_key=args.judge_api_key,
        embedder_api_key=args.embedder_api_key, max_tokens=args.max_tokens,
        timeout=args.timeout, disable_thinking=not args.enable_thinking,
        judge_context_tokens=getattr(args, "judge_context_tokens", None),
    )
    metrics = args.metrics or list(RagasAdapter.metric_names)
    if len(set(metrics)) != len(metrics):
        raise ValueError("Список метрик содержит повторы.")
    unknown = set(metrics) - set(RagasAdapter.metric_names)
    if unknown:
        raise ValueError(f"Неизвестные метрики: {sorted(unknown)}")
    config = {
        "schema_version": 1, "kind": "evaluation", "collection_sha256": collection_hash,
        "dataset_sha256": source["dataset_sha256"], "dataset_size": total,
        "models": models, "metrics": metrics, "judge": {
            "url": args.judge_url.rstrip("/"), "model": args.judge_model, "revision": args.judge_revision,
            "max_tokens": args.max_tokens, "temperature": 0.0, "disable_thinking": not args.enable_thinking,
        }, "embeddings": {"url": args.embedder_url.rstrip("/"), "model": args.embedder_model},
        "adapter": judge.fingerprint_payload(),
        "available_samples": available_samples,
    }
    ordered = []
    sample_order = getattr(args, "sample_order", "dataset")
    for model in models:
        model_samples = [sample for sample in captures if sample["model"] == model]
        if sample_order == "longest-context-first":
            model_samples.sort(key=lambda sample: (-sum(len(text) for text in sample["retrieved_contexts"]), sample["index"]))
        else:
            model_samples.sort(key=lambda sample: sample["index"])
        ordered.extend(model_samples[:args.limit] if args.limit else model_samples)
    captures = ordered
    jobs = [(f"{s['model']}:{s['sample_id']}:{metric}", {**s, "metric": metric}) for s in captures for metric in metrics]
    with Store(args.output_dir) as store:
        store.bind(config)
        try:
            async with RagasAdapter(judge) as adapter:

                async def operation(sample: dict) -> dict:
                    result = await adapter.score(sample["metric"], sample)
                    score = result.get("score")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError("Метрика не вернула конечное число.")
                    return {**sample, **result, "status": "complete", "evaluation_fingerprint": digest(config)}

                await run_jobs(
                    store, jobs, operation, concurrency=args.concurrency, attempts=args.attempts,
                    label="RAGAS, оценки метрик",
                )
        finally:
            export_evaluation(store, models, total)
        complete = sum(store.is_done(key) for key, _ in jobs)
        log.info("Оценки сохранены в %s; готово %d/%d", args.output_dir, complete, len(jobs))
        return 0 if complete == len(jobs) else 1
