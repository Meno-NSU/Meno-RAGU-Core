from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

from benchmarking.common import FatalRunError, atomic_json, digest, export_jsonl, load_dataset, run_jobs, safe_name
from benchmarking.storage import Store

log = logging.getLogger("benchmarking")


def bench_token(env_file: Path | None) -> str:
    value = os.environ.get("BENCH_API_TOKEN", "")
    if not value and env_file:
        value = str(dotenv_values(env_file).get("BENCH_API_TOKEN") or "")
    if not value:
        raise ValueError("Задайте BENCH_API_TOKEN в окружении или --backend-env с этим ключом.")
    return value


def validate_capture(data: dict) -> dict:
    trace = data.get("pipeline_trace")
    if not isinstance(trace, dict) or trace.get("schema_version") != 2:
        raise ValueError("Backend не вернул полную трассировку версии 2. Обновите и пересоберите core.")
    contexts = trace.get("retrieved_contexts")
    evidence = trace.get("final_evidence")
    if not isinstance(contexts, list) or not contexts or any(not isinstance(t, str) or not t.strip() for t in contexts):
        raise ValueError("Нет сохранённых доказательств, переданных генератору.")
    if not isinstance(evidence, list) or [block.get("text") for block in evidence] != contexts:
        raise ValueError("Текст контекста не совпадает со структурированными доказательствами.")
    context = trace.get("final_context")
    messages = trace.get("qa_messages")
    if not isinstance(context, str) or not context or not isinstance(messages, list) or not messages:
        raise ValueError("Отсутствует финальный промпт или контекст.")
    if not any(context in message.get("content", "") for message in messages):
        raise ValueError("Итоговый контекст отсутствует в сообщениях генератору.")
    previous_end = 0
    for block in evidence:
        start, end = block.get("context_start"), block.get("context_end")
        if (
            type(start) is not int or type(end) is not int or start < previous_end or end <= start
            or end > len(context) or context[start:end] != block["text"]
        ):
            raise ValueError("Доказательства не соответствуют позициям в итоговом контексте.")
        previous_end = end
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Некорректный ответ API.") from exc
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Генератор вернул пустой ответ.")
    generation = trace.get("generation") or {}
    if generation.get("finish_reason") != "stop":
        raise ValueError(f"Генерация не завершилась штатно: {generation.get('finish_reason')!r}.")
    return {"response": answer, "retrieved_contexts": contexts}


def export_collection(store: Store, models: list[str], total: int) -> dict:
    counts = {model: {"complete": 0, "failed": 0, "running": 0} for model in models}
    for record in store.records(failures=True):
        model = record["model"]
        if model in counts:
            counts[model][record.get("status", "complete")] += 1
    for model in models:
        directory = store.directory / safe_name(model)
        directory.mkdir(exist_ok=True)
        export_jsonl(directory / "responses.jsonl.gz", (r for r in store.records() if r["model"] == model))
        export_jsonl(
            directory / "errors.jsonl.gz",
            (r for r in store.records(failures=True) if r["model"] == model and r.get("status") != "complete"),
        )
        atomic_json(directory / "summary.json", {"model": model, "dataset_size": total, **counts[model]})
    summary = {"schema_version": 1, "config": store.metadata("config"), "models": counts}
    atomic_json(store.directory / "manifest.json", summary)
    return summary


async def collect(args) -> int:
    parsed_url = urlsplit(args.backend_url)
    if (
        parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc
        or parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment
    ):
        raise ValueError("--backend-url должен быть HTTP(S)-адресом без пароля и параметров.")
    dataset_hash, samples = load_dataset(args.dataset)
    selected = samples[:args.limit] if args.limit else samples
    token = bench_token(args.backend_env)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(
        base_url=args.backend_url.rstrip("/") + "/", headers=headers,
        timeout=args.timeout, trust_env=False,
        limits=httpx.Limits(max_connections=max(args.concurrencies.values()) + 4),
    ) as client:
        preflight = await client.get("v1/benchmark/config")
        if preflight.status_code != 200 or preflight.headers.get("X-Bench-Mode") != "1":
            raise ValueError(
                f"Проверка режима измерений не пройдена (HTTP {preflight.status_code}). "
                "Проверьте BENCH_API_TOKEN и обновление core. Запросы генерации не отправлены."
            )
        backend = preflight.json()
        if backend.get("schema_version") != 2 or not backend.get("fingerprint"):
            raise ValueError("Некорректная версия сведений backend.")
        client.headers["X-Benchmark-Fingerprint"] = backend["fingerprint"]
        catalog_response = await client.get("v1/models")
        catalog_response.raise_for_status()
        catalog = catalog_response.json()
        model_metadata = [
            {key: entry.get(key) for key in ("id", "endpoint", "context_length")}
            for entry in catalog.get("data", []) if entry.get("id") in args.models
        ]
        model_metadata.sort(key=lambda entry: (entry["id"], entry["endpoint"] or ""))
        discovered = {entry["id"] for entry in model_metadata}
        if set(args.models) - discovered:
            raise ValueError(f"Модели не обнаружены: {sorted(set(args.models) - discovered)}")
        smoke = getattr(args, "command", "collect") == "smoke"
        smoke_diagnostics = {}
        if smoke:
            readiness = await client.get("readyz")
            readiness.raise_for_status()
            health = readiness.json()
            if health.get("status") != "ok" or health.get("rag_ready") is not True:
                raise ValueError(f"Backend не готов: {health}")
            smoke_diagnostics = {"readyz": health, "models": sorted(discovered), "backend_fingerprint": backend["fingerprint"]}
            log.info("Smoke: API и граф готовы, модели обнаружены. Проверяю полный RAG для %s.", args.models)
        available = {item["id"] for item in backend.get("identity", {}).get("models", [])}
        if available and set(args.models) - available:
            raise ValueError(f"Модели отсутствуют в backend: {sorted(set(args.models) - available)}")
        config = {
            "schema_version": 1, "kind": "collection", "dataset_sha256": dataset_hash,
            "dataset_size": len(samples), "backend_url": args.backend_url.rstrip("/"),
            "backend_fingerprint": backend["fingerprint"], "backend_provenance": {
                "schema_version": backend["schema_version"], "fingerprint": backend["fingerprint"],
                "identity": {k: v for k, v in backend.get("identity", {}).items() if k != "models"},
            },
            "models": args.models, "model_revisions": args.model_revisions,
            "selected_models": model_metadata,
            "knowledge_base_id": args.knowledge_base_id, "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        with Store(args.output_dir) as store:
            store.bind(config)
            try:
                for model in args.models:
                    jobs = [(f"{model}:{s['sample_id']}", {"model": model, **s}) for s in selected]

                    async def operation(sample: dict) -> dict:
                        request = {
                            "model": sample["model"], "knowledge_base_id": args.knowledge_base_id,
                            "messages": [{"role": "user", "content": sample["question"]}],
                            "stream": False, "max_tokens": args.max_tokens, "temperature": args.temperature,
                        }
                        started = time.monotonic()
                        response = await client.post("v1/chat/completions", json=request)
                        if response.status_code in {401, 403, 404, 409}:
                            raise FatalRunError(f"Режим измерений недоступен или backend изменился: HTTP {response.status_code}")
                        if response.status_code >= 400:
                            raise ValueError(f"HTTP {response.status_code}: {response.text[:2000]}")
                        if response.headers.get("X-Bench-Mode") != "1":
                            raise FatalRunError("Ответ получен вне режима измерений, результат отклонён.")
                        if response.headers.get("X-Benchmark-Fingerprint") != backend["fingerprint"]:
                            raise FatalRunError("Backend изменился во время сбора, результат отклонён.")
                        raw = response.json()
                        prepared = validate_capture(raw)
                        if raw.get("model") != sample["model"]:
                            raise ValueError("Backend ответил другой моделью.")
                        result = {
                            **sample, **prepared, "status": "complete", "request": request,
                            "raw_response": raw, "elapsed_seconds": time.monotonic() - started,
                            "collected_at_unix": time.time(), "backend_fingerprint": backend["fingerprint"],
                        }
                        result["capture_sha256"] = digest(result)
                        return result

                    await run_jobs(
                        store, jobs, operation, concurrency=args.concurrencies[model], attempts=args.attempts,
                        label=f"Ответы {model}",
                    )
            finally:
                export_collection(store, args.models, len(samples))
            expected = len(selected) * len(args.models)
            actual = sum(store.is_done(key) for model in args.models for key in (f"{model}:{s['sample_id']}" for s in selected))
            log.info("Ответы и трассировки сохранены в %s; готово %d/%d", args.output_dir, actual, expected)
            if smoke:
                smoke_diagnostics["status"] = "passed" if actual == expected else "failed"
                smoke_diagnostics["results"] = [
                    {
                        "model": record["model"], "index": record["index"],
                        "elapsed_seconds": record["elapsed_seconds"],
                        "evidence_blocks": len(record["retrieved_contexts"]),
                        "answer_chars": len(record["response"]),
                        "pipeline": record["raw_response"].get("pipeline"),
                        "finish_reason": record["raw_response"]["pipeline_trace"]["generation"]["finish_reason"],
                    }
                    for record in store.records() if record["index"] == 0
                ]
                atomic_json(args.output_dir / "smoke_report.json", smoke_diagnostics)
                log.info("Smoke %s. Отчёт: %s", smoke_diagnostics["status"], args.output_dir / "smoke_report.json")
            return 0 if actual == expected else 1
