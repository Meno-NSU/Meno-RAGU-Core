"""Offline, read-only audit and paired analysis of exported RAGAS results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from benchmarking.common import atomic_json, digest, load_dataset
from benchmarking.result_diagnostics import analyze_diagnostics

MODELS = ("meno-lite", "gemma-4-12B-it")
METRICS = ("faithfulness", "factual_correctness", "answer_relevancy", "context_precision", "context_recall")


def fingerprint_file(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values), "mean": float(array.mean()) if values else None,
        "median": float(np.median(array)) if values else None,
        "zeros": int(np.count_nonzero(array == 0)), "ones": int(np.count_nonzero(array == 1)),
        "min": float(array.min()) if values else None, "max": float(array.max()) if values else None,
    }


def paired(left: dict, right: dict, repetitions: int, seed: int, selected: set | None = None) -> dict:
    keys = sorted(set(left) & set(right) & selected) if selected is not None else sorted(set(left) & set(right))
    a = np.array([left[key] for key in keys], dtype=float)
    b = np.array([right[key] for key in keys], dtype=float)
    difference = b - a
    rng = np.random.default_rng(seed)
    bootstrap = []
    if keys:
        for start in range(0, repetitions, 200):
            indices = rng.integers(0, len(keys), size=(min(200, repetitions - start), len(keys)))
            bootstrap.extend(difference[indices].mean(axis=1))
    return {
        "n": len(keys), "meno_mean": float(a.mean()) if keys else None,
        "gemma_mean": float(b.mean()) if keys else None,
        "gemma_minus_meno": float(difference.mean()) if keys else None,
        "delta_ci95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])] if bootstrap else None,
        "gemma_wins": int(np.count_nonzero(difference > 1e-12)),
        "ties": int(np.count_nonzero(np.abs(difference) <= 1e-12)),
        "meno_wins": int(np.count_nonzero(difference < -1e-12)),
    }


def inspect_calls(row: dict, audit: dict) -> int | None:
    details = row.get("details") or {}
    judge_calls = []
    flags = []
    for call in details.get("http_calls", []):
        audit["http_statuses"][str(call.get("status_code"))] += 1
        audit["http_kinds"][call.get("kind", "unknown")] += 1
        if call.get("kind") != "judge":
            continue
        judge_calls.append(call)
        request = call.get("request") or {}
        response = call.get("response") or {}
        audit["judge_models"][response.get("model", "unknown")] += 1
        if request.get("chat_template_kwargs", {}).get("enable_thinking") is not False:
            audit["judge_requests_without_explicit_thinking_off"] += 1
        for choice in response.get("choices", []):
            audit["finish_reasons"][choice.get("finish_reason", "unknown")] += 1
            message = choice.get("message") or {}
            audit["nonempty_reasoning"] += bool(message.get("reasoning"))
            audit["nonempty_refusal"] += bool(message.get("refusal"))
            try:
                parsed = json.loads(message["content"])
            except (KeyError, TypeError, ValueError):
                audit["unparseable_judge_json"] += 1
                continue
            for field in ("verdict", "attributed", "noncommittal"):
                if field in parsed and (type(parsed[field]) is not int or parsed[field] not in (0, 1)):
                    audit["nonbinary_judgments"] += 1
            if row["metric"] == "answer_relevancy" and "noncommittal" in parsed:
                flags.append(parsed["noncommittal"])
            for entry in parsed.get("statements", []) + parsed.get("classifications", []):
                if isinstance(entry, dict):
                    for field in ("verdict", "attributed"):
                        if field in entry and (type(entry[field]) is not int or entry[field] not in (0, 1)):
                            audit["nonbinary_judgments"] += 1
        usage = response.get("usage") or {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            audit["judge_usage"][field] += usage.get(field) or 0
    checks = details.get("budget_checks") or []
    for check in checks:
        if "prompt_tokens" in check:
            audit["max_prompt_tokens"] = max(audit["max_prompt_tokens"], check["prompt_tokens"])
            required = check["prompt_tokens"] + check["reserved"] + check["margin_tokens"]
            audit["max_reserved_total"] = max(audit["max_reserved_total"], required)
            if required > check["context_limit"]:
                audit["over_budget_checks"] += 1
    if len(checks) == len(judge_calls):
        for check, call in zip(checks, judge_calls, strict=True):
            audit["aligned_budget_checks"] += 1
            if check.get("prompt_tokens") != call.get("response", {}).get("usage", {}).get("prompt_tokens"):
                audit["tokenizer_usage_mismatches"] += 1
    else:
        audit["rows_with_unaligned_budget_count"] += 1
    if flags and all(type(flag) is int and flag in (0, 1) for flag in flags) and len(set(flags)) == 1:
        return int(flags[0])
    return None


def analyze(directory: Path, dataset: Path, repetitions: int = 10000, seed: int = 42) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    config = summary["config"]
    dataset_hash, samples = load_dataset(dataset)
    if config["dataset_sha256"] != dataset_hash or set(config["models"]) != set(MODELS):
        raise ValueError("Набор вопросов или модели не совпадают с протоколом отчёта.")
    if set(config["metrics"]) != set(METRICS):
        raise ValueError("Набор метрик не совпадает с протоколом отчёта.")
    expected_fingerprint = digest(config)
    scores: dict = {model: {metric: {} for metric in METRICS} for model in MODELS}
    records: dict = {model: {} for model in MODELS}
    hashes = {}
    failed = []
    seen = set()
    failed_exports = {}
    audit = {
        "http_statuses": Counter(), "http_kinds": Counter(), "judge_models": Counter(),
        "finish_reasons": Counter(), "judge_usage": Counter(), "nonempty_reasoning": 0,
        "nonempty_refusal": 0, "unparseable_judge_json": 0, "nonbinary_judgments": 0,
        "judge_requests_without_explicit_thinking_off": 0, "max_prompt_tokens": 0,
        "max_reserved_total": 0, "over_budget_checks": 0, "aligned_budget_checks": 0,
        "tokenizer_usage_mismatches": 0, "rows_with_unaligned_budget_count": 0,
    }
    files = [directory / "summary.json", directory / "checkpoints.sqlite3"]
    for file in sorted(directory.glob("*/scores.jsonl.gz")):
        files.append(file)
        with gzip.open(file, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                model, metric, sample_id = row["model"], row["metric"], row["sample_id"]
                original = samples[row["index"]]
                if (sample_id != original["sample_id"] or row["user_input"] != original["question"]
                        or row["reference"] != original["ground_truth"]):
                    raise ValueError(f"Нарушена привязка исходного вопроса: {model}, {row['index']}.")
                key = f"{model}:{sample_id}:{metric}"
                if key in seen:
                    raise ValueError(f"Повторная запись задания: {key}.")
                seen.add(key)
                if row.get("job_key", key) != key:
                    raise ValueError(f"Неверный ключ задания: {key}.")
                meta = records[model].setdefault(sample_id, {
                    "index": row["index"], "question": row["user_input"], "response": row["response"],
                    "reference": row["reference"], "noncommittal": None,
                    "context_chars": sum(map(len, row["retrieved_contexts"])),
                    "context_blocks": len(row["retrieved_contexts"]),
                })
                if meta["response"] != row["response"]:
                    raise ValueError(f"Различные ответы в метриках одного sample_id: {key}.")
                if row["status"] != "complete":
                    if row["status"] not in {"failed", "running"}:
                        raise ValueError(f"Неизвестный статус задания: {key}.")
                    failed_exports[key] = (row["status"], row.get("error"))
                    failed.append({"key": key, "model": model, "metric": metric, "index": row["index"],
                                   "sample_id": sample_id, "error": row.get("error")})
                    continue
                if key in hashes or not math.isfinite(row["score"]):
                    raise ValueError(f"Повторная или неопределённая успешная оценка: {key}.")
                if row["evaluation_fingerprint"] != expected_fingerprint:
                    raise ValueError(f"Неверный отпечаток оценки: {key}.")
                hashes[key] = hashlib.sha256(line.rstrip("\r\n").encode()).hexdigest()
                scores[model][metric][sample_id] = row["score"]
                flag = inspect_calls(row, audit)
                if metric == "answer_relevancy":
                    meta["noncommittal"] = flag
    database = sqlite3.connect(f"file:{directory / 'checkpoints.sqlite3'}?mode=ro", uri=True)
    try:
        db_counts = dict(database.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
        for key, payload in database.execute("SELECT key,payload FROM jobs WHERE status='complete'"):
            if hashes.get(key) != hashlib.sha256(gzip.decompress(payload)).hexdigest():
                raise ValueError(f"SQLite и JSONL не совпадают: {key}.")
        if len(hashes) != db_counts.get("complete", 0):
            raise ValueError("Число успешных строк SQLite и JSONL не совпадает.")
        if len(failed) != db_counts.get("failed", 0) + db_counts.get("running", 0):
            raise ValueError("Число ошибочных строк SQLite и JSONL не совпадает.")
        database_failed = {
            key: (status, error)
            for key, status, error in database.execute("SELECT key,status,error FROM jobs WHERE status!='complete'")
        }
        if failed_exports != database_failed:
            raise ValueError("Ошибочные записи SQLite и JSONL не совпадают по ключам, статусам и ошибкам.")
        attempts = database.execute(
            "SELECT COUNT(*),SUM(error IS NOT NULL),MIN(started),MAX(finished) FROM attempts"
        ).fetchone()
        error_attempts = [
            {"job_key": key, "error": error, "started": started, "finished": finished}
            for key, error, started, finished in database.execute(
                "SELECT job_key,error,started,finished FROM attempts WHERE error IS NOT NULL ORDER BY id"
            )
        ]
        all_attempt_audit = deepcopy(audit)
        for key, payload in database.execute("SELECT job_key,details FROM attempts WHERE error IS NOT NULL"):
            if payload:
                inspect_calls({"metric": key.rsplit(":", 1)[-1], "details": json.loads(gzip.decompress(payload))}, all_attempt_audit)
        full_run_times = database.execute("SELECT MIN(started),MAX(finished) FROM attempts WHERE id>10").fetchone()
    finally:
        database.close()
    all_stats = {}
    diagnostics = {}
    for model in MODELS:
        all_stats[model] = {}
        for metric in METRICS:
            result = stats(list(scores[model][metric].values()))
            if not math.isclose(result["mean"], summary["models"][model][metric]["mean"], abs_tol=1e-12):
                raise ValueError(f"Среднее не совпадает со сводкой: {model}, {metric}.")
            result["coverage"] = result["n"] / len(samples)
            all_stats[model][metric] = result
        rows = records[model]
        answered = {key for key, meta in rows.items() if meta["noncommittal"] == 0}
        noncommittal = {key for key, meta in rows.items() if meta["noncommittal"] == 1}
        diagnostics[model] = {
            "available_answers": len(rows), "missing_indices": sorted(set(range(len(samples))) - {r["index"] for r in rows.values()}),
            "noncommittal_count": len(noncommittal), "noncommittal_fraction": len(noncommittal) / len(rows),
            "noncommittal_unknown": sum(meta["noncommittal"] is None for meta in rows.values()),
            "context_chars": stats([row["context_chars"] for row in rows.values()]),
            "context_blocks": stats([row["context_blocks"] for row in rows.values()]),
            "noncommittal_false_cohort": {
                metric: stats([value for key, value in scores[model][metric].items() if key in answered])
                for metric in METRICS
            },
            "context_recall_zero_cohort_factual": stats([
                value for key, value in scores[model]["factual_correctness"].items()
                if scores[model]["context_recall"].get(key) == 0
            ]),
            "context_recall_at_least_08_cohort_factual": stats([
                value for key, value in scores[model]["factual_correctness"].items()
                if scores[model]["context_recall"].get(key, -1) >= 0.8
            ]),
        }
    paired_stats = {metric: paired(scores[MODELS[0]][metric], scores[MODELS[1]][metric], repetitions, seed) for metric in METRICS}
    both_answered = {key for key in records[MODELS[0]] if records[MODELS[0]][key]["noncommittal"] == 0
                     and records[MODELS[1]].get(key, {}).get("noncommittal") == 0}
    examples = {
        str(index): {
            model: {**meta, "sample_id": key, "scores": {metric: scores[model][metric].get(key) for metric in METRICS}}
            for model in MODELS for key, meta in records[model].items() if meta["index"] == index
        }
        for index in (50, 205, 218, 258, 463, 477, 785, 807)
    }
    return {
        "schema_version": 1, "dataset_size": len(samples), "config": config,
        "input_artifacts": [{"relative_path": str(file.relative_to(directory)), **fingerprint_file(file)} for file in files],
        "counts": db_counts, "failed_jobs": failed, "attempts": {
            "total": attempts[0], "with_error": attempts[1], "first_started_utc": datetime.fromtimestamp(attempts[2], UTC).isoformat(),
            "last_finished_utc": datetime.fromtimestamp(attempts[3], UTC).isoformat(),
            "last_finished_novosibirsk": datetime.fromtimestamp(attempts[3], ZoneInfo("Asia/Novosibirsk")).isoformat(),
            "error_history": error_attempts,
            "distinct_jobs_with_errors": len({row["job_key"] for row in error_attempts}),
            "full_run_attempts_after_first_10_smoke": {
                "started_utc": datetime.fromtimestamp(full_run_times[0], UTC).isoformat(),
                "finished_utc": datetime.fromtimestamp(full_run_times[1], UTC).isoformat(),
                "seconds": full_run_times[1] - full_run_times[0],
                "assumption": "first_10_attempts_are_the_verified_smoke",
            },
        },
        "all_successful_scores": all_stats, "paired": paired_stats,
        "bootstrap": {"repetitions": repetitions, "seed_reset_per_metric": seed, "method": "paired_percentile",
                      "quantiles": [0.025, 0.975], "sample_order": "lexicographic_sample_id", "numpy_version": np.__version__},
        "diagnostics": diagnostics,
        "paired_both_noncommittal_false": {
            metric: paired(scores[MODELS[0]][metric], scores[MODELS[1]][metric], repetitions, seed, both_answered)
            for metric in METRICS
        },
        "successful_job_http_audit": audit, "all_recorded_attempt_http_audit": all_attempt_audit,
        "evidence_diagnostics": analyze_diagnostics(directory),
        "illustrative_examples": examples,
    }


def plot(report: dict, directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Faithfulness", "Factual correctness", "Answer relevancy", "Context precision", "Context recall"]
    colors = ("#15616d", "#d3663c")
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(METRICS))
    for offset, model, color, label in zip((-0.18, 0.18), MODELS, colors, ("Meno Lite", "Gemma"), strict=True):
        values = [report["all_successful_scores"][model][metric]["mean"] for metric in METRICS]
        left.barh(y + offset, values, height=0.32, color=color, label=label)
        for position, value in zip(y + offset, values, strict=True):
            left.text(value + 0.015, position, f"{value:.3f}", va="center", fontsize=9)
    left.set(yticks=y, yticklabels=labels, xlim=(0, 1.03), xlabel="Среднее по успешным оценкам", title="Все успешные оценки, N = 953–954")
    left.invert_yaxis()
    left.legend(loc="lower right", frameon=False)
    differences = [report["paired"][metric]["gemma_minus_meno"] for metric in METRICS]
    intervals = np.array([report["paired"][metric]["delta_ci95"] for metric in METRICS])
    right.errorbar(differences, y, xerr=[np.array(differences) - intervals[:, 0], intervals[:, 1] - differences],
                   fmt="o", color="#334155", ecolor="#64748b", capsize=5, markersize=6)
    right.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
    right.set(yticks=y, yticklabels=labels, xlim=(-0.28, 0.41), xlabel="Gemma − Meno Lite", title="Общие вопросы, парный bootstrap 95%")
    right.invert_yaxis()
    right.set_ylim(left.get_ylim())
    for position, metric in enumerate(METRICS):
        right.text(0.99, position + 0.20, f"N={report['paired'][metric]['n']}", transform=right.get_yaxis_transform(),
                   ha="right", va="top", fontsize=8, color="#64748b")
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle("NSU v1 · RAGAS 0.4.3 · Qwen3.8-27B FP8", fontsize=14, y=1.02)
    fig.text(0.5, -0.02, "Интервалы отражают вариацию вопросов; ошибки автоматического судьи в них не учтены.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(directory / "metrics.svg", bbox_inches="tight")
    fig.savefig(directory / "metrics.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка и парный анализ сохранённых результатов RAGAS без HTTP-запросов.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data/benchmark_1000_verified.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap должен быть положительным")
    if args.output_dir.resolve() == args.input_dir.resolve():
        parser.error("Каталог отчёта должен отличаться от исходных результатов")
    report = analyze(args.input_dir, args.dataset, args.bootstrap, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "analysis.json", report)
    if args.plot:
        plot(report, args.output_dir)
    print(json.dumps({"counts": report["counts"], "paired": report["paired"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
