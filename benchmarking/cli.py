from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
from pathlib import Path

from benchmarking.collect import collect


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Значение должно быть положительным.")
    return number


def assignments(values: list[str], *, integers: bool = False) -> dict:
    result = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"Ожидается МОДЕЛЬ=ЗНАЧЕНИЕ, получено {item!r}")
        result[key] = positive(value) if integers else value
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Два прохода бенчмарка Meno RAGU с чекпоинтами.")
    commands = root.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect", aliases=["smoke"], help="Собрать ответы; smoke проверяет по одному вопросу.")
    capture.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data/benchmark_1000_verified.json")
    capture.add_argument("--backend-url", default="http://127.0.0.1:9007")
    capture.add_argument("--backend-env", type=Path)
    capture.add_argument("--knowledge-base-id", default="nsu-ragu")
    capture.add_argument("--models", nargs="+", default=["meno-lite", "gemma-4-12B-it"])
    capture.add_argument("--model-revision", action="append", default=[], metavar="МОДЕЛЬ=ВЕРСИЯ")
    capture.add_argument("--model-concurrency", action="append", default=[], metavar="МОДЕЛЬ=ЧИСЛО")
    capture.add_argument("--temperature", type=float, default=0.0)
    capture.add_argument("--max-tokens", type=positive, default=4096)
    judge = commands.add_parser("evaluate", help="Оценить сохранённые ответы, не вызывая RAG.")
    judge.add_argument("--collection-dir", type=Path, required=True)
    judge.add_argument("--judge-url", required=True)
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--judge-revision", required=True, help="Метка версии весов/настроек судьи для защиты кэша.")
    judge.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"), help=argparse.SUPPRESS)
    judge.add_argument("--embedder-url", default="http://127.0.0.1:8001/v1")
    judge.add_argument("--embedder-model", default="BAAI/bge-m3")
    judge.add_argument("--embedder-api-key", default=os.environ.get("EMBEDDER_API_KEY", "EMPTY"), help=argparse.SUPPRESS)
    judge.add_argument("--concurrency", type=positive, default=16, help="Одновременно вычисляемые метрики.")
    judge.add_argument("--metrics", nargs="+")
    judge.add_argument("--enable-thinking", action="store_true")
    judge.add_argument("--allow-partial", action="store_true")
    judge.add_argument("--max-tokens", type=positive, default=4096)
    for command in (capture, judge):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--timeout", type=positive, default=600)
        command.add_argument("--attempts", type=positive, default=3)
        command.add_argument("--limit", type=positive, help="Первые N вопросов для пробного запуска.")
        command.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        if args.command in {"collect", "smoke"}:
            if args.command == "smoke":
                args.limit = 1
            if len(set(args.models)) != len(args.models):
                raise ValueError("Список моделей содержит повторы.")
            args.concurrencies = {model: 8 if model == "gemma-4-12B-it" else 1 for model in args.models}
            overrides = assignments(args.model_concurrency, integers=True)
            if set(overrides) - set(args.models):
                raise ValueError("Параллелизм указан для модели вне --models.")
            args.concurrencies.update(overrides)
            args.model_revisions = assignments(args.model_revision)
            if set(args.model_revisions) - set(args.models):
                raise ValueError("Версия указана для модели вне --models.")
            if not math.isfinite(args.temperature) or args.temperature < 0:
                raise ValueError("Температура должна быть конечным неотрицательным числом.")
            status = asyncio.run(collect(args))
        else:
            from benchmarking.evaluate import evaluate

            status = asyncio.run(evaluate(args))
    except KeyboardInterrupt:
        logging.getLogger("benchmarking").info("Остановлено. Успешные записи сохранены, повторите команду для продолжения.")
        status = 130
    except Exception as exc:
        logging.getLogger("benchmarking").error("%s: %s", type(exc).__name__, exc)
        status = 1
    raise SystemExit(status)
