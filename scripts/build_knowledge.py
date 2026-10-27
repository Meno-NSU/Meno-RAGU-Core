#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

REPO_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_DIR / "code"
CORPUS_NAME = "chunked_texts_about_nsu_with_metadata_and_scores.jsonl"
YANDEX_API = "https://cloud-api.yandex.net/v1/disk/public/resources"
METADATA_RESUME_PREVIOUS_CODE = {
    "31cdf1e": {
        "driver": "227d547e2fc38979d07f625905010447bb8a4c1528bbbd422e0b12f6b448b5b3",
        "generate_title_and_annotation.py": "e3a2ffd86a88050ae6a8396eb988e196066dfb3e6e5c607cfd6028cc90cdf9c1",
    },
    "b94b09c": {
        "driver": "0d3703b8cf99e014bcc7a16edac78fe8fe4dd85e5a3286bff3138c42a67e9ed8",
        "generate_title_and_annotation.py": "24b8efc26c89b817f46c2f54106c3563cfd0c886c50a16479b83a3e2818c6f76",
    },
}


class BuildError(RuntimeError):
    pass


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Значение должно быть положительным.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Сборка корпуса НГУ и полного графа RAGU через HTTP-службы моделей.")
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Каталог data из набора meno_stand_actual либо его родитель."
    )
    parser.add_argument(
        "--source-url", help="Публичная ссылка на папку Яндекс Диска; нужные исходники скачиваются в --input-dir."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Новый каталог готовой базы знаний для backend.")
    parser.add_argument("--llm-url", required=True, help="Адрес OpenAI-совместимой LLM; суффикс /v1 можно опустить.")
    parser.add_argument("--llm-model", help="Имя модели в API; при единственной модели определяется автоматически.")
    parser.add_argument(
        "--tokenizer", required=True, help="Локальный путь или HF id токенизатора для разбиения корпуса."
    )
    parser.add_argument("--graph-tokenizer", help="Токенизатор внешней LLM; по умолчанию --tokenizer.")
    parser.add_argument("--embedder-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedder-model", default="BAAI/bge-m3")
    parser.add_argument("--embedder-tokenizer", help="Локальный токенизатор bge-m3; по умолчанию имя модели.")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--llm-key-env", default="MENO_INDEX_LLM_KEY", help="Имя переменной окружения с ключом LLM.")
    parser.add_argument("--embedder-key-env", default="MENO_INDEX_EMBEDDER_KEY")
    parser.add_argument("--summary", type=Path, help="Готовый JSONL справочника; переопределяет автоматический выбор.")
    parser.add_argument("--teachers", type=Path, help="Готовый JSONL преподавателей.")
    parser.add_argument("--base", type=Path, help="Проверенный JSONL сайта НГУ.")
    parser.add_argument(
        "--news",
        type=Path,
        action="append",
        help="Новостной JSONL; флаг повторяется. Иначе берутся все filtered_news_*.jsonl.",
    )
    parser.add_argument("--abbreviations", type=Path)
    parser.add_argument("--chunk-tokens", type=positive, default=350)
    parser.add_argument(
        "--llm-context-tokens",
        type=positive,
        help="Контекст LLM: по умолчанию max_model_len из /v1/models, иначе 16384.",
    )
    parser.add_argument("--max-completion-tokens", type=positive, default=6144)
    parser.add_argument("--embedder-tokens", type=positive, default=8192)
    parser.add_argument("--concurrency", type=positive, default=8)
    parser.add_argument("--embedding-batch", type=positive, default=16)
    parser.add_argument("--icl-examples", type=int, default=0)
    parser.add_argument("--chunk-header", choices=("none", "title", "title_annotation"), default="title")
    parser.add_argument("--chunk-header-sources", default="base,news")
    parser.add_argument("--no-news-dates", action="store_true")
    parser.add_argument("--timezone", default="Asia/Novosibirsk")
    parser.add_argument("--log-level", choices=("INFO", "DEBUG"), default="INFO", help="Подробность журналов RAGU.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать этапы без HTTP-запросов, загрузки токенизаторов и записи результатов.",
    )
    args = parser.parse_args(argv)
    if args.chunk_tokens < 45:
        parser.error("--chunk-tokens должен быть не меньше 45 (минимальный фрагмент — 30 токенов).")
    if args.icl_examples < 0:
        parser.error("--icl-examples не может быть отрицательным.")
    if args.llm_context_tokens is not None and args.llm_context_tokens <= args.max_completion_tokens:
        parser.error("--llm-context-tokens должен превышать --max-completion-tokens.")
    return args


def http_url(value, *, openai=False):
    parts = urlsplit(value.strip())
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise BuildError(
            "Нужен HTTP(S)-адрес без ключей, пароля, параметров и фрагмента; ключи задаются через окружение."
        )
    path = parts.path.rstrip("/")
    if openai and not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def tokenizer_path(value):
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


@dataclass
class Sources:
    summary: Path
    teachers: Path
    base: Path
    news: list[Path]
    abbreviations: Path
    summary_raw: bool = False
    teachers_raw: bool = False
    teachers_chunks: Path | None = None

    def files(self):
        return [
            self.summary,
            self.teachers,
            self.base,
            *self.news,
            self.abbreviations,
            *([self.teachers_chunks] if self.teachers_chunks else []),
        ]


def input_root(directory):
    root = directory.expanduser().resolve()
    if (root / "data" / "source").is_dir():
        root = root / "data"
    elif root.name == "source" and root.is_dir():
        root = root.parent
    return root


def find_sources(args):
    root = input_root(args.input_dir)
    source = root / "source"
    summary = args.summary or source / "nsu_text_corpus_20260706.jsonl"
    summary_raw = not summary.is_file() and args.summary is None
    if summary_raw:
        summary = root / "nsu_main_info.md"
    teachers = args.teachers
    if teachers is None:
        prepared = sorted(source.glob("teachers_corpus_????????.jsonl"))
        teachers = prepared[-1] if prepared else source / "teachers.json"
    teachers_raw = args.teachers is None and teachers.name == "teachers.json"
    news = args.news if args.news is not None else sorted(source.glob("filtered_news_????????_????????.jsonl"))
    if not news:
        raise BuildError(f"Не найдены новости в {source}; укажите --news.")
    for path in news:
        if not re.fullmatch(r"\w+_\d{8}_\d{8}\.jsonl", path.name):
            raise BuildError(f"Неверное имя новостного файла: {path.name}")
    result = Sources(
        summary.resolve(),
        teachers.resolve(),
        (args.base or source / "nsu_text_corpus_20250318_validated.jsonl").resolve(),
        [p.resolve() for p in news],
        (args.abbreviations or root / "knowledge" / "abbreviations.json").resolve(),
        summary_raw,
        teachers_raw,
        (source / "teachers_chunks.json").resolve() if teachers_raw else None,
    )
    for path in result.files():
        if not path.is_file() or path.stat().st_size == 0:
            raise BuildError(f"Не найден или пуст исходный файл: {path}")
    return result


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class YandexSource:
    def __init__(self, public_url, directory, client):
        self.public_url = http_url(public_url)
        self.root = input_root(directory)
        self.client = client

    def api(self, path, *, download=False, offset=0):
        import httpx

        url = YANDEX_API + ("/download" if download else "")
        params = {"public_key": self.public_url, "path": path}
        if not download:
            params.update(limit=1000, offset=offset)
        try:
            response = self.client.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise BuildError(f"API Яндекс Диска вернул HTTP {exc.response.status_code} для {path}.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise BuildError(f"Ошибка API Яндекс Диска для {path}: {type(exc).__name__}. Повторите команду.") from exc

    def listing(self, path):
        items = []
        while True:
            data = self.api(path, offset=len(items))
            if data.get("type") != "dir" or "_embedded" not in data:
                raise BuildError(f"Нужна публичная папка Яндекс Диска: {path}")
            page = data["_embedded"]
            current = page.get("items", [])
            items.extend(current)
            if not current or len(items) >= page["total"]:
                return {item["name"]: item for item in items}

    def selection(self):
        top = self.listing("/")
        prefix = "" if "source" in top else "/data"
        data = top if not prefix else self.listing(prefix)
        source = self.listing(prefix + "/source")
        knowledge = self.listing(prefix + "/knowledge")
        selected = []

        def include(folder, remote_folder, local_folder, name):
            item = folder.get(name, {})
            if item.get("type") != "file":
                raise BuildError(f"В наборе Яндекс Диска отсутствует {remote_folder}/{name}")
            sha256 = item.get("sha256", "")
            size = item.get("size", 0)
            if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256) or not isinstance(size, int) or size <= 0:
                raise BuildError(f"Нет размера или SHA-256 исходного файла: {name}")
            selected.append(
                {
                    "remote": f"{remote_folder}/{name}",
                    "local": f"{local_folder}{name}",
                    "size": size,
                    "sha256": sha256.lower(),
                }
            )

        def include_source(name):
            include(source, prefix + "/source", "source/", name)

        if "nsu_text_corpus_20260706.jsonl" in source:
            include_source("nsu_text_corpus_20260706.jsonl")
        else:
            include(data, prefix, "", "nsu_main_info.md")
        teachers = sorted(name for name in source if re.fullmatch(r"teachers_corpus_\d{8}\.jsonl", name))
        if teachers:
            include_source(teachers[-1])
        else:
            include_source("teachers.json")
            include_source("teachers_chunks.json")
        include_source("nsu_text_corpus_20250318_validated.jsonl")
        news = sorted(name for name in source if re.fullmatch(r"filtered_news_\d{8}_\d{8}\.jsonl", name))
        if not news:
            raise BuildError("В наборе Яндекс Диска нет файлов новостей.")
        for name in news:
            include_source(name)
        include(knowledge, prefix + "/knowledge", "knowledge/", "abbreviations.json")
        return selected

    def target(self, item):
        target = self.root / item["local"]
        if not target.resolve().is_relative_to(self.root) or target == self.root:
            raise BuildError("Путь загрузки выходит за пределы --input-dir.")
        return target

    def download(self, item):
        import httpx

        target = self.target(item)
        if target.exists():
            if target.stat().st_size == item["size"] and digest(target) == item["sha256"]:
                print(f"[ПРОПУСК ЗАГРУЗКИ] {item['local']}: SHA-256 проверен.", flush=True)
                return
            raise BuildError(f"Локальный файл отличается от исходника: {target}. Укажите новый --input-dir.")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        if partial.is_symlink():
            raise BuildError(f"Вместо временного файла найдена ссылка: {partial}")
        print(f"[ЗАГРУЗКА] {item['local']}: {item['size'] / 1_000_000:.1f} МБ", flush=True)
        for attempt in range(3):
            try:
                link = self.api(item["remote"], download=True)
                parts = urlsplit(link.get("href", ""))
                if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
                    raise BuildError(f"API не вернул HTTPS-ссылку для {item['local']}")
                checksum = hashlib.sha256()
                received = 0
                with self.client.stream("GET", link["href"], follow_redirects=True, timeout=60) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            received += len(chunk)
                            if received > item["size"]:
                                raise BuildError(f"Изменился размер исходника Яндекс Диска: {item['local']}")
                            stream.write(chunk)
                            checksum.update(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                if received != item["size"] or checksum.hexdigest() != item["sha256"]:
                    raise BuildError(f"Не совпали размер или SHA-256: {item['local']}. Файл не принят как готовый.")
                partial.replace(target)
                return
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise BuildError(
                        f"Не удалось скачать {item['local']}: {type(exc).__name__}. Повторите команду."
                    ) from exc
                time.sleep(attempt + 1)

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".download.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BuildError("Исходники в этом каталоге уже скачиваются.") from exc
            manifest = self.root / ".yandex-source.json"
            if manifest.exists():
                saved = json.loads(manifest.read_text(encoding="utf-8"))
                if saved["public_url"] != self.public_url:
                    raise BuildError("В --input-dir уже скачивался другой набор. Укажите новый каталог.")
                files = saved["files"]
            else:
                files = self.selection()
                atomic_json(manifest, {"public_url": self.public_url, "files": files})
            for item in files:
                self.download(item)


def scan_jsonl(path, required=(), *, metadata=False, corpus=False):
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                if not isinstance(doc, dict) or any(key not in doc for key in required):
                    raise ValueError("отсутствуют обязательные поля")
                if metadata and any(
                    not isinstance(doc[key], str) or not doc[key].strip() for key in ("doc_title", "doc_annotation")
                ):
                    raise ValueError("модель не создала название или аннотацию; повторите команду для продолжения")
                if corpus:
                    text = doc["doc_full_text"]
                    quality = float(doc["quality_score"])
                    if not math.isfinite(quality) or not 0 <= quality <= 1:
                        raise ValueError("неверная оценка качества")
                    if not doc["chunks"]:
                        raise ValueError("нет фрагментов")
                    for chunk in doc["chunks"]:
                        if not 0 <= chunk["start_char"] < chunk["end_char"] <= len(text):
                            raise ValueError("неверные границы фрагмента")
            except (ValueError, KeyError, TypeError) as exc:
                raise BuildError(f"{path.name}, строка {line_number}: {exc}") from exc
            count += 1
    if not count:
        raise BuildError(f"Пустой JSONL: {path}")
    return count


@dataclass
class Stage:
    name: str
    script: str
    args: list[str]
    inputs: list[Path]
    output: Path
    required: tuple[str, ...] = ()

    @property
    def command(self):
        return [sys.executable, "-u", str(CODE_DIR / self.script), *self.args]


def make_plan(args, sources, work):
    bundle = work / "bundle"
    intermediate = work / "stages"
    stages = []
    summary, teachers = sources.summary, sources.teachers
    if sources.summary_raw:
        summary = intermediate / "summary.jsonl"
        stages.append(
            Stage(
                "01-summary",
                "prepare_nsu_main_info_corpus.py",
                ["-i", str(sources.summary), "-o", str(summary)],
                [sources.summary],
                summary,
                ("text", "url"),
            )
        )
    if sources.teachers_raw:
        teachers = intermediate / "teachers.jsonl"
        stages.append(
            Stage(
                "02-teachers",
                "prepare_teachers_corpus.py",
                ["-i", str(sources.teachers), "-c", str(sources.teachers_chunks), "-o", str(teachers)],
                [sources.teachers, sources.teachers_chunks],
                teachers,
                ("text", "url"),
            )
        )
    united = intermediate / "united.jsonl"
    chunked = intermediate / "chunked.jsonl"
    lengths = intermediate / "lengths.jsonl"
    metadata = intermediate / "metadata.jsonl"
    scored = intermediate / "scored.jsonl"
    corpus = bundle / CORPUS_NAME
    mapping = bundle / "chunk_mapping_to_texts.json"
    graph = bundle / "knowledge" / "ragu_graph"
    news_args = [arg for path in sources.news for arg in ("-n", str(path))]
    stages.extend(
        [
            Stage(
                "03-combine",
                "combine_data.py",
                [
                    "-s",
                    str(summary),
                    "-t",
                    str(teachers),
                    "-b",
                    str(sources.base),
                    *news_args,
                    "--timezone",
                    args.timezone,
                    "-o",
                    str(united),
                ],
                [summary, teachers, sources.base, *sources.news],
                united,
                ("text", "url", "content_source"),
            ),
            Stage(
                "04-chunking",
                "apply_chunking.py",
                ["-i", str(united), "-o", str(chunked), "-m", args.tokenizer, "--maxtokens", str(args.chunk_tokens)],
                [united],
                chunked,
                ("text", "chunk_boundaries"),
            ),
            Stage(
                "05-token-counts",
                "calculate_num_tokens.py",
                ["-i", str(chunked), "-o", str(lengths), "-m", args.tokenizer],
                [chunked],
                lengths,
                ("text", "chunks"),
            ),
            Stage(
                "06-metadata",
                "generate_title_and_annotation.py",
                [
                    "-i",
                    str(lengths),
                    "-o",
                    str(metadata),
                    "-m",
                    args.llm_model,
                    "--url",
                    args.llm_url,
                    "--resume",
                    "--fail_on_error",
                    "--concurrency",
                    str(args.concurrency),
                    "--timeout",
                    "180",
                ],
                [lengths],
                metadata,
                ("doc_full_text", "doc_title", "doc_annotation", "chunks"),
            ),
            Stage(
                "07-quality",
                "evaluate_title_and_annotation.py",
                [
                    "-i",
                    str(metadata),
                    "-o",
                    str(scored),
                    "-m",
                    args.embedder_model,
                    "--url",
                    args.embedder_url,
                    "--minibatch",
                    str(args.embedding_batch),
                ],
                [metadata],
                scored,
                ("doc_full_text", "doc_title", "doc_annotation", "chunks", "quality_score"),
            ),
        ]
    )
    if not args.no_news_dates:
        stages.append(
            Stage(
                "08-news-dates",
                "update_title_for_news.py",
                ["-b", str(scored), *news_args, "-o", str(corpus), "--timezone", args.timezone],
                [scored, *sources.news],
                corpus,
                ("doc_full_text", "doc_title", "doc_annotation", "chunks", "quality_score"),
            )
        )
    stages.append(
        Stage("09-mapping", "prepare_chunk_mapping.py", ["-i", str(corpus), "-o", str(mapping)], [corpus], mapping)
    )
    stages.append(
        Stage(
            "10-graph",
            "prepare_graph_index.py",
            [
                "-i",
                str(corpus),
                "-r",
                str(mapping),
                "-o",
                str(graph),
                "-a",
                str(sources.abbreviations),
                "-e",
                args.embedder_model,
                "--embedder_url",
                args.embedder_url,
                "--embedder_tokenizer",
                args.embedder_tokenizer,
                "--embedder_tokens",
                str(args.embedder_tokens),
                "--llm",
                args.llm_model,
                "--url",
                args.llm_url,
                "--llm_tokenizer",
                args.graph_tokenizer,
                "--llm_context_tokens",
                str(args.llm_context_tokens),
                "--qdrant_location",
                args.qdrant_url,
                "--log_level",
                args.log_level,
                "--cache",
                str(work / "llm-cache"),
                "--chunk_header",
                args.chunk_header,
                "--chunk_header_sources",
                args.chunk_header_sources,
                "--icl_examples",
                str(args.icl_examples),
                "--max_completion_tokens",
                str(args.max_completion_tokens),
                "--rate_max_simultaneous",
                str(args.concurrency),
            ],
            [corpus, mapping, sources.abbreviations],
            graph,
        )
    )
    return stages


class Builder:
    def __init__(self, args, *, http_client=None):
        self.args = args
        self.sources = None if args.source_url else find_sources(args)
        args.output_dir = args.output_dir.expanduser().resolve()
        args.llm_url = http_url(args.llm_url, openai=True)
        args.embedder_url = http_url(args.embedder_url, openai=True)
        args.qdrant_url = http_url(args.qdrant_url)
        args.tokenizer = tokenizer_path(args.tokenizer)
        args.graph_tokenizer = tokenizer_path(args.graph_tokenizer or args.tokenizer)
        args.embedder_tokenizer = tokenizer_path(args.embedder_tokenizer or args.embedder_model)
        ZoneInfo(args.timezone)
        self.work = args.output_dir.parent / ("." + args.output_dir.name + ".build")
        self.bundle = self.work / "bundle"
        self.client = http_client
        self.owns_client = False
        self.state = {"stages": {}}
        self.input_hashes = {}

    def headers(self, kind):
        env = self.args.llm_key_env if kind == "llm" else self.args.embedder_key_env
        key = os.environ.get(env, "EMPTY")
        return {"Authorization": f"Bearer {key}"} if key else {}

    def qdrant_headers(self):
        key = os.environ.get("QDRANT_API_KEY", "")
        return {"api-key": key} if key else {}

    def request(self, method, url, **kwargs):
        import httpx

        try:
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BuildError(f"Не удалось обратиться к {url}: {exc}") from exc

    def preflight(self):
        from tokenization_utils.tokenization_utils import load_hf_tokenizer

        for name in dict.fromkeys((self.args.tokenizer, self.args.graph_tokenizer, self.args.embedder_tokenizer)):
            print(f"[ПРОВЕРКА] Токенизатор: {name}", flush=True)
            try:
                tokenizer = load_hf_tokenizer(name)
                if name == self.args.graph_tokenizer:
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Проверка"}], tokenize=True, add_generation_prompt=True
                    )
            except (OSError, ValueError) as exc:
                raise BuildError(f"Не удалось загрузить токенизатор {name}: {exc}") from exc
        models = self.request("GET", self.args.llm_url + "/models", headers=self.headers("llm"))
        ids = sorted({row["id"] for row in models.get("data", [])})
        if not self.args.llm_model:
            if len(ids) != 1:
                raise BuildError(f"Укажите --llm-model; API вернул модели: {ids}")
            self.args.llm_model = ids[0]
        if self.args.llm_model not in ids:
            raise BuildError(f"Модель {self.args.llm_model!r} отсутствует в API: {ids}")
        info = next(row for row in models["data"] if row["id"] == self.args.llm_model)
        served_context = info.get("max_model_len")
        if not isinstance(served_context, int) or served_context <= 0:
            served_context = None
        if self.args.llm_context_tokens is None:
            self.args.llm_context_tokens = served_context or 16384
        if served_context and self.args.llm_context_tokens > served_context:
            raise BuildError(
                f"Задан контекст {self.args.llm_context_tokens}, но API сообщает максимум {served_context}."
            )
        if self.args.llm_context_tokens <= self.args.max_completion_tokens:
            raise BuildError("Контекст службы не превышает --max-completion-tokens; уменьшите предел ответа.")
        print(f"[ПРОВЕРКА] LLM: {self.args.llm_model}; контекст: {self.args.llm_context_tokens} токенов.", flush=True)
        embedding = self.request(
            "POST",
            self.args.embedder_url + "/embeddings",
            headers=self.headers("embedder"),
            json={"model": self.args.embedder_model, "input": ["Проверка векторизации"]},
        )
        if not embedding.get("data") or not embedding["data"][0].get("embedding"):
            raise BuildError("Служба векторизации не вернула вектор.")
        self.request("GET", self.args.qdrant_url + "/collections", headers=self.qdrant_headers())

    def configuration(self, stages):
        files = sorted(set(self.sources.files()))
        self.input_hashes = {str(path): digest(path) for path in files}
        settings = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self.args).items()
            if key
            not in {
                "dry_run",
                "summary",
                "teachers",
                "base",
                "news",
                "abbreviations",
                "input_dir",
                "output_dir",
                "log_level",
            }
        }
        scripts = {stage.script for stage in stages}
        code_hashes = {name: digest(CODE_DIR / name) for name in sorted(scripts)}
        code_hashes["driver"] = digest(Path(__file__))
        code_hashes["dependencies"] = digest(REPO_DIR / "uv.lock")
        code_hashes["news_utils.py"] = digest(CODE_DIR / "news_utils.py")
        tokenizer_hashes = {}
        for name in {self.args.tokenizer, self.args.graph_tokenizer, self.args.embedder_tokenizer}:
            folder = Path(name)
            if folder.is_dir():
                tokenizer_hashes[name] = {
                    str(path.relative_to(folder)): digest(path)
                    for path in sorted(folder.iterdir())
                    if path.is_file() and path.suffix in {".json", ".txt", ".model", ".jinja"}
                }
        for folder in (
            "context_utils",
            "filter_utils",
            "graph_utils",
            "search_utils",
            "rewriting_utils",
            "tokenization_utils",
        ):
            for path in sorted((CODE_DIR / folder).glob("*.py")):
                code_hashes[str(path.relative_to(CODE_DIR))] = digest(path)
        config = {
            "version": 1,
            "settings": settings,
            "inputs": self.input_hashes,
            "code": code_hashes,
            "tokenizers": tokenizer_hashes,
        }
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return fingerprint, config

    def execute(self, stage, log_path):
        environment = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
        environment["MENO_INDEX_LLM_KEY"] = os.environ.get(self.args.llm_key_env, "EMPTY")
        environment["MENO_INDEX_EMBEDDER_KEY"] = os.environ.get(self.args.embedder_key_env, "EMPTY")
        with (
            log_path.open("a", encoding="utf-8") as log,
            subprocess.Popen(
                stage.command,
                cwd=CODE_DIR,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            ) as process,
        ):
            log.write(f"\n[{datetime.now(UTC).isoformat()}] {stage.name}: {shlex.join(stage.command)}\n")
            log.flush()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise BuildError(f"Этап {stage.name} завершился с кодом {returncode}. Журнал: {log_path}")

    def validate_stage(self, stage):
        if stage.required:
            count = scan_jsonl(stage.output, stage.required, metadata=stage.name == "06-metadata")
            if stage.name in {
                "05-token-counts",
                "06-metadata",
                "07-quality",
                "08-news-dates",
            } and count != scan_jsonl(stage.inputs[0]):
                raise BuildError(f"Этап {stage.name} изменил количество документов.")
        elif stage.name == "09-mapping":
            mapping = json.loads(stage.output.read_text())
            if not mapping or set(mapping) != {str(i) for i in range(len(mapping))}:
                raise BuildError("Отображение фрагментов пусто или нарушена нумерация.")
        else:
            self.verify_graph(stage.output)

    def stage_hash(self, stage):
        if stage.output.is_file():
            return digest(stage.output)
        names = ("knowledge_graph.gml", "kv_chunks.json", "menon_chunk_mapping.json", "qdrant_collections.json")
        return {name: digest(stage.output / name) for name in names}

    def verify_graph(self, graph):
        required = ("knowledge_graph.gml", "kv_chunks.json", "menon_chunk_mapping.json", "qdrant_collections.json")
        for name in required:
            if not (graph / name).is_file() or not (graph / name).stat().st_size:
                raise BuildError(f"Не завершена сборка графа: {graph / name}")
        mapping = json.loads((graph / "menon_chunk_mapping.json").read_text())
        chunks = json.loads((graph / "kv_chunks.json").read_text())
        reverse = json.loads((graph.parents[1] / "chunk_mapping_to_texts.json").read_text())
        if not mapping or set(mapping) != set(chunks) or any(str(index) not in reverse for index in mapping.values()):
            raise BuildError("Отображение фрагментов графа не соответствует корпусу.")
        collections = json.loads((graph / "qdrant_collections.json").read_text())
        for kind in ("vdb_entity", "vdb_relation", "vdb_chunk"):
            name = collections[kind]
            result = self.request(
                "POST",
                self.args.qdrant_url + f"/collections/{quote(name, safe='')}/points/count",
                headers=self.qdrant_headers(),
                json={"exact": True},
            )
            count = result["result"]["count"]
            if count <= 0 or (kind == "vdb_chunk" and count != len(mapping)):
                raise BuildError(f"Неполная коллекция Qdrant {name}: {count} элементов.")

    def run_stage(self, stage):
        command = stage.command
        if "--log_level" in command:
            index = command.index("--log_level")
            command = command[:index] + command[index + 2 :]
        signature = {"command": command, "inputs": {str(path): digest(path) for path in stage.inputs}}
        entry = self.state["stages"].get(stage.name)
        try:
            reusable = entry and entry["signature"] == signature and entry["output"] == self.stage_hash(stage)
        except (OSError, KeyError):
            reusable = False
        if reusable:
            if stage.name == "10-graph":
                self.verify_graph(stage.output)
            print(f"[ПРОПУСК] {stage.name}: результат проверен.", flush=True)
            return
        if entry and stage.name == "06-metadata":
            stage.output.unlink(missing_ok=True)
        print(f"[ЭТАП] {stage.name}", flush=True)
        self.execute(stage, self.work / "logs" / (stage.name + ".log"))
        self.validate_stage(stage)
        self.state["stages"][stage.name] = {"signature": signature, "output": self.stage_hash(stage)}
        atomic_json(self.work / "state.json", self.state)

    def build(self):
        if (
            self.args.output_dir.exists()
            and any(self.args.output_dir.iterdir())
            and not (self.args.output_dir / "build_manifest.json").is_file()
        ):
            raise BuildError("Каталог результата непустой и не создан этим сборщиком. Укажите новый --output-dir.")
        if self.args.dry_run:
            if self.args.source_url:
                print(f"00-download: {self.args.source_url} → {input_root(self.args.input_dir)}")
                try:
                    self.sources = find_sources(self.args)
                except BuildError:
                    print("Затем: объединение → фрагменты → метаданные → оценки → даты → отображения → граф RAGU.")
                    print("Точный план файлов будет определён при загрузке; dry-run не обращается к Яндекс Диску.")
                    return self.args.output_dir
            self.args.llm_model = self.args.llm_model or "<автоматически из /v1/models>"
            self.args.llm_context_tokens = self.args.llm_context_tokens or "<из /v1/models; иначе 16384>"
            for stage in make_plan(self.args, self.sources, self.work):
                print(f"{stage.name}: {shlex.join(stage.command)}")
            return self.args.output_dir
        import httpx

        if self.client is None:
            self.client = httpx.Client(timeout=30)
            self.owns_client = True
        try:
            if self.args.source_url:
                YandexSource(self.args.source_url, self.args.input_dir, self.client).run()
                self.sources = find_sources(self.args)
            self.preflight()
            stages = make_plan(self.args, self.sources, self.work)
            fingerprint, config = self.configuration(stages)
            self.work.mkdir(parents=True, exist_ok=True)
            with (self.work / "build.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise BuildError("Сборка в этом рабочем каталоге уже выполняется.") from exc
                if not self.check_completed(fingerprint):
                    self.run_locked(stages, fingerprint, config)
            return self.args.output_dir
        finally:
            if self.owns_client:
                self.client.close()

    def check_completed(self, fingerprint):
        completed = self.args.output_dir / "build_manifest.json"
        if not completed.is_file():
            return False
        saved = json.loads(completed.read_text())
        if saved["fingerprint"] != fingerprint:
            raise BuildError("Параметры или исходники изменились. Для новой сборки укажите новый --output-dir.")
        for name, checksum in saved["files"].items():
            if digest(self.args.output_dir / name) != checksum:
                raise BuildError(f"Изменён готовый файл: {name}; используйте новый каталог результата.")
        self.verify_graph(self.args.output_dir / "knowledge" / "ragu_graph")
        print(f"База уже собрана и проверена: {self.args.output_dir}")
        return True

    def run_locked(self, stages, fingerprint, config):
        state_path = self.work / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text())
            if self.state["fingerprint"] != fingerprint:
                previous_fingerprints = set()
                for previous_code in METADATA_RESUME_PREVIOUS_CODE.values():
                    previous_config = {**config, "code": {**config["code"], **previous_code}}
                    previous_fingerprints.add(hashlib.sha256(json.dumps(previous_config, sort_keys=True).encode()).hexdigest())
                compatible_metadata_fix = (
                    self.state["fingerprint"] in previous_fingerprints
                    and set(self.state["stages"]) <= {stage.name for stage in stages if stage.name < "06-metadata"}
                )
                if self.state["stages"] and not compatible_metadata_fix:
                    raise BuildError(
                        "Параметры, код или входные файлы изменились. Продолжение возможно с прежними параметрами; иначе используйте новый --output-dir."
                    )
                if compatible_metadata_fix:
                    print("[ОБНОВЛЕНИЕ] Исправлено восстановление метаданных; сохраняю подтверждённые этапы 01–05.", flush=True)
                    self.state["fingerprint"] = fingerprint
                else:
                    print(
                        "[ПОВТОР] Код или параметры изменились; подтверждённых этапов ещё нет, начинаю заново.", flush=True
                    )
                    self.state = {"fingerprint": fingerprint, "stages": {}}
                atomic_json(state_path, self.state)
        else:
            self.state = {"fingerprint": fingerprint, "stages": {}}
            atomic_json(state_path, self.state)
        for directory in (self.work / "stages", self.work / "logs", self.bundle / "knowledge"):
            directory.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            if stage.name == "09-mapping":
                corpus = self.bundle / CORPUS_NAME
                if self.args.no_news_dates:
                    shutil.copyfile(self.work / "stages" / "scored.jsonl", corpus)
                scan_jsonl(corpus, ("doc_full_text", "chunks", "quality_score"), metadata=True, corpus=True)
            self.run_stage(stage)
        if any(digest(Path(path)) != checksum for path, checksum in self.input_hashes.items()):
            raise BuildError("Во время сборки изменились исходные файлы. Результат не опубликован.")
        shutil.copyfile(self.sources.abbreviations, self.bundle / "knowledge" / "abbreviations.json")
        files = {
            str(path.relative_to(self.bundle)): digest(path)
            for path in sorted(self.bundle.rglob("*"))
            if path.is_file() and "llm_debug" not in path.parts and path.name != "build_manifest.json"
        }
        atomic_json(
            self.bundle / "build_manifest.json", {"fingerprint": fingerprint, "configuration": config, "files": files}
        )
        if self.args.output_dir.exists():
            self.args.output_dir.rmdir()
        self.bundle.rename(self.args.output_dir)
        print(
            f"Готовая база знаний: {self.args.output_dir}\nДля backend: STAND_RESOURCES_DIR={self.args.output_dir}\nQDRANT_URL={self.args.qdrant_url}",
            flush=True,
        )


def main(argv=None):
    try:
        Builder(parse_args(argv)).build()
        return 0
    except (BuildError, OSError, ValueError) as exc:
        print(f"Ошибка сборки: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Сборка прервана; повторите ту же команду для продолжения.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
