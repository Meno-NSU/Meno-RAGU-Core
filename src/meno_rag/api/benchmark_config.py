from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from meno_rag.api.bench import is_bench_request
from meno_rag.config import Settings

router = APIRouter()

_CONTENT_SETTINGS = (
    "default_model", "top_k", "rerank_top_k", "rerank_candidates_per_query", "rerank_weight",
    "min_document_quality", "max_history_answer_words", "max_output_tokens", "min_output_tokens",
    "generation_temperature", "max_rewrite_queries", "max_context_chunks", "max_qa_prompt_chars",
    "stand_compat_context_order", "qa_fewshots_enabled", "n_few_shots", "max_fewshots_chars",
    "rag_rewrite_rerank_model", "graph_top_k", "engine_rerank_top_k", "graph_related_candidates",
    "graph_source_documents", "graph_max_elements", "source_urls_per_document", "embedder_model",
    "engine_reranker_model", "source_attribution", "source_attribution_threshold", "max_sources",
    "rerank_candidates_per_page",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_url(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def source_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    files = {f"meno_rag/{path.relative_to(package_root)}": _file_digest(path)
             for path in sorted(package_root.rglob("*")) if path.suffix in {".py", ".json"}}
    try:
        distribution = metadata.distribution("meno-ragu-core")
        project_root = Path(str(distribution.locate_file("")))
    except metadata.PackageNotFoundError:
        project_root = package_root.parents[1]
    candidates = (package_root.parents[1] / "code", project_root)
    for base in candidates:
        for directory in sorted(base.glob("*_utils")):
            for path in sorted(directory.rglob("*.py")):
                files[f"research/{directory.name}/{path.relative_to(directory)}"] = _file_digest(path)
    dependencies: dict[str, str | None] = {}
    for name in ("graph_ragu", "qdrant-client", "openai", "fastembed", "transformers"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"sha256": _digest({"files": files, "dependencies": dependencies}), "dependencies": dependencies}


def knowledge_identity(settings: Settings) -> dict[str, Any]:
    manifest = settings.stand_resources_dir / "build_manifest.json"
    if manifest.is_file():
        return {"manifest_sha256": _file_digest(manifest), "provenance_complete": True}
    files = {}
    for name, path in (
        ("chunk_mapping", settings.chunk_mapping_path),
        ("graph_chunk_mapping", settings.graph_dir / "menon_chunk_mapping.json"),
        ("collections", settings.graph_dir / "qdrant_collections.json"),
        ("abbreviations", settings.abbreviations_path),
    ):
        files[name] = _file_digest(path) if path.is_file() else None
    return {"manifest_sha256": None, "fallback_files": files, "provenance_complete": False}


def runtime_identity(settings: Settings) -> dict[str, Any]:
    result = {name: getattr(settings, name) for name in _CONTENT_SETTINGS}
    result.update({name: _safe_url(getattr(settings, name)) for name in (
        "embedder_url", "engine_reranker_url", "qdrant_url",
    )})
    result["vllm_endpoints"] = sorted(_safe_url(url) for url in settings.vllm_endpoint_list)
    if settings.qa_fewshots_enabled:
        result["fewshots_sha256"] = _file_digest(settings.fewshots_path) if settings.fewshots_path.is_file() else None
    return result


def benchmark_payload(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    provenance = getattr(request.app.state, "benchmark_provenance", None)
    if provenance is None:
        provenance = {"code": source_identity(), "knowledge": knowledge_identity(settings)}
        request.app.state.benchmark_provenance = provenance
    registry = getattr(request.app.state, "vllm_registry", None)
    models = [
        {"id": record["id"], "endpoint": _safe_url(record.get("endpoint") or ""),
         "context_length": record.get("context_length")}
        for record in getattr(registry, "_cache", [])
    ]
    models.sort(key=lambda record: (record["id"], record["endpoint"]))
    identity = {
        "knowledge_base_id": "nsu-ragu", "code_sha256": provenance["code"]["sha256"],
        "dependencies": provenance["code"]["dependencies"], "knowledge": provenance["knowledge"],
        "runtime": runtime_identity(settings), "models": models,
    }
    fingerprint_identity = {key: value for key, value in identity.items() if key != "models"}
    return {"schema_version": 2, "fingerprint": _digest(fingerprint_identity), "identity": identity}


async def benchmark_guard(request: Request, call_next):
    if request.method != "POST" or request.url.path != "/v1/chat/completions":
        return await call_next(request)
    if not is_bench_request(request):
        if "X-Benchmark-Fingerprint" in request.headers:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return await call_next(request)
    fingerprint = benchmark_payload(request)["fingerprint"]
    headers = {"X-Bench-Mode": "1", "X-Benchmark-Fingerprint": fingerprint}
    expected = request.headers.get("X-Benchmark-Fingerprint")
    if expected and expected != fingerprint:
        return JSONResponse({"error": {"code": "benchmark_configuration_changed",
                                      "message": "Настройки backend изменились после начала бенчмарка."}},
                            status_code=409, headers=headers)
    response = await call_next(request)
    response.headers.update(headers)
    return response


@router.get("/v1/benchmark/config")
async def benchmark_config(request: Request):
    if not is_bench_request(request):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    if getattr(request.app.state, "pipeline", None) is None:
        return JSONResponse({"detail": "Конвейер RAG ещё не готов."}, status_code=503)
    payload = benchmark_payload(request)
    return JSONResponse(payload, headers={"X-Bench-Mode": "1", "X-Benchmark-Fingerprint": payload["fingerprint"]})
