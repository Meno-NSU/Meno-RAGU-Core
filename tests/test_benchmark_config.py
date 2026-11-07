from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from meno_rag.api import benchmark_config as config_mod
from meno_rag.config import Settings


def _app(tmp_path, **overrides):
    app = FastAPI()
    app.include_router(config_mod.router)
    app.middleware("http")(config_mod.benchmark_guard)
    app.state.settings = Settings(_env_file=None, BENCH_API_TOKEN="benchmark-secret",
                                  STAND_RESOURCES_DIR=tmp_path, **overrides)
    app.state.pipeline = object()
    app.state.vllm_registry = SimpleNamespace(_cache=[
        {"id": "meno-lite", "endpoint": "http://model:9020", "created": 123, "context_length": 16384}
    ])
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, "wrong-token"])
async def test_preflight_rejects_before_reading_resources(tmp_path, monkeypatch, token):
    app = _app(tmp_path)
    monkeypatch.setattr(config_mod, "source_identity", lambda: pytest.fail("Unexpected resource access"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.get("/v1/benchmark/config",
                                    headers={"Authorization": f"Bearer {token}"} if token else {})
    assert response.status_code == 404
    assert "X-Bench-Mode" not in response.headers


async def _config(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.get("/v1/benchmark/config", headers={"Authorization": "Bearer benchmark-secret"})
    assert response.status_code == 200
    assert response.headers["X-Bench-Mode"] == "1"
    return response.json()


@pytest.mark.asyncio
async def test_preflight_excludes_secrets_and_execution_settings(tmp_path):
    (tmp_path / "build_manifest.json").write_text('{"files":{"graph":"sha"}}')
    app = _app(tmp_path, DATABASE_URL="postgresql://user:db-secret@host/db",
               AUTH_JWT_SECRET="jwt-secret", OPENAI_API_KEY="model-secret",
               ENGINE_RERANKER_URL="http://user:rerank-secret@reranker:8002/v1?key=query-secret")
    first = await _config(app)
    assert first["schema_version"] == 2
    assert first["identity"]["knowledge"]["provenance_complete"] is True
    assert first["identity"]["runtime"]["engine_reranker_url"] == "http://reranker:8002/v1"
    for secret in ("db-secret", "jwt-secret", "model-secret", "benchmark-secret", "rerank-secret", "query-secret"):
        assert secret not in str(first)
    app.state.settings.bench_max_concurrent = 64
    app.state.settings.rerank_concurrency = 128
    app.state.settings.graph_timeout_seconds = 900
    app.state.vllm_registry._cache[0]["created"] = 999
    second = await _config(app)
    assert first["fingerprint"] == second["fingerprint"]
    app.state.settings.graph_source_documents += 1
    third = await _config(app)
    assert third["fingerprint"] != first["fingerprint"]


@pytest.mark.asyncio
async def test_preflight_captures_graph_manifest_and_model_context(tmp_path):
    app = _app(tmp_path)
    first = await _config(app)
    assert first["identity"]["knowledge"]["provenance_complete"] is False
    assert first["identity"]["models"][0]["context_length"] == 16384
    app.state.vllm_registry._cache[0]["context_length"] = 8192
    assert (await _config(app))["fingerprint"] == first["fingerprint"]
    app.state.vllm_registry._cache = []
    assert (await _config(app))["fingerprint"] == first["fingerprint"]
    (tmp_path / "build_manifest.json").write_text('{"files":{"graph":"sha"}}')
    fresh = _app(tmp_path)
    assert (await _config(fresh))["fingerprint"] != first["fingerprint"]


@pytest.mark.asyncio
async def test_preflight_requires_ready_pipeline(tmp_path):
    app = _app(tmp_path)
    app.state.pipeline = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.get("/v1/benchmark/config", headers={"Authorization": "Bearer benchmark-secret"})
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_benchmark_chat_preserves_upstream_finish_reason_and_usage():
    from unittest.mock import AsyncMock

    from meno_rag.api.main import _non_stream_response
    from meno_rag.schemas import ChatCompletionRequest
    from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime

    trace = {"generation": {"finish_reason": "length", "usage": {"completion_tokens": 4096}}}
    outcome = SimpleNamespace(trace=trace, sources=[], stage_durations_ms={})
    pipeline = SimpleNamespace(prepare=AsyncMock(return_value=outcome),
                               generate_text=AsyncMock(return_value="Незавершённый ответ"))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pipeline=pipeline, database=None)))
    response = await _non_stream_response(
        request=request, payload=ChatCompletionRequest(model="meno-lite", messages=[{"role": "user", "content": "Вопрос"}]),
        runtime=PipelineRuntime.uniform(ModelRuntime("meno-lite", "http://fake/v1")), completion_id="test-id",
        created_ts=1, session_id="test-session", max_tokens=4096, temperature=0, capture_trace=True, bench=True,
    )
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"] == {"completion_tokens": 4096}


@pytest.mark.asyncio
async def test_benchmark_fingerprint_guard_rejects_before_chat_handler(tmp_path):
    app = _app(tmp_path)
    calls = []

    @app.post("/v1/chat/completions")
    async def chat():
        calls.append(1)
        return {"answer": "ok"}

    config = await _config(app)
    headers = {"Authorization": "Bearer benchmark-secret", "X-Benchmark-Fingerprint": config["fingerprint"]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", headers=headers, json={})
        assert response.status_code == 200
        assert response.headers["X-Benchmark-Fingerprint"] == config["fingerprint"]
        app.state.settings.top_k += 1
        response = await client.post("/v1/chat/completions", headers=headers, json={})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "benchmark_configuration_changed"
    assert len(calls) == 1
    app.state.settings.bench_api_token = "changed-token"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", headers=headers, json={})
    assert response.status_code == 404
    assert len(calls) == 1
