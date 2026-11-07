"""Metric dispatch, cache isolation and concurrent judge auditing without models."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from benchmarking import ragas_adapter as module
from benchmarking.ragas_adapter import RagasAdapter, RagasConfig, RagasMetricError, UndefinedMetricError


@pytest.fixture
def config(tmp_path):
    return RagasConfig(
        judge_url="http://judge.invalid:9022/v1",
        judge_model="judge",
        judge_revision="weights-v1",
        embedder_url="http://embedder.invalid:8001/v1",
        embedder_model="BAAI/bge-m3",
        cache_dir=tmp_path / "cache",
        judge_api_key="private-judge-key",
        embedder_api_key="private-embedding-key",
    )


@pytest.fixture
def fake_components(monkeypatch):
    state = SimpleNamespace(
        cache_paths=[], closed_caches=[], clients=[], metric_calls=[],
        llm_kwargs=None, embedding_kwargs=None, value=0.75, error=None,
    )

    async def respond(request):
        await asyncio.sleep(0)
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "test-completion",
            "choices": [{"message": {"role": "assistant", "content": payload["messages"][0]["content"]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        })

    class FakeSdk:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.closed = False
            state.clients.append(self)

        async def close(self):
            self.closed = True
            await self.http_client.aclose()

    class FakeCache:
        def __init__(self, cache_dir):
            state.cache_paths.append(cache_dir)
            self.cache = SimpleNamespace(close=lambda: state.closed_caches.append(cache_dir))

    def llm_factory(model, **kwargs):
        state.llm_kwargs = kwargs
        return SimpleNamespace(model=model, **kwargs)

    def embedding_factory(provider, **kwargs):
        state.embedding_kwargs = kwargs
        return SimpleNamespace(provider=provider, **kwargs)

    def metric_class(label):
        class Metric:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            async def ascore(self, **kwargs):
                state.metric_calls.append((label, kwargs))
                content = kwargs.get("user_input", kwargs.get("reference"))
                await self.llm.client.http_client.post(
                    self.llm.client.base_url + "/chat/completions",
                    json={"model": "judge", "messages": [{"role": "user", "content": content}]},
                    headers={"Authorization": "Bearer never-store-this-key"},
                )
                if state.error:
                    raise state.error
                return SimpleNamespace(value=state.value, reason="test explanation")

        return Metric

    monkeypatch.setattr(module, "_load_components", lambda: {
        "http_client": lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs),
        "openai_client": FakeSdk,
        "cache": FakeCache,
        "llm_factory": llm_factory,
        "embedding_factory": embedding_factory,
        "metric_classes": tuple(metric_class(name) for name in module.METRIC_NAMES),
    })
    return state


def sample(question="Какие факультеты есть в НГУ?"):
    return {
        "user_input": question,
        "response": "В НГУ есть механико-математический факультет.",
        "reference": "Механико-математический факультет входит в НГУ.",
        "retrieved_contexts": ["Сущность: ММФ. Факультет НГУ.", "ММФ — механико-математический факультет."],
        "unrelated_capture_field": "must not reach the metric",
    }


def test_module_import_does_not_load_ragas_or_clients():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import benchmarking.ragas_adapter; "
            "assert not {'ragas','openai','httpx'} & sys.modules.keys()"
        )],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fingerprint_excludes_credentials_and_execution_settings(config, tmp_path):
    payload = module.fingerprint_payload(config)
    serialized = json.dumps(payload)
    assert "private" not in serialized
    changed = replace(config, judge_api_key="rotated", embedder_api_key="rotated", timeout=700, cache_dir=tmp_path)
    assert changed.fingerprint_payload() == payload
    assert RagasAdapter(config).fingerprint_payload == payload
    assert replace(config, judge_url=config.judge_url + "/").fingerprint_payload() == payload


@pytest.mark.parametrize("name,value", [
    ("judge_url", "http://user:secret@judge.invalid/v1"),
    ("embedder_url", "http://embedder.invalid/v1?api_key=secret"),
    ("judge_url", "file:///tmp/model"),
    ("judge_revision", ""),
    ("max_tokens", 0),
    ("temperature", float("nan")),
    ("timeout", float("inf")),
])
def test_config_rejects_invalid_or_secret_bearing_settings(config, name, value):
    with pytest.raises(ValueError):
        replace(config, **{name: value})


async def test_separate_cache_namespaces_follow_model_identity(config, fake_components):
    async with RagasAdapter(config):
        pass
    initial_judge, initial_embeddings = fake_components.cache_paths
    async with RagasAdapter(replace(config, judge_revision="weights-v2")):
        pass
    assert fake_components.cache_paths[2] != initial_judge
    assert fake_components.cache_paths[3] == initial_embeddings
    async with RagasAdapter(replace(config, embedder_model="another-embedder")):
        pass
    assert fake_components.cache_paths[4] == initial_judge
    assert fake_components.cache_paths[5] != initial_embeddings
    assert all(client.closed for client in fake_components.clients)
    assert sorted(fake_components.closed_caches) == sorted(fake_components.cache_paths)


async def test_factories_use_http_bge_and_thinking_setting(config, fake_components):
    async with RagasAdapter(config):
        assert fake_components.llm_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert fake_components.embedding_kwargs["model"] == "BAAI/bge-m3"
        assert fake_components.embedding_kwargs["interface"] == "modern"
        assert fake_components.embedding_kwargs["client"].base_url == config.embedder_url
    async with RagasAdapter(replace(config, disable_thinking=False)):
        assert "extra_body" not in fake_components.llm_kwargs


@pytest.mark.parametrize("metric_name,fields", [
    ("faithfulness", {"user_input", "response", "retrieved_contexts"}),
    ("factual_correctness", {"response", "reference"}),
    ("answer_relevancy", {"user_input", "response"}),
    ("context_precision", {"user_input", "reference", "retrieved_contexts"}),
    ("context_recall", {"user_input", "reference", "retrieved_contexts"}),
])
async def test_metric_dispatch_and_judge_output_audit(config, fake_components, metric_name, fields):
    async with RagasAdapter(config) as adapter:
        result = await adapter.score(metric_name, sample())
    assert result["score"] == 0.75
    assert result["reason"] == "test explanation"
    assert set(fake_components.metric_calls[0][1]) == fields
    details = result["details"]
    assert details["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert details["usage_scope"] == "uncached_http_calls_only"
    assert details["http_calls"][0]["response"]["choices"][0]["message"]["content"]
    assert "never-store-this-key" not in json.dumps(result)
    assert "private" not in json.dumps(result)


async def test_parallel_scores_keep_judge_audits_separate(config, fake_components):
    async with RagasAdapter(config) as adapter:
        first, second = await asyncio.gather(
            adapter.score("faithfulness", sample("Первый вопрос")),
            adapter.score("faithfulness", sample("Второй вопрос")),
        )
    for result, expected in ((first, "Первый вопрос"), (second, "Второй вопрос")):
        calls = result["details"]["http_calls"]
        assert len(calls) == 1
        assert calls[0]["request"]["messages"][0]["content"] == expected


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), "invalid", True])
async def test_undefined_metric_is_failure_not_zero(config, fake_components, value):
    fake_components.value = value
    async with RagasAdapter(config) as adapter:
        with pytest.raises(UndefinedMetricError) as failure:
            await adapter.score("faithfulness", sample())
    assert failure.value.metric_name == "faithfulness"
    assert len(failure.value.details["http_calls"]) == 1


async def test_zero_is_a_valid_score(config, fake_components):
    fake_components.value = 0.0
    async with RagasAdapter(config) as adapter:
        assert (await adapter.score("faithfulness", sample()))["score"] == 0.0


async def test_judge_failure_keeps_completed_call_audit(config, fake_components):
    fake_components.error = ValueError("malformed structured response")
    async with RagasAdapter(config) as adapter:
        with pytest.raises(RagasMetricError) as failure:
            await adapter.score("faithfulness", sample())
    assert "malformed structured response" in str(failure.value)
    assert len(failure.value.details["http_calls"]) == 1


async def test_upstream_echoed_credentials_are_redacted(config, fake_components):
    async with RagasAdapter(config) as adapter:
        result = await adapter.score("faithfulness", sample(config.judge_api_key))
        assert config.judge_api_key not in json.dumps(result)
        fake_components.error = ValueError(f"Unauthorized key {config.embedder_api_key}")
        with pytest.raises(RagasMetricError) as failure:
            await adapter.score("faithfulness", sample(config.embedder_api_key))
    assert config.embedder_api_key not in str(failure.value)
    assert config.embedder_api_key not in json.dumps(failure.value.details)
    assert "[REDACTED]" in str(failure.value)


async def test_missing_contexts_do_not_issue_judge_requests(config, fake_components):
    async with RagasAdapter(config) as adapter:
        bad_sample = sample()
        bad_sample["retrieved_contexts"] = []
        with pytest.raises(RagasMetricError, match="retrieved_contexts"):
            await adapter.score("faithfulness", bad_sample)
    assert not fake_components.metric_calls


async def test_adapter_requires_open_context(config):
    with pytest.raises(RuntimeError, match="context manager"):
        await RagasAdapter(config).score("faithfulness", sample())


@pytest.mark.skipif(importlib.util.find_spec("ragas") is None, reason="RAGAS evaluation extra is not installed")
async def test_actual_ragas_metrics_and_disk_cache_without_model_network(config, monkeypatch):
    components = module._load_components()
    transport_calls = []

    async def respond(request):
        transport_calls.append(request)
        payload = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            inputs = payload["input"]
            inputs = inputs if isinstance(inputs, list) else [inputs]
            return httpx.Response(200, json={
                "object": "list", "model": config.embedder_model,
                "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.5]} for i, _ in enumerate(inputs)],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            })
        messages = json.dumps(payload["messages"])
        verdict = {"statement": "ММФ входит в НГУ.", "reason": "Указано в источнике.", "verdict": 1}
        if "StatementGeneratorOutput" in messages:
            output = {"statements": ["ММФ входит в НГУ."]}
        elif "ClaimDecompositionOutput" in messages:
            output = {"claims": ["ММФ входит в НГУ."]}
        elif "NLIStatementOutput" in messages:
            output = {"statements": [verdict]}
        elif "ContextRecallOutput" in messages:
            output = {"classifications": [{"statement": verdict["statement"], "reason": verdict["reason"], "attributed": 1}]}
        elif "ContextPrecisionOutput" in messages:
            output = {"reason": verdict["reason"], "verdict": 1}
        elif "AnswerRelevanceOutput" in messages:
            output = {"question": "Какие факультеты есть в НГУ?", "noncommittal": 0}
        else:
            raise AssertionError(f"Unexpected structured judge schema: {messages[:500]}")
        return httpx.Response(200, json={
            "id": "mock-judge", "object": "chat.completion", "created": 1, "model": config.judge_model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(output, ensure_ascii=False),
            }}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        })

    components["http_client"] = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)
    monkeypatch.setattr(module, "_load_components", lambda: components)
    async with RagasAdapter(config) as adapter:
        for name in adapter.metric_names:
            result = await adapter.score(name, sample())
            assert result["score"] == pytest.approx(1.0)
    assert transport_calls
    assert any(request.url.path.endswith("/embeddings") for request in transport_calls)
    first_pass_count = len(transport_calls)
    async with RagasAdapter(config) as adapter:
        for name in adapter.metric_names:
            result = await adapter.score(name, sample())
            assert result["score"] == pytest.approx(1.0)
            assert result["details"]["http_calls"] == []
    assert len(transport_calls) == first_pass_count
