"""Unit tests for VLLMClient. Use a fake httpx transport so no network calls happen."""

import json
from unittest.mock import AsyncMock, call

import httpx
import pytest
import structlog
from structlog.testing import capture_logs

from meno_rag.llm import client as client_module
from meno_rag.llm.client import VLLMClient


def _fake_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        body = {"choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_chat_completion_forwards_seed_when_provided():
    captured: dict = {}
    transport = _fake_transport(captured)
    async with httpx.AsyncClient(transport=transport) as http:
        client = VLLMClient(http_client=http)
        await client.chat_completion(
            base_url="http://example/v1",
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            seed=42,
        )
    assert captured["payload"]["seed"] == 42


@pytest.mark.asyncio
async def test_chat_completion_omits_seed_when_none():
    captured: dict = {}
    transport = _fake_transport(captured)
    async with httpx.AsyncClient(transport=transport) as http:
        client = VLLMClient(http_client=http)
        await client.chat_completion(
            base_url="http://example/v1",
            model="x",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert "seed" not in captured["payload"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError])
async def test_transient_transport_failures_retry_with_same_payload_and_headers(monkeypatch, error_type):
    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr("meno_rag.llm.client.asyncio.sleep", sleep)

    def handler(request):
        requests.append(request)
        if len(requests) < 3:
            raise error_type("reset", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http, endpoint_api_keys={"http://x": "endpoint-key"})
        result = await client.chat_completion(
            base_url="http://x/v1", model="m", messages=[{"role": "user", "content": "hi"}], timeout=12.0,
        )
    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(requests) == 3
    assert len({request.content for request in requests}) == 1
    assert all(request.headers["Authorization"] == "Bearer endpoint-key" for request in requests)
    assert all(request.extensions["timeout"]["read"] == 12.0 for request in requests)
    assert sleep.await_args_list == [call(0.25), call(0.5)]


@pytest.mark.asyncio
async def test_transport_retries_stop_after_three_attempts(monkeypatch):
    attempts = []
    sleep = AsyncMock()
    monkeypatch.setattr("meno_rag.llm.client.asyncio.sleep", sleep)

    def handler(request):
        attempts.append(request)
        raise httpx.ReadError("reset", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        with pytest.raises(httpx.ReadError):
            await client.chat_completion(base_url="http://x/v1", model="m", messages=[])
    assert len(attempts) == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 429, 500])
async def test_http_status_errors_are_not_retried(monkeypatch, status):
    attempts = []
    sleep = AsyncMock()
    monkeypatch.setattr("meno_rag.llm.client.asyncio.sleep", sleep)

    def handler(request):
        attempts.append(request)
        return httpx.Response(status, json={"error": "failed"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat_completion(base_url="http://x/v1", model="m", messages=[])
    assert len(attempts) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_is_not_replayed_after_delivering_content(monkeypatch):
    attempts = []
    sleep = AsyncMock()
    monkeypatch.setattr("meno_rag.llm.client.asyncio.sleep", sleep)

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("reset")

    def handler(request):
        attempts.append(request)
        return httpx.Response(200, stream=BrokenStream())

    chunks = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        with pytest.raises(httpx.ReadError):
            async for content in client.stream_chat_completion(base_url="http://x/v1", model="m", messages=[]):
                chunks.append(content)
    assert chunks == ["partial"]
    assert len(attempts) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_tokenization_uses_root_endpoint_auth_and_chat_template(monkeypatch):
    attempts = []
    sleep = AsyncMock()
    monkeypatch.setattr("meno_rag.llm.client.asyncio.sleep", sleep)

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ReadError("reset", request=request)
        return httpx.Response(200, json={"count": 123, "max_model_len": 16384})

    messages = [{"role": "user", "content": "Вопрос"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http, endpoint_api_keys={"http://x/prefix": "endpoint-key"})
        result = await client.tokenize_messages(
            base_url="http://x/prefix/v1/", model="meno-lite", messages=messages,
            chat_template_kwargs={"enable_thinking": False}, timeout=9.0,
        )
    assert result == (123, 16384)
    assert len(attempts) == 2
    assert all(str(request.url) == "http://x/prefix/tokenize" for request in attempts)
    assert attempts[-1].headers["Authorization"] == "Bearer endpoint-key"
    assert attempts[-1].extensions["timeout"]["read"] == 9.0
    assert json.loads(attempts[-1].content) == {
        "model": "meno-lite", "messages": messages, "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 405])
async def test_missing_tokenization_is_cached_per_endpoint(status):
    attempts = []

    def handler(request):
        attempts.append(request)
        if request.url.host == "unsupported":
            return httpx.Response(status)
        return httpx.Response(200, json={"count": 12, "max_model_len": 60000})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        for model in ("first", "second"):
            assert await client.tokenize_messages(base_url="http://unsupported/v1/", model=model, messages=[]) is None
        assert await client.tokenize_messages(base_url="http://supported/v1", model="m", messages=[]) == (12, 60000)
    assert len(attempts) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"count": True, "max_model_len": 60000},
    {"count": 2, "max_model_len": None},
    {"count": -1, "max_model_len": 60000},
])
async def test_invalid_tokenization_values_fail_explicitly(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as http:
        client = VLLMClient(http_client=http)
        with pytest.raises(ValueError):
            await client.tokenize_messages(base_url="http://x/v1", model="m", messages=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens,logprobs,warn", [(1, True, False), (1, False, True), (20, True, True)])
async def test_only_intentional_single_token_classification_suppresses_truncation(monkeypatch, max_tokens, logprobs, warn):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "2"}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 1},
        })

    with capture_logs() as logs:
        monkeypatch.setattr(client_module, "logger", structlog.get_logger(__name__))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = VLLMClient(http_client=http)
            await client.chat_completion(
                base_url="http://x/v1", model="m", messages=[], max_tokens=max_tokens, logprobs=logprobs,
            )
    assert bool([log for log in logs if log["event"] == "generation_truncated"]) is warn
    assert any(log["event"] == "vllm_response" for log in logs)
