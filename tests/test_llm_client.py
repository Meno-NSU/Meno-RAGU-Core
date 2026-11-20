"""Unit tests for VLLMClient. Use a fake httpx transport so no network calls happen."""

import asyncio
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
@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 408, 409, 422, 429, 500])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [502, 503, 504])
async def test_gateway_errors_retry_without_changing_request(monkeypatch, status):
    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)

    def handler(request):
        requests.append(request)
        if len(requests) < 3:
            return httpx.Response(status, json={"error": "temporarily unavailable"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http, endpoint_api_keys={"http://x": "endpoint-key"})
        result = await client.chat_completion(
            base_url="http://x/v1", model="m", messages=[{"role": "user", "content": "Вопрос"}],
            seed=42, timeout=12,
        )
    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(requests) == 3
    assert len({request.content for request in requests}) == 1
    assert all(request.headers["Authorization"] == "Bearer endpoint-key" for request in requests)
    assert all(request.extensions["timeout"]["read"] == 12 for request in requests)
    assert sleep.await_args_list == [call(0.25), call(0.5)]


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [("transport", 503, 200), (503, "transport", 200), (502, 503, 504), (503, "transport", "transport")])
async def test_transport_and_gateway_failures_share_three_attempt_cap(monkeypatch, sequence):
    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)

    def handler(request):
        step = sequence[len(requests)]
        requests.append(request)
        if step == "transport":
            raise httpx.ReadError("reset", request=request)
        return httpx.Response(step, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        if sequence[-1] == 200:
            assert (await client.chat_completion(base_url="http://x/v1", model="m", messages=[]))["choices"]
        else:
            error = httpx.ReadError if sequence[-1] == "transport" else httpx.HTTPStatusError
            with pytest.raises(error):
                await client.chat_completion(base_url="http://x/v1", model="m", messages=[])
    assert len(requests) == 3
    assert sleep.await_args_list == [call(0.25), call(0.5)]


@pytest.mark.asyncio
async def test_gateway_response_is_closed_before_backoff_and_cancel_stops_retry(monkeypatch):
    events = []

    class UnreadBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"unavailable"

        async def aclose(self):
            events.append("closed")

    request = httpx.Request("POST", "http://x/v1/chat/completions")
    response = httpx.Response(503, request=request, stream=UnreadBody())
    http = AsyncMock(spec=httpx.AsyncClient)
    http.post.return_value = response

    async def cancel_during_backoff(_delay):
        assert response.is_closed
        events.append("backoff")
        raise asyncio.CancelledError

    monkeypatch.setattr(client_module.asyncio, "sleep", cancel_during_backoff)
    with pytest.raises(asyncio.CancelledError):
        await VLLMClient(http_client=http).chat_completion(base_url="http://x/v1", model="m", messages=[])
    assert http.post.await_count == 1
    assert events == ["closed", "backoff"]


@pytest.mark.asyncio
async def test_tokenize_gateway_retry_does_not_change_404_cache(monkeypatch):
    requests = []
    monkeypatch.setattr(client_module.asyncio, "sleep", AsyncMock())

    def handler(request):
        requests.append(request)
        return httpx.Response(503 if len(requests) == 1 else 404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        for _ in range(2):
            assert await client.tokenize_messages(base_url="http://x/v1", model="m", messages=[]) is None
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_stream_gateway_error_is_never_replayed(monkeypatch):
    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)

    def handler(request):
        requests.append(request)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in client.stream_chat_completion(base_url="http://x/v1", model="m", messages=[]):
                pass
    assert len(requests) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_hundred_healthy_responses_keep_request_bytes_and_results(monkeypatch):
    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)

    def handler(request):
        requests.append(request)
        i = len(requests) - 1
        return httpx.Response(200, content=json.dumps({"choices": [{"message": {"content": f" Ответ {i} "}, "finish_reason": "stop"}]}).encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        for i in range(100):
            messages = [{"role": "user", "content": f"Вопрос {i}"}]
            result = await client.chat_completion(base_url="http://x/v1", model="m", messages=messages, seed=42)
            expected = {"model": "m", "messages": messages, "stream": False, "seed": 42}
            assert requests[-1].content == httpx.Request("POST", "http://x/v1/chat/completions", json=expected).content
            assert result == {"choices": [{"message": {"content": f" Ответ {i} "}, "finish_reason": "stop"}]}
    assert len(requests) == 100
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("event,expected", [
    ('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n', []),
    ('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', ["partial"]),
])
async def test_clean_eof_without_finish_reason_is_error_and_closes_response(event, expected):
    responses = []

    def handler(request):
        response = httpx.Response(200, content=event.encode())
        responses.append(response)
        return response

    received = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)
        with pytest.raises(RuntimeError, match="finish_reason"):
            async for token in client.stream_chat_completion(base_url="http://x/v1", model="m", messages=[]):
                received.append(token)
    assert received == expected
    assert len(responses) == 1
    assert responses[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["stop", "length"])
async def test_stream_with_explicit_finish_keeps_text_and_length_warning(monkeypatch, finish):
    content = ('data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
               'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
               f'data: {{"choices":[{{"delta":{{}},"finish_reason":"{finish}"}}]}}\n\n'
               'data: [DONE]\n\n')
    with capture_logs() as logs:
        monkeypatch.setattr(client_module, "logger", structlog.get_logger(__name__))
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content.encode()))) as http:
            received = []
            async def consume():
                async for part in VLLMClient(http_client=http).stream_chat_completion(base_url="http://x/v1", model="m", messages=[]):
                    received.append(part)
            if finish == "length":
                with pytest.raises(RuntimeError, match="finish_reason=length"):
                    await consume()
            else:
                await consume()
    assert received == ["hello ", "world"]
    assert bool([entry for entry in logs if entry["event"] == "generation_truncated"]) == (finish == "length")


@pytest.mark.asyncio
async def test_sse_error_is_not_replayed_and_closes_response():
    responses = []

    def handler(request):
        response = httpx.Response(200, content=b'data: {"error":{"message":"provider failed"}}\n\n')
        responses.append(response)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(RuntimeError, match="provider failed"):
            async for _ in VLLMClient(http_client=http).stream_chat_completion(base_url="http://x/v1", model="m", messages=[]):
                pass
    assert len(responses) == 1
    assert responses[0].is_closed


@pytest.mark.asyncio
async def test_stream_cancellation_after_text_closes_response_without_retry():
    entered = asyncio.Event()
    closed = []
    requests = []

    class HangingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=HangingBody())

    received = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMClient(http_client=http)

        async def consume():
            async for part in client.stream_chat_completion(base_url="http://x/v1", model="m", messages=[]):
                received.append(part)

        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert received == ["partial"]
    assert len(requests) == 1
    assert closed == [True]
