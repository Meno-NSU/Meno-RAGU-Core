from copy import deepcopy

import httpx
import pytest

from benchmarking import judge_budget as module
from benchmarking.judge_budget import JudgeBudgetGuard, JudgeContextError


def payload(**changes):
    return {
        "model": "qwen3.8-27b", "messages": [{"role": "system", "content": "JSON schema"},
                                             {"role": "user", "content": "Все доказательства НГУ"}],
        "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False}, **changes,
    }


@pytest.fixture
def tokenizer(monkeypatch):
    state = {"requests": [], "count": 1000, "max_model_len": 15000, "status": 200, "body": None}
    original = httpx.AsyncClient

    def handle(request):
        import json

        state["requests"].append((str(request.url), json.loads(request.content), dict(request.headers)))
        body = state["body"] if state["body"] is not None else {
            "count": state["count"], "max_model_len": state["max_model_len"],
        }
        return httpx.Response(state["status"], json=body)

    def create(**kwargs):
        assert kwargs["trust_env"] is False
        return original(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", create)
    return state


@pytest.mark.asyncio
async def test_checks_exact_messages_template_and_prefix_without_mutating_input(tokenizer):
    data = payload()
    before = deepcopy(data)
    async with JudgeBudgetGuard("http://judge/prefix/v1/", "secret", 15000, 30) as guard:
        info = await guard.check(data)
    assert info == {
        "prompt_tokens": 1000, "reserved": 4096, "margin_tokens": 32,
        "context_limit": 15000, "tokenizer_context_limit": 15000, "cached": False,
    }
    assert data == before
    url, request, headers = tokenizer["requests"][0]
    assert url == "http://judge/prefix/tokenize"
    assert request == {"model": data["model"], "messages": data["messages"],
                       "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True}
    assert headers["authorization"] == "Bearer secret"
    assert all("/chat/completions" not in entry[0] for entry in tokenizer["requests"])


@pytest.mark.asyncio
@pytest.mark.parametrize("configured,served,expected", [(15000, 60000, 15000), (60000, 15000, 15000),
                                                     (15000, None, 15000)])
async def test_uses_smallest_known_window(tokenizer, configured, served, expected):
    tokenizer["max_model_len"] = served
    async with JudgeBudgetGuard("http://judge/v1", "", configured, 30) as guard:
        assert (await guard.check(payload()))["context_limit"] == expected


@pytest.mark.asyncio
async def test_oversize_request_fails_without_truncation_or_generation(tokenizer):
    tokenizer["count"] = 11000
    data = payload()
    before = deepcopy(data)
    async with JudgeBudgetGuard("http://judge/v1", "", 15000, 30) as guard:
        with pytest.raises(JudgeContextError) as error:
            await guard.check(data)
    assert error.value.details == {
        "code": "judge_context_length_exceeded", "prompt_tokens": 11000, "reserved": 4096,
        "margin_tokens": 32, "context_limit": 15000, "tokenizer_context_limit": 15000, "cached": False,
    }
    assert data == before
    assert len(tokenizer["requests"]) == 1


@pytest.mark.asyncio
async def test_exact_boundary_and_completion_token_precedence(tokenizer):
    tokenizer["count"] = 12920
    async with JudgeBudgetGuard("http://judge/v1", "", 15000, 30) as guard:
        result = await guard.check(payload(max_completion_tokens=2048))
        assert result["reserved"] == 2048
        with pytest.raises(JudgeContextError):
            await guard.check(payload(max_completion_tokens=2049))
    assert len(tokenizer["requests"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"count": True}, {"count": -1}, {"count": 1.0},
                                 {"count": 2, "max_model_len": False}, {"count": 2, "max_model_len": 0},
                                 {"count": 2, "max_model_len": 15000.0}, [1]])
async def test_invalid_tokenizer_response_fails_closed(tokenizer, body):
    tokenizer["body"] = body
    async with JudgeBudgetGuard("http://judge/v1", "", 15000, 30) as guard:
        with pytest.raises(JudgeContextError) as error:
            await guard.check(payload())
    assert error.value.details["code"] == "judge_tokenizer_invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.0, "4096"])
async def test_missing_or_invalid_reserve_never_contacts_server(tokenizer, value):
    async with JudgeBudgetGuard("http://judge/v1", "", 15000, 30) as guard:
        with pytest.raises(JudgeContextError) as error:
            await guard.check(payload(max_tokens=value))
    assert error.value.details["code"] == "judge_invalid_request_budget"
    assert tokenizer["requests"] == []


@pytest.mark.asyncio
async def test_tokenizer_http_error_is_explicit(tokenizer):
    tokenizer["status"] = 404
    async with JudgeBudgetGuard("http://judge/v1", "", 15000, 30) as guard:
        with pytest.raises(JudgeContextError) as error:
            await guard.check(payload())
    assert error.value.details["code"] == "judge_tokenizer_unavailable"
    assert error.value.details["status_code"] == 404


@pytest.mark.asyncio
async def test_count_cache_reuses_prompt_but_checks_changed_reserve_and_template(tokenizer):
    async with JudgeBudgetGuard("http://judge/v1", "", 15000, 30) as guard:
        first = await guard.check(payload())
        second = await guard.check(payload(max_tokens=3000))
        assert not first["cached"] and second["cached"]
        assert second["reserved"] == 3000
        await guard.check(payload(chat_template_kwargs={"enable_thinking": True}))
        await guard.check(payload(model="another-model"))
        await guard.check(payload(messages=[{"role": "user", "content": "Другой вопрос"}]))
    assert len(tokenizer["requests"]) == 4


@pytest.mark.asyncio
async def test_cache_is_bounded_lru(tokenizer):
    async with JudgeBudgetGuard("http://judge/v1", "", 15000, 30) as guard:
        guard.cache_size = 2
        await guard.check(payload(model="one"))
        await guard.check(payload(model="two"))
        assert (await guard.check(payload(model="one")))["cached"]
        await guard.check(payload(model="three"))
        assert not (await guard.check(payload(model="two")))["cached"]
        assert len(guard._cache) == 2
    assert len(tokenizer["requests"]) == 4


@pytest.mark.parametrize("value", [0, -1, True, 15000.0])
def test_invalid_configured_window_is_rejected(value):
    with pytest.raises(ValueError):
        JudgeBudgetGuard("http://judge/v1", "", value, 30)
