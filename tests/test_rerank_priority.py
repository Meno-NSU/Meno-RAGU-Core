"""RERANK_PRIORITY (audit of 19.11, П3, the client side): vLLM's request priority of the LLM-rerank calls.

vLLM rejects a non-zero ``priority`` unless it runs with ``--scheduling-policy priority``, so at 0 (the default) the
field must not be sent at all: the requests stay byte-for-byte what they were."""

import asyncio
import json

import httpx
import pytest

from meno_rag.config import Settings
from meno_rag.llm.client import VLLMClient
from meno_rag.llm.router import LLMRouter
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from tests import ragu_fakes

QUESTION = [ChatMessage(role="user", content="Какие факультеты есть в НГУ?")]


class RecordingLLM(ragu_fakes.LLM):
    """Records every call; guided choice fails when ``fail_guided`` so the JSON fallback runs."""

    def __init__(self, fail_guided=False):
        super().__init__()
        self.calls, self.fail_guided = [], fail_guided

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_guided and "logprobs" in kwargs:
            raise ValueError("guided choice unsupported")
        if "logprobs" not in kwargs:
            return {"choices": [{"message": {"content": '{"label": "2"}'}}]}
        return await super().chat_completion(**kwargs)

    async def chat_completion_text(self, **kwargs):
        self.calls.append(kwargs)
        return await super().chat_completion_text(**kwargs)


def run(pipeline, provider="vllm"):
    runtime = PipelineRuntime.uniform(ModelRuntime("fake-model", "http://fake/v1", provider=provider))
    return asyncio.run(pipeline.prepare(messages=QUESTION, runtime=runtime))


def rerank_calls(llm):
    return [call for call in llm.calls if call.get("stage") == "rerank"]


@pytest.mark.parametrize("fail_guided", [False, True])
def test_by_default_no_call_carries_a_priority(fail_guided):
    pipeline = ragu_fakes.pipeline()
    pipeline.llm_router = llm = RecordingLLM(fail_guided)
    run(pipeline)
    assert rerank_calls(llm)
    assert all("priority" not in (call.get("extra_body") or {}) and "priority" not in call for call in llm.calls)


@pytest.mark.parametrize("fail_guided", [False, True])
def test_a_set_priority_goes_with_every_rerank_call_and_only_with_them(fail_guided):
    pipeline = ragu_fakes.pipeline(rerank_priority=3)
    pipeline.llm_router = llm = RecordingLLM(fail_guided)
    run(pipeline)
    reranks = rerank_calls(llm)
    assert reranks and all(call["extra_body"]["priority"] == 3 for call in reranks)
    if fail_guided:  # the JSON fallback is the same rerank work: it carries the priority too
        assert any("logprobs" not in call for call in reranks)
    others = [call for call in llm.calls if call.get("stage") != "rerank"]
    assert others and all("priority" not in (call.get("extra_body") or {}) for call in others)
    # the rest of the rerank request is unchanged
    assert all(call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False} for call in reranks)


def test_a_provider_other_than_vllm_never_gets_the_vllm_field():
    pipeline = ragu_fakes.pipeline(rerank_priority=3)
    pipeline.llm_router = llm = RecordingLLM()
    run(pipeline, provider="openrouter")
    assert all("priority" not in (call.get("extra_body") or {}) for call in rerank_calls(llm))


def test_the_setting_is_read_from_the_environment_name():
    assert Settings(_env_file=None).rerank_priority == 0
    assert Settings(_env_file=None, RERANK_PRIORITY="2").rerank_priority == 2


@pytest.mark.parametrize("priority, expected", [(0, None), (1, 1)])
def test_the_json_sent_to_vllm_has_the_priority_field_only_when_it_is_set(priority, expected):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/tokenize"):  # the prompt-length check of the stand
            return httpx.Response(200, json={"count": 100, "max_model_len": 60000})
        bodies.append(body)
        if body.get("max_tokens") == 1:
            top = [{"token": "2", "logprob": 0.0}]
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "2"},
                            "logprobs": {"content": [{"token": "2", "logprob": 0.0, "top_logprobs": top}]},
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "поисковый запрос"}}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            pipeline = ragu_fakes.pipeline(rerank_priority=priority)
            pipeline.llm_router = LLMRouter(vllm=VLLMClient(http_client=http), openrouter=None)
            runtime = PipelineRuntime.uniform(ModelRuntime("fake-model", "http://fake/v1"))
            await pipeline.prepare(messages=QUESTION, runtime=runtime)

    asyncio.run(go())
    reranks = [body for body in bodies if body.get("max_tokens") == 1]
    assert reranks and all(body.get("priority") == expected for body in reranks)
    if expected is None:
        assert all("priority" not in body for body in bodies)
