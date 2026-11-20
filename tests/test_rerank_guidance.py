"""Explicit vLLM guidance selection, with unchanged legacy defaults and JSON fallback."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from meno_rag.config import Settings
from meno_rag.llm.client import VLLMClient
from meno_rag.llm.router import LLMRouter
from meno_rag.stand.pipeline import ModelRuntime
from tests import ragu_fakes


def test_guidance_is_explicit_and_legacy_by_default():
    assert Settings(_env_file=None).rerank_guidance == "legacy"
    assert Settings(_env_file=None, RERANK_GUIDANCE="structured").rerank_guidance == "structured"


@pytest.mark.parametrize("value", ["none", "auto", "STRUCTURED", ""])
def test_unknown_guidance_mode_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, RERANK_GUIDANCE=value)


def wire_requests(mode, failure=None):
    bodies = []

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 100, "max_model_len": 60000})
        body = json.loads(request.content)
        bodies.append(body)
        if body["max_tokens"] == 1:
            if failure == "http400":
                return httpx.Response(400, json={"error": {"message": "Unsupported choice protocol"}})
            token = "<|channel>" if failure == "missing_labels" else "2"
            return httpx.Response(200, json={"choices": [{"message": {"content": token}, "logprobs": {"content": [
                {"token": token, "top_logprobs": [{"token": token, "logprob": 0.0}]}]}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"label": "2"}'}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            pipeline = ragu_fakes.pipeline(rerank_guidance=mode, rerank_priority=3)
            pipeline.llm_router = LLMRouter(vllm=VLLMClient(http_client=http), openrouter=None)
            return await pipeline._score_document_with_llm(
                "Вопрос", "", "Документ", ModelRuntime("fake-model", "http://fake/v1"), "test-chunk",
            )

    assert asyncio.run(run()) > .99
    return bodies


def test_wire_requests_differ_only_in_the_explicit_vllm_constraint():
    legacy, = wire_requests("legacy")
    structured, = wire_requests("structured")
    assert legacy.pop("guided_choice") == ["0", "1", "2"]
    assert structured.pop("structured_outputs") == {"choice": ["0", "1", "2"]}
    assert legacy == structured
    assert legacy["priority"] == 3
    assert legacy["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("mode", ["legacy", "structured"])
@pytest.mark.parametrize("failure", ["http400", "missing_labels"])
def test_json_fallback_keeps_priority_and_does_not_guess_a_protocol(mode, failure):
    first, fallback = wire_requests(mode, failure)
    assert first["max_tokens"] == 1 and first["priority"] == 3
    assert ("structured_outputs" in first) == (mode == "structured")
    assert ("guided_choice" in first) == (mode == "legacy")
    assert fallback["max_tokens"] == 20 and fallback["priority"] == 3
    assert "response_format" in fallback
    assert "guided_choice" not in fallback and "structured_outputs" not in fallback
    assert fallback["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("mode", ["legacy", "structured"])
def test_other_providers_keep_their_existing_request_format(mode):
    class RecordingLLM(ragu_fakes.LLM):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            return await super().chat_completion(**kwargs)

    pipeline = ragu_fakes.pipeline(rerank_guidance=mode, rerank_priority=3)
    pipeline.llm_router = llm = RecordingLLM()
    asyncio.run(pipeline._score_document_with_llm(
        "Вопрос", "", "Документ", ModelRuntime("fake-model", "http://fake/v1", provider="openrouter"), "test-chunk",
    ))
    assert len(llm.calls) == 1
    assert llm.calls[0]["extra_body"] == {
        "guided_choice": ["0", "1", "2"], "chat_template_kwargs": {"enable_thinking": False},
    }
