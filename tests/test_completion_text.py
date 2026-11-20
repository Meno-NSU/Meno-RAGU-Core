"""Missing model content must not become the visible answer 'None' or '[]'."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from meno_rag.config import Settings
from meno_rag.llm.client import VLLMClient
from meno_rag.llm.openrouter_client import OpenRouterClient
from meno_rag.schemas import PipelineOutcome
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StandRagPipeline


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [VLLMClient, OpenRouterClient])
@pytest.mark.parametrize("content", [None, " \n ", [], {}])
async def test_text_clients_reject_empty_or_nontext_content(client_class, content):
    client = object.__new__(client_class)
    client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": content}}]})
    with pytest.raises(ValueError, match="no non-empty text content"):
        await client.chat_completion_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [VLLMClient, OpenRouterClient])
@pytest.mark.parametrize("content", ["\n0\n", "  Два  пробела внутри.\n", "В источниках нет нужных сведений."])
async def test_valid_text_preserves_existing_trim_and_internal_spaces(client_class, content):
    client = object.__new__(client_class)
    client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": content}}]})
    assert await client.chat_completion_text() == content.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "\n", [], " Ответ "])
async def test_traced_generation_validates_the_actual_message(content):
    response = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    router = SimpleNamespace(chat_completion=AsyncMock(return_value=response))
    pipeline = StandRagPipeline(
        settings=Settings(_env_file=None), resources=None, llm_router=router,
        rewrite_semaphore=asyncio.Semaphore(1), rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1), embed_semaphore=asyncio.Semaphore(1),
    )
    outcome = PipelineOutcome(
        question="q", prepared_dialogue_history="", search_queries=[], context="", sources=[],
        qa_messages=[{"role": "user", "content": "q"}], stage_durations_ms={}, stage_details={}, trace={},
    )
    runtime = PipelineRuntime.uniform(ModelRuntime("test", "http://closed.invalid/v1"))
    if isinstance(content, str) and content.strip():
        assert await pipeline.generate_text(outcome=outcome, runtime=runtime) == content.strip()
        assert outcome.trace["generation"]["message"]["content"] == content
    else:
        with pytest.raises(ValueError, match="no non-empty text content"):
            await pipeline.generate_text(outcome=outcome, runtime=runtime)
        assert outcome.trace["generation"]["message"]["content"] == content


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [VLLMClient, OpenRouterClient])
async def test_text_clients_reject_known_truncation(client_class):
    client = object.__new__(client_class)
    client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]})
    with pytest.raises(RuntimeError, match="finish_reason=length"):
        await client.chat_completion_text()


@pytest.mark.asyncio
async def test_traced_generation_rejects_truncated_response():
    response = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    pipeline = StandRagPipeline(
        settings=Settings(_env_file=None), resources=None,
        llm_router=SimpleNamespace(chat_completion=AsyncMock(return_value=response)),
        rewrite_semaphore=asyncio.Semaphore(1), rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1), embed_semaphore=asyncio.Semaphore(1),
    )
    outcome = PipelineOutcome(question="q", prepared_dialogue_history="", search_queries=[], context="", sources=[], qa_messages=[{"role":"user","content":"q"}], stage_durations_ms={}, stage_details={}, trace={})
    runtime = PipelineRuntime.uniform(ModelRuntime("test", "http://closed.invalid/v1"))
    with pytest.raises(RuntimeError, match="finish_reason=length"):
        await pipeline.generate_text(outcome=outcome, runtime=runtime)
    assert outcome.trace["generation"]["finish_reason"] == "length"
    assert outcome.trace["generation"]["message"]["content"] == "partial"
