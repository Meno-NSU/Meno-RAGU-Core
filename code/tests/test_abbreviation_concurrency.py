import asyncio
import json

import httpx
import pytest
from diskcache import Cache
from openai import AsyncOpenAI
from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

from graph_utils.graph_utils import ABBREVIATION_REWRITE_SYSTEM_PROMPT, AbbreviationExpandingExtractor
import graph_utils.graph_utils as graph_utils


ABBREVIATIONS = {'нгу': {'abbreviation': 'НГУ', 'explanation': ['Новосибирский государственный университет']}}
EXPANSION = 'Новосибирский государственный университет'


def extractor(llm, concurrency):
    return AbbreviationExpandingExtractor(
        llm=llm, embedder=None, icl_config=None, ontology=None,
        abbreviations=ABBREVIATIONS, rewrite_concurrency=concurrency,
    )


def chunks(count):
    return [Chunk(content=f'НГУ готовит студентов по программе номер {i}.', chunk_order_idx=i,
                  doc_id=f'doc_{i}', num_tokens=10) for i in range(count)]


@pytest.mark.parametrize('concurrency', [1, 3, 8])
def test_bounded_parallel_rewrites_preserve_order_ids_and_validation(concurrency):
    async def run():
        class Model:
            active = 0
            peak = 0

            async def chat_completion(self, conversation, **kwargs):
                self.active += 1
                self.peak = max(self.peak, self.active)
                text = conversation[-1]['content'].split('ТЕКСТ ФРАГМЕНТА:\n', 1)[1]
                try:
                    await asyncio.sleep(0.001 * (9 - int(text.split()[-1].rstrip('.'))))
                    return text.replace('НГУ', EXPANSION)
                finally:
                    self.active -= 1

        model = Model()
        original = chunks(9)
        builder = extractor(model, concurrency)
        messages = []
        sink = logger.add(lambda message: messages.append(str(message)), format='{message}', level='INFO')
        try:
            result = await builder._expand_abbreviations(original)
        finally:
            logger.remove(sink)
        assert model.peak == concurrency
        assert model.active == 0
        assert builder.num_rewritten_chunks == len(original)
        assert builder.num_rejected_rewrites == builder.num_failed_rewrites == 0
        assert any('9/9 (100.0%)' in message for message in messages)
        for old, new in zip(original, result):
            assert new.content == old.content.replace('НГУ', EXPANSION)
            assert (new.id, new.doc_id, new.chunk_order_idx, new.num_tokens) == (
                old.id, old.doc_id, old.chunk_order_idx, old.num_tokens
            )
    asyncio.run(run())


def test_cancelled_rewrite_cleans_up_workers_and_progress_monitor():
    async def run():
        started = asyncio.Event()
        blocked = asyncio.Event()

        class Model:
            active = 0

            async def chat_completion(self, **kwargs):
                self.active += 1
                if self.active == 3:
                    started.set()
                try:
                    await blocked.wait()
                finally:
                    self.active -= 1

        model = Model()
        task = asyncio.create_task(extractor(model, 3)._expand_abbreviations(chunks(100)))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.active == 0
        assert len(asyncio.all_tasks()) == 1
    asyncio.run(run())


def test_parallel_restart_reuses_legacy_disk_cache_without_new_requests(tmp_path):
    async def run():
        requested = []

        async def respond(request):
            payload = json.loads(request.content)
            text = payload['messages'][-1]['content'].split('ТЕКСТ ФРАГМЕНТА:\n', 1)[1]
            requested.append(text)
            await asyncio.sleep(0)
            return httpx.Response(200, json={
                'id': 'test', 'object': 'chat.completion', 'created': 1, 'model': 'graph-model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                    'role': 'assistant', 'content': text.replace('НГУ', EXPANSION),
                }}],
            })

        original = chunks(6)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            api = AsyncOpenAI(base_url='http://llm.test/v1', api_key='EMPTY', http_client=http)
            with Cache(str(tmp_path / 'llm-cache')) as cache:
                client = CachedAsyncOpenAI(client=api, cache=cache, max_completion_tokens=6144)
                for chunk in original[:4]:
                    await client.chat_completion(
                        model_name='graph-model', temperature=0.0, output_schema=str,
                        conversation=[
                            {'role': 'system', 'content': ABBREVIATION_REWRITE_SYSTEM_PROMPT},
                            {'role': 'user', 'content': 'СЛОВАРЬ АББРЕВИАТУР:\nНГУ — ' + EXPANSION
                             + '\n\nТЕКСТ ФРАГМЕНТА:\n' + chunk.content},
                        ],
                    )
                assert len(cache) == 4
                model = LLMOpenAI(client=client, model_name='graph-model', temperature=0.0)
                result = await extractor(model, 3)._expand_abbreviations(original)
                assert requested == [chunk.content for chunk in original]
                assert len(cache) == 6
                replay = await extractor(model, 3)._expand_abbreviations(original)
                assert len(requested) == 6
                assert [(c.id, c.content) for c in result] == [(c.id, c.content) for c in replay]
    asyncio.run(run())


def test_request_failure_keeps_original_chunk_and_reports_failure():
    class Model:
        async def chat_completion(self, **kwargs):
            raise RuntimeError('endpoint unavailable')
    original = chunks(3)
    builder = extractor(Model(), 2)
    assert asyncio.run(builder._expand_abbreviations(original)) == original
    assert builder.num_failed_rewrites == 3


def test_progress_is_reported_while_requests_are_still_running(monkeypatch):
    async def run():
        reported = asyncio.Event()

        class Model:
            async def chat_completion(self, conversation, **kwargs):
                await reported.wait()
                return conversation[-1]['content'].split('ТЕКСТ ФРАГМЕНТА:\n', 1)[1].replace('НГУ', EXPANSION)

        def capture(message):
            if '0/2 (0.0%)' in str(message):
                reported.set()

        monkeypatch.setattr(graph_utils, 'ABBREVIATION_PROGRESS_INTERVAL', 0.001)
        sink = logger.add(capture, format='{message}', level='INFO')
        try:
            result = await asyncio.wait_for(extractor(Model(), 2)._expand_abbreviations(chunks(2)), timeout=1)
        finally:
            logger.remove(sink)
        assert reported.is_set()
        assert len(result) == 2
    asyncio.run(run())


@pytest.mark.parametrize('concurrency', [0, -1])
def test_invalid_concurrency_is_rejected(concurrency):
    with pytest.raises(ValueError, match='positive integer'):
        extractor(None, concurrency)
