import asyncio
import copy

import httpx
import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.ragu_pipeline import GraphSelection
from tests.ragu_fakes import LLM, pipeline


class BudgetLLM(LLM):
    def __init__(self, context=8000):
        super().__init__()
        self.context = context

    async def tokenize_messages(self, *, messages, **kwargs):
        return sum(len(message['content']) for message in messages) // 2 + 4, self.context

    async def chat_completion(self, **kwargs):
        count, context = await self.tokenize_messages(messages=kwargs['messages'])
        assert count + kwargs['max_tokens'] + 32 <= context
        return await super().chat_completion(**kwargs)


@pytest.mark.asyncio
async def test_large_rerank_document_is_scored_in_full_with_fitting_prompts():
    p = pipeline()
    llm = BudgetLLM()
    p.llm_router = llm
    text = 'Сведения о факультете. ' * 5000
    score = await p._score_document_with_llm('Факультеты?', '', text, ModelRuntime('meno-lite', 'http://model/v1'), 'entity')
    assert score > 0
    assert len(llm.prompts) > 1
    parts = [messages[1]['content'].split('DOCUMENT 1\n==========\n\n', 1)[1]
             .split('\n\n==========\nCURRENT QUESTION', 1)[0] for messages in llm.prompts]
    assert ''.join(parts).replace(' ', '') == text.replace(' ', '')


@pytest.mark.asyncio
@pytest.mark.parametrize('context', [8000, 16384, 60000])
async def test_qa_reserves_requested_output_and_keeps_question_and_matching_sources(context):
    p = pipeline()
    p.llm_router = BudgetLLM(context)
    original = p.resources.graph.search

    async def search(query):
        result = await original(query)
        result['entities'][0]['description'] = 'Длинный факт. ' * 1500
        return result

    p.resources.graph.search = search
    outcome = await p.prepare(
        messages=[ChatMessage(role='user', content='Какие факультеты есть в НГУ?')],
        runtime=PipelineRuntime.uniform(ModelRuntime('meno-lite', 'http://model/v1')),
        generation_max_tokens=4096,
    )
    count, context = await p.llm_router.tokenize_messages(messages=outcome.qa_messages)
    assert count + 4096 + 32 <= context
    assert 'Какие факультеты есть в НГУ?' in outcome.qa_messages[1]['content']
    assert outcome.context in outcome.qa_messages[1]['content']
    assert {source['source_url'] for source in outcome.sources} <= {'https://example.org/a', 'https://example.org/b'}


@pytest.mark.asyncio
async def test_oversized_question_fails_before_sending_rerank_request():
    p = pipeline()
    p.llm_router = BudgetLLM()
    with pytest.raises(ValueError, match='инструкции реранкера'):
        await p._score_document_with_llm('Вопрос' * 10000, '', 'текст', ModelRuntime('meno-lite', 'http://model/v1'), 0)
    assert p.llm_router.prompts == []


def test_graph_context_has_bounded_references_without_mutating_graph_provenance():
    p = pipeline()
    sources = []
    for index in range(20):
        doc = dict(p.resources.documents[0], doc_title=f'Название {index}', url=f'https://example.org/{index}')
        p.resources.documents.append(doc)
        p.resources.chunk_mapping[str(index + 1)] = {'doc_index': index + 1, 'local_chunk_index': 0}
        sources.append(index + 1)
    element = {'kind': 'entity', 'name': 'НГУ', 'type': 'ORG', 'description': 'Факт', 'sources': sources}
    before = copy.deepcopy(element)
    # all 20 chunks were scored for the question, later corpus positions higher: the element names the 5 best
    scored = {chunk_id: chunk_id / 100 for chunk_id in sources}
    selection = GraphSelection([], [element], scored=20, candidate_scores=scored, graph_trace={})
    context, shown = p._assemble_context(selection)
    assert 'Название 19' in context and 'Название 15' in context
    assert 'Название 14' not in context and 'Название 0;' not in context
    assert shown == []  # no document is in the context, so none is shown as a source
    assert element == before


@pytest.mark.asyncio
async def test_failed_rerank_cancels_other_scores_without_json_retry():
    p = pipeline()
    state = {'started': 0, 'cancelled': 0}

    class FailedLLM(LLM):
        async def chat_completion(self, **kwargs):
            state['started'] += 1
            if state['started'] == 1:
                await asyncio.sleep(0)
                raise httpx.ReadError('connection interrupted')
            try:
                await asyncio.sleep(30)
            finally:
                state['cancelled'] += 1

    p.llm_router = FailedLLM()
    batches = p._fuse(await p._retrieve(['ФИТ']))
    with pytest.raises(httpx.ReadError):
        await p._rerank(batches, 'Вопрос', '', ModelRuntime('meno-lite', 'http://model/v1'))
    assert 2 <= state['started'] <= 3
    assert state['cancelled'] == state['started'] - 1
