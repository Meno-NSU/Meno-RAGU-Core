import asyncio
import dataclasses
import json

import httpx
import pytest
from diskcache import Cache
from graph_utils.bounded_summaries import (
    BudgetedCommunitySummarizer,
    BudgetedEntitySummarizer,
    BudgetedRelationSummarizer,
    SummaryBudget,
    SummaryBudgetError,
)
from graph_utils.checkpoints import (
    CheckpointedExtractor,
    CheckpointedGraphBuilder,
    GraphCheckpoints,
    previous_summary_configuration,
)
from openai import AsyncOpenAI
from prepare_graph_index import ContextBudgetLLM
from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import (
    ClusterSummarizationModel,
    CommunityReportModel,
    EntityDescriptionModel,
    RelationDescriptionModel,
)
from ragu.graph.artifacts_summarizer import EntitySummarizer, RelationSummarizer
from ragu.graph.community_summarizer import CommunitySummarizer
from ragu.graph.graph_builder_pipeline import BuilderArguments, InMemoryGraphBuilder
from ragu.graph.types import Community, Entity, Relation
from ragu.models.openai import CachedAsyncOpenAI


class Tokenizer:
    calls = 0

    def apply_chat_template(self, conversation, **kwargs):
        self.calls += 1
        return range(sum(len(message['content']) for message in conversation) + 4)


def response(schema, text='Краткое описание'):
    if schema is EntityDescriptionModel:
        return schema(entity_name='НГУ', description=text)
    if schema is RelationDescriptionModel:
        return schema(subject_name='Иванов', object_name='НГУ', description=text)
    if schema is ClusterSummarizationModel:
        return schema(content=text)
    assert schema is CommunityReportModel
    return schema(title='Сообщество', summary=text, rating=1, rating_explanation='Описание', findings=[])


class Client:
    def __init__(self, text='Краткое описание'):
        self.requests = []
        self.text = text

    async def chat_completion(self, model_name, conversation, output_schema, **kwargs):
        self.requests.append(conversation)
        text = self.text(conversation) if callable(self.text) else self.text
        return response(output_schema, text)


def model(client=None, context=1000, completion=100):
    return ContextBudgetLLM(client=client or Client(), model_name='graph-model', tokenizer=Tokenizer(),
                            context_tokens=context, completion_tokens=completion, temperature=0.0)


def artifacts(text):
    entities = [Entity('НГУ', 'ORGANIZATION', text, ['chunk-a']),
                Entity('НГУ', 'ORGANIZATION', text + '.', ['chunk-b'])]
    relations = [Relation('person', 'nsu', 'Иванов', 'НГУ', 'WORKPLACE', text, source_chunk_id=['chunk-a']),
                 Relation('person', 'nsu', 'Иванов', 'НГУ', 'WORKPLACE', text + '.', source_chunk_id=['chunk-b'])]
    return entities, relations


@pytest.mark.parametrize('kind', ['entity', 'relation', 'community'])
def test_large_descriptions_are_reduced_within_real_context_guard_without_changing_source_references(kind):
    async def run():
        client = Client('Сжатое описание')
        llm = model(client)
        entities, relations = artifacts('x' * 5000)
        before = [dataclasses.asdict(item) for item in entities + relations]
        if kind == 'entity':
            result = await BudgetedEntitySummarizer(llm, summarize_only_if_more_than=0).run(entities)
            assert len(result) == 1
            assert set(result[0].source_chunk_id) == {'chunk-a', 'chunk-b'}
            assert result[0].entity_name == 'НГУ'
        elif kind == 'relation':
            result = await BudgetedRelationSummarizer(llm, summarize_only_if_more_than=0).run(relations)
            assert len(result) == 1
            assert set(result[0].source_chunk_id) == {'chunk-a', 'chunk-b'}
            assert (result[0].subject_id, result[0].object_id) == ('person', 'nsu')
        else:
            community = Community(0, 7, entities, relations)
            result = await BudgetedCommunitySummarizer(llm).summarize([community])
            assert len(result) == 1 and result[0].id == community.id
        assert len(client.requests) > 1
        assert all(llm.count_prompt_tokens(request) + llm.completion_tokens <= llm.context_tokens
                   for request in client.requests)
        assert llm.overflow_count == 0
        assert [dataclasses.asdict(item) for item in entities + relations] == before
    asyncio.run(run())


def test_625k_input_is_reduced_over_multiple_levels_without_increasing_60k_context():
    async def run():
        client = Client('с' * 4000)
        llm = model(client, context=60000, completion=6144)
        entities = [Entity('НГУ', 'ORGANIZATION', 'x' * 625000, ['chunk-1'])]
        result = await BudgetedEntitySummarizer(llm, summarize_only_if_more_than=0).run(entities)
        assert len(result) == 1 and result[0].source_chunk_id == ['chunk-1']
        assert len(client.requests) > 32
        assert all(llm.count_prompt_tokens(request) + 6144 <= 60000 for request in client.requests)
        assert llm.overflow_count == 0
    asyncio.run(run())


def test_upgrade_restores_extracted_graph_and_bounds_summary_without_repeating_extraction(tmp_path):
    async def run():
        chunk = Chunk('НГУ', 0, 'doc-0')
        entities = [Entity('НГУ', 'ORGANIZATION', 'x' * 5000, [chunk.id])]
        configuration = {'code': {'graph_utils/bounded_summaries.py': 'new'}}
        with GraphCheckpoints(tmp_path, previous_summary_configuration(configuration)) as old:
            old.save('06-artifacts', (entities, []), (Entity, Relation))
            original = old.manifest['stages']['06-artifacts']['sha256']
        client = Client()
        llm = model(client)
        with GraphCheckpoints(tmp_path, configuration) as store:
            parent = InMemoryGraphBuilder(
                llm=llm, embedder=object(),
                artifact_extractor=CheckpointedExtractor(llm=llm, ontology=None),
                build_parameters=BuilderArguments(use_clustering=False, make_community_summary=False),
            )
            parent.artifact_extractor.checkpoints = store
            graph = CheckpointedGraphBuilder.from_pipeline(parent, store)
            result = await graph.extract_graph([chunk])
            assert result[0][0].source_chunk_id == [chunk.id]
            assert llm.overflow_count == 0
            assert store.manifest['stages']['06-artifacts']['sha256'] == original
            assert '11-graph-ready' in store.manifest['stages']
            count = len(client.requests)
            await graph.extract_graph([chunk])
            assert len(client.requests) == count
    asyncio.run(run())


def test_split_preserves_every_character_in_order_including_last_fragment():
    async def run():
        client = Client('s')
        llm = model(client, context=100, completion=10)
        text = '\n'.join(f'Запись {index}: имя и связь. 🦉' for index in range(20))
        renderer = lambda value: [{'role': 'user', 'content': value}]
        batches = []
        original = llm.batch_chat_completion

        async def batch(conversations, **kwargs):
            batches.append(conversations)
            return await original(conversations, **kwargs)

        llm.batch_chat_completion = batch
        result = await SummaryBudget(llm).reduce(text, renderer, EntityDescriptionModel, 'description', label='test')
        assert result
        assert ''.join(request[0]['content'] for request in batches[0]) == text
        assert llm.overflow_count == 0
    asyncio.run(run())


@pytest.mark.parametrize('bad', ['', 'repeat'])
def test_empty_or_nonshrinking_reduction_fails_without_overflow_or_silent_truncation(bad):
    async def run():
        text = 'x' * 1000
        client = Client((lambda request: request[0]['content']) if bad == 'repeat' else '')
        llm = model(client, context=100, completion=10)
        renderer = lambda value: [{'role': 'user', 'content': value}]
        with pytest.raises(SummaryBudgetError):
            await SummaryBudget(llm).reduce(text, renderer, EntityDescriptionModel, 'description', label='test')
        assert all(llm.count_prompt_tokens(request) <= 90 for request in client.requests)
        assert llm.overflow_count == 0
    asyncio.run(run())


def test_split_at_two_character_whitespace_boundary_makes_progress():
    async def run():
        llm = model(Client('s'), context=17, completion=10)
        renderer = lambda value: [{'role': 'user', 'content': value}]
        with pytest.raises(SummaryBudgetError, match='не уменьшается'):
            await SummaryBudget(llm).reduce('a\nb ', renderer, EntityDescriptionModel, 'description', label='test')
        assert all(len(request[0]['content']) == 1 for request in llm.client.requests)
    asyncio.run(run())


@pytest.mark.parametrize('kind', ['entity', 'relation', 'community'])
def test_fitting_prompts_reuse_real_legacy_response_cache_without_http_requests(tmp_path, kind):
    async def run():
        requests = []
        schemas = {cls.__name__: cls for cls in (
            EntityDescriptionModel, RelationDescriptionModel, CommunityReportModel,
        )}

        async def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            schema = schemas[payload['response_format']['json_schema']['name']]
            return httpx.Response(200, json={
                'id': 'test', 'object': 'chat.completion', 'created': 1, 'model': 'graph-model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                    'role': 'assistant', 'content': response(schema).model_dump_json(),
                }}],
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            api = AsyncOpenAI(base_url='http://llm.test/v1', api_key='EMPTY', http_client=http)
            with Cache(str(tmp_path / 'llm-cache')) as cache:
                llm = model(CachedAsyncOpenAI(client=api, cache=cache), context=60000)
                entities, relations = artifacts('Университет')
                if kind == 'entity':
                    legacy = EntitySummarizer(llm, summarize_only_if_more_than=0)
                    updated = BudgetedEntitySummarizer(llm, summarize_only_if_more_than=0)
                    data = entities
                    old = await legacy.run(data)
                    actual = await updated.run(data)
                elif kind == 'relation':
                    legacy = RelationSummarizer(llm, summarize_only_if_more_than=0)
                    updated = BudgetedRelationSummarizer(llm, summarize_only_if_more_than=0)
                    data = relations
                    old = await legacy.run(data)
                    actual = await updated.run(data)
                else:
                    data = [Community(0, 7, entities, relations)]
                    old = await CommunitySummarizer(llm).summarize(data)
                    actual = await BudgetedCommunitySummarizer(llm).summarize(data)
                assert len(requests) == 1
                assert [dataclasses.asdict(item) for item in actual] == [dataclasses.asdict(item) for item in old]
    asyncio.run(run())
