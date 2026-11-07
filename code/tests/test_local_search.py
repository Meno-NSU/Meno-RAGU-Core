import asyncio
import random
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graph_utils.local_search import EfficientLocalSearchEngine, find_related_text_units
from ragu.graph.types import Entity, Relation
from ragu.search_engine.local_search import LocalSearchEngine
from ragu.search_engine.search_functional import _find_most_related_text_unit_from_entities


def fixture():
    first = Entity('НГУ', 'ORGANIZATION', 'Описание', ['c1', 'c2', 'c3', 'missing', 'c1'], id='a')
    second = Entity('Факультет', 'ORGANIZATION', 'Описание', ['c2', 'c4', 'c5'], id='b')

    def edge(subject, target, kind='RELATED'):
        return Relation(subject, target, subject, target, kind, 'Связь')

    edges = {'a': [edge('a', 'n1'), edge('a', 'n1', 'SECOND'), edge('n2', 'a'),
                   edge('a', 'a'), edge('outside', 'n1'), None],
             'b': [edge('b', 'n1'), edge('b', 'n3')]}
    neighbors = {'n1': ['c2', 'c2', 'c3'], 'n2': ['c3'], 'a': first.source_chunk_id,
                 'n3': ['c5', 'c5']}
    payloads = {f'c{index}': {'content': f'Текст c{index}', 'chunk_order_idx': index, 'doc_id': f'doc-{index}'}
                for index in range(1, 6)}
    return [None, first, second, first], edges, neighbors, payloads


def graph(payloads):
    storage = SimpleNamespace(get_by_ids=AsyncMock(side_effect=lambda ids: [payloads.get(item) for item in ids]))
    return SimpleNamespace(index=SimpleNamespace(chunks_kv_storage=storage))


def test_chunk_selection_matches_upstream_with_duplicates_self_loops_and_missing_payloads():
    async def run():
        entities, edges, neighbors, payloads = fixture()
        original, optimized = graph(payloads), graph(payloads)
        before = await _find_most_related_text_unit_from_entities(entities, original, edges, neighbors)
        after = await find_related_text_units(entities, optimized, edges, neighbors)
        assert [asdict(chunk) for chunk in after] == [asdict(chunk) for chunk in before]
        assert [chunk.content for chunk in after] == ['Текст c3', 'Текст c2', 'Текст c1', 'Текст c5', 'Текст c4']
        assert original.index.chunks_kv_storage.get_by_ids.call_args == optimized.index.chunks_kv_storage.get_by_ids.call_args
    asyncio.run(run())


@pytest.mark.parametrize('payload', [{}, {'content': 'bad', 'unknown': True}, 'bad'])
def test_invalid_payloads_raise_instead_of_silently_dropping_chunks(payload):
    async def run():
        entities, edges, neighbors, payloads = fixture()
        payloads['c1'] = payload
        for operation in (_find_most_related_text_unit_from_entities, find_related_text_units):
            with pytest.raises(TypeError):
                await operation(entities, graph(payloads), edges, neighbors)
    asyncio.run(run())


@pytest.mark.parametrize('entities', [[], [None], [Entity('Empty', 'ORGANIZATION', '', [], id='empty')]])
def test_empty_chunk_sets_return_no_chunks(entities):
    assert asyncio.run(find_related_text_units(entities, graph({}), {}, {})) == []


@pytest.mark.parametrize('rerank', [False, True])
def test_subclass_assembles_identical_context_and_metrics_to_upstream(rerank):
    async def run():
        entities, edges, neighbors, payloads = fixture()
        entities = [entity for entity in entities[:3] if entity]
        before = object.__new__(LocalSearchEngine)
        after = object.__new__(EfficientLocalSearchEngine)
        for engine in (before, after):
            engine.knowledge_graph = graph(payloads)
            engine.reranker = (SimpleNamespace(score=AsyncMock(
                side_effect=lambda query, texts: [(index, float(index)) for index in reversed(range(len(texts)))],
            )) if rerank else None)
        hits = [SimpleNamespace(distance=0.8), SimpleNamespace(distance=0.5)]
        original = await before._assemble('вопрос', entities, hits, entities, edges, neighbors)
        optimized = await after._assemble('вопрос', entities, hits, entities, edges, neighbors)
        assert asdict(optimized) == asdict(original)
        if rerank:
            assert before.reranker.score.call_args_list == after.reranker.score.call_args_list
    asyncio.run(run())


def test_chunk_selection_can_be_cancelled_before_reading_chunk_payloads():
    async def run():
        entities, edges, neighbors, payloads = fixture()
        knowledge_graph = graph(payloads)
        task = asyncio.create_task(find_related_text_units(entities, knowledge_graph, edges, neighbors))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        knowledge_graph.index.chunks_kv_storage.get_by_ids.assert_not_awaited()
    asyncio.run(run())


def test_related_candidate_budget_applies_before_reranker_and_keeps_source_references():
    async def run():
        entities, edges, neighbors, payloads = fixture()
        entities = [entity for entity in entities[:3] if entity]
        engine = object.__new__(EfficientLocalSearchEngine)
        engine.knowledge_graph = graph(payloads)
        engine.max_related_candidates = 2
        engine.reranker = SimpleNamespace(score=AsyncMock(
            side_effect=lambda query, texts: [(index, 1.0) for index in range(len(texts))],
        ))
        hits = [SimpleNamespace(distance=0.8), SimpleNamespace(distance=0.5)]
        result = await engine._assemble('вопрос', entities, hits, entities, edges, neighbors)
        assert [chunk.content for chunk in result.result.chunks] == ['Текст c3', 'Текст c2']
        assert len(result.result.relations) == 2
        assert result.result.entities == entities
        assert all(len(call.args[1]) <= 2 for call in engine.reranker.score.call_args_list)
    asyncio.run(run())


@pytest.mark.parametrize('seed', range(5))
def test_chunk_order_matches_upstream_across_overlapping_graph_neighborhoods(seed):
    async def run():
        rng = random.Random(seed)
        ids = [f'c{index}' for index in range(20)]
        entities = [Entity(f'E{index}', 'TYPE', '', rng.choices(ids, k=12), id=f'e{index}')
                    for index in range(5)]
        edges = {}
        neighbors = {entity.id: entity.source_chunk_id for entity in entities}
        for entity in entities:
            edges[entity.id] = []
            for index in range(10):
                other = rng.choice(entities)
                subject, target = (entity.id, other.id) if index % 2 else (other.id, entity.id)
                edges[entity.id].append(Relation(subject, target, subject, target, f'K{index}', ''))
        payloads = {chunk_id: {'content': chunk_id, 'chunk_order_idx': 0, 'doc_id': 'doc'}
                    for chunk_id in ids if rng.randrange(4)}
        original = await _find_most_related_text_unit_from_entities(entities, graph(payloads), edges, neighbors)
        optimized = await find_related_text_units(entities, graph(payloads), edges, neighbors)
        assert [asdict(item) for item in optimized] == [asdict(item) for item in original]
    asyncio.run(run())
