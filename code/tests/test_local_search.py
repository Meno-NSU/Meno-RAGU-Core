import asyncio
import random
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graph_utils.local_search import EfficientLocalSearchEngine, find_related_text_units
from ragu.graph.types import Entity, Relation
from ragu.search_engine.local_search import LocalParams, LocalSearchEngine
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
        optimized.metrics.pop('chunk_scores')
        optimized.metrics.pop('relation_scores')
        for metric in optimized.metrics['entities']:
            assert metric.pop('rerank_score') is None
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


def scored_reranker(scores, default=0.2):
    """A cross-encoder stub: the score of a text is looked up by a substring of it, best first like ScorerOpenAI."""
    def score(query, texts):
        values = [next((value for key, value in scores.items() if key in text), default) for text in texts]
        return sorted(enumerate(values), key=lambda pair: -pair[1])
    return SimpleNamespace(score=AsyncMock(side_effect=score))


def test_the_cross_encoder_scores_are_kept_next_to_the_order_they_give():
    # The engine already scores every related chunk and relation with the cross-encoder; the scores are what puts
    # the graph channel and the chunk channel on one scale, so they are reported with the items, best first.
    async def run():
        entities, edges, neighbors, payloads = fixture()
        entities = [entity for entity in entities[:3] if entity]
        engine = object.__new__(EfficientLocalSearchEngine)
        engine.knowledge_graph = graph(payloads)
        engine.reranker = scored_reranker({'Текст c1': 0.9, 'Текст c2': 0.1, 'Текст c3': 0.5, 'Текст c4': 0.7,
                                           'Текст c5': 0.3, 'SECOND': 0.6})
        hits = [SimpleNamespace(distance=0.8), SimpleNamespace(distance=0.5)]
        result = await engine._assemble('вопрос', entities, hits, entities, edges, neighbors,
                                        entity_scores={'a': 0.8, 'b': 0.4})
        assert [chunk.content for chunk in result.result.chunks] == [
            'Текст c1', 'Текст c4', 'Текст c3', 'Текст c5', 'Текст c2']
        assert result.metrics['chunk_scores'] == [0.9, 0.7, 0.5, 0.3, 0.1]
        assert result.result.relations[0].relation_type == 'SECOND'
        assert result.metrics['relation_scores'][0] == 0.6
        assert len(result.metrics['relation_scores']) == len(result.result.relations)
        assert [m['rerank_score'] for m in result.metrics['entities']] == [0.8, 0.4]
        assert [m['relevance_score'] for m in result.metrics['entities']] == [0.8, 0.5]
    asyncio.run(run())


def test_without_a_cross_encoder_no_scores_are_reported():
    async def run():
        entities, edges, neighbors, payloads = fixture()
        entities = [entity for entity in entities[:3] if entity]
        engine = object.__new__(EfficientLocalSearchEngine)
        engine.knowledge_graph = graph(payloads)
        engine.reranker = None
        hits = [SimpleNamespace(distance=0.8), SimpleNamespace(distance=0.5)]
        result = await engine._assemble('вопрос', entities, hits, entities, edges, neighbors)
        assert result.metrics.get('chunk_scores') is None and result.metrics.get('relation_scores') is None
        assert all(m.get('rerank_score') is None for m in result.metrics['entities'])
    asyncio.run(run())


def test_search_passes_the_cross_encoder_scores_of_the_anchor_entities(monkeypatch):
    async def run():
        entities, edges, neighbors, payloads = fixture()
        entities = [entity for entity in entities[:3] if entity]
        engine = object.__new__(EfficientLocalSearchEngine)
        engine.knowledge_graph = graph(payloads)
        engine.reranker = scored_reranker({'Факультет': 0.95, 'НГУ': 0.4})
        hits = [SimpleNamespace(distance=0.8), SimpleNamespace(distance=0.5)]
        engine.retriever = SimpleNamespace(query_entities=AsyncMock(return_value=[(entities, hits)]))
        monkeypatch.setattr('graph_utils.local_search._prefetch_entity_edges',
                            AsyncMock(return_value=(edges, neighbors)))
        [result] = await engine.batch_search(['вопрос'], params=LocalParams(top_k=2, rerank_top_k=1))
        assert [entity.id for entity in result.result.entities] == ['b']
        assert result.metrics['entities'] == [{'id': 'b', 'name': 'Факультет', 'rank': 0, 'relevance_score': 0.5,
                                               'rerank_score': 0.95}]
    asyncio.run(run())
