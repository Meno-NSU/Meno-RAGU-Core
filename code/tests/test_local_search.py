import asyncio
import random
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graph_utils.local_search import (
    EfficientLocalSearchEngine,
    build_hub_index,
    find_related_text_units,
    hub_candidates,
)
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


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('limit', [1, 3, 7, 50])
def test_a_limit_gives_exactly_the_first_chunks_of_the_full_selection(seed, limit):
    # The engine keeps only the first `max_related_candidates` chunks; computing just those must not change them.
    async def run():
        rng = random.Random(seed)
        ids = [f'c{index}' for index in range(40)]
        entities = [Entity(f'E{index}', 'TYPE', '', rng.choices(ids, k=15), id=f'e{index}') for index in range(6)]
        neighbors = {entity.id: entity.source_chunk_id for entity in entities}
        edges = {}
        for entity in entities:
            edges[entity.id] = []
            for index in range(12):
                other = rng.choice(entities)
                subject, target = (entity.id, other.id) if index % 2 else (other.id, entity.id)
                edges[entity.id].append(Relation(subject, target, subject, target, f'K{index}', ''))
        payloads = {chunk_id: {'content': chunk_id, 'chunk_order_idx': 0, 'doc_id': 'doc'}
                    for chunk_id in ids if rng.randrange(5)}
        full = await _find_most_related_text_unit_from_entities(entities, graph(payloads), edges, neighbors)
        limited = await find_related_text_units(entities, graph(payloads), edges, neighbors, limit=limit)
        assert [asdict(item) for item in limited] == [asdict(item) for item in full[:limit]]
    asyncio.run(run())


def test_a_hub_entity_reads_only_the_chunks_it_keeps():
    async def run():
        ids = [f'c{index}' for index in range(5000)]
        hub = Entity('НГУ', 'ORGANIZATION', '', ids, id='hub')
        edges = {'hub': [Relation('hub', f'n{index}', 'hub', f'n{index}', 'R', '') for index in range(3000)]}
        neighbors = {f'n{index}': [ids[index], ids[index + 1]] for index in range(3000)}
        payloads = {chunk_id: {'content': chunk_id, 'chunk_order_idx': 0, 'doc_id': 'doc'} for chunk_id in ids}
        knowledge_graph = graph(payloads)
        chunks = await find_related_text_units([hub], knowledge_graph, edges, neighbors, limit=200)
        read = sum(len(call.args[0]) for call in knowledge_graph.index.chunks_kv_storage.get_by_ids.call_args_list)
        assert len(chunks) == 200 and read == 200
        full = await _find_most_related_text_unit_from_entities([hub], graph(payloads), edges, neighbors)
        assert [chunk.content for chunk in chunks] == [chunk.content for chunk in full[:200]]
    asyncio.run(run())


class HubGraph:
    """A knowledge graph with hubs, read through the backend calls ragu's search makes: an edge between two entities
    is in both incident lists, relation strengths repeat (ties), some edges are self-loops, some neighbours have no
    node and some chunks no payload."""

    def __init__(self, seed, entities=80, hubs=(0, 1, 2), hub_edges=600, edges=400):
        rng = random.Random(seed)
        ids = [f'c{index}' for index in range(500)]
        self.nodes = {}
        for index in range(entities):
            count = 300 if index in hubs else rng.randint(1, 12)
            self.nodes[f'e{index}'] = Entity(f'E{index}', 'TYPE', f'описание {index}', rng.choices(ids, k=count),
                                             id=f'e{index}')
        missing = {f'e{index}' for index in rng.sample(range(10, entities), 5)}
        self.incident = {node: [] for node in self.nodes}
        strengths = [1.0, 2.0, 2.0, 5.0, 7.5] + [round(rng.uniform(0, 10), 1) for _ in range(20)]
        pairs = [(f'e{rng.choice(hubs)}', f'e{rng.randrange(entities)}') for _ in range(hub_edges)]
        pairs += [(f'e{rng.randrange(entities)}', f'e{rng.randrange(entities)}') for _ in range(edges)]
        for number, (left, right) in enumerate(pairs):
            subject, target = (left, right) if number % 3 else (right, left)
            relation = Relation(subject, target, subject, target, f'K{number % 4}', f'связь {number}',
                                relation_strength=rng.choice(strengths), id=f'r{number}')
            self.incident[subject].append(relation)
            if target != subject:
                self.incident[target].append(relation)
        for node in missing:
            self.nodes.pop(node)
        self.payloads = {chunk_id: {'content': chunk_id, 'chunk_order_idx': 0, 'doc_id': 'doc'}
                         for chunk_id in ids if rng.randrange(6)}

        async def get_all_edges_for_nodes(node_ids):
            return [list(self.incident.get(node, [])) for node in node_ids]

        async def get_nodes(node_ids):
            return [self.nodes.get(node) for node in node_ids]

        storage = SimpleNamespace(get_by_ids=AsyncMock(side_effect=lambda ids: [self.payloads.get(i) for i in ids]))
        backend = SimpleNamespace(get_all_edges_for_nodes=AsyncMock(side_effect=get_all_edges_for_nodes),
                                  _graph=SimpleNamespace(degree=lambda: [(node, len(edges))
                                                                         for node, edges in self.incident.items()]))
        self.knowledge_graph = SimpleNamespace(index=SimpleNamespace(graph_backend=backend, get_nodes=get_nodes,
                                                                     chunks_kv_storage=storage))


def search_engine(hub_graph, max_related, hub_index=None):
    engine = object.__new__(EfficientLocalSearchEngine)
    engine.knowledge_graph = hub_graph.knowledge_graph
    engine.max_related_candidates = max_related
    engine.hub_index = hub_index
    engine.reranker = scored_reranker({f'E{index}': 0.1 + index / 100 for index in range(80)}, default=0.05)
    return engine


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('max_related', [1, 5, 40, 200])
def test_searches_with_the_hub_index_find_the_same_pools(seed, max_related):
    async def run():
        hub_graph = HubGraph(seed)
        hubs = hub_candidates(hub_graph.knowledge_graph, 100)
        assert {'e0', 'e1', 'e2'} <= set(hubs)
        index = await build_hub_index(hub_graph.knowledge_graph, hubs, max_related)
        rng = random.Random(seed)
        for query in range(12):
            anchors = [hub_graph.nodes[node] for node in rng.sample(sorted(hub_graph.nodes), 20)]
            hubs_first = [hub_graph.nodes[node] for node in ('e0', 'e1', 'e2')]
            if query % 4 == 0:  # only hubs: every relation and chunk of the pools comes from the precomputed ones
                anchors = hubs_first
            elif query % 4 == 1:  # the hubs first, then others that share many of their relations
                anchors = hubs_first + anchors[:17]
            else:
                anchors[rng.randrange(20)] = hub_graph.nodes['e0']
            hits = [SimpleNamespace(distance=0.9 - position / 100) for position in range(len(anchors))]
            results = []
            for hub_index in (None, index):
                engine = search_engine(hub_graph, max_related, hub_index)
                engine.retriever = SimpleNamespace(query_entities=AsyncMock(return_value=[(list(anchors), hits)]))
                [result] = await engine.batch_search(['вопрос'], params=LocalParams(top_k=20, rerank_top_k=20))
                results.append((result, engine.reranker.score.call_args_list))
            (before, calls_before), (after, calls_after) = results
            assert [r.id for r in after.result.relations] == [r.id for r in before.result.relations]
            assert [asdict(c) for c in after.result.chunks] == [asdict(c) for c in before.result.chunks]
            assert after.metrics == before.metrics
            assert calls_after == calls_before  # the same cross-encoder calls with the same texts
    asyncio.run(run())


def test_a_hub_query_reads_no_hub_edges_or_neighbours():
    async def run():
        hub_graph = HubGraph(0)
        index = await build_hub_index(hub_graph.knowledge_graph, ['e0', 'e1', 'e2'], 200)
        backend = hub_graph.knowledge_graph.index.graph_backend
        backend.get_all_edges_for_nodes.reset_mock()
        engine = search_engine(hub_graph, 200, index)
        anchors = [hub_graph.nodes['e0'], hub_graph.nodes['e1']]
        engine.retriever = SimpleNamespace(query_entities=AsyncMock(return_value=[
            (anchors, [SimpleNamespace(distance=0.9), SimpleNamespace(distance=0.8)])]))
        await engine.batch_search(['вопрос'], params=LocalParams(top_k=2, rerank_top_k=2))
        read = [node for call in backend.get_all_edges_for_nodes.call_args_list for node in call.args[0]]
        assert 'e0' not in read and 'e1' not in read
    asyncio.run(run())


def test_the_hub_index_is_not_used_when_the_search_keeps_more_relations_than_it_holds():
    hub_graph = HubGraph(1)
    index = asyncio.run(build_hub_index(hub_graph.knowledge_graph, ['e0'], 5))
    assert search_engine(hub_graph, 5, index)._usable_hub_index() is index
    assert search_engine(hub_graph, 6, index)._usable_hub_index() is None
    assert search_engine(hub_graph, 0, index)._usable_hub_index() is None  # no cut, no precomputed list
    assert hub_candidates(hub_graph.knowledge_graph, 0) == []
