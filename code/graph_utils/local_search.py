import asyncio
import time
from collections import Counter

from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.search_engine.local_search import LocalParams, LocalSearchEngine, LocalSearchResult, LocalSearchRetrieve
from ragu.search_engine.search_functional import (
    _find_documents_id,
    _find_most_related_community_from_entities,
    _find_most_related_edges_from_entities,
    _prefetch_entity_edges,
)

from graph_utils import search_timings


async def rerank_with_scores(query, items, text_getter, reranker, kind=None):
    """ragu's ``_rerank_items`` (the items in the cross-encoder's order, the same scorer call) that also returns
    the cross-encoder's scores, aligned with the items; ``None`` without a reranker. ``kind`` names the candidates
    in the search timings (search_timings)."""
    if reranker is None or not items:
        return items, None
    if kind is not None:
        search_timings.record_pairs(kind, len(items))
    ranked = [(index, float(score)) for index, score in await reranker.score(query, [text_getter(item) for item in items])
              if 0 <= index < len(items)]
    return [items[index] for index, _ in ranked], [score for _, score in ranked]


async def _relation_counts(entity_id, edges, neighbor_chunks, source_set, chunk_set):
    """For each chunk of ``source_set``: how many of the entity's relations lead to a neighbor that shares it (ragu's
    count, each relation once per neighbor chunk). Only the neighbors in ``neighbor_chunks`` count."""
    neighbor_counts = Counter()
    for index, edge in enumerate(edges):
        if index and index % 4096 == 0:
            await asyncio.sleep(0)
        if not edge:
            continue
        if edge.subject_id == entity_id:
            neighbor_id = edge.object_id
        elif edge.object_id == entity_id:
            neighbor_id = edge.subject_id
        else:
            continue
        if neighbor_id in neighbor_chunks:
            neighbor_counts[neighbor_id] += 1
    counts = Counter()
    for index, (neighbor_id, multiplicity) in enumerate(neighbor_counts.items()):
        if index and index % 128 == 0:
            await asyncio.sleep(0)
        # both are sets: `&` walks the smaller one (a neighbor has a few chunks, a hub tens of thousands)
        for chunk_id in chunk_set(neighbor_id) & source_set:
            counts[chunk_id] += multiplicity
    return counts


async def find_related_text_units(entities, knowledge_graph, edges_by_entity, neighbor_chunks, limit=None,
                                  hub_counts=None):
    """The source chunks of the entities, in entity order and, within an entity, by how many of its relations lead
    to neighbors that share the chunk (ragu's ordering). With ``limit`` only the first ``limit`` chunks are built: the
    caller keeps no more than that, and the order puts every chunk of an earlier entity first, so entities after the
    one that fills the limit cannot change the result. A hub entity (the university, a faculty) has tens of thousands
    of relations and source chunks; building all of them took 10–15 s of CPU on the one search thread.

    ``hub_counts`` (HubIndex.chunk_counts): the counts of a hub's chunks computed at load, used instead of counting
    over its edges, which a hub's entry of ``edges_by_entity`` then does not need to hold. A chunk's count does not
    depend on the other entities, so it is the count computed here."""
    lookup = {}
    neighbor_sets = {}
    kept = 0

    def chunk_set(neighbor_id):
        found = neighbor_sets.get(neighbor_id)
        if found is None:
            found = neighbor_sets[neighbor_id] = frozenset(neighbor_chunks[neighbor_id])
        return found

    for order, entity in enumerate(entity for entity in entities if entity and entity.id):
        await asyncio.sleep(0)
        source_ids = dict.fromkeys(chunk_id for chunk_id in entity.source_chunk_id if chunk_id not in lookup)
        if not source_ids:
            continue
        source_set = set(source_ids)
        counts = hub_counts.get(entity.id) if hub_counts is not None else None
        if counts is None:
            counts = await _relation_counts(entity.id, edges_by_entity.get(entity.id, []), neighbor_chunks,
                                            source_set, chunk_set)
        entity_chunks = [{'id': chunk_id, 'order': order, 'relation_counts': counts.get(chunk_id, 0)}
                         for chunk_id in source_ids]
        if limit is not None:
            entity_chunks.sort(key=lambda item: -item['relation_counts'])  # stable: ties keep the source order
            # chunks without a stored payload are dropped below, so the limit counts only chunks that exist
            for start in range(0, len(entity_chunks), max(limit, 1)):
                batch = entity_chunks[start:start + max(limit, 1)]
                payloads = await knowledge_graph.index.chunks_kv_storage.get_by_ids([item['id'] for item in batch])
                for item, payload in zip(batch, payloads, strict=False):
                    item['data'] = payload
                    lookup[item['id']] = item
                    kept += payload is not None
                    if kept >= limit:
                        break
                if kept >= limit:
                    break
            if kept >= limit:
                break
            for item in entity_chunks:
                lookup.setdefault(item['id'], {**item, 'data': None})
        else:
            for item in entity_chunks:
                lookup[item['id']] = item
    if not lookup:
        return []
    if limit is None:
        payloads = await knowledge_graph.index.chunks_kv_storage.get_by_ids(list(lookup))
        for item, payload in zip(lookup.values(), payloads, strict=False):
            item['data'] = payload
    ordered = sorted(lookup.values(), key=lambda item: (item['order'], -item['relation_counts']))
    chunks = []
    for index, item in enumerate(ordered):
        if index and index % 4096 == 0:
            await asyncio.sleep(0)
        if item['data'] is not None:
            chunks.append(Chunk(**item['data']))
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


class EfficientLocalSearchEngine(LocalSearchEngine):
    """ragu's local search with a cheaper chunk lookup, a budget of related candidates before the cross-encoder,
    and the cross-encoder's scores kept in the metrics (``chunk_scores``, ``relation_scores``, the anchor entities'
    ``rerank_score``): the same model scores the chunk channel, so these put both channels on one scale.

    ``hub_index`` (build_hub_index, at load): the hub entities' neighbourhoods precomputed, so a query does not read
    their tens of thousands of edges and neighbours; the pools are the same."""
    max_related_candidates: int | None = None
    hub_index: 'HubIndex | None' = None

    def _usable_hub_index(self):
        hubs = self.hub_index
        if hubs is None or not self.max_related_candidates or hubs.top_relations < self.max_related_candidates:
            return None
        return hubs

    async def batch_search(self, queries, params=None):
        if not queries:
            return []
        params = params or LocalParams()
        entity_results = await self.retriever.query_entities(queries, top_k=params.top_k)
        ranked = await asyncio.gather(*[
            self._rank_entities_scored(query, entities, params.rerank_top_k)
            for query, (entities, _) in zip(queries, entity_results, strict=True)
        ])
        prefetch_started = time.perf_counter()
        seeds = [entity.id for entities, _ in ranked for entity in entities if entity and entity.id]
        hubs = self._usable_hub_index()
        edges_by_entity, neighbor_chunks = await _prefetch_entity_edges(
            [entity_id for entity_id in seeds if hubs is None or entity_id not in hubs],
            self.knowledge_graph,
        )
        if hubs is not None:
            for entity_id in seeds:
                if entity_id in hubs:
                    edges_by_entity[entity_id] = hubs.edges[entity_id]
        search_timings.record('graph_cpu_ms', prefetch_started)
        return list(await asyncio.gather(*[
            self._assemble(query, entities, entity_hits, anchors, edges_by_entity, neighbor_chunks,
                           entity_scores=scores)
            for query, (entities, entity_hits), (anchors, scores) in zip(queries, entity_results, ranked, strict=True)
        ]))

    async def _rank_entities_scored(self, query, entities, rerank_top_k=None):
        """ragu's ``_rank_entities`` that also returns the anchors' cross-encoder scores by entity id."""
        entities, scores = await rerank_with_scores(
            query, entities, lambda entity: f'{entity.entity_name}\n{entity.entity_type}\n{entity.description}',
            self.reranker, kind='entity')
        if self.reranker is not None and rerank_top_k is not None:
            entities = entities[:rerank_top_k]
        by_id = {entity.id: score for entity, score in zip(entities, scores or [], strict=False)
                 if entity and entity.id}
        return entities, by_id

    async def _rank_entities(self, query, entities, rerank_top_k=None):
        return (await self._rank_entities_scored(query, entities, rerank_top_k))[0]

    async def _assemble(self, query, entities, entity_hits, ranked_entities, edges_by_entity, neighbor_chunks,
                        entity_scores=None):
        started = time.monotonic()
        prep_started = time.perf_counter()
        entity_scores_by_id = {
            entity.id: hit.distance
            for entity, hit in zip(entities, entity_hits, strict=False)
            if entity and entity.id
        }
        entities = ranked_entities
        relations, chunks, summaries, documents_id = await asyncio.gather(
            _find_most_related_edges_from_entities(entities, edges_by_entity),
            find_related_text_units(entities, self.knowledge_graph, edges_by_entity, neighbor_chunks,
                                    limit=self.max_related_candidates or None,
                                    hub_counts=hubs.chunk_counts if (hubs := self._usable_hub_index()) else None),
            _find_most_related_community_from_entities(entities, self.knowledge_graph),
            _find_documents_id(entities),
        )
        search_timings.record('graph_cpu_ms', prep_started)
        relations = [relation for relation in relations if relation is not None]
        chunks = [chunk for chunk in chunks if chunk is not None]
        summaries = [summary for summary in summaries if summary is not None]
        logger.info('Локальный поиск RAGU: подготовлено сущностей {}, связей {}, фрагментов {}, '
                    'описаний сообществ {}; {:.2f} с. Начинаю переранжирование.',
                    len(entities), len(relations), len(chunks), len(summaries), time.monotonic() - started)
        if self.max_related_candidates:
            relations = relations[:self.max_related_candidates]
            summaries = summaries[:self.max_related_candidates]
            chunks = chunks[:self.max_related_candidates]
            logger.info('Локальный поиск RAGU: пул для переранжирования ограничен {} кандидатами каждого вида.',
                        self.max_related_candidates)
        (relations, relation_scores), (summaries, _), (chunks, chunk_scores) = await asyncio.gather(
            rerank_with_scores(query, relations, lambda relation: (
                f'{relation.subject_name}\n{relation.relation_type}\n'
                f'{relation.object_name}\n{relation.description}'
            ), self.reranker, kind='relation'),
            rerank_with_scores(query, summaries, lambda summary: summary.summary, self.reranker, kind='summary'),
            rerank_with_scores(query, chunks, lambda chunk: chunk.content, self.reranker, kind='chunk'),
        )
        logger.info('Локальный поиск RAGU: переранжирование завершено; всего {:.2f} с.',
                    time.monotonic() - started)
        return LocalSearchRetrieve(
            query=query,
            result=LocalSearchResult(
                entities=entities, relations=relations, summaries=summaries,
                chunks=chunks, documents_id=documents_id,
            ),
            metrics={
                'entities': [
                    {'id': entity.id, 'name': entity.entity_name, 'rank': index,
                     'relevance_score': entity_scores_by_id.get(entity.id),
                     'rerank_score': (entity_scores or {}).get(entity.id)}
                    for index, entity in enumerate(entities)
                ],
                'chunk_scores': chunk_scores,
                'relation_scores': relation_scores,
            },
        )


class HubIndex:
    """The neighbourhoods of the hub entities (those with at least GRAPH_HUB_MIN_EDGES incident edges), computed once
    at load (speed audit of 19.11, П7): per hub, its first ``top_relations`` relations in the order ragu's
    _find_most_related_edges_from_entities gives them (deduplicated, by relation strength, ties in incident order),
    and the relation count of each of its source chunks as find_related_text_units counts them.

    The pools are the same. The engine cuts the relations of a query to ``max_related_candidates`` after one stable
    sort of all its anchors' relations; a relation among the first ``top_relations`` of that sort is among the first
    ``top_relations`` of its own entity, so the cut lists of the hubs give the same first relations. A chunk's count
    does not depend on the other anchors. tests/test_local_search.py compares whole searches with and without the
    index."""

    def __init__(self, top_relations):
        self.top_relations = top_relations
        self.edges = {}
        self.chunk_counts = {}

    def __contains__(self, entity_id):
        return entity_id in self.edges

    def __len__(self):
        return len(self.edges)


def hub_candidates(knowledge_graph, min_edges):
    """The ids of the entities with at least ``min_edges`` incident edges (the networkx graph backend's degree, which
    counts a self-loop twice); none for another backend or ``min_edges`` 0."""
    graph = getattr(getattr(knowledge_graph.index, 'graph_backend', None), '_graph', None)
    if graph is None or min_edges <= 0:
        return []
    return [str(node) for node, degree in graph.degree() if degree >= min_edges]


async def build_hub_index(knowledge_graph, entity_ids, top_relations):
    """HubIndex of the entities ``entity_ids``, each read through the same functions as a query reads it."""
    index = HubIndex(top_relations)
    for entity_id in dict.fromkeys(entity_ids):
        [entity] = await knowledge_graph.index.get_nodes([entity_id])
        if entity is None or not entity.id:
            continue
        edges_by_entity, neighbor_chunks = await _prefetch_entity_edges([entity.id], knowledge_graph)
        ordered = await _find_most_related_edges_from_entities([entity], edges_by_entity)
        neighbor_sets = {}

        def chunk_set(neighbor_id, neighbor_sets=neighbor_sets, neighbor_chunks=neighbor_chunks):
            found = neighbor_sets.get(neighbor_id)
            if found is None:
                found = neighbor_sets[neighbor_id] = frozenset(neighbor_chunks[neighbor_id])
            return found

        counts = await _relation_counts(entity.id, edges_by_entity.get(entity.id, []), neighbor_chunks,
                                        set(entity.source_chunk_id), chunk_set)
        index.edges[entity.id] = [edge for edge in ordered if edge is not None][:top_relations]
        index.chunk_counts[entity.id] = dict(counts)
    return index
