import asyncio
import itertools
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


async def rerank_with_scores(query, items, text_getter, reranker):
    """ragu's ``_rerank_items`` (the items in the cross-encoder's order, the same scorer call) that also returns
    the cross-encoder's scores, aligned with the items; ``None`` without a reranker."""
    if reranker is None or not items:
        return items, None
    ranked = [(index, float(score)) for index, score in await reranker.score(query, [text_getter(item) for item in items])
              if 0 <= index < len(items)]
    return [items[index] for index, _ in ranked], [score for _, score in ranked]


async def find_related_text_units(entities, knowledge_graph, edges_by_entity, neighbor_chunks, limit=None,
                                  balanced=False):
    """The source chunks of the entities, in entity order and, within an entity, by how many of its relations lead
    to neighbors that share the chunk (ragu's ordering). With ``limit`` only the first ``limit`` chunks are built: the
    caller keeps no more than that, and the order puts every chunk of an earlier entity first, so entities after the
    one that fills the limit cannot change the result. A hub entity (the university, a faculty) has tens of thousands
    of relations and source chunks; building all of them took 10–15 s of CPU on the one search thread.

    ``balanced`` (with ``limit``): the seed entities give their chunks in turn, each in its own order, instead of
    one after another, so a hub that comes first no longer fills the whole budget with its own chunks and the
    chunks of the other seed entities reach the cross-encoder too (v6 graph audit, 18.11.2026)."""
    lookup = {}
    neighbor_sets = {}
    kept = 0
    per_entity = []  # balanced: each seed entity's unclaimed chunks, in its own order

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
        neighbor_counts = Counter()
        for index, edge in enumerate(edges_by_entity.get(entity.id, [])):
            if index and index % 4096 == 0:
                await asyncio.sleep(0)
            if not edge:
                continue
            if edge.subject_id == entity.id:
                neighbor_id = edge.object_id
            elif edge.object_id == entity.id:
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
        entity_chunks = [{'id': chunk_id, 'order': order, 'relation_counts': counts[chunk_id]}
                         for chunk_id in source_ids]
        if balanced and limit is not None:
            entity_chunks.sort(key=lambda item: -item['relation_counts'])  # stable: ties keep the source order
            per_entity.append(entity_chunks)
            for item in entity_chunks:  # a chunk belongs to the first seed entity that has it
                lookup[item['id']] = {**item, 'data': None}
        elif limit is not None:
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
            for item in entity_chunks:  # the rest of this entity's chunks stay claimed, as without the limit
                lookup.setdefault(item['id'], {**item, 'data': None})
            if kept >= limit:
                break
        else:
            for item in entity_chunks:
                lookup[item['id']] = item
    if balanced and limit is not None:
        return await _take_in_turn(per_entity, knowledge_graph, limit)
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


async def _take_in_turn(per_entity, knowledge_graph, limit):
    """Up to ``limit`` chunks with a stored payload, taking the seed entities' chunks in turn."""
    def interleaved():
        positions = [0] * len(per_entity)
        while True:
            progressed = False
            for index, items in enumerate(per_entity):
                if positions[index] < len(items):
                    yield items[positions[index]]
                    positions[index] += 1
                    progressed = True
            if not progressed:
                return

    queue = interleaved()
    chunks = []
    while len(chunks) < limit:
        batch = list(itertools.islice(queue, max(limit - len(chunks), 1)))
        if not batch:
            break
        payloads = await knowledge_graph.index.chunks_kv_storage.get_by_ids([item['id'] for item in batch])
        for payload in payloads:
            if payload is not None:
                chunks.append(Chunk(**payload))
                if len(chunks) >= limit:
                    break
    return chunks


def related_edges_balanced(entities, edges_by_entity, limit):
    """The relations of the seed entities, deduplicated as ragu does, each entity's by descending
    ``relation_strength``, taken in turn up to ``limit``: sorted by strength across all the seeds, a hub's tens of
    thousands of relations fill the budget before the relations of any other seed entity."""
    seen = set()
    per_entity = []
    for entity in (entity for entity in entities if entity and entity.id):
        group = []
        for edge in edges_by_entity.get(entity.id, []):
            if not edge:
                continue
            key = edge.id or (edge.subject_id, edge.object_id, edge.relation_type, edge.description)
            if key in seen:
                continue
            seen.add(key)
            group.append(edge)
        group.sort(key=lambda edge: edge.relation_strength, reverse=True)  # stable: ties keep the stored order
        per_entity.append(group)
    picked = []
    positions = [0] * len(per_entity)
    while len(picked) < limit:
        progressed = False
        for index, group in enumerate(per_entity):
            if positions[index] < len(group):
                picked.append(group[positions[index]])
                positions[index] += 1
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
    return picked


async def _related_edges_balanced(entities, edges_by_entity, limit):
    return related_edges_balanced(entities, edges_by_entity, limit)


class EfficientLocalSearchEngine(LocalSearchEngine):
    """ragu's local search with a cheaper chunk lookup, a budget of related candidates before the cross-encoder,
    and the cross-encoder's scores kept in the metrics (``chunk_scores``, ``relation_scores``, the anchor entities'
    ``rerank_score``): the same model scores the chunk channel, so these put both channels on one scale."""
    max_related_candidates: int | None = None
    balanced_pool: bool = False  # the related candidates of the seed entities taken in turn (GRAPH_BALANCED_POOL)

    async def batch_search(self, queries, params=None):
        if not queries:
            return []
        params = params or LocalParams()
        entity_results = await self.retriever.query_entities(queries, top_k=params.top_k)
        ranked = await asyncio.gather(*[
            self._rank_entities_scored(query, entities, params.rerank_top_k)
            for query, (entities, _) in zip(queries, entity_results, strict=True)
        ])
        edges_by_entity, neighbor_chunks = await _prefetch_entity_edges(
            [entity.id for entities, _ in ranked for entity in entities if entity and entity.id],
            self.knowledge_graph,
        )
        return list(await asyncio.gather(*[
            self._assemble(query, entities, entity_hits, anchors, edges_by_entity, neighbor_chunks,
                           entity_scores=scores)
            for query, (entities, entity_hits), (anchors, scores) in zip(queries, entity_results, ranked, strict=True)
        ]))

    async def _rank_entities_scored(self, query, entities, rerank_top_k=None):
        """ragu's ``_rank_entities`` that also returns the anchors' cross-encoder scores by entity id."""
        entities, scores = await rerank_with_scores(
            query, entities, lambda entity: f'{entity.entity_name}\n{entity.entity_type}\n{entity.description}',
            self.reranker)
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
        entity_scores_by_id = {
            entity.id: hit.distance
            for entity, hit in zip(entities, entity_hits, strict=False)
            if entity and entity.id
        }
        entities = ranked_entities
        relations, chunks, summaries, documents_id = await asyncio.gather(
            (_related_edges_balanced(entities, edges_by_entity, self.max_related_candidates)
             if self.balanced_pool and self.max_related_candidates
             else _find_most_related_edges_from_entities(entities, edges_by_entity)),
            find_related_text_units(entities, self.knowledge_graph, edges_by_entity, neighbor_chunks,
                                    limit=self.max_related_candidates or None, balanced=self.balanced_pool),
            _find_most_related_community_from_entities(entities, self.knowledge_graph),
            _find_documents_id(entities),
        )
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
            ), self.reranker),
            rerank_with_scores(query, summaries, lambda summary: summary.summary, self.reranker),
            rerank_with_scores(query, chunks, lambda chunk: chunk.content, self.reranker),
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
