import asyncio
import time
from collections import Counter

from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.search_engine.local_search import LocalSearchEngine, LocalSearchResult, LocalSearchRetrieve
from ragu.search_engine.search_functional import (
    _find_documents_id,
    _find_most_related_community_from_entities,
    _find_most_related_edges_from_entities,
    _rerank_items,
)


async def find_related_text_units(entities, knowledge_graph, edges_by_entity, neighbor_chunks):
    lookup = {}
    for order, entity in enumerate(entity for entity in entities if entity and entity.id):
        await asyncio.sleep(0)
        source_ids = dict.fromkeys(chunk_id for chunk_id in entity.source_chunk_id if chunk_id not in lookup)
        if not source_ids:
            continue
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
            for chunk_id in set(neighbor_chunks[neighbor_id]).intersection(source_ids):
                counts[chunk_id] += multiplicity
        for chunk_id in source_ids:
            lookup[chunk_id] = {'order': order, 'relation_counts': counts[chunk_id]}
    if not lookup:
        return []
    payloads = await knowledge_graph.index.chunks_kv_storage.get_by_ids(list(lookup))
    for chunk_id, payload in zip(lookup, payloads, strict=False):
        lookup[chunk_id]['data'] = payload
    ordered = sorted(lookup.values(), key=lambda item: (item['order'], -item['relation_counts']))
    chunks = []
    for index, item in enumerate(ordered):
        if index and index % 4096 == 0:
            await asyncio.sleep(0)
        if item['data'] is not None:
            chunks.append(Chunk(**item['data']))
    return chunks


class EfficientLocalSearchEngine(LocalSearchEngine):
    max_related_candidates: int | None = None

    async def _assemble(self, query, entities, entity_hits, ranked_entities, edges_by_entity, neighbor_chunks):
        started = time.monotonic()
        entity_scores_by_id = {
            entity.id: hit.distance
            for entity, hit in zip(entities, entity_hits, strict=False)
            if entity and entity.id
        }
        entities = ranked_entities
        relations, chunks, summaries, documents_id = await asyncio.gather(
            _find_most_related_edges_from_entities(entities, edges_by_entity),
            find_related_text_units(entities, self.knowledge_graph, edges_by_entity, neighbor_chunks),
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
        relations, summaries, chunks = await asyncio.gather(
            _rerank_items(query, relations, lambda relation: (
                f'{relation.subject_name}\n{relation.relation_type}\n'
                f'{relation.object_name}\n{relation.description}'
            ), self.reranker),
            _rerank_items(query, summaries, lambda summary: summary.summary, self.reranker),
            _rerank_items(query, chunks, lambda chunk: chunk.content, self.reranker),
        )
        logger.info('Локальный поиск RAGU: переранжирование завершено; всего {:.2f} с.',
                    time.monotonic() - started)
        return LocalSearchRetrieve(
            query=query,
            result=LocalSearchResult(
                entities=entities, relations=relations, summaries=summaries,
                chunks=chunks, documents_id=documents_id,
            ),
            metrics={'entities': [
                {'id': entity.id, 'name': entity.entity_name, 'rank': index,
                 'relevance_score': entity_scores_by_id.get(entity.id)}
                for index, entity in enumerate(entities)
            ]},
        )
