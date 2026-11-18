import asyncio
import math
import time
from collections import Counter
from types import SimpleNamespace

from graph_utils.alias_table import stems
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


async def find_related_text_units(entities, knowledge_graph, edges_by_entity, neighbor_chunks, limit=None):
    """The source chunks of the entities, in entity order and, within an entity, by how many of its relations lead
    to neighbors that share the chunk (ragu's ordering). With ``limit`` only the first ``limit`` chunks are built: the
    caller keeps no more than that, and the order puts every chunk of an earlier entity first, so entities after the
    one that fills the limit cannot change the result. A hub entity (the university, a faculty) has tens of thousands
    of relations and source chunks; building all of them took 10–15 s of CPU on the one search thread."""
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
            for item in entity_chunks:  # the rest of this entity's chunks stay claimed, as without the limit
                lookup.setdefault(item['id'], {**item, 'data': None})
            if kept >= limit:
                break
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


# The ontology's relation type for "a stated relation that no other type names": such an edge says nothing about
# what kind of neighbour it leads to, so the expansion does not follow it.
UNTYPED_RELATION = 'RELATED_TO'
NEIGHBOR_CHUNKS_PER_EDGE = 3


def lexical_scores(query, token_sets):
    """How much of the query each text covers: the sum, over the query's stems the text has, of the stem's rarity
    among these texts (ln(1 + N / texts with the stem)). A word every candidate has (the anchor's own name) adds
    almost nothing, so no stop-word list is needed."""
    wanted = set(stems(query))
    counts = Counter(stem for tokens in token_sets for stem in tokens & wanted)
    total = len(token_sets)
    weights = {stem: math.log(1 + total / count) for stem, count in counts.items()}
    return [sum(weights[stem] for stem in tokens & wanted) for tokens in token_sets]


def round_robin(lists, limit):
    """Items of several ranked lists taken in turns (first of each, then second of each...), without repeats,
    up to ``limit``."""
    taken, seen = [], set()
    for position in range(max((len(items) for items in lists), default=0)):
        for items in lists:
            if position < len(items) and items[position] not in seen:
                seen.add(items[position])
                taken.append(items[position])
                if len(taken) >= limit:
                    return taken
    return taken


def neighbor_of(edge, entity_id):
    if edge.subject_id == entity_id:
        return edge.object_id
    if edge.object_id == entity_id:
        return edge.subject_id
    return None


class NavigationPlan:
    """The candidates of the graph channel when the search walks the graph (``expansion_candidates`` > 0):

    * every anchor entity gets its turn: its own source chunks (ragu's order, by shared neighbours) and its
      relations are taken in turns across the anchors instead of filling the budget from the first anchor on;
    * the relations of an anchor are ordered by how much of the query their text covers, then typed before untyped,
      then by ``relation_strength`` (79% of the relations of graph v6 have strength 4 or 5);
    * one step along typed relations: the chunks the relation was extracted from and the first chunks of the
      neighbour, for neighbours with at most ``hub_chunks`` source chunks (a hub neighbours everything).
    """

    def __init__(self, query, entities, edges_by_entity, neighbor_chunks, edge_tokens):
        self.entities = [entity for entity in entities if entity and entity.id]
        self.neighbor_chunks = neighbor_chunks
        self.ranked_edges = {}
        for entity in self.entities:
            if entity.id in self.ranked_edges:
                continue
            edges = list({(edge.id or id(edge)): edge for edge in edges_by_entity.get(entity.id, []) if edge
                          and neighbor_of(edge, entity.id) is not None}.values())
            scores = lexical_scores(query, [edge_tokens(edge) for edge in edges])
            order = sorted(range(len(edges)), key=lambda index: (
                -scores[index], edges[index].relation_type == UNTYPED_RELATION, -edges[index].relation_strength))
            self.ranked_edges[entity.id] = [edges[index] for index in order]

    def relations(self, limit):
        chosen = round_robin([[edge.id or id(edge) for edge in self.ranked_edges[entity.id]]
                              for entity in self.entities], limit)
        by_key = {(edge.id or id(edge)): edge for edges in self.ranked_edges.values() for edge in edges}
        return [by_key[key] for key in chosen]

    def own_chunks(self, limit):
        lists = []
        for entity in self.entities:
            source_ids = list(dict.fromkeys(entity.source_chunk_id))
            source_set = set(source_ids)
            counts = Counter()
            multiplicity = Counter(neighbor_of(edge, entity.id) for edge in self.ranked_edges[entity.id])
            for neighbor_id, times in multiplicity.items():
                for chunk_id in set(self.neighbor_chunks.get(neighbor_id, ())) & source_set:
                    counts[chunk_id] += times
            lists.append(sorted(source_ids, key=lambda chunk_id: -counts[chunk_id]))  # stable: ties keep the order
        return round_robin(lists, limit)

    def expansion_chunks(self, limit, hub_chunks, exclude):
        lists = []
        for entity in self.entities:
            found = []
            for edge in self.ranked_edges[entity.id]:
                if edge.relation_type == UNTYPED_RELATION:
                    continue
                chunks = self.neighbor_chunks.get(neighbor_of(edge, entity.id))
                if chunks is None or len(chunks) > hub_chunks:
                    continue
                found.extend(edge.source_chunk_id)
                found.extend(list(dict.fromkeys(chunks))[:NEIGHBOR_CHUNKS_PER_EDGE])
            lists.append([chunk_id for chunk_id in dict.fromkeys(found) if chunk_id not in exclude])
        return round_robin(lists, limit)


async def load_chunks(knowledge_graph, chunk_ids):
    payloads = await knowledge_graph.index.chunks_kv_storage.get_by_ids(list(chunk_ids)) if chunk_ids else []
    return [Chunk(**payload) for payload in payloads if payload is not None]


class EfficientLocalSearchEngine(LocalSearchEngine):
    """ragu's local search with a cheaper chunk lookup, a budget of related candidates before the cross-encoder,
    and the cross-encoder's scores kept in the metrics (``chunk_scores``, ``relation_scores``, the anchor entities'
    ``rerank_score``): the same model scores the chunk channel, so these put both channels on one scale."""
    max_related_candidates: int | None = None
    # > 0: the search walks the graph (NavigationPlan), with up to this many neighbour chunks per query
    expansion_candidates: int = 0
    expansion_hub_chunks: int = 200
    # graph_utils.alias_table.AliasTable: the entities the question names join the entities of the vector search
    alias_table = None
    max_linked_entities: int = 8
    _edge_tokens: dict | None = None

    def edge_tokens(self, edge):
        cache = self._edge_tokens
        if cache is None or len(cache) > 500_000:
            cache = self._edge_tokens = {}
        key = edge.id or id(edge)
        tokens = cache.get(key)
        if tokens is None:
            tokens = cache[key] = frozenset(stems(f'{edge.subject_name} {edge.object_name} {edge.description}'))
        return tokens

    async def _with_named_entities(self, queries, entity_results):
        """The entities the queries name (alias table), before those of the vector search; the cross-encoder then
        ranks all of them together."""
        named = [self.alias_table.node_ids(query)[:self.max_linked_entities] for query in queries]
        unique_ids = list(dict.fromkeys(node_id for node_ids in named for node_id in node_ids))
        if not unique_ids:
            return entity_results
        nodes = dict(zip(unique_ids, await self.knowledge_graph.index.get_nodes(unique_ids), strict=True))
        merged = []
        for node_ids, (entities, hits) in zip(named, entity_results, strict=True):
            known = {entity.id for entity in entities if entity}
            added = [nodes[node_id] for node_id in node_ids if nodes[node_id] is not None and node_id not in known]
            merged.append((added + list(entities), [SimpleNamespace(distance=None)] * len(added) + list(hits)))
        return merged

    async def batch_search(self, queries, params=None):
        if not queries:
            return []
        params = params or LocalParams()
        entity_results = await self.retriever.query_entities(queries, top_k=params.top_k)
        if self.alias_table is not None:
            entity_results = await self._with_named_entities(queries, entity_results)
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
        if self.expansion_candidates > 0:
            plan = NavigationPlan(query, entities, edges_by_entity, neighbor_chunks, self.edge_tokens)
            budget = self.max_related_candidates or 200
            relations = plan.relations(budget)
            own = plan.own_chunks(budget)
            expanded = plan.expansion_chunks(self.expansion_candidates, self.expansion_hub_chunks, set(own))
            chunks, summaries, documents_id = await asyncio.gather(
                load_chunks(self.knowledge_graph, own + expanded),
                _find_most_related_community_from_entities(entities, self.knowledge_graph),
                _find_documents_id(entities),
            )
            return await self._rerank_and_pack(query, entities, entity_scores_by_id, entity_scores, relations,
                                               chunks, summaries, documents_id, started, cut=False)
        relations, chunks, summaries, documents_id = await asyncio.gather(
            _find_most_related_edges_from_entities(entities, edges_by_entity),
            find_related_text_units(entities, self.knowledge_graph, edges_by_entity, neighbor_chunks,
                                    limit=self.max_related_candidates or None),
            _find_most_related_community_from_entities(entities, self.knowledge_graph),
            _find_documents_id(entities),
        )
        return await self._rerank_and_pack(query, entities, entity_scores_by_id, entity_scores, relations, chunks,
                                           summaries, documents_id, started, cut=True)

    async def _rerank_and_pack(self, query, entities, entity_scores_by_id, entity_scores, relations, chunks,
                               summaries, documents_id, started, cut):
        relations = [relation for relation in relations if relation is not None]
        chunks = [chunk for chunk in chunks if chunk is not None]
        summaries = [summary for summary in summaries if summary is not None]
        logger.info('Локальный поиск RAGU: подготовлено сущностей {}, связей {}, фрагментов {}, '
                    'описаний сообществ {}; {:.2f} с. Начинаю переранжирование.',
                    len(entities), len(relations), len(chunks), len(summaries), time.monotonic() - started)
        if cut and self.max_related_candidates:
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
