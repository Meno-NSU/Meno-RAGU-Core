from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

from context_utils.context_utils import graph_element_to_text

from meno_rag.api.events import StageName
from meno_rag.stand.attribution import norm_url
from meno_rag.stand.context import normalize_urls
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline, _RerankOutput
from meno_rag.stand.rerank import rerank_merge_score


@dataclass(frozen=True)
class Candidate:
    key: str
    score: float
    chunk_id: int | None = None
    element: dict[str, Any] | None = None


class GraphSelection(_RerankOutput):
    def __init__(self, chunks, elements, *, scored, candidate_scores, graph_trace):
        super().__init__(chunks)
        self.elements = elements
        self.scored_candidates = scored
        self.candidate_scores = candidate_scores
        self.graph_trace = graph_trace


class RaguRagPipeline(StandRagPipeline):
    def _priority(self, candidate: Candidate, merged: float) -> float:
        """The order of relevant candidates, as Menon 1.1 orders them (v2/prompting.py priority, prod
        V2_SOURCE_PRIOR_WEIGHT=2): the log-odds of the merged score plus SOURCE_PRIOR_WEIGHT * ln(quality of the
        source document), so a comparably relevant official page goes before a post (posts of the KB have quality
        about .55, NSU pages .73-.95) and a clearly better post stays ahead. Graph elements have prior 1; weight 0
        keeps the merged score itself."""
        weight = self.settings.source_prior_weight
        if weight <= 0:
            return merged
        prior = 1.0
        if candidate.chunk_id is not None:
            mapping = self.resources.chunk_mapping.get(str(candidate.chunk_id))
            if mapping is not None:
                prior = float(self.resources.documents[mapping["doc_index"]].get("quality_score") or 1.0)
        p = min(max(merged, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p)) + weight * math.log(min(max(prior, 0.05), 1.0))

    async def _retrieve(self, search_queries):
        batches = []
        for query in search_queries:
            async with self.embed_semaphore:
                pool = await self.resources.graph.search(query)
            batches.append({"query": query, "graph": pool})
        return batches

    def _fuse(self, retrieval_batches):
        fused = []
        for batch in retrieval_batches:
            pool = batch["graph"]
            candidates = [Candidate(f"chunk:{cid}", float(score), chunk_id=cid) for cid, score in pool["chunks"]]
            candidates.extend(
                Candidate(f"{item['kind']}:{item['key']}", float(item["score"]), element=item)
                for item in [*pool["entities"], *pool["relations"]]
            )
            unique = {}
            for candidate in candidates:
                previous = unique.get(candidate.key)
                if previous is None or candidate.score > previous.score:
                    unique[candidate.key] = candidate
            fused.append({"query": batch["query"], "pool": sorted(unique.values(), key=lambda c: (-c.score, c.key))})
        return fused

    async def _rerank(self, fused_batches, user_question, dialogue_history, runtime: ModelRuntime):
        batches = [self._candidate_pool(batch["pool"]) for batch in fused_batches]
        unique = {c.key: c for batch in batches for c in batch}

        async def score(candidate):
            if candidate.chunk_id is not None:
                return await self._score_chunk_with_llm(user_question, dialogue_history, candidate.chunk_id, runtime)
            text = graph_element_to_text(candidate.element, self.resources.documents,
                                         self.resources.chunk_mapping, max_source_titles=0)
            return await self._score_document_with_llm(user_question, dialogue_history, text, runtime, candidate.key)

        if self.settings.llm_rerank:
            tasks = [asyncio.create_task(score(candidate)) for candidate in unique.values()]
            try:
                values = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # as Menon 1.1: the engine's cross-encoder score (bge-reranker) is the relevance, no LLM call
            values = [float(candidate.score) for candidate in unique.values()]
        scores = dict(zip(unique, values, strict=True))
        order = self._priority

        def relevant(candidate):
            floor = (self.settings.context_min_rerank_score if candidate.chunk_id is not None
                     else self.settings.graph_min_rerank_score)
            return scores[candidate.key] >= floor

        selected: dict[str, tuple[Candidate, float]] = {}
        for batch in batches:
            ranked = [(c, rerank_merge_score(c.score, scores[c.key], self.settings.rerank_weight)) for c in batch
                      if relevant(c)]
            ranked = sorted((p for p in ranked if p[1] > 0), key=lambda p: (-order(*p), p[0].key))
            # A query keeps its best chunks and its best graph elements, each kind up to rerank_top_k: elements carry
            # no source prior, so in one shared cut they outranked every penalised chunk of the query.
            for is_chunk in (True, False):
                for candidate, merged in [p for p in ranked if (p[0].chunk_id is not None) == is_chunk][
                    : self.settings.rerank_top_k
                ]:
                    previous = selected.get(candidate.key)
                    if previous is None or merged > previous[1]:
                        selected[candidate.key] = (candidate, merged)
        # Document chunks and graph elements have separate limits: in one shared list the short element
        # blocks pushed documents out of the context (on NSU v1, 5.9 elements and 4.5 documents per answer).
        ranked_all = sorted(selected.values(), key=lambda p: (-order(*p), p[0].key))
        kept = sorted(
            self._capped_chunks([p for p in ranked_all if p[0].chunk_id is not None])
            + [p for p in ranked_all if p[0].chunk_id is None][: self.settings.graph_max_elements],
            key=lambda p: (-order(*p), p[0].key),
        )
        kept_by_key = {candidate.key: (rank, merged) for rank, (candidate, merged) in enumerate(kept)}
        chunks = [(c.chunk_id, merged) for c, merged in kept if c.chunk_id is not None]
        elements = [dict(c.element, score=merged) for c, merged in kept if c.element is not None]
        graph_trace = {
            "per_query": [
                {"query": b["query"], "candidates": [self._trace_candidate(c) for c in b["pool"]]}
                for b in fused_batches
            ],
            "rerank": [
                {**self._trace_candidate(c), "rerank_score": scores[key], "kept": key in kept_by_key,
                 "rank": kept_by_key[key][0] if key in kept_by_key else None,
                 "merged_score": kept_by_key[key][1] if key in kept_by_key else None}
                for key, c in unique.items()
            ],
            "selected": [{**self._trace_candidate(c), "merged_score": merged} for c, merged in kept],
            "selected_elements": elements,
        }
        return GraphSelection(
            chunks,
            elements,
            scored=len(unique),
            candidate_scores={c.chunk_id: scores[c.key] for c in unique.values() if c.chunk_id is not None},
            graph_trace=graph_trace,
        )

    def _capped_chunks(self, ranked):
        """The best chunks for the context, at most ``max_context_chunks``, and at most ``rerank_candidates_per_page``
        of one page across all the queries (the per-query pools hold that cap each, so several queries could fill the
        context with one page)."""
        per_page = self.settings.rerank_candidates_per_page
        taken: dict[str, int] = {}
        kept = []
        for pair in ranked:
            if len(kept) >= self.settings.max_context_chunks:
                break
            if per_page > 0:
                page = self._page_of(pair[0].chunk_id)
                if page is not None:
                    if taken.get(page, 0) >= per_page:
                        continue
                    taken[page] = taken.get(page, 0) + 1
            kept.append(pair)
        return kept

    def _candidate_pool(self, pool):
        """The query's best candidates for the reranker, at most ``rerank_candidates_per_page`` chunks of one page."""
        cap = self.settings.rerank_candidates_per_query
        per_page = self.settings.rerank_candidates_per_page
        if per_page <= 0:
            return pool[:cap] if cap > 0 else list(pool)
        taken: dict[str, int] = {}
        selected = []
        for candidate in pool:
            if cap > 0 and len(selected) >= cap:
                break
            page = self._page_of(candidate.chunk_id) if candidate.chunk_id is not None else None
            if page is not None:
                if taken.get(page, 0) >= per_page:
                    continue
                taken[page] = taken.get(page, 0) + 1
            selected.append(candidate)
        return selected

    def _page_of(self, chunk_id):
        mapping = self.resources.chunk_mapping.get(str(chunk_id))
        if mapping is None:
            return None
        urls = normalize_urls(self.resources.documents[mapping["doc_index"]].get("url"))
        return norm_url(urls[0]) if urls else f"document:{mapping['doc_index']}"

    @staticmethod
    def _trace_candidate(candidate):
        return {
            "key": candidate.key,
            "score": candidate.score,
            "chunk_id": candidate.chunk_id,
            "element": candidate.element,
        }

    def _assemble_context(self, chunks, budget_override=None):
        budget = self.settings.max_qa_prompt_chars if budget_override is None else budget_override
        elements = getattr(chunks, "elements", [])
        blocks = []
        kept_elements = []
        graph_evidence = []
        capture_evidence = getattr(chunks, "capture_evidence", False)
        used = 0
        heading = "**ЭЛЕМЕНТЫ ГРАФА ЗНАНИЙ**"
        # An element names only documents retrieved for this question: the kept chunks first, then the other
        # chunks the reranker scored. Its full source list is in corpus order, so its first documents are
        # arbitrary for the question (a hub entity such as the university occurs in tens of thousands).
        retrieved: dict[int, int] = {}
        for chunk_id, _score in chunks:
            retrieved.setdefault(chunk_id, len(retrieved))
        for chunk_id, _score in sorted((getattr(chunks, "candidate_scores", None) or {}).items(), key=lambda kv: -kv[1]):
            retrieved.setdefault(chunk_id, len(retrieved))
        for element in elements:
            references = []
            documents = set()
            for chunk_id in sorted((c for c in element.get("sources", []) if c in retrieved), key=retrieved.get):
                mapping = self.resources.chunk_mapping.get(str(chunk_id))
                if mapping is None or mapping["doc_index"] in documents:
                    continue
                documents.add(mapping["doc_index"])
                references.append(chunk_id)
                if len(references) == self.settings.graph_source_documents:
                    break
            element = dict(element, sources=references)
            text = graph_element_to_text(element, self.resources.documents, self.resources.chunk_mapping)
            cost = len(text) + 2 + (len(heading) + 2 if not blocks else 0)
            if used + cost > budget:
                continue
            blocks.append(text)
            kept_elements.append(element)
            if capture_evidence:
                source_references = []
                source_rows = []
                for chunk_id in references:
                    mapping = self.resources.chunk_mapping[str(chunk_id)]
                    document = self.resources.documents[mapping["doc_index"]]
                    urls = normalize_urls(document.get("url"))
                    source_references.append({
                        "chunk_id": chunk_id, "document_index": mapping["doc_index"],
                        "local_chunk_index": mapping["local_chunk_index"],
                        "title": document.get("doc_title", ""), "urls": urls,
                    })
                    source_rows.extend({"document_title": document.get("doc_title", ""), "source_url": url}
                                       for url in urls or [""])
                graph_evidence.append({
                    "kind": element.get("kind", "entity"), "element_key": element.get("key"),
                    "text": text, "chunk_ids": list(references), "sources": source_rows,
                    "source_references": source_references,
                })
            used += cost
        graph_context = heading + "\n\n" + "\n\n".join(blocks) if blocks else ""
        remaining = max(0, budget - len(graph_context) - (2 if graph_context else 0))
        context, sources = super()._assemble_context(chunks, budget_override=remaining) if remaining else ("", [])
        if len(context) > remaining:
            context, sources = "", []
            if capture_evidence:
                chunks.final_evidence = []
        if capture_evidence:
            offset = (len(context) + 2 if context else 0) + len(heading) + 2
            for evidence in graph_evidence:
                evidence["context_start"] = offset
                evidence["context_end"] = offset + len(evidence["text"])
                offset = evidence["context_end"] + 2
            chunks.final_evidence = (chunks.final_evidence or []) + graph_evidence if remaining else graph_evidence
        if graph_context:
            context = "\n\n".join(p for p in (context, graph_context) if p)
        # Shown sources are the documents of the context. Element references are retrieved chunks, so their
        # documents are either in the context already or were judged less relevant than it.
        return context, sources

    def _stage_detail(self, stage_name, result):
        if stage_name == StageName.RETRIEVAL:
            return {
                "chunks_found": sum(len(b["graph"]["chunks"]) for b in result),
                "entities": sum(len(b["graph"]["entities"]) for b in result),
                "relations": sum(len(b["graph"]["relations"]) for b in result),
            }
        if stage_name == StageName.FUSION:
            return {"candidates": sum(len(b["pool"]) for b in result)}
        if stage_name == StageName.RERANK:
            return {"kept": len(result) + len(result.elements), "from": result.scored_candidates}
        return super()._stage_detail(stage_name, result)
