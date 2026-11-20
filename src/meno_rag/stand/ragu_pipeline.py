from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

from context_utils.context_utils import graph_element_to_text

from meno_rag.api.events import StageName
from meno_rag.stand.attribution import norm_url
from meno_rag.stand.context import graph_element_as_note, normalize_urls
from meno_rag.stand.pipeline import (
    ModelRuntime,
    StandRagPipeline,
    _RerankOutput,
    counting_rerank_calls,
    new_rerank_calls,
    rerank_timings,
)
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


class EarlyQuery:
    """The first search query's search and rerank scores of one turn, started while the rewrite call runs."""

    def __init__(self, query, dialogue_history, *, rerank_question=None, runtime=None):
        self.query = query
        self.rerank_question = query if rerank_question is None else rerank_question
        self.dialogue_history = dialogue_history
        self.runtime = runtime
        self.retrieval = None  # task: the retrieval batches of [query]
        self.started = None  # task: waits for the search, then starts the scoring tasks
        self.scores = {}  # candidate key -> scoring task
        self.calls = new_rerank_calls()

    async def close(self):
        """Cancel what is still running (an unused or failed turn) and wait for it."""
        pending = [t for t in (self.retrieval, self.started, *self.scores.values()) if t is not None and not t.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in (self.retrieval, self.started, *self.scores.values()):
            if task is not None and task.done() and not task.cancelled():
                task.exception()  # retrieved: an unused failure is not reported as "never retrieved"


class RaguRagPipeline(StandRagPipeline):
    def _priority(self, candidate: Candidate, merged: float) -> float:
        """The order of relevant candidates: the merged relevance times the quality of the source document to the
        power SOURCE_PRIOR_WEIGHT, in logs (ln merged + weight * ln quality), so a comparably relevant official page
        goes before a post (posts of the KB have quality about .55, NSU pages .73-.95) and a clearly better post
        stays ahead.

        Not the log-odds of the merged score (Menon 1.1's priority): the LLM labels of the stand saturate at 1.0 and
        the cross-encoder near .99, and the log-odds stretch that end, so .999 against .997 of the retrieval score
        outweighed the source (17.11, "Как поступить в НГУ?": seven posts kept, nsu.ru/n/education/apply/ with LLM
        score 1.0 left out). Graph elements have prior 1; weight 0 keeps the merged score itself."""
        weight = self.settings.source_prior_weight
        if weight <= 0:
            return merged
        return math.log(min(max(merged, 1e-6), 1.0)) + weight * math.log(self._source_quality(candidate))

    def _source_quality(self, candidate: Candidate) -> float:
        """The quality of the candidate's source document (KB scale, .05-1); 1 for graph elements."""
        if candidate.chunk_id is None:
            return 1.0
        mapping = self.resources.chunk_mapping.get(str(candidate.chunk_id))
        if mapping is None:
            return 1.0
        prior = float(self.resources.documents[mapping["doc_index"]].get("quality_score") or 1.0)
        return min(max(prior, 0.05), 1.0)

    async def _retrieve(self, search_queries, early=None):
        use_early = (early is not None and early.retrieval is not None
                     and bool(search_queries) and search_queries[0] == early.query)
        # The queries of one question run at once (still within embed_semaphore): the search mostly waits for the
        # embedder, Qdrant and the cross-encoder, and three rewrites used to wait for three searches in a row.
        async def one(query):
            if use_early and query == early.query:
                return (await early.retrieval)[0]
            async with self.embed_semaphore:
                return {"query": query, "graph": await self.resources.graph.search(query)}

        tasks = [asyncio.create_task(one(query)) for query in search_queries]
        try:
            return list(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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

    async def _score_candidate(self, candidate, user_question, dialogue_history, runtime: ModelRuntime):
        """The LLM rerank score of one candidate for the user question (not for the search query that found it)."""
        if candidate.chunk_id is not None:
            return await self._score_chunk_with_llm(user_question, dialogue_history, candidate.chunk_id, runtime)
        text = graph_element_to_text(candidate.element, self.resources.documents,
                                     self.resources.chunk_mapping, max_source_titles=0)
        return await self._score_document_with_llm(user_question, dialogue_history, text, runtime, candidate.key)

    def _start_early_query(self, question, dialogue_history, runtime: ModelRuntime, *, rerank_question=None):
        """Start the first retrieval during the rewrite; EARLY_RERANK additionally starts its LLM scores.

        The question is the first search query and its pool depends on its own search only, so later stages reuse
        these results unchanged. Retrieval alone leaves every LLM score to the normal rerank stage.
        """
        early = EarlyQuery(question, dialogue_history, rerank_question=rerank_question, runtime=runtime)
        early.retrieval = asyncio.create_task(self._retrieve([question]))
        if not self.settings.early_rerank:
            return early

        async def start_scores():
            pool = self._candidate_pool(self._fuse(await early.retrieval)[0]["pool"])
            if not self.settings.llm_rerank or not pool:
                return
            with counting_rerank_calls(early.calls):
                if self._rerank_warm_up(len(pool)):  # question-first prompts: the first one fills the cache
                    first = asyncio.create_task(self._score_candidate(
                        pool[0], early.rerank_question, dialogue_history, runtime))
                    early.scores[pool[0].key] = first
                    await first
                for candidate in pool:
                    if candidate.key not in early.scores:
                        early.scores[candidate.key] = asyncio.create_task(
                            self._score_candidate(candidate, early.rerank_question, dialogue_history, runtime))

        early.started = asyncio.create_task(start_scores())
        return early

    async def _rerank(self, fused_batches, user_question, dialogue_history, runtime: ModelRuntime, early=None):
        batches = [self._candidate_pool(batch["pool"]) for batch in fused_batches]
        unique = {c.key: c for batch in batches for c in batch}

        async def score(candidate):
            return await self._score_candidate(candidate, user_question, dialogue_history, runtime)

        use_early = (early is not None and early.started is not None and bool(fused_batches)
                     and fused_batches[0]["query"] == early.query and user_question == early.rerank_question
                     and dialogue_history == early.dialogue_history and runtime == early.runtime)
        calls = new_rerank_calls()
        if self.settings.llm_rerank:
            candidates = list(unique.values())
            tasks: list[asyncio.Task] = []
            try:
                if use_early:
                    await early.started  # the first query's scoring tasks exist from here on
                cached_scores = early.scores if use_early else {}
                if self._rerank_warm_up(len(candidates)):  # question-first prompts: the first one fills the cache
                    with counting_rerank_calls(calls):
                        tasks.append(cached_scores.get(candidates[0].key)
                                     or asyncio.create_task(score(candidates[0])))
                    await tasks[0]
                with counting_rerank_calls(calls):
                    tasks.extend(cached_scores.get(candidate.key) or asyncio.create_task(score(candidate))
                                 for candidate in candidates[len(tasks):])
                values = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # as Menon 1.1: the engine's cross-encoder score (bge-reranker) is the relevance, no LLM call
            values = [float(candidate.score) for candidate in unique.values()]
        if use_early:
            for name, value in early.calls.items():
                calls[name] = calls.get(name, 0) + value
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
        element_pairs = [p for p in ranked_all if p[0].chunk_id is None]
        if self.settings.graph_element_evidence:
            # an element enters the context only with evidence: one of the chunks it was extracted from was
            # retrieved for the question, so its description can be cited; the slot goes to the next element
            retrieved = {c.chunk_id for c in unique.values() if c.chunk_id is not None}
            element_pairs = [p for p in element_pairs
                             if retrieved.intersection((p[0].element or {}).get("sources") or [])]
        kept = sorted(
            self._capped_chunks([p for p in ranked_all if p[0].chunk_id is not None])
            + element_pairs[: self.settings.graph_max_elements],
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
        selection = GraphSelection(
            chunks,
            elements,
            scored=len(unique),
            candidate_scores={c.chunk_id: scores[c.key] for c in unique.values() if c.chunk_id is not None},
            graph_trace=graph_trace,
        )
        selection.llm_calls = calls
        return selection

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
        sources_style = self.settings.context_style == "sources"
        heading = "**СПРАВКИ ПО МАТЕРИАЛАМ**" if sources_style else "**ЭЛЕМЕНТЫ ГРАФА ЗНАНИЙ**"
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
            if not references and self.settings.graph_element_evidence:
                continue  # no retrieved chunk backs it: its description could not be cited or checked
            element = dict(element, sources=references)
            if sources_style:
                text = graph_element_as_note(element, self.resources.documents, self.resources.chunk_mapping,
                                             chunk_addresses=self.resources.chunk_addresses or None)
            else:
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
                    for address in self.resources.chunk_addresses.get(str(chunk_id), ()):
                        other = self.resources.documents[address["doc_index"]]
                        other_urls = [url for url in normalize_urls(other.get("url")) if url not in urls]
                        if not other_urls:
                            continue
                        source_references.append({
                            "chunk_id": chunk_id, "document_index": address["doc_index"],
                            "local_chunk_index": address["local_chunk_index"],
                            "title": other.get("doc_title", ""), "urls": other_urls,
                        })
                        source_rows.extend({"document_title": other.get("doc_title", ""), "source_url": url}
                                           for url in other_urls)
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
            detail: dict[str, Any] = {
                "chunks_found": sum(len(b["graph"]["chunks"]) for b in result),
                "entities": sum(len(b["graph"]["entities"]) for b in result),
                "relations": sum(len(b["graph"]["relations"]) for b in result),
            }
            # where each query's search spent its time (graph_utils.search_timings), in the order of the queries
            per_query = [b["graph"].get("timings") for b in result]
            if any(timings is not None for timings in per_query):
                detail["timings"] = {"per_query": per_query}
            return detail
        if stage_name == StageName.FUSION:
            return {"candidates": sum(len(b["pool"]) for b in result)}
        if stage_name == StageName.RERANK:
            detail = {"kept": len(result) + len(result.elements), "from": result.scored_candidates}
            if getattr(result, "llm_calls", None) is not None:
                detail["timings"] = rerank_timings(result.llm_calls)
            return detail
        return super()._stage_detail(stage_name, result)
