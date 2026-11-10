from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from context_utils.context_utils import graph_element_to_text

from meno_rag.api.events import StageName
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
        cap = self.settings.rerank_candidates_per_query
        batches = [batch["pool"][:cap] if cap > 0 else batch["pool"] for batch in fused_batches]
        unique = {c.key: c for batch in batches for c in batch}

        async def score(candidate):
            if candidate.chunk_id is not None:
                return await self._score_chunk_with_llm(user_question, dialogue_history, candidate.chunk_id, runtime)
            text = graph_element_to_text(candidate.element, self.resources.documents,
                                         self.resources.chunk_mapping, max_source_titles=0)
            return await self._score_document_with_llm(user_question, dialogue_history, text, runtime, candidate.key)

        tasks = [asyncio.create_task(score(candidate)) for candidate in unique.values()]
        try:
            values = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        scores = dict(zip(unique, values, strict=True))
        selected: dict[str, tuple[Candidate, float]] = {}
        for batch in batches:
            ranked = [(c, rerank_merge_score(c.score, scores[c.key], self.settings.rerank_weight)) for c in batch]
            ranked = sorted((p for p in ranked if p[1] > 0), key=lambda p: (-p[1], p[0].key))
            for candidate, merged in ranked[: self.settings.rerank_top_k]:
                previous = selected.get(candidate.key)
                if previous is None or merged > previous[1]:
                    selected[candidate.key] = (candidate, merged)
        # Document chunks and graph elements have separate limits: in one shared list the short element
        # blocks pushed documents out of the context (on NSU v1, 5.9 elements and 4.5 documents per answer).
        ranked_all = sorted(selected.values(), key=lambda p: (-p[1], p[0].key))
        kept = sorted(
            [p for p in ranked_all if p[0].chunk_id is not None][: self.settings.max_context_chunks]
            + [p for p in ranked_all if p[0].chunk_id is None][: self.settings.graph_max_elements],
            key=lambda p: (-p[1], p[0].key),
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
