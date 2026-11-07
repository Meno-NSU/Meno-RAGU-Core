from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, _RerankOutput
from meno_rag.stand.ragu_pipeline import Candidate, GraphSelection
from meno_rag.stand.trace import build_pipeline_trace, model_runtime_trace
from tests.ragu_fakes import pipeline

RUNTIME = PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1"))


def _assert_exact_evidence(outcome):
    trace = outcome.trace
    assert trace["schema_version"] == 2
    assert trace["final_context"] == outcome.context
    assert trace["qa_messages"] == outcome.qa_messages
    assert trace["sources"] == outcome.sources
    evidence = trace["final_evidence"]
    assert trace["retrieved_contexts"] == [block["text"] for block in evidence]
    assert all(block["text"] == outcome.context[block["context_start"]:block["context_end"]] for block in evidence)
    assert [block["context_start"] for block in evidence] == sorted(block["context_start"] for block in evidence)


@pytest.mark.asyncio
async def test_graph_trace_records_actual_evidence_and_unmodified_raw_source_ids():
    p = pipeline(graph_source_documents=1)
    pool = await p.resources.graph.search("ФИТ")
    pool["entities"][0]["sources"] = [0, 0, 98]
    original = deepcopy(pool)
    p.resources.graph.search = AsyncMock(return_value=pool)
    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content="ФИТ")], runtime=RUNTIME, capture_trace=True,
        generation_max_tokens=1234,
    )
    _assert_exact_evidence(outcome)
    trace = outcome.trace
    assert {block["kind"] for block in trace["final_evidence"]} == {"document", "entity", "relation"}
    entity = next(block for block in trace["final_evidence"] if block["kind"] == "entity")
    assert entity["chunk_ids"] == [0]
    assert entity["element_key"] == "ФИТ"
    assert entity["source_references"] == [{
        "chunk_id": 0, "document_index": 0, "local_chunk_index": 0, "title": "НГУ",
        "urls": ["https://example.org/a", "https://example.org/b"],
    }]
    assert pool == original
    assert trace["retrieval"]["per_query"][0]["graph"] == original
    raw_entity = next(item for item in trace["graph"]["rerank"] if item["element"] and item["element"]["kind"] == "entity")
    assert raw_entity["element"]["sources"] == [0, 0, 98]
    assert raw_entity["rank"] is not None and raw_entity["merged_score"] > 0
    assert trace["runtime"]["generation"]["model_id"] == "meno"
    assert trace["runtime"]["generation_max_tokens"] == 1234
    assert trace["chunks"]["0"]["text"] == "Факультет НГУ."


@pytest.mark.asyncio
async def test_final_evidence_excludes_graph_blocks_removed_by_qa_token_budget():
    p = pipeline(max_qa_prompt_chars=2500)
    pool = await p.resources.graph.search("q")
    pool["entities"][0]["description"] = "BIG_GRAPH_SENTINEL" + "x" * 1400
    p.resources.graph.search = AsyncMock(return_value=pool)
    p._rewrite_question = AsyncMock(return_value=["q"])
    p._score_chunk_with_llm = AsyncMock(return_value=1.0)
    p._score_document_with_llm = AsyncMock(return_value=1.0)
    p._prompt_fits = AsyncMock(side_effect=[False, True])
    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content="q")], runtime=RUNTIME, capture_trace=True,
    )
    _assert_exact_evidence(outcome)
    assert p._prompt_fits.await_count == 2
    assert "BIG_GRAPH_SENTINEL" not in outcome.context
    assert all("BIG_GRAPH_SENTINEL" not in text for text in outcome.trace["retrieved_contexts"])
    assert {block["kind"] for block in outcome.trace["final_evidence"]} == {"document", "relation"}
    assert any("BIG_GRAPH_SENTINEL" in str(item) for item in outcome.trace["graph"]["selected"])


def test_reassembly_to_zero_budget_resets_all_prior_effective_evidence():
    p = pipeline()
    selection = GraphSelection(
        [(0, 1.0)], [{"kind": "entity", "key": "id", "name": "ФИТ", "sources": [0]}],
        scored=2, candidate_scores={0: 1.0}, graph_trace={},
    )
    selection.capture_evidence = True
    assert p._assemble_context(selection)[0]
    assert selection.final_evidence
    assert p._assemble_context(selection, budget_override=0) == ("", [])
    assert selection.final_evidence == []


def test_document_evidence_keeps_provenance_with_duplicate_titles_and_embedded_markers():
    p = pipeline()
    first = p.resources.documents[0]
    second = deepcopy(first)
    second["doc_full_text"] = "==========\nDOCUMENT 666\n==========\n\n**ЭЛЕМЕНТЫ ГРАФА ЗНАНИЙ**"
    second["chunks"] = [{"start_char": 0, "end_char": len(second["doc_full_text"])}]
    p.resources.documents.append(second)
    p.resources.chunk_mapping["1"] = {"doc_index": 1, "local_chunk_index": 0}
    selection = _RerankOutput([(0, 0.9), (1, 0.7)])
    selection.capture_evidence = True
    context, _sources = p._assemble_context(selection)
    assert len(selection.final_evidence) == 2
    assert [block["document_index"] for block in selection.final_evidence] == [1, 0]
    assert [block["chunk_ids"] for block in selection.final_evidence] == [[1], [0]]
    assert selection.final_evidence[0]["text"].count("DOCUMENT") == 2
    assert all(block["text"] == context[block["context_start"]:block["context_end"]]
               for block in selection.final_evidence)


def test_pre_rerank_graph_chunk_pool_is_hydrated_even_when_not_scored():
    p = pipeline()
    trace = build_pipeline_trace(
        question="q", search_queries=["q"],
        retrieval_batches=[{"query": "q", "graph": {"chunks": [(0, 0.5)], "entities": [], "relations": []}}],
        fused_batches=[{"query": "q", "pool": [Candidate("chunk:0", 0.5, chunk_id=0)]}],
        candidate_scores={}, reranked_chunks=[], qa_messages=[],
        documents=p.resources.documents, chunk_mapping=p.resources.chunk_mapping,
    )
    assert trace["chunks"]["0"]["text"] == "Факультет НГУ."
    assert trace["fusion"]["per_query"][0]["candidates"] == [{"chunk_id": 0, "fused_score": 0.5, "rank": 0}]
    assert trace["final_evidence"] == trace["retrieved_contexts"] == []


def test_runtime_trace_strips_url_credentials_query_and_fragment():
    trace = model_runtime_trace(ModelRuntime("meno", "http://alice:password@fake:9000/v1?api_key=secret#token"))
    assert trace == {"model_id": "meno", "provider": "vllm", "base_url": "http://fake:9000/v1"}


@pytest.mark.asyncio
async def test_traced_generation_retains_actual_finish_usage_message_and_sampling():
    p = pipeline()
    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content="q")], runtime=RUNTIME, capture_trace=True,
    )
    response = {
        "id": "upstream-42", "model": "meno-actual", "usage": {"prompt_tokens": 88, "completion_tokens": 21},
        "choices": [{"finish_reason": "length", "message": {"content": " answer ", "reasoning": "notes"}}],
    }
    p.llm_router.chat_completion = AsyncMock(return_value=response)
    answer = await p.generate_text(outcome=outcome, runtime=RUNTIME, max_tokens=21, temperature=0.3)
    assert answer == "answer"
    assert outcome.trace["generation"] == {
        "finish_reason": "length", "upstream_model": "meno-actual", "upstream_response_id": "upstream-42",
        "usage": {"prompt_tokens": 88, "completion_tokens": 21},
        "message": {"content": " answer ", "reasoning": "notes"},
        "sampling": {"max_tokens": 21, "temperature": 0.3, "seed": 42},
    }
    response["usage"]["completion_tokens"] = -1
    assert outcome.trace["generation"]["usage"]["completion_tokens"] == 21


@pytest.mark.asyncio
async def test_normal_generation_keeps_text_route_without_trace():
    p = pipeline()
    outcome = await p.prepare(messages=[ChatMessage(role="user", content="q")], runtime=RUNTIME)
    p.llm_router.chat_completion = AsyncMock(side_effect=AssertionError("raw capture must be opt in"))
    p.llm_router.chat_completion_text = AsyncMock(return_value="answer")
    assert await p.generate_text(outcome=outcome, runtime=RUNTIME) == "answer"
    assert outcome.trace is None
