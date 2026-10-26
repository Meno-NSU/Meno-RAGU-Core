import asyncio

import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.resources import _validate_mapping
from tests.ragu_fakes import pipeline


@pytest.mark.asyncio
async def test_graph_elements_reranked_and_grounded_with_all_source_urls():
    p = pipeline()
    events = []

    async def sink(event):
        events.append(event)

    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content="ФИТ")],
        runtime=PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1")),
        stage_sink=sink,
        capture_trace=True,
    )
    assert "ЭЛЕМЕНТЫ ГРАФА ЗНАНИЙ" in outcome.context
    assert "Факт о ФИТ" in outcome.context
    assert {s["source_url"] for s in outcome.sources} == {"https://example.org/a", "https://example.org/b"}
    assert len(outcome.sources) == 2
    assert outcome.stage_details["rerank"] == {"kept": 3, "from": 3}
    assert outcome.trace["graph"]["selected_elements"]
    assert any("Элемент графа знаний" in str(messages) for messages in p.llm_router.prompts)
    assert events


@pytest.mark.asyncio
async def test_graph_only_context_is_a_valid_qa_prompt():
    p = pipeline()
    pool = await p.resources.graph.search("ФИТ")
    pool["chunks"] = []
    p.resources.graph.search = lambda query: asyncio.sleep(0, result=pool)
    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content="ФИТ")],
        runtime=PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1")),
    )
    assert "Факт о ФИТ" in outcome.qa_messages[1]["content"]
    assert outcome.sources


@pytest.mark.asyncio
async def test_concurrent_requests_keep_graph_context_separate():
    p = pipeline()
    runtime = PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1"))
    a, b = await asyncio.gather(
        *(p.prepare(messages=[ChatMessage(role="user", content=q)], runtime=runtime) for q in ("ФИТ", "ММФ"))
    )
    assert "Факт о ФИТ" in a.context and "Факт о ММФ" not in a.context
    assert "Факт о ММФ" in b.context and "Факт о ФИТ" not in b.context


@pytest.mark.asyncio
async def test_graph_search_concurrency_is_bounded():
    p = pipeline()
    p.embed_semaphore = asyncio.Semaphore(1)
    state = {"active": 0, "maximum": 0}
    original = p.resources.graph.search

    async def bounded(query):
        state["active"] += 1
        state["maximum"] = max(state["maximum"], state["active"])
        await asyncio.sleep(0.005)
        result = await original(query)
        state["active"] -= 1
        return result

    p.resources.graph.search = bounded
    await asyncio.gather(*(p._retrieve([str(i)]) for i in range(6)))
    assert state["maximum"] == 1


def test_corrupt_mapping_fails_before_network_access():
    with pytest.raises(ValueError):
        _validate_mapping({"0": {"doc_index": -1, "local_chunk_index": 0}}, [{}])


def test_context_budget_drops_oversize_graph_and_its_sources():
    p = pipeline(max_qa_prompt_chars=20)
    from meno_rag.stand.ragu_pipeline import GraphSelection

    selection = GraphSelection(
        [], [{"kind": "entity", "name": "X" * 200, "sources": [0]}], scored=1, candidate_scores={}, graph_trace={}
    )
    context, sources = p._assemble_context(selection)
    assert len(context) <= 20
    assert sources == []
