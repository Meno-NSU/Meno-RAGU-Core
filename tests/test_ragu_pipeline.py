import asyncio

import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.resources import _validate_mapping
from tests.ragu_fakes import pipeline


@pytest.mark.asyncio
async def test_graph_elements_reranked_and_grounded_with_the_document_link():
    p = pipeline(context_style="legacy")
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
    # one link per document (SOURCE_URLS_PER_DOCUMENT=1): the document has two
    assert outcome.sources == [{"document_title": "НГУ", "source_url": "https://example.org/a"}]
    rerank = dict(outcome.stage_details["rerank"])
    assert rerank.pop("timings")["llm_calls"] == 3  # sub-stage timings (tests/test_substage_timings.py)
    assert rerank == {"kept": 3, "from": 3}
    assert outcome.trace["graph"]["selected_elements"]
    assert any("Элемент графа знаний" in str(messages) for messages in p.llm_router.prompts)
    assert events


@pytest.mark.asyncio
async def test_graph_elements_are_notes_with_their_documents_in_the_sources_style():
    p = pipeline()
    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content="ФИТ")],
        runtime=PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1")),
        capture_trace=True,
    )
    assert "**СПРАВКИ ПО МАТЕРИАЛАМ**" in outcome.context and "ЭЛЕМЕНТЫ ГРАФА ЗНАНИЙ" not in outcome.context
    assert "**ФИТ.** Факт о ФИТ.\nПо материалам: «НГУ»." in outcome.context
    assert outcome.sources == [{"document_title": "НГУ", "source_url": "https://example.org/a"}]
    # the LLM rerank still sees the elements as they are: the style is the answer prompt's only
    assert any("Элемент графа знаний" in str(messages) for messages in p.llm_router.prompts)


@pytest.mark.asyncio
async def test_graph_only_context_is_a_valid_qa_prompt():
    # the earlier mode, where an element enters the context without a retrieved source (GRAPH_ELEMENT_EVIDENCE=false)
    p = pipeline(graph_element_evidence=False)
    pool = await p.resources.graph.search("ФИТ")
    pool["chunks"] = []
    p.resources.graph.search = lambda query: asyncio.sleep(0, result=pool)
    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content="ФИТ")],
        runtime=PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1")),
    )
    assert "Факт о ФИТ" in outcome.qa_messages[1]["content"]
    # no document was retrieved, so the elements name none and no source is shown
    assert outcome.sources == []
    assert "Встречается во фрагментах документов" not in outcome.qa_messages[1]["content"]


@pytest.mark.asyncio
async def test_without_retrieved_chunks_no_graph_element_has_evidence():
    p = pipeline()
    pool = await p.resources.graph.search("ФИТ")
    pool["chunks"] = []
    p.resources.graph.search = lambda query: asyncio.sleep(0, result=pool)
    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content="ФИТ")],
        runtime=PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1")),
    )
    assert "Факт о ФИТ" not in outcome.context


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


@pytest.mark.asyncio
async def test_the_search_queries_of_one_question_run_at_once_and_keep_their_order():
    # They used to run one after another: a question with three rewrites waited for three searches in a row.
    p = pipeline()
    p.embed_semaphore = asyncio.Semaphore(8)
    state = {"active": 0, "maximum": 0}
    original = p.resources.graph.search

    async def slow(query):
        state["active"] += 1
        state["maximum"] = max(state["maximum"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return await original(query)

    p.resources.graph.search = slow
    batches = await p._retrieve(["ФИТ", "ММФ", "ФИТ НГУ"])
    assert [b["query"] for b in batches] == ["ФИТ", "ММФ", "ФИТ НГУ"]
    assert state["maximum"] == 3


@pytest.mark.asyncio
async def test_a_short_prompt_is_not_sent_to_the_tokenizer_once_the_context_length_is_known():
    # A token is at least one character, so a prompt shorter than the context in characters always fits; the
    # tokenizer round trip (one per reranked candidate) is needed only near the limit.
    p = pipeline()
    calls = []

    async def tokenize(*, runtime, messages, chat_template_kwargs=None):
        calls.append(sum(len(m["content"]) for m in messages))
        return sum(len(m["content"]) for m in messages), 1000

    p.llm_router.tokenize_messages = tokenize
    runtime = ModelRuntime("meno", "http://fake/v1")
    short = [{"role": "user", "content": "а" * 100}]
    long = [{"role": "user", "content": "а" * 990}]
    assert await p._prompt_fits(short, runtime, 20)  # first call learns the context length
    assert await p._prompt_fits(short, runtime, 20)
    assert len(calls) == 1
    assert not await p._prompt_fits(long, runtime, 20)  # near the limit the tokenizer still decides
    assert len(calls) == 2
