import asyncio

import pytest
from nltk.stem.snowball import SnowballStemmer

from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.ragu_pipeline import RaguRagPipeline
from meno_rag.stand.resources import StandResources, _validate_mapping


class Graph:
    async def search(self, query):
        await asyncio.sleep(0)
        return {
            "chunks": [(0, 0.5)],
            "entities": [
                {
                    "key": query,
                    "kind": "entity",
                    "name": query,
                    "type": "ORG",
                    "description": f"Факт о {query}",
                    "score": 1.0,
                    "sources": [0],
                }
            ],
            "relations": [
                {
                    "key": query,
                    "kind": "relation",
                    "subject": query,
                    "object": "НГУ",
                    "relation_type": "часть",
                    "description": "Связь",
                    "score": 0.7,
                    "sources": [0],
                }
            ],
        }


class LLM:
    def __init__(self):
        self.prompts = []

    async def chat_completion_text(self, **kwargs):
        return "поисковый запрос"

    async def chat_completion(self, **kwargs):
        self.prompts.append(kwargs["messages"])
        return {
            "choices": [
                {
                    "message": {"content": "2"},
                    "logprobs": {
                        "content": [{"token": "2", "logprob": 0.0, "top_logprobs": [{"token": "2", "logprob": 0.0}]}]
                    },
                }
            ]
        }


def pipeline(**overrides):
    text = "Факультет НГУ."
    resources = StandResources(
        documents=[
            {
                "doc_full_text": text,
                "doc_title": "НГУ",
                "doc_annotation": "",
                "url": ["https://example.org/a", "https://example.org/b"],
                "quality_score": 1.0,
                "chunks": [{"start_char": 0, "end_char": len(text)}],
            }
        ],
        chunk_mapping={"0": {"doc_index": 0, "local_chunk_index": 0}},
        graph=Graph(),
        stemmer=SnowballStemmer("russian"),
        abbreviations={},
    )
    return RaguRagPipeline(
        settings=Settings(_env_file=None, max_rewrite_queries=1, **overrides),
        resources=resources,
        llm_router=LLM(),
        rewrite_semaphore=asyncio.Semaphore(2),
        rerank_semaphore=asyncio.Semaphore(2),
        generation_semaphore=asyncio.Semaphore(2),
        embed_semaphore=asyncio.Semaphore(2),
    )


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
