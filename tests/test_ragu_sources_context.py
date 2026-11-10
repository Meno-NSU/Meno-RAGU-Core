"""Shown sources, references of graph elements and the context share of graph elements (RAGU pipeline)."""
import asyncio

import pytest
from nltk.stem.snowball import SnowballStemmer

from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.ragu_pipeline import RaguRagPipeline
from meno_rag.stand.resources import StandResources
from tests.ragu_fakes import LLM


def _doc(text, title, url):
    return {"doc_full_text": text, "doc_title": title, "doc_annotation": "", "url": url, "quality_score": 1.0,
            "chunks": [{"start_char": 0, "end_char": len(text)}]}


DOCS = [
    _doc("Учебные дисциплины преподавателя ФИТ.", "Документ А", "https://a.test"),
    _doc("Расписание занятий преподавателя ФИТ.", "Документ Б", "https://b.test"),
    _doc("Посторонний текст о другом.", "Посторонний", "https://z.test"),
]
MAPPING = {str(i): {"doc_index": i, "local_chunk_index": 0} for i in range(len(DOCS))}


class Graph:
    """Entities mention chunk 2 (never retrieved) before chunk 0 (retrieved), as corpus order does."""

    def __init__(self, n_entities=1, chunks=((0, 0.5),)):
        self.n_entities = n_entities
        self.chunks = list(chunks)

    async def search(self, query):
        await asyncio.sleep(0)
        return {
            "chunks": list(self.chunks),
            "entities": [
                {"key": f"e{i}-{query}", "kind": "entity", "name": f"Сущность {i}", "type": "ORG",
                 "description": f"Факт {i} о {query}", "score": 1.0 / (i + 1), "sources": [0, 2]}
                for i in range(self.n_entities)
            ],
            "relations": [],
        }


def _pipeline(graph, **settings):
    resources = StandResources(documents=DOCS, chunk_mapping=MAPPING, graph=graph,
                               stemmer=SnowballStemmer("russian"), abbreviations={})
    return RaguRagPipeline(
        settings=Settings(_env_file=None, max_rewrite_queries=1, **settings),
        resources=resources,
        llm_router=LLM(),
        rewrite_semaphore=asyncio.Semaphore(2),
        rerank_semaphore=asyncio.Semaphore(2),
        generation_semaphore=asyncio.Semaphore(2),
        embed_semaphore=asyncio.Semaphore(2),
    )


async def _ask(p, question="ФИТ"):
    return await p.prepare(
        messages=[ChatMessage(role="user", content=question)],
        runtime=PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1")),
        capture_trace=True,
    )


@pytest.mark.asyncio
async def test_shown_sources_are_the_documents_of_the_context():
    outcome = await _ask(_pipeline(Graph()))
    assert "Факт 0 о ФИТ" in outcome.context
    assert [s["document_title"] for s in outcome.sources] == ["Документ А"]


@pytest.mark.asyncio
async def test_element_references_come_from_the_retrieved_chunks():
    outcome = await _ask(_pipeline(Graph()))
    assert "Посторонний" not in outcome.context
    assert "**Встречается во фрагментах документов:** Документ А." in outcome.context


@pytest.mark.asyncio
async def test_element_without_retrieved_references_names_no_documents():
    outcome = await _ask(_pipeline(Graph(chunks=((1, 0.5),))))
    assert "Факт 0 о ФИТ" in outcome.context
    assert "Встречается во фрагментах документов" not in outcome.context
    assert [s["document_title"] for s in outcome.sources] == ["Документ Б"]


@pytest.mark.asyncio
async def test_graph_elements_do_not_take_document_slots():
    p = _pipeline(Graph(n_entities=5, chunks=((0, 0.5), (1, 0.4))), max_context_chunks=2, graph_max_elements=1)
    outcome = await _ask(p)
    assert {s["document_title"] for s in outcome.sources} == {"Документ А", "Документ Б"}
    assert outcome.context.count("**Элемент графа знаний") == 1


@pytest.mark.asyncio
async def test_graph_elements_can_be_switched_off():
    outcome = await _ask(_pipeline(Graph(n_entities=3), graph_max_elements=0))
    assert "Элемент графа знаний" not in outcome.context
    assert [s["document_title"] for s in outcome.sources] == ["Документ А"]
