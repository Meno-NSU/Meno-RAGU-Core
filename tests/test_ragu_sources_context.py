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
    outcome = await _ask(_pipeline(Graph(), context_style="legacy"))
    assert "Посторонний" not in outcome.context
    assert "**Встречается во фрагментах документов:** Документ А." in outcome.context


@pytest.mark.asyncio
async def test_note_references_come_from_the_retrieved_chunks():
    outcome = await _ask(_pipeline(Graph()))
    assert "Посторонний" not in outcome.context
    assert "По материалам: «Документ А»." in outcome.context


@pytest.mark.asyncio
async def test_element_without_retrieved_references_names_no_documents():
    # the earlier mode (GRAPH_ELEMENT_EVIDENCE=false): the element is shown, without documents it cannot name
    outcome = await _ask(_pipeline(Graph(chunks=((1, 0.5),)), graph_element_evidence=False))
    assert "Факт 0 о ФИТ" in outcome.context
    assert "Встречается во фрагментах документов" not in outcome.context
    assert [s["document_title"] for s in outcome.sources] == ["Документ Б"]


@pytest.mark.asyncio
async def test_element_without_retrieved_references_stays_out_of_the_context():
    outcome = await _ask(_pipeline(Graph(chunks=((1, 0.5),))))
    assert "Факт 0 о ФИТ" not in outcome.context
    assert [s["document_title"] for s in outcome.sources] == ["Документ Б"]


@pytest.mark.asyncio
async def test_graph_elements_do_not_take_document_slots():
    p = _pipeline(Graph(n_entities=5, chunks=((0, 0.5), (1, 0.4))), max_context_chunks=2, graph_max_elements=1,
                  context_style="legacy")
    outcome = await _ask(p)
    assert {s["document_title"] for s in outcome.sources} == {"Документ А", "Документ Б"}
    assert outcome.context.count("**Элемент графа знаний") == 1


@pytest.mark.asyncio
async def test_notes_do_not_take_document_slots():
    p = _pipeline(Graph(n_entities=5, chunks=((0, 0.5), (1, 0.4))), max_context_chunks=2, graph_max_elements=1)
    outcome = await _ask(p)
    assert {s["document_title"] for s in outcome.sources} == {"Документ А", "Документ Б"}
    assert outcome.context.count("\nПо материалам: ") == 1


@pytest.mark.asyncio
async def test_graph_elements_can_be_switched_off():
    outcome = await _ask(_pipeline(Graph(n_entities=3), graph_max_elements=0))
    assert "Элемент графа знаний" not in outcome.context
    assert [s["document_title"] for s in outcome.sources] == ["Документ А"]


class Relations(Graph):
    async def search(self, query):
        found = await super().search(query)
        found["relations"] = [{"key": f"r-{query}", "kind": "relation", "subject": "Сущность 0", "object": "НГУ",
                               "relation_type": "PART_OF", "description": "Сущность 0 входит в НГУ.", "score": 0.9,
                               "sources": [0]}]
        return found


@pytest.mark.asyncio
async def test_sources_style_shows_the_context_as_sources_without_internal_labels():
    """CONTEXT_STYLE=sources: the prompt holds sources and notes, not numbered DOCUMENT sections, «элементы графа
    знаний» or ontology types; the model echoed those to users («в документе 12», «согласно графу знаний»)."""
    outcome = await _ask(_pipeline(Relations(), context_style="sources", graph_max_elements=3))
    prompt = outcome.qa_messages[-1]["content"]
    for internal in ("DOCUMENT", "ГРАФ", "граф", "(тип:", "—[", "<Фрагмент 1>", "Элемент"):
        assert internal not in prompt, internal
    assert "==========\nИСТОЧНИК\n==========" in prompt and "**Название документа:** Документ А." in prompt
    assert "**СПРАВКИ ПО МАТЕРИАЛАМ**" in outcome.context
    assert "**Сущность 0.** Факт 0 о ФИТ.\nПо материалам: «Документ А»." in outcome.context
    assert "**Сущность 0 — НГУ.** Сущность 0 входит в НГУ.\nПо материалам: «Документ А»." in outcome.context
    assert "опираясь только на факты из разделов ИСТОЧНИК и СПРАВКИ ПО МАТЕРИАЛАМ." in prompt


@pytest.mark.asyncio
async def test_the_sources_style_is_the_default():
    """Adopted on 18.11 (docs/root-fixes-2026-11-18): label leaks of the answers fell from 10.5% to 0.9% on 1271
    paired questions, with the judge's vital recall and contradictions unchanged on the same retrieval."""
    outcome = await _ask(_pipeline(Graph()))
    prompt = outcome.qa_messages[-1]["content"]
    assert "==========\nИСТОЧНИК\n==========" in prompt and "DOCUMENT" not in prompt


@pytest.mark.asyncio
async def test_the_legacy_style_can_still_be_chosen():
    outcome = await _ask(_pipeline(Graph(), context_style="legacy"))
    prompt = outcome.qa_messages[-1]["content"]
    assert "==========\nDOCUMENT 1\n==========" in prompt and "**ЭЛЕМЕНТЫ ГРАФА ЗНАНИЙ**" in outcome.context
