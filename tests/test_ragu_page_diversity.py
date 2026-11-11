import asyncio

import pytest

from meno_rag.stand.attribution import page_titles
from meno_rag.stand.ragu_pipeline import Candidate
from tests.ragu_fakes import pipeline

TIMETABLE = "https://table.nsu.ru/teacher/1"


def document(title, url, text):
    return {"doc_full_text": text, "doc_title": title, "doc_annotation": "", "url": url, "quality_score": 1.0,
            "chunks": [{"start_char": 0, "end_char": len(text)}]}


def split_record():
    parts = [document(f"Публикация {i} Иванова", TIMETABLE, f"# Иванов Иван Иванович\n\n## Публикации\n\nСтатья {i}.")
             for i in range(10)]
    others = [document("Кафедра физики", "https://www.nsu.ru/n/physics/", "Иванов читает курс физики."),
              document("Программа магистратуры", "https://education.nsu.ru/mlbd/", "Курс ведёт Иванов.")]
    return parts + others


def test_parts_of_one_page_are_titled_by_their_common_heading():
    titles = page_titles([
        *split_record(),
        document("Новость 1", "https://www.nsu.ru/n/news/", "Первая часть новости."),
        document("Новость 2", "https://www.nsu.ru/n/news", "Вторая часть новости."),
    ])
    assert titles == {"table.nsu.ru/teacher/1": "Иванов Иван Иванович", "nsu.ru/n/news": "Новость 1"}


def test_a_page_takes_a_limited_share_of_each_query_pool():
    p = pipeline(rerank_candidates_per_query=6, rerank_candidates_per_page=2)
    documents = split_record()
    p.resources.documents[:] = documents
    p.resources.chunk_mapping.clear()
    p.resources.chunk_mapping.update({str(i): {"doc_index": i, "local_chunk_index": 0} for i in range(len(documents))})
    pool = [Candidate(f"chunk:{i}", 1.0 - i / 100, chunk_id=i) for i in range(len(documents))]
    pool.append(Candidate("entity:x", 0.5, element={"kind": "entity", "key": "x", "name": "x", "sources": []}))
    scored = []

    async def score(question, history, chunk_id, runtime):
        scored.append(chunk_id)
        return 2

    p._score_chunk_with_llm = score

    async def element_score(*args):
        return 2

    p._score_document_with_llm = element_score
    asyncio.run(p._rerank([{"query": "q", "pool": pool}], "q", "", None))
    assert sorted(scored) == [0, 1, 10, 11]


def test_without_a_page_limit_the_pool_is_cut_by_score_only():
    p = pipeline(rerank_candidates_per_query=3, rerank_candidates_per_page=0)
    documents = split_record()
    p.resources.documents[:] = documents
    p.resources.chunk_mapping.clear()
    p.resources.chunk_mapping.update({str(i): {"doc_index": i, "local_chunk_index": 0} for i in range(len(documents))})
    scored = []

    async def score(question, history, chunk_id, runtime):
        scored.append(chunk_id)
        return 2

    p._score_chunk_with_llm = score
    pool = [Candidate(f"chunk:{i}", 1.0 - i / 100, chunk_id=i) for i in range(len(documents))]
    asyncio.run(p._rerank([{"query": "q", "pool": pool}], "q", "", None))
    assert sorted(scored) == [0, 1, 2]


@pytest.mark.asyncio
async def test_sources_of_a_split_page_carry_the_page_title():
    async def scorer(sentence, texts):
        return [0.9] * len(texts)

    p = pipeline(source_scorer=scorer)
    documents = split_record()
    p.resources.documents[:] = documents
    p.resources.chunk_mapping.clear()
    p.resources.chunk_mapping.update({str(i): {"doc_index": i, "local_chunk_index": 0} for i in range(len(documents))})

    class Picked(list):
        final_evidence = [{"kind": "document", "chunk_ids": [3, 10]}]

    units = p._evidence_units(Picked())
    assert [(u["title"], u["url"]) for u in units] == [
        ("Иванов Иван Иванович", TIMETABLE), ("Кафедра физики", "https://www.nsu.ru/n/physics/")]
