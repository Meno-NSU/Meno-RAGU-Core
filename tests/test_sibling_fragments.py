import asyncio
from types import SimpleNamespace

from meno_rag.stand.ragu_pipeline import RaguRagPipeline


def pipeline(budget):
    text = "Общие сведения о программе. Стоимость обучения 235 000 руб. Контакты деканата. Другая страница."
    bounds = [(0, 27), (28, 60), (61, 79), (80, 96)]
    documents = [
        {"doc_full_text": text, "chunks": [{"start_char": a, "end_char": b} for a, b in bounds[:3]]},
        {"doc_full_text": text, "chunks": [{"start_char": bounds[3][0], "end_char": bounds[3][1]}]},
    ]
    mapping = {"10": {"doc_index": 0, "local_chunk_index": 0}, "11": {"doc_index": 0, "local_chunk_index": 1},
               "12": {"doc_index": 0, "local_chunk_index": 2}, "20": {"doc_index": 1, "local_chunk_index": 0}}
    scored = []

    async def score_texts(query, texts):
        scored.append(list(texts))
        return [0.9] * len(texts)

    async def search(query):
        return {"chunks": [(10, 0.7)], "entities": [], "relations": []}

    made = object.__new__(RaguRagPipeline)
    made.settings = SimpleNamespace(sibling_fragment_candidates=budget, rerank_candidates_per_query=40)
    made.resources = SimpleNamespace(documents=documents, chunk_mapping=mapping,
                                     graph=SimpleNamespace(search=search, score_texts=score_texts))
    made.embed_semaphore = asyncio.Semaphore(1)
    return made, scored


def test_sibling_fragments_are_off_by_default_and_add_the_matching_chunk_of_a_found_page():
    off, scored_off = pipeline(0)
    assert asyncio.run(off._retrieve(["стоимость обучения"]))[0]["graph"]["chunks"] == [(10, 0.7)]
    assert scored_off == []
    on, scored_on = pipeline(4)
    # the chunk of the found page that covers the query joins the pool; the chunk without the query's words and
    # the other page do not
    assert asyncio.run(on._retrieve(["стоимость обучения"]))[0]["graph"]["chunks"] == [(10, 0.7), (11, 0.9)]
    assert scored_on == [["Стоимость обучения 235 000 руб."]]
