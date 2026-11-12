import asyncio

from meno_rag.stand.attribution import norm_url, page_titles
from meno_rag.stand.ragu_pipeline import Candidate
from meno_rag.stand.routing import SOURCE_AUTHORITY_RULE, with_route_rules
from tests.ragu_fakes import pipeline


def document(title, url, quality):
    text = f"{title}. Подробные сведения о стоимости обучения."
    return {"doc_full_text": text, "doc_title": title, "doc_annotation": "", "url": url, "quality_score": quality,
            "chunks": [{"start_char": 0, "end_char": len(text)}]}


def order(weight, llm_scores):
    """The chunk order the rerank keeps for a post (chunk 0, quality .55) and an NSU page (chunk 1, quality .9)."""
    p = pipeline(source_prior_weight=weight, rerank_candidates_per_page=0)
    documents = [document("Пост ВКонтакте", "https://vk.com/wall-1_2", 0.55),
                 document("Страница НГУ", "https://www.nsu.ru/n/economics-department/", 0.9)]
    p.resources.documents[:] = documents
    p.resources.chunk_mapping.clear()
    p.resources.chunk_mapping.update({str(i): {"doc_index": i, "local_chunk_index": 0} for i in range(len(documents))})

    async def score(question, history, chunk_id, runtime):
        return llm_scores[chunk_id]

    p._score_chunk_with_llm = score
    pool = [Candidate("chunk:0", 0.9, chunk_id=0), Candidate("chunk:1", 0.9, chunk_id=1)]
    kept = asyncio.run(p._rerank([{"query": "q", "pool": pool}], "q", "", None))
    return [chunk_id for chunk_id, _ in kept]


def test_a_comparably_relevant_official_page_goes_before_a_post():
    assert order(2.0, {0: 0.6, 1: 0.6}) == [1, 0]


def test_a_clearly_better_post_stays_ahead():
    assert order(2.0, {0: 0.95, 1: 0.3}) == [0, 1]


def test_without_a_prior_weight_the_order_is_the_rerank_score_alone():
    assert order(0.0, {0: 0.6, 1: 0.6}) == [0, 1]


def kept_of_one_query(weight, top_k):
    """What the rerank keeps of a query that found one graph entity and one chunk of a post (quality .6), both
    judged fully relevant."""
    p = pipeline(source_prior_weight=weight, rerank_top_k=top_k, rerank_candidates_per_page=0)
    p.resources.documents[:] = [document("Пост ВКонтакте", "https://vk.com/wall-1_2", 0.6)]

    async def score_chunk(question, history, chunk_id, runtime):
        return 1.0

    async def score_element(question, history, text, runtime, key):
        return 1.0

    p._score_chunk_with_llm, p._score_document_with_llm = score_chunk, score_element
    entity = {"key": "НГУ", "kind": "entity", "name": "НГУ", "type": "ORG", "description": "Университет",
              "score": 1.0, "sources": [0]}
    pool = [Candidate("entity:НГУ", 1.0, element=entity), Candidate("chunk:0", 0.9, chunk_id=0)]
    kept = asyncio.run(p._rerank([{"query": "q", "pool": pool}], "q", "", None))
    return [chunk_id for chunk_id, _ in kept], [element["key"] for element in kept.elements]


def test_graph_elements_take_no_chunk_slots_of_a_query():
    """Elements have no source prior, so under one shared per-query cut they pushed every penalised chunk out (the
    question about E. Vityaev kept 2 chunks at weight 2 against 11 at weight 0)."""
    assert kept_of_one_query(2.0, 1) == ([0], ["НГУ"])
    assert kept_of_one_query(0.0, 1) == ([0], ["НГУ"])


def test_an_anchor_names_its_own_page_part():
    first = "https://www.nsu.ru/n/sveden/employees/#Мулляджанов_Рустам_Илхамович"
    second = "https://www.nsu.ru/n/sveden/employees/#Абалмасова_Юлия_Михайловна"
    assert norm_url(first) != norm_url(second)
    assert norm_url("https://www.nsu.ru/n/a/") == norm_url("http://nsu.ru/n/a")
    titles = page_titles([document("Мулляджанов Рустам Илхамович — педагогический состав НГУ", first, 0.9),
                          document("Абалмасова Юлия Михайловна — педагогический состав НГУ", second, 0.9)])
    assert titles == {}


def test_the_rule_that_official_documents_come_first_joins_questions_answered_from_documents():
    assert SOURCE_AUTHORITY_RULE in with_route_rules("S", "nsu", True, authority=True)
    assert SOURCE_AUTHORITY_RULE in with_route_rules("S", "general", True, authority=True)
    assert SOURCE_AUTHORITY_RULE not in with_route_rules("S", "nsu", True)
    assert SOURCE_AUTHORITY_RULE not in with_route_rules("S", "chitchat", False, authority=True)
