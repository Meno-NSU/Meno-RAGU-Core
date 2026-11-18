"""Graph elements in the context need evidence: an entity or relation goes into the prompt only when one of the chunks
it was extracted from was retrieved for the question, so its description can be cited and checked. Without that,
268 of 989 graph blocks of the core set reached the prompt with no source (audit of stand v6k, 18.11.2026)."""
import asyncio

from meno_rag.stand.ragu_pipeline import Candidate
from tests.ragu_fakes import pipeline


def document(title, url):
    text = f"{title}. Подробные сведения о приёме."
    return {"doc_full_text": text, "doc_title": title, "doc_annotation": "", "url": url, "quality_score": 0.9,
            "chunks": [{"start_char": 0, "end_char": len(text)}]}


def entity(key, score, sources):
    return {"key": key, "kind": "entity", "name": key, "type": "ORG", "description": f"Описание {key}",
            "score": score, "sources": sources}


def kept_elements(evidence_required, max_elements=1):
    p = pipeline(rerank_candidates_per_page=0, graph_max_elements=max_elements,
                 graph_element_evidence=evidence_required)
    p.resources.documents[:] = [document("Приём в НГУ", "https://www.nsu.ru/n/education/apply/"),
                                document("Посторонняя страница", "https://example.org/x")]
    p.resources.chunk_mapping.clear()
    p.resources.chunk_mapping.update({str(i): {"doc_index": i, "local_chunk_index": 0} for i in range(2)})

    async def score_chunk(question, history, chunk_id, runtime):
        return 1.0

    async def score_element(question, history, text, runtime, key):
        return 1.0

    p._score_chunk_with_llm, p._score_document_with_llm = score_chunk, score_element
    unsupported = entity("Без доказательств", 1.0, [1])  # its only source chunk was not retrieved
    supported = entity("С доказательством", 0.9, [0])
    pool = [Candidate("entity:Без доказательств", 1.0, element=unsupported),
            Candidate("entity:С доказательством", 0.9, element=supported),
            Candidate("chunk:0", 0.9, chunk_id=0)]
    kept = asyncio.run(p._rerank([{"query": "q", "pool": pool}], "q", "", None))
    return [element["key"] for element in kept.elements]


def test_an_element_whose_sources_were_not_retrieved_gives_its_slot_to_one_with_evidence():
    assert kept_elements(True) == ["С доказательством"]


def test_without_the_requirement_the_better_scored_element_is_kept():
    assert kept_elements(False) == ["Без доказательств"]


def test_an_element_without_evidence_is_not_rendered_into_the_context():
    p = pipeline(graph_element_evidence=True)
    p.resources.documents[:] = [document("Приём в НГУ", "https://www.nsu.ru/n/education/apply/")]
    p.resources.chunk_mapping.clear()
    p.resources.chunk_mapping.update({"0": {"doc_index": 0, "local_chunk_index": 0}})

    class Chunks(list):
        elements = [entity("Без доказательств", 1.0, [5])]
        candidate_scores = {0: 0.9}

    context, _ = p._assemble_context(Chunks([(0, 0.9)]))
    assert "Описание Без доказательств" not in context
