"""One chunk id, several addresses. A chunk id is the hash of the chunk text, so the same text in two documents
(a methodical guide published for the bachelor's and the master's programme, a publication in the records of two
teachers) is one id and one point of the chunk collection, and the chunk mapping names one of the documents. The
optional chunk_addresses.json next to the corpus lists the others, so the text is attributed to all its pages.
Without the file nothing changes."""
import asyncio
import json

import pytest
from nltk.stem.snowball import SnowballStemmer

from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.context import flatten_sources, graph_element_as_note, prepare_context
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.ragu_pipeline import RaguRagPipeline
from meno_rag.stand.resources import StandResources, _load_chunk_addresses
from tests.ragu_fakes import LLM

TEXT = "Руководитель оказывает студенту помощь в выборе темы ВКР ФИТ."
BACHELOR, MASTER = "https://nsu.test/09-03-01/gia.pdf", "https://nsu.test/09-04-01/gia.pdf"


def _doc(title, url, quality=1.0, text=TEXT):
    return {"doc_full_text": text, "doc_title": title, "doc_annotation": "", "url": url, "quality_score": quality,
            "chunks": [{"start_char": 0, "end_char": len(text)}]}


DOCS = [_doc("ГИА, бакалавриат", BACHELOR), _doc("ГИА, магистратура", MASTER, quality=0.6),
        _doc("Посторонний документ", "https://z.test", text="Совсем другой текст о другом.")]
MAPPING = {str(i): {"doc_index": i, "local_chunk_index": 0} for i in range(len(DOCS))}
ADDRESSES = {"0": [{"doc_index": 1, "local_chunk_index": 0}]}  # the id of chunk 0 is also the text of document 1


def test_a_chunk_with_two_addresses_is_attributed_to_both_urls_its_own_first():
    _texts, sources = prepare_context([0], [0.9], DOCS, MAPPING, 0.0, chunk_addresses=ADDRESSES)
    assert sources == [{"document_title": "ГИА, бакалавриат", "source_urls": [BACHELOR],
                        "other_addresses": [{"document_title": "ГИА, магистратура", "source_urls": [MASTER]}]}]
    assert flatten_sources(sources, max_urls=1) == [
        {"document_title": "ГИА, бакалавриат", "source_url": BACHELOR},
        {"document_title": "ГИА, магистратура", "source_url": MASTER}]


def test_without_addresses_the_context_and_the_sources_are_what_they_were():
    plain = prepare_context([0, 2], [0.9, 0.5], DOCS, MAPPING, 0.0)
    assert plain == prepare_context([0, 2], [0.9, 0.5], DOCS, MAPPING, 0.0, chunk_addresses=None)
    assert plain == prepare_context([0, 2], [0.9, 0.5], DOCS, MAPPING, 0.0, chunk_addresses={})
    assert plain[1][1] == {"document_title": "ГИА, бакалавриат", "source_urls": [BACHELOR]}
    with_addresses = prepare_context([0, 2], [0.9, 0.5], DOCS, MAPPING, 0.0, chunk_addresses=ADDRESSES)
    assert with_addresses[0] == plain[0]  # the text shown to the model and the order of the documents do not change
    assert [s["document_title"] for s in with_addresses[1]] == [s["document_title"] for s in plain[1]]


def test_a_note_of_a_graph_element_names_the_documents_of_every_address():
    element = {"kind": "entity", "name": "ВКР", "description": "Выпускная работа", "sources": [0]}
    assert graph_element_as_note(element, DOCS, MAPPING).endswith("По материалам: «ГИА, бакалавриат».")
    note = graph_element_as_note(element, DOCS, MAPPING, chunk_addresses=ADDRESSES)
    assert note.endswith("По материалам: «ГИА, бакалавриат»; «ГИА, магистратура».")
    capped = graph_element_as_note(element, DOCS, MAPPING, max_source_titles=1, chunk_addresses=ADDRESSES)
    assert capped.endswith("По материалам: «ГИА, бакалавриат».")


def test_the_addresses_file_is_optional_and_checked_against_the_corpus(tmp_path):
    path = tmp_path / "chunk_addresses.json"
    assert _load_chunk_addresses(path, MAPPING, DOCS) == {}
    path.write_text(json.dumps({"0": [{"doc_index": 1, "local_chunk_index": 0, "url": MASTER},
                                      {"doc_index": 0, "local_chunk_index": 0}]}), encoding="utf-8")
    assert _load_chunk_addresses(path, MAPPING, DOCS) == ADDRESSES  # the mapping's own document is not repeated
    path.write_text(json.dumps({"0": [{"doc_index": 7, "local_chunk_index": 0}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_chunk_addresses(path, MAPPING, DOCS)
    path.write_text(json.dumps({"9": [{"doc_index": 1, "local_chunk_index": 0}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_chunk_addresses(path, MAPPING, DOCS)


class Graph:
    async def search(self, query):
        await asyncio.sleep(0)
        return {"chunks": [(0, 0.5)], "relations": [],
                "entities": [{"key": "e", "kind": "entity", "name": "ВКР", "type": "DOC",
                              "description": "Выпускная работа ФИТ", "score": 1.0, "sources": [0]}]}


def _pipeline(addresses, source_scorer=None, **settings):
    resources = StandResources(documents=DOCS, chunk_mapping=MAPPING, graph=Graph(),
                               stemmer=SnowballStemmer("russian"), abbreviations={}, chunk_addresses=addresses)
    return RaguRagPipeline(
        settings=Settings(_env_file=None, max_rewrite_queries=1, **settings), resources=resources, llm_router=LLM(),
        rewrite_semaphore=asyncio.Semaphore(2), rerank_semaphore=asyncio.Semaphore(2),
        generation_semaphore=asyncio.Semaphore(2), embed_semaphore=asyncio.Semaphore(2), source_scorer=source_scorer)


async def _ask(p):
    return await p.prepare(messages=[ChatMessage(role="user", content="ВКР ФИТ")],
                           runtime=PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1")), capture_trace=True)


@pytest.mark.asyncio
async def test_the_answer_of_the_stand_shows_every_url_of_a_multi_address_chunk():
    plain, both = await _ask(_pipeline({})), await _ask(_pipeline(ADDRESSES))
    assert [s["source_url"] for s in plain.sources] == [BACHELOR]
    assert [s["source_url"] for s in both.sources] == [BACHELOR, MASTER]
    assert [s["document_title"] for s in both.sources] == ["ГИА, бакалавриат", "ГИА, магистратура"]
    assert both.context.replace("; «ГИА, магистратура»", "") == plain.context  # only the note's titles differ
    element = next(piece for piece in both.trace["final_evidence"] if piece["kind"] == "entity")
    assert [(r["document_index"], r["urls"]) for r in element["source_references"]] == [(0, [BACHELOR]), (1, [MASTER])]


@pytest.mark.asyncio
async def test_the_evidence_units_cover_every_address_with_the_prior_of_its_own_document():
    async def scorer(sentence, texts):
        return [0.9] * len(texts)

    class Picked(list):
        final_evidence = [{"kind": "document", "chunk_ids": [0]}]

    plain = _pipeline({}, source_scorer=scorer)._evidence_units(Picked())
    assert plain == [{"text": TEXT, "url": BACHELOR, "prior": 1.0, "title": "ГИА, бакалавриат"}]
    units = _pipeline(ADDRESSES, source_scorer=scorer)._evidence_units(Picked())
    assert units == [*plain, {"text": TEXT, "url": MASTER, "prior": 0.6, "title": "ГИА, магистратура"}]
