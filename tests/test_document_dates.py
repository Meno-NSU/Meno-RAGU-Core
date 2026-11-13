"""Document dates in the RAGU context, as Menon 1.1 shows them in each block's header: the answer rules ask to
check the documents' dates against today's, which needs the dates. The stand corpus has none; the KB v3.8 chunk
metadata dates 155K of 690K chunks (79% of posts, 37% of news), joined by kb_doc_id into document_dates.json."""
import json

from meno_rag.stand.context import document_to_text, format_document_date
from meno_rag.stand.resources import _load_documents

DOC = {"doc_full_text": "Прощание состоится 14 октября.", "chunks": [{"start_char": 0, "end_char": 30}],
       "doc_title": "Ушёл из жизни профессор", "doc_annotation": "Новость факультета"}


def test_a_dated_document_shows_its_date_after_the_title():
    text = document_to_text(0, [0], [dict(DOC, date="2026-10-13")])
    assert text.startswith("**Название документа:** Ушёл из жизни профессор.\n\n**Дата документа:** 13.10.2026.")


def test_an_undated_document_shows_no_date_line():
    assert "Дата документа" not in document_to_text(0, [0], [DOC])


def test_dates_are_formatted_as_menon_formats_them():
    assert format_document_date("2026-10-13") == "13.10.2026"
    assert format_document_date("2026-10-13T08:00:00") == "13.10.2026"
    assert format_document_date("2026-10") == "10.2026"
    assert format_document_date("2026") == "2026 г."
    assert format_document_date("") == ""


def test_the_dates_sidecar_next_to_the_corpus_dates_the_documents(tmp_path):
    corpus = tmp_path / "chunked_texts_about_nsu_with_metadata_and_scores.jsonl"
    corpus.write_text("\n".join(json.dumps(dict(DOC, doc_title=f"Д{i}"), ensure_ascii=False) for i in range(3)) + "\n",
                      encoding="utf-8")
    (tmp_path / "document_dates.json").write_text(json.dumps({"0": "2026-10-13", "2": "2025-05-01"}), encoding="utf-8")
    documents, _ = _load_documents(corpus)
    assert [d.get("date") for d in documents] == ["2026-10-13", None, "2025-05-01"]


def test_without_a_sidecar_documents_stay_undated(tmp_path):
    corpus = tmp_path / "chunked_texts_about_nsu_with_metadata_and_scores.jsonl"
    corpus.write_text(json.dumps(DOC, ensure_ascii=False) + "\n", encoding="utf-8")
    documents, _ = _load_documents(corpus)
    assert "date" not in documents[0]
