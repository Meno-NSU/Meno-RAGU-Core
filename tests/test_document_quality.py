"""One authority scale for the stand corpus. The KB v3.8 documents carry the KB's quality (posts about .55, NSU pages
.8-.9); the graph corpus of NSU v1 carries its own, squeezed into .70-.80 (its VK posts .77, its nsu.ru pages .73), so
under the source prior its posts outranked official KB pages. A document takes the KB's quality of its page, else the
KB's median quality of the nearest section of the site (URL directory) with enough KB documents, else keeps its own."""
import json

from meno_rag.stand.quality import QualityScale
from meno_rag.stand.resources import _load_documents


def scale(min_documents=2):
    rows = [("https://www.nsu.ru/n/education/apply/", 0.9), ("https://nsu.ru/n/education/master/", 0.8),
            ("https://nsu.ru/n/education/phd/", 0.85), ("https://vk.com/wall-1_1", 0.55),
            ("https://vk.com/wall-1_2", 0.6), ("https://vk.com/wall-2_5", 0.5),
            ("https://www.nsu.ru/n/media/news/a/", 0.45), ("https://www.nsu.ru/n/media/news/b/", 0.45)]
    return QualityScale.from_kb(rows, min_documents=min_documents)


def test_a_page_of_the_kb_takes_the_kb_quality():
    assert scale().quality_of("https://nsu.ru/n/education/apply", 0.73) == 0.9


def test_a_page_outside_the_kb_takes_the_median_of_its_nearest_section():
    assert scale().quality_of("https://vk.com/wall-3_7", 0.77) == 0.55
    assert scale().quality_of("https://www.nsu.ru/n/education/bachelor/", 0.73) == 0.85
    assert scale().quality_of("https://www.nsu.ru/n/media/news/c/", 0.75) == 0.45


def test_a_section_with_too_few_kb_documents_defers_to_a_wider_one():
    assert scale(min_documents=4).quality_of("https://www.nsu.ru/n/education/bachelor/", 0.73) == 0.8


def test_a_site_unknown_to_the_kb_keeps_its_own_quality():
    assert scale().quality_of("https://example.org/page", 0.74) == 0.74
    assert scale().quality_of("", 0.74) == 0.74


def test_the_quality_sidecar_next_to_the_corpus_sets_the_documents_quality(tmp_path):
    corpus = tmp_path / "chunked_texts_about_nsu_with_metadata_and_scores.jsonl"
    docs = [{"doc_full_text": "Текст.", "quality_score": q} for q in (0.77, 0.73, 0.9)]
    corpus.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    (tmp_path / "document_quality.json").write_text(json.dumps({"0": 0.6, "1": 0.9}), encoding="utf-8")
    documents, _ = _load_documents(corpus)
    assert [d["quality_score"] for d in documents] == [0.6, 0.9, 0.9]
