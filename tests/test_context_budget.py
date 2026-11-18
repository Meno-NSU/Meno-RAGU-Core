"""The QA context under a binding character budget: the documents are shown weakest first (the strongest next to the
question), but the budget must keep the strongest ones. The loop used to walk the display order and stop at the first
document that did not fit, so one long weak document emptied the context (audit of 18.11)."""
from meno_rag.stand.pipeline import _RerankOutput
from tests.ragu_fakes import pipeline

LONG = "Длинный слабый документ о другом. " * 200


def _two_documents(p):
    strong = p.resources.documents[0]
    weak = dict(strong, doc_full_text=LONG, doc_title="Слабый",
                chunks=[{"start_char": 0, "end_char": len(LONG)}])
    p.resources.documents.append(weak)
    p.resources.chunk_mapping["1"] = {"doc_index": 1, "local_chunk_index": 0}


def test_the_strongest_document_stays_when_a_long_weak_one_does_not_fit():
    p = pipeline()
    _two_documents(p)
    context, sources = p._assemble_context(_RerankOutput([(0, 0.98), (1, 0.21)]), budget_override=2000)
    assert "Факультет НГУ." in context and "Длинный слабый" not in context
    assert [s["document_title"] for s in sources] == ["НГУ"]


def test_a_document_that_does_not_fit_is_skipped_and_the_rest_keep_the_display_order():
    p = pipeline()
    _two_documents(p)
    short_weak = dict(p.resources.documents[0], doc_full_text="Короткий слабый документ.", doc_title="Короткий",
                      chunks=[{"start_char": 0, "end_char": 25}])
    p.resources.documents.append(short_weak)
    p.resources.chunk_mapping["2"] = {"doc_index": 2, "local_chunk_index": 0}
    selection = _RerankOutput([(0, 0.98), (1, 0.5), (2, 0.1)])
    selection.capture_evidence = True
    context, sources = p._assemble_context(selection, budget_override=2000)
    assert [s["document_title"] for s in sources] == ["Короткий", "НГУ"]  # weakest first, as without a budget
    assert context.index("Короткий слабый") < context.index("Факультет НГУ.")
    assert [block["document_index"] for block in selection.final_evidence] == [2, 0]
    assert all(block["text"] == context[block["context_start"]:block["context_end"]]
               for block in selection.final_evidence)
