import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from tests.ragu_fakes import pipeline

RUNTIME = PipelineRuntime.uniform(ModelRuntime("meno", "http://fake/v1"))


def scorer(value, calls=None):
    async def score(sentence, texts):
        if calls is not None:
            calls.append((sentence, list(texts)))
        if isinstance(value, Exception):
            raise value
        return [value] * len(texts)
    return score


async def prepared(p):
    p.resources.documents[0]["url"] = "https://example.org/a"
    return await p.prepare(messages=[ChatMessage(role="user", content="ФИТ")], runtime=RUNTIME)


@pytest.mark.asyncio
async def test_context_chunks_become_evidence_when_sources_are_attributed():
    outcome = await prepared(pipeline(source_scorer=scorer(0.9)))
    assert outcome.evidence == [{"text": "Факультет НГУ.", "title": "НГУ", "url": "https://example.org/a"}]


@pytest.mark.asyncio
async def test_sources_are_the_pages_the_answer_rests_on():
    calls = []
    p = pipeline(source_scorer=scorer(0.9, calls))
    outcome = await prepared(p)
    sources = await p.finalize_sources(outcome, "На факультете учатся студенты НГУ.")
    assert sources == outcome.sources == [{"document_title": "НГУ", "source_url": "https://example.org/a"}]
    assert calls == [("На факультете учатся студенты НГУ.", ["Факультет НГУ."])]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0.1, RuntimeError("reranker down")])
async def test_an_answer_resting_on_nothing_or_a_failed_attribution_shows_no_sources(value):
    p = pipeline(source_scorer=scorer(value))
    outcome = await prepared(p)
    assert await p.finalize_sources(outcome, "На факультете учатся студенты НГУ.") == []
    assert outcome.sources == []


@pytest.mark.asyncio
async def test_without_attribution_the_sources_stay_the_context_documents():
    p = pipeline()
    outcome = await prepared(p)
    assert not p.late_sources and outcome.evidence == []
    assert await p.finalize_sources(outcome, "Ответ про факультет НГУ здесь.") == outcome.sources == [
        {"document_title": "НГУ", "source_url": "https://example.org/a"}]
