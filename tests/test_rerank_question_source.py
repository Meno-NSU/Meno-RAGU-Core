import asyncio
import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand import pipeline as pipeline_module
from meno_rag.stand.pipeline import ModelRuntime
from tests.ragu_fakes import LLM, Graph, pipeline
from tests.test_ragu_sources_late import RUNTIME

ORIGINAL = "А с ним можно сделать проект и кто может быть внешним руководителем?"
RESOLVED = "Можно ли сделать проект с научным центром?"
HISTORY = [{"role": "user", "content": "Меня интересует научный центр и работа под его руководством."},
           {"role": "assistant", "content": "Вы обсуждаете научный центр."}]


class RecordingGraph(Graph):
    def __init__(self):
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return await super().search(query)


class RecordingLLM(LLM):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.score_calls = []

    async def chat_completion_text(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stage") == "turn_routing":
            return json.dumps({"route": "nsu", "standalone_question": RESOLVED,
                               "subqueries": [], "chain": False, "focus": "научный центр"})
        await asyncio.sleep(0.01)
        return "другой поисковый запрос"

    async def chat_completion(self, **kwargs):
        self.score_calls.append(kwargs)
        return await super().chat_completion(**kwargs)


def test_source_mode_is_experimental_validated_and_keeps_the_existing_default(monkeypatch):
    monkeypatch.delenv("RERANK_QUESTION_SOURCE", raising=False)
    assert Settings(_env_file=None).rerank_question_source == "resolved"
    monkeypatch.setenv("RERANK_QUESTION_SOURCE", "original_resolved")
    assert Settings(_env_file=None).rerank_question_source == "original_resolved"
    with pytest.raises(ValidationError, match="RERANK_QUESTION_SOURCE"):
        Settings(_env_file=None, RERANK_QUESTION_SOURCE="search_query")


def test_original_intents_resolved_references_and_history_are_preserved_without_duplicate_identical_questions():
    p = pipeline(rerank_question_source="original_resolved")
    text, history = p._rerank_inputs(ORIGINAL, RESOLVED, "история", has_memory=True)
    assert ORIGINAL in text and RESOLVED in text and history == "история"
    assert p._rerank_inputs(ORIGINAL, ORIGINAL, "история", has_memory=True) == (ORIGINAL, "история")
    p.settings = p.settings.model_copy(update={"rerank_question_source": "resolved"})
    assert p._rerank_inputs(ORIGINAL, RESOLVED, "история", has_memory=True) == (RESOLVED, "")
    assert p._rerank_inputs(ORIGINAL, RESOLVED, "история", has_memory=False) == (RESOLVED, "история")


@pytest.mark.asyncio
@pytest.mark.parametrize("early_rerank", [True, False])
@pytest.mark.parametrize("order", ["legacy", "question_first"])
@pytest.mark.parametrize("guidance", ["legacy", "structured"])
async def test_experimental_question_source_has_exact_full_and_retrieval_only_scheduling_parity(
    monkeypatch, early_rerank, order, guidance
):
    monkeypatch.setattr(pipeline_module, "prompt_now", lambda _zone: datetime(2026, 11, 20, tzinfo=UTC))

    async def one(early):
        p = pipeline(rerank_question_source="original_resolved", early_first_query=early,
                     early_rerank=early_rerank, rerank_prompt_order=order, rerank_guidance=guidance)
        p.settings = p.settings.model_copy(update={"max_rewrite_queries": 2})
        graph, llm = RecordingGraph(), RecordingLLM()
        p.resources = replace(p.resources, graph=graph)
        p.llm_router = llm
        outcome = await p.prepare(messages=[ChatMessage(**m) for m in HISTORY]
                                  + [ChatMessage(role="user", content=ORIGINAL)], runtime=RUNTIME, capture_trace=True)
        return outcome, graph, llm

    late, late_graph, late_llm = await one(False)
    early, early_graph, early_llm = await one(True)
    assert early_graph.queries == late_graph.queries
    assert early_graph.queries[0] == RESOLVED
    assert all(ORIGINAL not in query for query in early_graph.queries)
    assert early.context == late.context and early.sources == late.sources and early.evidence == late.evidence
    assert early.qa_messages == late.qa_messages
    assert Counter(map(repr, early_llm.score_calls)) == Counter(map(repr, late_llm.score_calls))
    assert len(early_llm.score_calls) == len({repr(c) for c in early_llm.score_calls})
    assert len(early_llm.calls) == len(late_llm.calls) == 2
    for call in early_llm.score_calls:
        body = call["messages"][-1]["content"]
        assert ORIGINAL in body and RESOLVED in body
        assert HISTORY[0]["content"] in body
    assert early.trace["runtime"]["rerank_question"] == late.trace["runtime"]["rerank_question"]
    assert ORIGINAL in early.trace["runtime"]["rerank_question"]
    assert early.trace["runtime"]["settings"]["rerank_question_source"] == "original_resolved"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["question", "history", "runtime"])
async def test_a_mismatched_early_scoring_input_is_never_reused(changed):
    p = pipeline(early_rerank=True)
    calls = []

    async def score(candidate, question, history, runtime):
        calls.append((candidate.key, question, history, runtime))
        return 0.8

    p._score_candidate = score
    runtime = ModelRuntime("old-model", "http://model/v1")
    early = p._start_early_query("search", "old-history", runtime, rerank_question="old-question")
    try:
        batches = await early.retrieval
        await early.started
        await asyncio.gather(*early.scores.values())
        first_count = len(calls)
        question = "new-question" if changed == "question" else "old-question"
        history = "new-history" if changed == "history" else "old-history"
        later_runtime = replace(runtime, model_id="new-model") if changed == "runtime" else runtime
        await p._rerank(p._fuse(batches), question, history, later_runtime, early=early)
        assert len(calls) == 2 * first_count
        assert all(q == question and h == history and r == later_runtime for _, q, h, r in calls[first_count:])
    finally:
        await early.close()
