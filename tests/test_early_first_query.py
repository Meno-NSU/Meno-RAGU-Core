"""EARLY_FIRST_QUERY: the standalone question is searched and its candidates are scored while the rewrite call runs.
The first search query is always that question (``_rewrite_question`` puts it first) and a rerank score is judged
against the question, not the search query, so the prompts, the context and the sources must not change."""

import asyncio
from dataclasses import replace

import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StageName
from meno_rag.stand.ragu_pipeline import EarlyQuery
from tests.ragu_fakes import LLM, Graph, pipeline

QUESTION = "Какие факультеты есть в НГУ?"


class RecordingLLM(LLM):
    """Analysis falls back to the question itself; the rewrite takes a while and adds two queries."""

    def __init__(self, fail_rewrite=False):
        super().__init__()
        self.events = []
        self.fail_rewrite = fail_rewrite

    async def chat_completion_text(self, **kwargs):
        if kwargs.get("stage") == StageName.QUERY_REWRITE:
            await asyncio.sleep(0.05)
            self.events.append("rewrite_done")
            if self.fail_rewrite:
                raise RuntimeError("rewrite failed")
            return "второй запрос\nтретий запрос"
        return "не JSON"

    async def chat_completion(self, **kwargs):
        self.events.append("rerank")
        return await super().chat_completion(**kwargs)


async def run(early, llm, **overrides):
    p = pipeline(early_first_query=early, **overrides)
    p.settings = p.settings.model_copy(update={"max_rewrite_queries": 3})
    p.llm_router = llm
    outcome = await p.prepare(
        messages=[ChatMessage(role="user", content=QUESTION)],
        runtime=PipelineRuntime.uniform(ModelRuntime("meno-lite", "http://model/v1")),
    )
    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [{}, {"rerank_prompt_order": "question_first"}, {"llm_rerank": False}])
async def test_early_first_query_changes_nothing_but_the_timing(overrides):
    late_llm, early_llm = RecordingLLM(), RecordingLLM()
    late = await run(False, late_llm, **overrides)
    early = await run(True, early_llm, **overrides)
    assert early.context == late.context
    assert early.sources == late.sources
    # the same rerank prompts, each once: the first query's candidates are not scored twice
    assert sorted(map(repr, early_llm.prompts)) == sorted(map(repr, late_llm.prompts))
    assert len(early_llm.prompts) == len({repr(p) for p in early_llm.prompts})
    if overrides.get("llm_rerank") is False:
        assert early_llm.prompts == late_llm.prompts == []
        return
    # without the option every rerank call waits for the rewrite; with it scoring starts during the rewrite
    assert late_llm.events.index("rewrite_done") < late_llm.events.index("rerank")
    assert early_llm.events.index("rerank") < early_llm.events.index("rewrite_done")


@pytest.mark.asyncio
async def test_failed_rewrite_leaves_no_early_task_running():
    llm = RecordingLLM(fail_rewrite=True)
    with pytest.raises(RuntimeError, match="rewrite failed"):
        await run(True, llm)
    await asyncio.sleep(0.01)
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert pending == []


@pytest.mark.asyncio
async def test_other_queries_start_while_the_early_retrieval_is_still_running():
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingGraph(Graph):
        async def search(self, query):
            if query == QUESTION:
                first_started.set()
                await release_first.wait()
            else:
                second_started.set()
            return await super().search(query)

    p = pipeline()
    p.resources = replace(p.resources, graph=BlockingGraph())
    early = EarlyQuery(QUESTION, "")
    early.retrieval = asyncio.create_task(p._retrieve([QUESTION]))
    await first_started.wait()
    retrieval = asyncio.create_task(p._retrieve([QUESTION, "second"], early=early))
    try:
        await asyncio.wait_for(second_started.wait(), timeout=1)
    finally:
        release_first.set()
        result = await retrieval
        await early.close()
    assert [batch["query"] for batch in result] == [QUESTION, "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_rewrite_completion_failure_cleans_up_the_early_retrieval(failure):
    retrieval_started = asyncio.Event()
    retrieval_cancelled = asyncio.Event()

    class BlockingGraph(Graph):
        async def search(self, query):
            retrieval_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                retrieval_cancelled.set()

    async def stage_sink(event):
        if event.stage == StageName.QUERY_REWRITE and event.status == "completed":
            await retrieval_started.wait()
            raise failure("rewrite completion failed")

    p = pipeline(early_first_query=True)
    p.resources = replace(p.resources, graph=BlockingGraph())
    p.llm_router = RecordingLLM()
    with pytest.raises(failure, match="rewrite completion failed"):
        await p.prepare(
            messages=[ChatMessage(role="user", content=QUESTION)],
            runtime=PipelineRuntime.uniform(ModelRuntime("meno-lite", "http://model/v1")),
            stage_sink=stage_sink,
        )
    assert retrieval_cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_first_pool", [False, True])
async def test_question_first_keeps_its_warm_up_when_the_first_pool_is_small(monkeypatch, empty_first_pool):
    class SmallFirstPoolGraph(Graph):
        async def search(self, query):
            result = await super().search(query)
            if query == QUESTION:
                result["entities"], result["relations"] = [], []
                if empty_first_pool:
                    result["chunks"] = []
            return result

    p = pipeline(rerank_prompt_order="question_first")
    p.resources = replace(p.resources, graph=SmallFirstPoolGraph())
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    first_finished = False
    warm_up_key = "entity:second" if empty_first_pool else "chunk:0"

    async def score(candidate, user_question, dialogue_history, runtime):
        nonlocal first_finished
        if candidate.key == warm_up_key:
            first_started.set()
            await release_first.wait()
            first_finished = True
        else:
            assert first_finished, "Other candidates started before the prefix was warmed up."
        return 0.9

    async def release():
        await first_started.wait()
        await asyncio.sleep(0.01)
        release_first.set()

    monkeypatch.setattr(p, "_score_candidate", score)
    runtime = ModelRuntime("meno-lite", "http://model/v1")
    early = p._start_early_query(QUESTION, "", runtime)
    await early.started
    batches = await p._retrieve([QUESTION, "second"], early=early)
    release_task = asyncio.create_task(release())
    try:
        await p._rerank(p._fuse(batches), QUESTION, "", runtime, early=early)
    finally:
        release_first.set()
        await release_task
        await early.close()
