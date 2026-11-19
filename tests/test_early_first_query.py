"""EARLY_FIRST_QUERY: the standalone question is searched and its candidates are scored while the rewrite call runs.
The first search query is always that question (``_rewrite_question`` puts it first) and a rerank score is judged
against the question, not the search query, so the prompts, the context and the sources must not change."""

import asyncio

import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StageName
from tests.ragu_fakes import LLM, pipeline

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
