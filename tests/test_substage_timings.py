"""Sub-stage timings (audit of 19.11, §4.0): recorded in the stage details, nothing else of a turn changes.

The golden snapshot (tests/test_pipeline_snapshot.py) checks that every other field of every stage, the context, the
QA prompt and the sources stay as they were; these tests check what the timings say."""

import asyncio
import copy

from meno_rag.schemas import ChatMessage
from meno_rag.stand.attribution import EvidenceUnit, attribute, to_sources
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.trace import substage_timings
from tests import ragu_fakes

RUNTIME = PipelineRuntime.uniform(ModelRuntime("fake-model", "http://fake"))
QUESTION = [ChatMessage(role="user", content="Какие факультеты есть в НГУ?")]


class TimedGraph(ragu_fakes.Graph):
    """The fake graph with the per-search timings the engines report (graph_utils.search_timings)."""

    async def search(self, query):
        pool = await super().search(query)
        return {**pool, "timings": {"search_ms": 12.5, "embed_calls": 1, "ce_pairs": 7, "query": query}}


class UsageLLM(ragu_fakes.LLM):
    """Rerank responses with usage; cached tokens reported for every second call only."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def chat_completion(self, **kwargs):
        response = copy.deepcopy(await super().chat_completion(**kwargs))
        self.calls += 1
        usage = {"prompt_tokens": 100, "completion_tokens": 1}
        if self.calls % 2 == 0:
            usage["prompt_tokens_details"] = {"cached_tokens": 60}
        response["usage"] = usage
        return response


class FallbackLLM(ragu_fakes.LLM):
    """Guided choice fails (a client error), the JSON fallback answers."""

    async def chat_completion(self, **kwargs):
        self.prompts.append(kwargs["messages"])
        if "logprobs" in kwargs:
            raise ValueError("guided choice unsupported")
        return {"choices": [{"message": {"content": '{"label": "2"}'}}]}


def prepare(pipeline, capture_trace=False):
    return asyncio.run(pipeline.prepare(messages=QUESTION, runtime=RUNTIME, capture_trace=capture_trace))


def with_queries(pipeline, queries):
    """The fixture caps the rewrite at one query; the question plus the fake rewrite make two."""
    pipeline.settings = pipeline.settings.model_copy(update={"max_rewrite_queries": queries})
    return pipeline


def attributable(pipeline):
    """One link per document, so the fixture's chunk is an evidence unit (as tests/test_ragu_sources_late.py)."""
    pipeline.resources.documents[0]["url"] = "https://example.org/a"
    return pipeline


def test_the_rewrite_stage_reports_the_memory_analysis_and_rewrite_times():
    outcome = prepare(ragu_fakes.pipeline())
    timings = outcome.stage_details["query_rewrite"]["timings"]
    assert set(timings) == {"memory_ms", "analysis_ms", "rewrite_ms"}
    assert all(isinstance(value, float) and value >= 0 for value in timings.values())


def test_without_dialogue_memory_the_route_and_rewrite_calls_are_timed():
    outcome = prepare(ragu_fakes.pipeline(dialogue_memory=False))
    assert set(outcome.stage_details["query_rewrite"]["timings"]) == {"route_ms", "rewrite_ms"}


def test_the_retrieval_stage_keeps_each_query_s_search_timings_in_query_order():
    pipeline = with_queries(ragu_fakes.pipeline(), 2)
    object.__setattr__(pipeline.resources, "graph", TimedGraph())  # a frozen dataclass: swap the fixture's graph
    outcome = prepare(pipeline)
    per_query = outcome.stage_details["retrieval"]["timings"]["per_query"]
    assert [item["query"] for item in per_query] == outcome.search_queries
    assert all(item["embed_calls"] == 1 and item["ce_pairs"] == 7 for item in per_query)


def test_a_graph_without_timings_adds_no_timings_to_the_retrieval_stage():
    outcome = prepare(ragu_fakes.pipeline())
    assert "timings" not in outcome.stage_details["retrieval"]


def test_the_rerank_stage_counts_its_llm_calls_and_reports_no_tokens_the_server_did_not_send():
    outcome = prepare(ragu_fakes.pipeline())
    timings = outcome.stage_details["rerank"]["timings"]
    assert timings["llm_calls"] == outcome.stage_details["rerank"]["from"] == 3
    assert timings["json_fallback_calls"] == 0
    assert timings["prompt_tokens"] is None and timings["cached_tokens"] is None


def test_prompt_and_cached_tokens_are_summed_from_the_responses_that_report_them():
    pipeline = ragu_fakes.pipeline()
    pipeline.llm_router = UsageLLM()
    timings = prepare(pipeline).stage_details["rerank"]["timings"]
    assert timings["llm_calls"] == 3
    assert (timings["usage_reported"], timings["prompt_tokens"]) == (3, 300)
    assert (timings["cached_reported"], timings["cached_tokens"]) == (1, 60)


def test_json_fallback_calls_are_counted_next_to_the_failed_guided_calls():
    pipeline = ragu_fakes.pipeline()
    pipeline.llm_router = FallbackLLM()
    timings = prepare(pipeline).stage_details["rerank"]["timings"]
    assert (timings["llm_calls"], timings["json_fallback_calls"]) == (6, 3)


def test_concurrent_turns_count_their_own_rerank_calls():
    one, two = ragu_fakes.pipeline(), with_queries(ragu_fakes.pipeline(), 2)

    async def both():
        return await asyncio.gather(*(p.prepare(messages=QUESTION, runtime=RUNTIME) for p in (one, two)))

    first, second = asyncio.run(both())
    assert first.stage_details["rerank"]["timings"]["llm_calls"] == first.stage_details["rerank"]["from"] == 3
    assert second.stage_details["rerank"]["timings"]["llm_calls"] == second.stage_details["rerank"]["from"] == 5


def test_source_attribution_reports_its_time_and_pairs_without_changing_the_sources():
    async def scorer(sentence, texts):
        return [0.9 for _ in texts]

    answer = "В НГУ есть много факультетов и институтов. Подробности есть на сайте университета."
    pipeline = attributable(ragu_fakes.pipeline(source_scorer=scorer))
    outcome = prepare(pipeline, capture_trace=True)
    expected = copy.deepcopy(outcome)
    sources = asyncio.run(pipeline.finalize_sources(outcome, answer))
    timings = outcome.stage_details["source_attribution"]["timings"]
    assert timings["sentences"] == 2 and timings["scorer_calls"] == 2
    assert timings["units"] == len(outcome.evidence) and timings["pairs"] == 2 * len(outcome.evidence)
    assert timings["attribution_ms"] >= 0
    assert outcome.trace["stage_details"]["source_attribution"]["timings"] == timings
    assert "source_attribution" not in outcome.stage_durations_ms  # not a stage: the stage list and its sum stay
    # the same sources as the attribution gives on its own
    units = [EvidenceUnit(**unit) for unit in expected.evidence]
    scored = asyncio.run(
        attribute(
            answer,
            units,
            scorer,
            pipeline.settings.source_attribution_threshold,
            prior_weight=pipeline.settings.source_prior_weight,
        )
    )
    assert sources == to_sources(scored, pipeline.settings.max_sources)


def test_a_failed_attribution_is_timed_and_marked():
    async def scorer(sentence, texts):
        raise RuntimeError("cross-encoder down")

    pipeline = attributable(ragu_fakes.pipeline(source_scorer=scorer))
    outcome = prepare(pipeline)
    assert asyncio.run(pipeline.finalize_sources(outcome, "В НГУ есть много факультетов и институтов.")) == []
    assert outcome.stage_details["source_attribution"]["timings"]["failed"] is True


def test_substage_timings_collects_the_timings_of_every_stage_that_has_them():
    outcome = prepare(ragu_fakes.pipeline())
    collected = substage_timings(outcome)
    assert set(collected) == {"query_rewrite", "rerank"}
    assert collected["rerank"] is outcome.stage_details["rerank"]["timings"]
