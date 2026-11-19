"""Smooth reranker scoring (#2a) + answer-usefulness prompt (#2b).

The old scoring returned P(label 2) or else 0.0, so any chunk the reranker
judged "partially relevant" (label 1) was zeroed and dropped — the mechanism
that evicted the price-bearing «Регламент» chunk for cost questions. The new
aggregate score gives label-1 chunks ~0.5 so they survive.
"""

import asyncio
import math

import pytest

from meno_rag.stand.pipeline import ModelRuntime
from meno_rag.stand.rerank import (
    SYSTEM_PROMPT_FOR_RELEVANCE,
    aggregate_logprobs_to_score,
    build_prompt,
    score_from_logprobs,
)
from tests import ragu_fakes


def _dominant(label: int) -> dict[int, float]:
    return {lbl: (0.0 if lbl == label else -100.0) for lbl in (0, 1, 2)}


def test_aggregate_maps_labels_to_0_half_1():
    assert aggregate_logprobs_to_score(_dominant(0)) == pytest.approx(0.0, abs=1e-6)
    assert aggregate_logprobs_to_score(_dominant(1)) == pytest.approx(0.5, abs=1e-6)
    assert aggregate_logprobs_to_score(_dominant(2)) == pytest.approx(1.0, abs=1e-6)


def test_aggregate_is_monotonic_and_smooth():
    # A confident-2 must outrank a confident-1 must outrank a confident-0.
    s0 = aggregate_logprobs_to_score({0: -0.1, 1: -3.0, 2: -5.0})
    s1 = aggregate_logprobs_to_score({0: -3.0, 1: -0.1, 2: -3.0})
    s2 = aggregate_logprobs_to_score({0: -5.0, 1: -3.0, 2: -0.1})
    assert 0.0 <= s0 < s1 < s2 <= 1.0


def test_aggregate_rejects_wrong_labels():
    with pytest.raises(ValueError):
        aggregate_logprobs_to_score({0: 0.0, 1: 0.0})  # missing label 2


def test_label1_chunk_survives_not_zeroed():
    # Regression guard for the eviction bug: a label-1-dominant chunk must get a
    # POSITIVE score (old behaviour returned 0.0 → filtered out).
    choice = {
        "logprobs": {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "1", "logprob": -0.05},
                        {"token": "2", "logprob": -3.0},
                        {"token": "0", "logprob": -3.5},
                    ]
                }
            ]
        }
    }
    score = score_from_logprobs(choice)
    assert score > 0.0
    assert 0.4 < score < 0.6  # ~0.5, not 0


def _choice(tokens):
    return {"logprobs": {"content": [{"top_logprobs": [
        {"token": token, "logprob": logprob} for token, logprob in tokens
    ]}]}}


@pytest.mark.parametrize("dominant", [0, 2])
def test_duplicate_decoded_digits_do_not_overwrite_the_likely_token(dominant):
    entries = [(str(label), 0.0 if label == dominant else -20.0) for label in (0, 1, 2)]
    entries.extend([(str(dominant), -39.0), (str(2 - dominant), -40.0)])
    assert score_from_logprobs(_choice(entries)) == pytest.approx(dominant / 2, abs=1e-7)
    assert score_from_logprobs(_choice(list(reversed(entries)))) == pytest.approx(dominant / 2, abs=1e-7)


def test_probability_mass_of_distinct_tokens_with_the_same_digit_is_added():
    entries = [("2", math.log(.3)), ("1", math.log(.4)), ("0", math.log(.1)), ("2", math.log(.2))]
    assert score_from_logprobs(_choice(entries)) == pytest.approx(.7)
    assert score_from_logprobs(_choice(list(reversed(entries)))) == pytest.approx(.7)


@pytest.mark.parametrize("entries", [
    [("0", -.1), ("1", -3.), ("2", -5.)],
    [("1", -.1), ("2", -2.)],
    [("0", -.1)],
])
def test_unique_digit_tokens_keep_the_previous_score(entries):
    previous = aggregate_logprobs_to_score({label: dict(entries).get(str(label), -100.) for label in (0, 1, 2)})
    assert score_from_logprobs(_choice(entries)) == previous


@pytest.mark.parametrize("entries", [[], [(" ", -.1)], [(" 2", -.1)], [("<|channel>", -.1)]])
def test_missing_digit_labels_raise_instead_of_becoming_half_relevant(entries):
    with pytest.raises(ValueError, match="no finite digit label"):
        score_from_logprobs(_choice(entries))


@pytest.mark.parametrize("invalid", [math.nan, math.inf])
def test_nonfinite_label_logprobs_are_rejected(invalid):
    with pytest.raises(ValueError, match="nonfinite"):
        score_from_logprobs(_choice([("0", -.1), ("1", invalid)]))


def test_negative_infinity_has_zero_probability_mass():
    assert score_from_logprobs(_choice([("0", -math.inf), ("1", 0.), ("2", -math.inf)])) == .5
    with pytest.raises(ValueError, match="no finite digit label"):
        score_from_logprobs(_choice([("0", -math.inf), ("1", -math.inf), ("2", -math.inf)]))


def test_the_pipeline_uses_json_fallback_when_no_digit_label_is_returned():
    class MissingLabelsLLM(ragu_fakes.LLM):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["max_tokens"] == 1:
                return {"choices": [_choice([("<|channel>", -.1)])]}
            return {"choices": [{"message": {"content": '{"label": "2"}'}}]}

    pipeline = ragu_fakes.pipeline()
    pipeline.llm_router = llm = MissingLabelsLLM()
    score = asyncio.run(pipeline._score_document_with_llm(
        "Вопрос", "", "Текст документа", ModelRuntime("fake-model", "http://fake/v1"), "test-chunk",
    ))
    assert score == pytest.approx(1.)
    assert len(llm.calls) == 2
    assert llm.calls[1]["max_tokens"] == 20 and "response_format" in llm.calls[1]


def test_build_prompt_structure_minimal():
    msgs = build_prompt("Сколько стоит общежитие?", "", "**Название документа:** Регламент\n\n1300 руб", {}, None)
    assert msgs[0]["content"] == SYSTEM_PROMPT_FOR_RELEVANCE
    user = msgs[1]["content"]
    assert "DOCUMENT 1" in user
    assert "CURRENT QUESTION" in user and "Сколько стоит общежитие?" in user
    assert "INSTRUCTION" in user
    assert "1300 руб" in user
    assert "ABBREVIATION DICTIONARY" not in user  # empty abbr dict
    assert "DIALOGUE HISTORY" not in user  # empty history


def test_build_prompt_includes_history_when_present():
    msgs = build_prompt("А сколько именно?", "Пользователь: про общежитие", "doc text", {}, None)
    assert "DIALOGUE HISTORY" in msgs[1]["content"]


def test_build_prompt_rejects_empty_document():
    with pytest.raises(ValueError):
        build_prompt("вопрос", "", "", {}, None)


def test_build_prompt_json_mode_switches_format_section():
    msgs = build_prompt("вопрос", "", "doc", {}, None, is_json=True)
    assert '{"label"' in msgs[0]["content"]
