"""Sources attributed while the answer streams (speed audit of 19.11, П5): the same sources as attributing the whole
answer at the end, for any answer and any way the stream cuts it into tokens."""

import asyncio
import copy
import hashlib
import random

from meno_rag.schemas import ChatMessage
from meno_rag.stand.attribution import EvidenceUnit, StreamingAttribution, attribute, split_sentences, to_sources
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from tests import ragu_fakes

SENTENCES = [
    "НГУ принимает документы с двадцатого июня до конца июля.",
    "Общежитие предоставляется всем иногородним студентам первого курса.",
    "На ФИТ можно поступить по результатам трёх экзаменов ЕГЭ.",
    "Стоимость обучения указана на странице приёмной комиссии университета.",
    "Да.",
    "Источник: сайт приёмной комиссии НГУ.",
    "_Ответ составлен по документам базы знаний._",
    "Подробнее смотрите расписание экзаменов на сайте факультета.",
    "Т. е. срок подачи заявлений один и тот же для всех программ.",
]
BREAKS = ["\n", "\n\n", "\r\n", "\r", " ", " ", " ", " "]


def answer(rng):
    parts = []
    for _ in range(rng.randint(1, 12)):
        sentence = rng.choice(SENTENCES)
        parts.append(rng.choice(["", "", "- ", "1. ", "**Важно:** "]) + sentence + rng.choice(BREAKS))
    text = "".join(parts)
    return text if rng.random() < 0.5 else text.rstrip()


def chunks(rng, text):
    out, position = [], 0
    while position < len(text):
        size = rng.randint(1, 9)
        out.append(text[position : position + size])
        position += size
    return out


def scorer_with(calls):
    async def score(sentence, texts):
        calls.append(sentence)
        await asyncio.sleep(0)
        return [int(hashlib.sha256(f"{sentence}|{text}".encode()).hexdigest()[:6], 16) / 0xFFFFFF for text in texts]

    return score


UNITS = [
    EvidenceUnit(
        text=f"Фрагмент {index} о поступлении в НГУ и общежитиях.",
        title=f"Страница {index}",
        url=f"https://example.org/{index % 4}",
        prior=0.5 + index / 20,
    )
    for index in range(7)
]


def test_streamed_attribution_equals_the_whole_answer_s_for_any_stream():
    rng = random.Random(11)

    async def run():
        for _ in range(300):
            text = answer(rng)
            reference_calls, streamed_calls = [], []
            expected = await attribute(text, UNITS, scorer_with(reference_calls), 0.5, prior_weight=1.0)
            streaming = StreamingAttribution(UNITS, scorer_with(streamed_calls), 0.5, prior_weight=1.0)
            so_far = ""
            for token in chunks(rng, text):
                so_far += token
                streaming.feed(so_far)
                await asyncio.sleep(0)
            got = await streaming.finish(text)
            assert got == expected
            assert to_sources(got, 5) == to_sources(expected, 5)
            # only the answer's own sentences are scored, each once
            assert sorted(streamed_calls) == sorted(set(split_sentences(text)))

    asyncio.run(run())


def test_finished_lines_are_scored_before_the_answer_ends():
    calls = []

    async def run():
        streaming = StreamingAttribution(UNITS, scorer_with(calls), 0.5)
        streaming.feed("НГУ принимает документы с двадцатого июня до конца июля.\nОбщежитие предоставля")
        await asyncio.sleep(0)
        assert calls == ["НГУ принимает документы с двадцатого июня до конца июля."]
        text = (
            "НГУ принимает документы с двадцатого июня до конца июля.\n"
            "Общежитие предоставляется всем иногородним студентам первого курса."
        )
        await streaming.finish(text)
        assert calls[1:] == ["Общежитие предоставляется всем иногородним студентам первого курса."]

    asyncio.run(run())


def test_a_scoring_error_while_streaming_gives_no_sources_as_before():
    async def failing(sentence, texts):
        raise RuntimeError("cross-encoder down")

    pipeline = ragu_fakes.pipeline(source_scorer=failing)
    pipeline.resources.documents[0]["url"] = "https://example.org/a"
    runtime = PipelineRuntime.uniform(ModelRuntime("fake-model", "http://fake/v1"))
    outcome = asyncio.run(pipeline.prepare(messages=[ChatMessage(role="user", content="ФИТ")], runtime=runtime))

    async def run():
        streaming = pipeline.start_source_attribution(outcome)
        text = "На факультете учатся студенты НГУ.\nЕщё одно длинное предложение об этом."
        streaming.feed(text)
        return await pipeline.finalize_sources(outcome, text, streaming=streaming)

    assert asyncio.run(run()) == []
    assert outcome.stage_details["source_attribution"]["timings"]["failed"] is True


def test_cancel_stops_the_scoring_still_running():
    async def run():
        started = asyncio.Event()

        async def slow(sentence, texts):
            started.set()
            await asyncio.sleep(10)
            return [1.0] * len(texts)

        streaming = StreamingAttribution(UNITS, slow, 0.5)
        streaming.feed("НГУ принимает документы с двадцатого июня до конца июля.\n")
        await started.wait()
        streaming.cancel()
        await asyncio.sleep(0)
        return [task.cancelled() for task in streaming._scores.values()]

    assert asyncio.run(run()) == [True]


def test_the_pipeline_s_streamed_sources_are_its_sources_at_the_end():
    calls = []
    pipeline = ragu_fakes.pipeline(source_scorer=scorer_with(calls))
    pipeline.resources.documents[0]["url"] = "https://example.org/a"
    runtime = PipelineRuntime.uniform(ModelRuntime("fake-model", "http://fake/v1"))
    outcome = asyncio.run(pipeline.prepare(messages=[ChatMessage(role="user", content="ФИТ")], runtime=runtime))
    text = "На факультете учатся студенты НГУ.\n- Ещё одно длинное предложение об НГУ.\nИсточник: сайт."
    at_end = copy.deepcopy(outcome)

    async def run():
        streaming = pipeline.start_source_attribution(outcome)
        so_far = ""
        for token in chunks(random.Random(3), text):
            so_far += token
            streaming.feed(so_far)
            await asyncio.sleep(0)
        streamed = await pipeline.finalize_sources(outcome, text, streaming=streaming)
        whole = await pipeline.finalize_sources(at_end, text)
        return streamed, whole

    streamed, whole = asyncio.run(run())
    assert streamed == whole == outcome.sources == at_end.sources
    streamed_timings = outcome.stage_details["source_attribution"]["timings"]
    whole_timings = at_end.stage_details["source_attribution"]["timings"]
    assert streamed_timings["streamed"] == 2  # both claims end with a line break: scored while streaming
    assert {key: streamed_timings[key] for key in ("sentences", "units", "pairs")} == {
        key: whole_timings[key] for key in ("sentences", "units", "pairs")
    }
