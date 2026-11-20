import asyncio
import json
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand import clock as clock_module
from meno_rag.stand import pipeline as pipeline_module
from meno_rag.stand.clock import prompt_now
from tests.ragu_fakes import LLM, pipeline
from tests.test_ragu_sources_late import RUNTIME


@pytest.mark.parametrize("hour, minute, expected", [
    (16, 59, "2026-11-19T23:59:00+07:00"),
    (17, 0, "2026-11-20T00:00:00+07:00"),
    (17, 30, "2026-11-20T00:30:00+07:00"),
])
def test_prompt_clock_uses_novosibirsk_calendar_at_utc_day_boundary(hour, minute, expected):
    local = prompt_now("Asia/Novosibirsk", moment=datetime(2026, 11, 19, hour, minute, tzinfo=UTC))
    assert local.isoformat() == expected
    if hour >= 17:
        assert local.date() == date(2026, 11, 20)
        assert local.date() + timedelta(days=1) == date(2026, 11, 21)


def test_prompt_clock_accepts_an_explicit_alternative_timezone_and_rejects_naive_instants():
    instant = datetime(2026, 11, 19, 17, 30, tzinfo=UTC)
    assert prompt_now("UTC", moment=instant).isoformat() == "2026-11-19T17:30:00+00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        prompt_now("Asia/Novosibirsk", moment=instant.replace(tzinfo=None))


def test_default_clock_reads_one_explicit_utc_instant(monkeypatch):
    calls = []

    class RecordingDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            return datetime(2026, 11, 19, 17, 30, tzinfo=UTC)

    monkeypatch.setattr(clock_module, "datetime", RecordingDatetime)
    assert prompt_now("Asia/Novosibirsk").isoformat() == "2026-11-20T00:30:00+07:00"
    assert calls == [UTC]


def test_prompt_timezone_setting_is_explicit_and_validated(monkeypatch):
    monkeypatch.delenv("PROMPT_TIMEZONE", raising=False)
    assert Settings(_env_file=None).prompt_timezone == "Asia/Novosibirsk"
    monkeypatch.setenv("PROMPT_TIMEZONE", "UTC")
    assert Settings(_env_file=None).prompt_timezone == "UTC"
    for invalid in ("Mars/NSU", "../UTC", ""):
        with pytest.raises(ValidationError, match="PROMPT_TIMEZONE"):
            Settings(_env_file=None, PROMPT_TIMEZONE=invalid)


@pytest.mark.parametrize("timezone, expected_date", [
    ("Asia/Novosibirsk", "2026-11-20"),
    ("UTC", "2026-11-19"),
])
def test_analysis_and_qa_share_one_local_clock_instant(monkeypatch, timezone, expected_date):
    clocks = []
    instant = datetime(2026, 11, 19, 17, 30, tzinfo=UTC)

    def clock(zone):
        clocks.append(zone)
        # A second clock read would cross another local day; a turn must keep its original reference date.
        return prompt_now(zone, moment=instant + timedelta(days=len(clocks) - 1))

    class RecordingLLM(LLM):
        analysis_system = ""

        async def chat_completion_text(self, **kwargs):
            if kwargs.get("stage") == "turn_routing":
                self.analysis_system = kwargs["messages"][0]["content"]
                return json.dumps({"route": "nsu", "standalone_question": "События НГУ сегодня и завтра",
                                   "subqueries": [], "chain": False, "focus": ""})
            return await super().chat_completion_text(**kwargs)

    monkeypatch.setattr(pipeline_module, "prompt_now", clock)
    p = pipeline(prompt_timezone=timezone)
    p.llm_router = RecordingLLM()
    outcome = asyncio.run(p.prepare(messages=[ChatMessage(role="user", content="Что в НГУ сегодня и завтра?")],
                                    runtime=RUNTIME, capture_trace=True))
    assert clocks == [timezone]
    assert f"Сегодня {expected_date}." in p.llm_router.analysis_system
    assert prompt_now(timezone, moment=instant).isoformat() in outcome.qa_messages[0]["content"]
    assert outcome.trace["runtime"]["settings"]["prompt_timezone"] == timezone


def test_database_clock_remains_aware_utc():
    from meno_rag.db.orm import utcnow

    assert utcnow().utcoffset() == timedelta(0)
