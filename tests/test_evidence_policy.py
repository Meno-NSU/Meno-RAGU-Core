from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand import pipeline as pipeline_module
from meno_rag.stand.evidence_policy import DOCUMENT_FIRST_RULE, with_evidence_policy
from tests.ragu_fakes import LLM, pipeline
from tests.test_ragu_sources_late import RUNTIME


def test_legacy_policy_keeps_prompt_bytes(monkeypatch):
    monkeypatch.delenv("QA_EVIDENCE_POLICY", raising=False)
    assert Settings(_env_file=None).qa_evidence_policy == "legacy"
    assert with_evidence_policy("SYSTEM\n\n", "legacy", "nsu", True) == "SYSTEM\n\n"


@pytest.mark.parametrize("route", ["nsu", "general", None])
def test_document_policy_is_applied_only_to_answers_with_sources(route):
    system = with_evidence_policy("SYSTEM\n\n", "document_first", route, True)
    assert system.startswith("SYSTEM\n\n")
    assert system.count(DOCUMENT_FIRST_RULE) == 1
    assert with_evidence_policy("SYSTEM\n\n", "document_first", route, False) == "SYSTEM\n\n"


@pytest.mark.parametrize("route", ["meta_dialog", "rework", "chitchat", "off_topic", "prompt_injection"])
def test_no_source_policy_is_added_to_dialogue_and_non_search_routes(route):
    assert with_evidence_policy("SYSTEM", "document_first", route, True) == "SYSTEM"


def test_policy_setting_is_validated(monkeypatch):
    monkeypatch.setenv("QA_EVIDENCE_POLICY", "document_first")
    assert Settings(_env_file=None).qa_evidence_policy == "document_first"
    with pytest.raises(ValidationError, match="QA_EVIDENCE_POLICY"):
        Settings(_env_file=None, QA_EVIDENCE_POLICY="any")


@pytest.mark.asyncio
async def test_policy_changes_only_qa_system_and_trace_not_search_or_model_calls(monkeypatch):
    monkeypatch.setattr(pipeline_module, "prompt_now", lambda _zone: datetime(2026, 11, 20, tzinfo=UTC))

    class RecordingLLM(LLM):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def chat_completion_text(self, **kwargs):
            self.calls.append(("text", kwargs))
            return await super().chat_completion_text(**kwargs)

        async def chat_completion(self, **kwargs):
            self.calls.append(("chat", kwargs))
            return await super().chat_completion(**kwargs)

    async def one(policy):
        p = pipeline(qa_evidence_policy=policy)
        llm = RecordingLLM()
        p.llm_router = llm
        outcome = await p.prepare(messages=[ChatMessage(role="user", content="Где НГУ?")],
                                  runtime=RUNTIME, capture_trace=True)
        return outcome, llm.calls

    legacy, before_calls = await one("legacy")
    revised, after_calls = await one("document_first")
    assert before_calls == after_calls
    assert revised.context == legacy.context and revised.evidence == legacy.evidence
    assert revised.search_queries == legacy.search_queries and revised.sources == legacy.sources
    assert revised.qa_messages[1:] == legacy.qa_messages[1:]
    assert DOCUMENT_FIRST_RULE in revised.qa_messages[0]["content"]
    assert DOCUMENT_FIRST_RULE not in legacy.qa_messages[0]["content"]
    assert revised.trace["runtime"]["settings"]["qa_evidence_policy"] == "document_first"
