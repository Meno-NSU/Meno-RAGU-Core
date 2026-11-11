import asyncio
import dataclasses

import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.routing import NO_CONTEXT_RULES, ROUTE_RULES, STYLE_RULE, parse_route, with_route_rules
from tests.ragu_fakes import LLM, pipeline
from tests.test_ragu_sources_late import RUNTIME


def test_a_route_is_read_from_the_json_reply_and_short_talk_about_nsu_counts_as_a_question():
    assert parse_route('{"route": "general"}', "объясните производную") == "general"
    assert parse_route('{"route": "chitchat"}', "спасибо!") == "chitchat"
    assert parse_route('{"route": "chitchat"}', "спасибо, а где общежитие?") == "nsu"
    assert parse_route('ответ: {"route": "off_topic"}', " ".join(["слово"] * 13)) == "nsu"
    with pytest.raises(ValueError):
        parse_route("поисковый запрос", "ФИТ")
    with pytest.raises(ValueError):
        parse_route('{"route": "weather"}', "ФИТ")


def test_the_rules_follow_the_route_and_found_documents():
    assert ROUTE_RULES["general"] in with_route_rules("SYSTEM", "general", has_context=True)
    assert NO_CONTEXT_RULES["general"] in with_route_rules("SYSTEM", "general", has_context=False)
    chitchat = with_route_rules("SYSTEM", "chitchat", has_context=False)
    assert chitchat.startswith("SYSTEM") and ROUTE_RULES["chitchat"] in chitchat and STYLE_RULE in chitchat


class RoutedLLM(LLM):
    def __init__(self, route):
        super().__init__()
        self.route = route

    async def chat_completion_text(self, **kwargs):
        if kwargs.get("stage") == "turn_routing":
            return f'{{"route": "{self.route}"}}'
        return await super().chat_completion_text(**kwargs)


class CountingGraph:
    def __init__(self, graph):
        self.graph, self.calls = graph, 0

    async def search(self, query):
        self.calls += 1
        return await self.graph.search(query)


def prepare(route, question="Спасибо!"):
    p = pipeline()
    p.llm_router = RoutedLLM(route)
    p.resources = dataclasses.replace(p.resources, graph=CountingGraph(p.resources.graph))
    outcome = asyncio.run(p.prepare(messages=[ChatMessage(role="user", content=question)], runtime=RUNTIME))
    return p, outcome


def test_a_greeting_is_answered_without_search_by_its_rules():
    p, outcome = prepare("chitchat")
    assert p.resources.graph.calls == 0
    assert outcome.context == "" and outcome.sources == [] and outcome.search_queries == []
    assert ROUTE_RULES["chitchat"] in outcome.qa_messages[0]["content"]
    assert outcome.stage_details["query_rewrite"]["route"] == "chitchat"


def test_a_general_question_still_searches_and_may_use_general_knowledge():
    p, outcome = prepare("general", "Объясните, что такое производная")
    assert p.resources.graph.calls > 0 and outcome.context
    assert ROUTE_RULES["general"] in outcome.qa_messages[0]["content"]


def test_a_failed_routing_call_keeps_the_previous_behaviour():
    p = pipeline()
    p.resources = dataclasses.replace(p.resources, graph=CountingGraph(p.resources.graph))
    outcome = asyncio.run(p.prepare(messages=[ChatMessage(role="user", content="ФИТ")], runtime=RUNTIME))
    assert p.resources.graph.calls > 0 and outcome.context
    assert ROUTE_RULES["nsu"] in outcome.qa_messages[0]["content"]
    assert outcome.stage_details["query_rewrite"]["route"] == "nsu"


def test_routing_can_be_switched_off():
    p = pipeline(turn_routing=False)
    p.llm_router = RoutedLLM("chitchat")
    outcome = asyncio.run(p.prepare(messages=[ChatMessage(role="user", content="Спасибо!")], runtime=RUNTIME))
    assert outcome.context and "# Как отвечать на эту реплику" not in outcome.qa_messages[0]["content"]


def test_a_history_of_several_messages_of_one_side_in_a_row_becomes_whole_turns():
    from meno_rag.stand.pipeline import extract_question_and_history

    messages = [ChatMessage(role="assistant", content="Здравствуйте!"),
                ChatMessage(role="user", content="Где ФИТ?"), ChatMessage(role="user", content="И как туда попасть?"),
                ChatMessage(role="assistant", content="ФИТ в главном корпусе."),
                ChatMessage(role="assistant", content="Вход с Пирогова."),
                ChatMessage(role="user", content="Без ответа"),
                ChatMessage(role="user", content="А часы работы?")]
    question, history = extract_question_and_history(messages)
    assert question == "А часы работы?"
    assert history == [{"role": "user", "content": "Где ФИТ?\nИ как туда попасть?"},
                       {"role": "assistant", "content": "ФИТ в главном корпусе.\nВход с Пирогова."}]


def test_a_question_after_an_unanswered_one_is_still_answered():
    p = pipeline()
    messages = [ChatMessage(role="user", content="Привет"), ChatMessage(role="user", content="ФИТ")]
    outcome = asyncio.run(p.prepare(messages=messages, runtime=RUNTIME))
    assert outcome.question == "ФИТ" and outcome.prepared_dialogue_history == ""
