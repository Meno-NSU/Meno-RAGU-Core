"""Dialogue handling of the RAGU stand, as Menon 1.1 has it: the user turns that got no answer stay in the history,
the recent turns are whole (a budget in tokens, not nine words of each answer), older turns become a summary, and one
analysis call gives the route, a question that stands without the history and the sub-queries the search uses."""
import asyncio
import json

from meno_rag.schemas import ChatMessage
from meno_rag.stand.dialogue_memory import render_history, split_history
from meno_rag.stand.pipeline import extract_question_and_history
from meno_rag.stand.prompts import QA_SYSTEM_PROMPT
from tests.ragu_fakes import LLM, pipeline
from tests.test_ragu_sources_late import RUNTIME

LONG_ANSWER = ("В лаборатории есть три прибора: генератор Г3-112, осциллограф Tektronix TDS2024 и USB-осциллограф "
               "Handyscope HS3 с частотой дискретизации до 100 МВыб/с и разрядностью АЦП 8/12/14/16 бит.")


def messages(*turns):
    return [ChatMessage(role=role, content=content) for role, content in turns]


def test_user_turns_that_got_no_answer_stay_in_the_history():
    question, history = extract_question_and_history(messages(
        ("user", "Когда начинается учёба?"), ("user", "А какого числа?")))
    assert question == "А какого числа?"
    assert history == [{"role": "user", "content": "Когда начинается учёба?"}]
    _, history = extract_question_and_history(messages(
        ("user", "Привет"), ("assistant", "Здравствуйте!"), ("user", "аываы"), ("user", "Что я спрашивал вначале?")))
    assert [m["content"] for m in history] == ["Привет", "Здравствуйте!", "аываы"]


def test_recent_answers_stay_whole_and_older_turns_go_to_the_summary():
    history = [{"role": "user", "content": "Какие приборы есть в лаборатории?"},
               {"role": "assistant", "content": LONG_ANSWER}]
    recent, older = split_history(history, budget_tokens=5000)
    assert older == [] and LONG_ANSWER in render_history(recent, "")
    many = [m for i in range(40) for m in ({"role": "user", "content": f"вопрос {i} " + "слово " * 60},
                                           {"role": "assistant", "content": f"ответ {i} " + "слово " * 60})]
    recent, older = split_history(many, budget_tokens=500)
    assert older and recent and recent[-1]["content"].startswith("ответ 39")
    assert "Сводка: о чём шла речь" in render_history(recent, "о чём шла речь")


class DialogueLLM(LLM):
    """The analysis call answers with the given JSON; every other call records its messages."""

    def __init__(self, analysis):
        super().__init__()
        self.analysis, self.calls = analysis, []

    async def chat_completion_text(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stage") == "turn_routing":
            return json.dumps(self.analysis, ensure_ascii=False)
        return await super().chat_completion_text(**kwargs)


def prepare(analysis, turns):
    p = pipeline()
    p.llm_router = DialogueLLM(analysis)
    return p, asyncio.run(p.prepare(messages=messages(*turns), runtime=RUNTIME))


def test_the_search_uses_the_question_that_stands_without_the_history_and_its_subqueries():
    p, outcome = prepare(
        {"route": "nsu", "standalone_question": "Какая частота дискретизации у USB-осциллографа Handyscope HS3?",
         "subqueries": ["Разрядность АЦП Handyscope HS3"], "chain": False, "focus": "лаборатория"},
        [("user", "Какие приборы есть в лаборатории?"), ("assistant", LONG_ANSWER),
         ("user", "а у третьего какая частота дискретизации?")])
    analysis_call = next(c for c in p.llm_router.calls if c.get("stage") == "turn_routing")
    assert LONG_ANSWER[:150] in analysis_call["messages"][1]["content"]
    rewrite_calls = [c for c in p.llm_router.calls if c.get("stage") == "query_rewrite"]
    assert rewrite_calls and "Handyscope HS3" in rewrite_calls[-1]["messages"][-1]["content"]
    assert "Разрядность АЦП Handyscope HS3" in outcome.search_queries
    assert LONG_ANSWER in outcome.qa_messages[1]["content"]
    assert outcome.stage_details["query_rewrite"]["standalone_question"].startswith("Какая частота")


def test_a_question_about_the_dialogue_is_answered_from_the_whole_history_without_search():
    p, outcome = prepare(
        {"route": "meta_dialog", "standalone_question": "О чём был первый вопрос?", "subqueries": [], "chain": False,
         "focus": ""},
        [("user", "Привет, как ты?"), ("user", "аываы"), ("user", "А в самом начале диалога, что я спрашивал?")])
    assert outcome.search_queries == [] and outcome.context == ""
    assert "Привет, как ты?" in outcome.qa_messages[1]["content"]


def test_the_answer_prompt_asks_to_check_the_dates_of_the_documents():
    assert "не выдавайте прошедшее за текущее" in QA_SYSTEM_PROMPT
    assert "опирайтесь на более свежий" in QA_SYSTEM_PROMPT


def test_a_question_about_the_dialogue_gets_only_what_was_asked():
    # The judge scored RAGU's correct answers to "what did I ask first?" at ~0.43 against Menon's ~0.99: RAGU named the
    # question and then retold its own answer to it ("Я ответил вам, что…"), which the user did not ask for.
    from meno_rag.stand.qa import prepare_prompt_for_question_answering
    from meno_rag.stand.routing import with_route_rules

    system = with_route_rules(QA_SYSTEM_PROMPT, "meta_dialog", has_context=False)
    assert "не пересказывайте свой ответ" in system
    user = prepare_prompt_for_question_answering(
        user_question="О чём был мой первый вопрос?", dialogue_history="Пользователь: Привет", context="",
        abbr_dict={}, route="meta_dialog")
    assert "только на то, о чём спрашивают" in user
