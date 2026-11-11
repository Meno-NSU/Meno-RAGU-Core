import asyncio
import json

import httpx
import pytest

from meno_rag.stand.attribution import (
    EvidenceUnit,
    RerankerScorer,
    attribute,
    document_units,
    norm_url,
    split_sentences,
    to_sources,
)

SUMMARY = {
    "doc_title": "Руководство студента",
    "url": ["https://www.nsu.ru/n/education/stipendii/", "https://www.nsu.ru/n/life/campus/"],
}


def test_sentences_skip_short_ones_and_appended_markers():
    answer = "Да.\nПара длится полтора часа с перерывом.\n_Ответ сформирован автоматически._\nИсточник: сайт НГУ и его страницы."
    assert split_sentences(answer) == ["Пара длится полтора часа с перерывом."]


def test_norm_url_ignores_scheme_www_case_and_trailing_slash():
    assert norm_url("https://WWW.nsu.ru/n/faq/") == norm_url("http://nsu.ru/n/faq") == "nsu.ru/n/faq"
    assert norm_url("https://table.nsu.ru/teacher/a?x=1") == "table.nsu.ru/teacher/a?x=1"


def test_single_page_document_gives_its_chunks_with_the_page_link():
    document = {"doc_title": "Общежития", "url": "https://www.nsu.ru/n/life/campus/dormitories/"}
    assert document_units(document, ["Общежитие №1 находится на ул. Пирогова.", "Заселение в августе."]) == [
        EvidenceUnit("Общежитие №1 находится на ул. Пирогова.", "Общежития", "https://www.nsu.ru/n/life/campus/dormitories/"),
        EvidenceUnit("Заселение в августе.", "Общежития", "https://www.nsu.ru/n/life/campus/dormitories/"),
    ]


def test_page_compilation_gives_each_line_the_link_written_next_to_it():
    chunk = ("**Расписание занятий**\n"
             "- Расписание — на [table.nsu.ru](http://table.nsu.ru/), пара длится полтора часа.\n"
             "- Стипендии назначаются по итогам сессии https://www.nsu.ru/n/education/stipendii/ каждый семестр.\n"
             "- Нагрузка составляет около тридцати часов в неделю.")
    assert document_units(SUMMARY, [chunk]) == [
        EvidenceUnit("- Расписание — на [table.nsu.ru](http://table.nsu.ru/), пара длится полтора часа.",
                     "Руководство студента", "http://table.nsu.ru/"),
        EvidenceUnit("- Стипендии назначаются по итогам сессии https://www.nsu.ru/n/education/stipendii/ каждый семестр.",
                     "Руководство студента", "https://www.nsu.ru/n/education/stipendii/"),
        EvidenceUnit("- Нагрузка составляет около тридцати часов в неделю.", "Руководство студента", ""),
    ]


def test_document_without_a_link_gives_units_without_one():
    assert document_units({"doc_title": "Памятка", "url": None}, ["Текст памятки для студентов."]) == [
        EvidenceUnit("Текст памятки для студентов.", "Памятка", "")]


def scorer_from(table):
    async def score(sentence, texts):
        return [table.get((sentence, text), 0.0) for text in texts]
    return score


def test_attribution_keeps_units_some_sentence_rests_on_best_first():
    units = [EvidenceUnit("a", "A", "https://a"), EvidenceUnit("b", "B", "https://b"), EvidenceUnit("c", "C", "https://c")]
    answer = "Первое утверждение ответа здесь. Второе утверждение ответа тоже здесь."
    s1, s2 = split_sentences(answer)
    scored = asyncio.run(attribute(answer, units, scorer_from({(s1, "a"): 0.6, (s2, "a"): 0.2, (s2, "c"): 0.9,
                                                                (s1, "b"): 0.4}), 0.5))
    assert scored == [(units[2], 0.9), (units[0], 0.6)]


def test_attribution_of_an_answer_without_claims_is_empty():
    units = [EvidenceUnit("a", "A", "https://a")]
    assert asyncio.run(attribute("Привет!", units, scorer_from({}), 0.5)) == []
    assert asyncio.run(attribute("Первое утверждение ответа здесь.", [], scorer_from({}), 0.5)) == []


def test_sources_are_one_row_per_page_best_first_and_skip_pieces_without_a_link():
    scored = [
        (EvidenceUnit("x", "Расписание Иванова", "https://table.nsu.ru/teacher/1"), 0.95),
        (EvidenceUnit("y", "Публикации Иванова", "https://table.nsu.ru/teacher/1/"), 0.9),
        (EvidenceUnit("z", "Руководство студента", ""), 0.85),
        (EvidenceUnit("w", "Стипендии", "https://www.nsu.ru/n/education/stipendii/"), 0.8),
        (EvidenceUnit("v", "Общежития", "https://www.nsu.ru/n/life/campus/"), 0.7),
    ]
    assert to_sources(scored, max_sources=2) == [
        {"document_title": "Расписание Иванова", "source_url": "https://table.nsu.ru/teacher/1"},
        {"document_title": "Стипендии", "source_url": "https://www.nsu.ru/n/education/stipendii/"},
    ]


def test_reranker_scorer_posts_one_request_per_sentence_and_keeps_the_text_order():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, body["model"], body["text_1"], body["text_2"]))
        return httpx.Response(200, json={"data": [{"index": 1, "score": 0.2}, {"index": 0, "score": 0.7}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            scorer = RerankerScorer(client, "http://reranker:8002/v1", "BAAI/bge-reranker-v2-m3", "EMPTY")
            return await scorer("вопрос", ["первый", "второй"])

    assert asyncio.run(run()) == [0.7, 0.2]
    assert seen == [("/v1/score", "BAAI/bge-reranker-v2-m3", "вопрос", ["первый", "второй"])]


def test_reranker_scorer_raises_on_service_errors():
    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "down"}))
        async with httpx.AsyncClient(transport=transport) as client:
            await RerankerScorer(client, "http://reranker:8002/v1", "m", "EMPTY")("q", ["t"])

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())


def test_a_named_link_of_a_compilation_gives_its_own_title():
    chunk = ("- Действует [Режим занятий обучающихся в НГУ](https://www.nsu.ru/upload/rezhim.pdf) с 2015 года.\n"
             "- Расписание публикуется на [table.nsu.ru](http://table.nsu.ru/) перед семестром.")
    assert [(unit.title, unit.url) for unit in document_units(SUMMARY, [chunk])] == [
        ("Режим занятий обучающихся в НГУ", "https://www.nsu.ru/upload/rezhim.pdf"),
        ("Руководство студента", "http://table.nsu.ru/"),
    ]


def test_escaped_brackets_inside_a_link_stay_part_of_it():
    chunk = ("- Его [профиль на исследовательском портале НГУ]"
             "(https://pure.nsu.ru/portal/ru/persons/--\\(0a947727-a6bd\\).html) содержит публикации.")
    assert [(unit.title, unit.url) for unit in document_units(SUMMARY, [chunk])] == [
        ("профиль на исследовательском портале НГУ", "https://pure.nsu.ru/portal/ru/persons/--(0a947727-a6bd).html")]


TEACHER = (
    "# Иванов Иван Иванович\n\n"
    "## Учебные дисциплины\n\n- Теория вероятностей и математическая статистика;\n\n"
    "## Расписание (часть 1/2)\n\nПонедельник 9:00 практическое занятие Физика 2, аудитория 411 ГК\n"
    "Вторник 10:50 лекция Физика 2, аудитория 220 ГК\n\n"
    "## Публикации (часть 3/9)\n\nСтатья о волновых пакетах в журнале Physics of Fluids.\n"
)
SECTIONS = [
    {"heading": "Учебные дисциплины", "url": "https://www.nsu.ru/n/sveden/employees/", "title": "Педагогический состав НГУ"},
    {"heading": "Расписание", "url": "https://table.nsu.ru/teacher/1", "title": "Расписание: Иванов Иван Иванович"},
    {"heading": "Публикации", "url": "", "title": ""},
]


def test_a_document_of_several_sources_gives_each_section_the_page_it_was_taken_from():
    document = {"doc_title": "Иванов Иван Иванович", "url": "https://table.nsu.ru/teacher/1",
                "doc_full_text": TEACHER, "sections": SECTIONS}
    units = document_units(document, [TEACHER.strip()])
    assert [(unit.url, unit.title) for unit in units] == [
        ("https://www.nsu.ru/n/sveden/employees/", "Педагогический состав НГУ"),
        ("https://table.nsu.ru/teacher/1", "Расписание: Иванов Иван Иванович"),
        ("", "Иванов Иван Иванович"),  # no page of its own: no source rather than a wrong one
    ]
    assert units[1].text.startswith("## Расписание (часть 1/2)") and "Вторник" in units[1].text


def test_a_chunk_that_starts_inside_a_section_keeps_that_section_page():
    document = {"doc_title": "Иванов Иван Иванович", "url": "https://table.nsu.ru/teacher/1",
                "doc_full_text": TEACHER, "sections": SECTIONS}
    units = document_units(document, ["Вторник 10:50 лекция Физика 2, аудитория 220 ГК"])
    assert [(unit.url, unit.title) for unit in units] == [("https://table.nsu.ru/teacher/1", "Расписание: Иванов Иван Иванович")]
