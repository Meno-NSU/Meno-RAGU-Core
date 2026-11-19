"""RERANK_PROMPT_ORDER (speed audit of 19.11, П1): "legacy" is the prompt as it was, byte for byte; "question_first"
has the same sections with the same texts, the shared ones first, and the turn's first candidate is scored alone
before the others (the warm-up that lets vLLM's prefix cache hold the shared part)."""

import asyncio
import random

import pytest
from nltk.stem.snowball import SnowballStemmer

from meno_rag.api import benchmark_config as config_mod
from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.stand.prompts import SYSTEM_PROMPT_FOR_RELEVANCE
from meno_rag.stand.rerank import build_prompt
from meno_rag.stand.rewriting import find_candidates_to_abbreviations
from tests import ragu_fakes

STEMMER = SnowballStemmer("russian")
ABBREVIATIONS = {
    key.lower(): {"abbreviation": key, "explanation": [f"расшифровка {key}"]} for key in ["НГУ", "ФИТ", "ММФ", "ИИ"]
}
WORDS_TEXT = "Факультет ФИТ НГУ ММФ ИИ приём студентов 2026 года общежитие кафедра расписание экзаменов «НГУ» — (ФИТ)."
WORDS = WORDS_TEXT.split()


def sample(rng):
    def text(low, high):
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(low, high)))

    history = rng.choice(["", "Пользователь: а на ММФ?\nАссистент: На ММФ тоже.", text(1, 10)])
    return text(1, 8) + "?", history, text(1, 60) or "Документ."


def old_build_prompt(
    user_question: str,
    dialogue_history: str,
    estimated_document: str,
    abbr_dict: dict[str, dict[str, str | list[str]]],
    stemmer: SnowballStemmer | None = None,
    is_json: bool = False,
) -> list[dict[str, str]]:
    """Build the reranker prompt that scores a candidate document by its USEFULNESS
    as a factual basis for answering the user's question (with dialogue history and
    the abbreviation dictionary), mirroring the QA prompt structure — rather than
    abstract topical relevance to a search query."""
    if not estimated_document:
        raise ValueError("The estimated document is empty.")
    found_abbreviations = find_candidates_to_abbreviations(
        source_text=estimated_document + "\n\n" + dialogue_history + "\n\n" + user_question,
        all_abbreviations=abbr_dict,
        stemmer=stemmer,
    )
    user_prompt = ""
    if found_abbreviations:
        user_prompt += "==========\nABBREVIATION DICTIONARY\n==========\n\n"
        user_prompt += (
            "**Природа данных:** предположительные расшифровки, подобранные автоматически"
            " по совпадению буквенной формы. Записи могут быть лишними, ошибочными или "
            "неоднозначными (омонимия). Используйте словарь как подсказку, а не как "
            "достоверный источник.\n\n"
        )
        user_prompt += "**Записи:**\n" + found_abbreviations + "\n\n"
    user_prompt += "==========\nDOCUMENT 1\n==========\n\n" + estimated_document.strip() + "\n\n"
    if dialogue_history:
        user_prompt += "==========\nDIALOGUE HISTORY\n==========\n\n" + dialogue_history + "\n\n"
    user_prompt += "==========\nCURRENT QUESTION\n==========\n\n" + user_question + "\n\n"
    user_prompt += "==========\nINSTRUCTION\n==========\n\n"
    user_prompt += (
        "Оцените, пожалуйста, возможность ответа на вопрос из раздела CURRENT QUESTION, "
        "опираясь только на факты из раздела DOCUMENT 1."
    )
    if dialogue_history:
        user_prompt += (
            "\n\nРаздел DIALOGUE HISTORY используйте исключительно для разрешения местоимений, "
            "уточнений и восстановления темы — не извлекайте из него фактические сведения."
        )
    if found_abbreviations:
        user_prompt += "\n\nРаздел ABBREVIATION DICTIONARY используйте по следующим правилам:\n"
        user_prompt += (
            "1. Применяйте расшифровку из словаря, только если сокращение действительно встречается в CURRENT QUESTION"
        )
        user_prompt += ", DIALOGUE HISTORY или DOCUMENT 1" if dialogue_history else " или DOCUMENT 1"
        user_prompt += " и по контексту является аббревиатурой.\n"
        user_prompt += (
            "2. Если одной аббревиатуре соответствует несколько расшифровок (омонимия), "
            "выберите ту, которая согласуется с темой диалога и содержанием документов.\n"
        )
        user_prompt += (
            "3. Если в тексте встречается буквосочетание, совпадающее с ключом словаря, "
            "но по контексту аббревиатурой не является (например, это команда, междометие, "
            "имя собственное, часть слова), — игнорируйте запись словаря.\n"
        )
        user_prompt += (
            "4. Если подходящая расшифровка в словаре отсутствует, а в документах её тоже нет, — "
            "не придумывайте расшифровку; используйте сокращение как есть.\n"
        )
        user_prompt += (
            "5. Словарь не является источником фактов: сам факт наличия записи «X — Y» не доказывает, "
            "что Y упомянут в документах или релевантен вопросу."
        )
    if is_json:
        modified_chapter_name = "## Формат ответа"
        found_idx = SYSTEM_PROMPT_FOR_RELEVANCE.find(modified_chapter_name)
        if found_idx < 0:
            raise RuntimeError(f"The system prompt does not contain the chapter `{modified_chapter_name}`.")
        modified_system_prompt = SYSTEM_PROMPT_FOR_RELEVANCE[:found_idx] + modified_chapter_name + "\n\n"
        modified_system_prompt += (
            'Ответьте в формате JSON вида `{"label": <your_label>}`, где '
            "`<your_label>` — это цифра 2, 1 или 0. Не добавляйте никаких пояснений, "
            "комментариев или дополнительного текста.\n"
        )
        return [{"role": "system", "content": modified_system_prompt}, {"role": "user", "content": user_prompt}]
    return [{"role": "system", "content": SYSTEM_PROMPT_FOR_RELEVANCE}, {"role": "user", "content": user_prompt}]


SECTIONS = ("ABBREVIATION DICTIONARY", "DOCUMENT 1", "DIALOGUE HISTORY", "CURRENT QUESTION", "INSTRUCTION")


def sections(user_prompt):
    """{section name: its text, without the blank line after it} and the order of the names."""
    found = sorted(
        (user_prompt.index(f"==========\n{name}\n=========="), name)
        for name in SECTIONS
        if f"==========\n{name}\n==========" in user_prompt
    )
    bodies = {}
    for (start, name), following in zip(found, [*found[1:], (len(user_prompt), None)], strict=True):
        bodies[name] = user_prompt[start : following[0]].rstrip("\n")
    return bodies, [name for _, name in found]


@pytest.mark.parametrize("is_json", [False, True])
def test_the_legacy_order_is_the_old_prompt_byte_for_byte(is_json):
    rng = random.Random(7)
    for _ in range(500):
        question, history, document = sample(rng)
        assert build_prompt(question, history, document, ABBREVIATIONS, STEMMER, is_json=is_json) == old_build_prompt(
            question, history, document, ABBREVIATIONS, STEMMER, is_json=is_json
        )
    assert build_prompt("ФИТ?", "", "Документ НГУ.", ABBREVIATIONS) == build_prompt(
        "ФИТ?", "", "Документ НГУ.", ABBREVIATIONS, order="legacy"
    )


@pytest.mark.parametrize("is_json", [False, True])
def test_question_first_has_the_same_sections_and_texts_with_the_shared_ones_first(is_json):
    rng = random.Random(8)
    for _ in range(300):
        question, history, document = sample(rng)
        legacy = build_prompt(question, history, document, ABBREVIATIONS, STEMMER, is_json=is_json)
        first = build_prompt(
            question, history, document, ABBREVIATIONS, STEMMER, is_json=is_json, order="question_first"
        )
        assert first[0] == legacy[0]  # the system prompt
        legacy_sections, _ = sections(legacy[1]["content"])
        first_sections, order = sections(first[1]["content"])
        assert first_sections == legacy_sections
        expected = [
            name
            for name in ("DIALOGUE HISTORY", "CURRENT QUESTION", "INSTRUCTION", "ABBREVIATION DICTIONARY", "DOCUMENT 1")
            if name in legacy_sections
        ]
        assert order == expected


def test_question_first_prompts_of_one_turn_share_everything_before_the_documents():
    question = "Как поступить на ФИТ НГУ?"
    prompts = [
        build_prompt(question, "", document, ABBREVIATIONS, STEMMER, order="question_first")[1]["content"]
        for document in ("Приём на ФИТ ведёт приёмная комиссия.", "Общежитие для студентов ММФ.")
    ]
    shared = 0
    while prompts[0][shared] == prompts[1][shared]:
        shared += 1
    assert shared >= prompts[0].index("==========\nABBREVIATION DICTIONARY")
    legacy = [
        build_prompt(question, "", document, ABBREVIATIONS, STEMMER)[1]["content"]
        for document in ("Приём на ФИТ ведёт приёмная комиссия.", "Общежитие для студентов ММФ.")
    ]
    shared_legacy = 0
    while legacy[0][shared_legacy] == legacy[1][shared_legacy]:
        shared_legacy += 1
    assert shared > shared_legacy  # the old order shared only the start of the dictionary section


def test_an_unknown_order_is_refused():
    with pytest.raises(ValueError):
        build_prompt("ФИТ?", "", "Документ.", {}, order="document_last")


class OrderLLM(ragu_fakes.LLM):
    """Records when every rerank call starts and ends; the same score for any prompt."""

    def __init__(self):
        super().__init__()
        self.events = []

    async def chat_completion(self, **kwargs):
        key = kwargs["messages"][1]["content"]
        self.events.append(("start", key))
        await asyncio.sleep(0.01)
        self.events.append(("end", key))
        return await super().chat_completion(**kwargs)


def turn(order):
    pipeline = ragu_fakes.pipeline(rerank_prompt_order=order)
    pipeline.llm_router = llm = OrderLLM()
    runtime = PipelineRuntime.uniform(ModelRuntime("fake-model", "http://fake/v1"))
    outcome = asyncio.run(
        pipeline.prepare(messages=[ChatMessage(role="user", content="Какие факультеты есть в НГУ?")], runtime=runtime)
    )
    return outcome, llm


def test_question_first_scores_the_first_candidate_alone_then_the_rest():
    _, llm = turn("question_first")
    starts = [index for index, (kind, _) in enumerate(llm.events) if kind == "start"]
    first_end = llm.events.index(("end", llm.events[starts[0]][1]))
    assert len(starts) == 3 and all(start > first_end for start in starts[1:])
    assert all(
        "==========\nCURRENT QUESTION" in prompt[1]["content"].split("==========\nDOCUMENT 1")[0]
        for prompt in llm.prompts
    )


def test_legacy_keeps_the_calls_concurrent_and_both_orders_select_the_same_with_the_same_scores():
    legacy, legacy_llm = turn("legacy")
    first, _ = turn("question_first")
    starts = [index for index, (kind, _) in enumerate(legacy_llm.events) if kind == "start"]
    # no warm-up: the second call starts before the first ends (the fixture's rerank semaphore lets two run)
    assert starts[:2] == [0, 1]
    assert first.context == legacy.context and first.sources == legacy.sources
    assert first.stage_details["rerank"] == legacy.stage_details["rerank"]


def test_the_order_is_part_of_the_benchmark_fingerprint():
    def runtime(order):
        return config_mod.runtime_identity(Settings(_env_file=None, rerank_prompt_order=order))

    assert runtime("legacy") != runtime("question_first")
    assert Settings(_env_file=None).rerank_prompt_order == "legacy"
