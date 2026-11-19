from concurrent.futures import ThreadPoolExecutor
from itertools import product

import pytest
from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.abbreviation_cache import AbbreviationTokenCache
from meno_rag.stand.rewriting import find_candidates_to_abbreviations

ABBR = {"нгу": {"abbreviation": "НГУ", "explanation": ["Новосибирский университет"]},
        "фит": {"abbreviation": "ФИТ", "explanation": ["Факультет ИТ", "Другая расшифровка"]},
        "музей": {"abbreviation": "Музей", "explanation": ["Музей"]},
        "вк": {"abbreviation": "ВК", "explanation": ["Военная кафедра"]}}


@pytest.mark.parametrize("stemmer", [None, SnowballStemmer("russian")])
def test_fragment_union_matches_original_at_token_boundaries(stemmer):
    cache = AbbreviationTokenCache(stemmer)
    fragments = ["", "НГУ", "ФИТ", "музея", "ВК", "НГУ-", "-фит", "1.", "2", "a@example.ru", "ё\u0308", "музеи\n", "<a>", "нгу/фит"]
    for document, history, question in product(fragments, repeat=3):
        assert cache.find(document, history, question, ABBR) == find_candidates_to_abbreviations(
            document + "\n\n" + history + "\n\n" + question, ABBR, stemmer)


def test_current_dictionary_keys_and_explanations_are_never_cached():
    cache = AbbreviationTokenCache(SnowballStemmer("russian"))
    dictionary = {}
    assert cache.find("НГУ ФИТ", "", "", dictionary) == ""
    dictionary.update(ABBR)
    before = cache.find("НГУ ФИТ", "", "", dictionary)
    dictionary["нгу"] = {"abbreviation": "Changed", "explanation": ["Changed meaning"]}
    dictionary.pop("фит")
    after = cache.find("НГУ ФИТ", "", "", dictionary)
    assert before != after
    assert after == find_candidates_to_abbreviations("НГУ ФИТ\n\n\n\n", dictionary, cache.stemmer)


def test_actual_question_and_history_are_part_of_the_token_input():
    cache = AbbreviationTokenCache(None)
    assert cache.find("text", "", "НГУ", ABBR) != cache.find("text", "", "ФИТ", ABBR)
    assert cache.find("text", "ВК", "НГУ", ABBR) != cache.find("text", "", "НГУ", ABBR)


def test_cache_is_bounded_and_oversized_entries_are_not_retained():
    cache = AbbreviationTokenCache(None, max_entries=2, max_bytes=4096)
    for index in range(10):
        cache.tokens(str(index))
    assert cache.stats()["entries"] <= 2
    assert cache.stats()["accounted_bytes"] <= 4096
    before = cache.stats()["entries"]
    cache.tokens("large " * 1000)
    assert cache.stats()["entries"] == before


def test_concurrent_cache_access_keeps_the_same_result_and_budget():
    cache = AbbreviationTokenCache(SnowballStemmer("russian"), max_entries=4, max_bytes=16384)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: cache.find("ФИТ музея", "", "НГУ", ABBR), range(100)))
    assert len(set(results)) == 1
    assert cache.stats()["entries"] <= 4
    assert cache.stats()["accounted_bytes"] <= 16384


@pytest.mark.parametrize("order", ["legacy", "question_first"])
@pytest.mark.parametrize("is_json", [False, True])
def test_complete_messages_keep_the_same_bytes(order, is_json):
    from meno_rag.stand.rerank import build_prompt

    stemmer = SnowballStemmer("russian")
    cache = AbbreviationTokenCache(stemmer)
    args = ("НГУ ФИТ?", "Расскажи о музее.", "ФИТ НГУ: музеи и ВК.\n\nДругой абзац.", ABBR, stemmer)
    assert build_prompt(*args, order=order, is_json=is_json, abbreviation_cache=cache) == build_prompt(
        *args, order=order, is_json=is_json)


def test_cache_belongs_to_one_resource_instance_and_its_stemmer():
    from dataclasses import replace

    from tests.ragu_fakes import pipeline

    first = pipeline().resources
    second = replace(first)
    assert first.abbreviation_tokens is not second.abbreviation_tokens
    assert first.abbreviation_tokens.stemmer is first.stemmer
    first.abbreviation_tokens.tokens("НГУ")
    assert second.abbreviation_tokens.stats()["entries"] == 0
