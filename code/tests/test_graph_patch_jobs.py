import json

from graph_utils import graph_patch_jobs as jobs


def side(name):
    return {'name': name, 'type': 'FACULTY', 'description': f'{name} НГУ.', 'source': f'Декан: {name}.'}


def test_pairs_prompt_lists_every_pair_with_its_sources_and_the_stated_alias():
    prompt = jobs.pairs_prompt([{'a': side('ФИТ'), 'b': side('Факультет информационных технологий'), 'stated': ''},
                                {'a': side('ФФ'), 'b': side('ММФ'), 'stated': 'ФФ — другое название ММФ.'}])
    assert prompt.count('\nПара ') == 2 and 'A: «ФИТ» (тип FACULTY)' in prompt and 'Из источника B: Декан: ММФ.' in prompt
    assert prompt.count('Граф утверждает') == 1 and prompt.rstrip().endswith('...]')
    assert jobs.batches(list(range(25)), 10) == [list(range(10)), list(range(10, 20)), list(range(20, 25))]


def test_around_shows_the_part_of_the_chunk_that_has_the_name():
    text = 'x' * 500 + ' Кафедру возглавляет Иванов И. И. с 2020 года.' + 'y' * 500
    assert 'Иванов И. И. с 2020 года' in jobs.around(text, ['Петров', 'Иванов И. И.'], 200)
    assert jobs.around('Нет имён.', ['Иванов'], 200) == 'Нет имён.'


def test_other_prompts_state_the_question_and_the_answer_format():
    direction = jobs.direction_prompt('PART_OF', 'часть входит в целое', 'НГУ', 'ФИТ', 'ФИТ входит в НГУ.', 'ФИТ НГУ')
    assert '«НГУ» —PART_OF→ «ФИТ»' in direction and '"as_is|reversed|other"' in direction
    assert '«Марчук И. В.»' in jobs.person_chunk_prompt('Марчук И. В.', 'Декан.', '100-летие Г. И. Марчука')
    prompt = jobs.abbreviation_prompt('ИФП', ['Институт философии и права', 'Институт физики полупроводников'], 'ИФП СО РАН', 'лазеры')
    assert '2. Институт физики полупроводников' in prompt and '0. Ни один' in prompt


def test_answers_parse_with_fences_and_words_around():
    assert jobs.parse_answer('```json\n[{"n": 1, "verdict": "same"}]\n```') == [{'n': 1, 'verdict': 'same'}]
    assert jobs.parse_answer('Ответ: {"verdict": "reversed", "why": "x"}.') == {'verdict': 'reversed', 'why': 'x'}
    assert jobs.parse_answer('не знаю') is None and jobs.parse_answer('') is None


def test_the_runner_asks_once_caches_and_resumes(tmp_path):
    cache, out = tmp_path / 'cache.jsonl', tmp_path / 'out.jsonl'
    rows = [{'id': 'a', 'kind': 'direction', 'prompt': 'p1'}, {'id': 'b', 'kind': 'direction', 'prompt': 'p2'},
            {'id': 'c', 'kind': 'direction', 'prompt': 'p1'}]
    asked = []

    def call(prompt):
        asked.append(prompt)
        if prompt == 'p2' and asked.count('p2') <= 3:
            raise OSError('busy')
        return '{"verdict": "as_is"}' if prompt == 'p1' else 'что-то'

    counts = jobs.run_jobs(rows, cache, out, 'm', call, concurrency=1, retries=2)
    assert counts == {'jobs': 3, 'cached': 0, 'asked': 1, 'failed': 1, 'unparsed': 0}  # the same prompt is asked once
    assert [json.loads(r)['id'] for r in out.read_text().splitlines()] == ['a', 'c']
    counts = jobs.run_jobs(rows, cache, out, 'm', call, concurrency=1)
    assert counts == {'jobs': 3, 'cached': 2, 'asked': 1, 'failed': 0, 'unparsed': 1} and asked.count('p1') == 1
    results = [json.loads(r) for r in out.read_text().splitlines()]
    assert [r['parsed'] for r in results] == [{'verdict': 'as_is'}, None, {'verdict': 'as_is'}] and 'prompt' not in results[0]
    assert jobs.run_jobs(rows, cache, out, 'other-model', lambda p: '[]', concurrency=2)['asked'] == 2  # the model is in the key


def test_merge_candidates():
    from graph_utils import graph_patch as gp
    names = {1: 'Факультет информационных технологий (ФИТ)', 2: 'Факультет информационных технологий', 3: 'ФИТ',
             4: 'Столовая Новосибирского государственного университета', 5: 'Столовые Новосибирский государственный университет',
             6: 'Академпарк', 7: 'академпарк.', 8: 'Физика'}
    assert jobs.bracket_pairs(names) == {(1, 2), (1, 3)}
    assert jobs.stem_pairs(names) == {(4, 5)}
    assert jobs.same_key_pairs(names) == {(6, 7)}
    people = {1: 'Михаил Федорук', 2: 'Михаил Петрович Федорук', 3: 'В. Кузнецов', 4: 'Кузнецов Виктор Владимирович',
              5: 'Кузнецов Владимир Васильевич', 6: 'Иван Петров', 7: 'Петров И. С.'}
    assert jobs.person_pairs(people, gp.readings, gp.fits, gp.detail) == {(1, 2), (6, 7)}  # В. Кузнецов fits two people
