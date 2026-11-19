"""
The model passes of graph v6.1, prepared and run apart from the patch: a job is a row {id, kind, prompt, ...} whose
answer is one JSON value; the runner asks an OpenAI-compatible endpoint, keeps every answer in a cache keyed by the
model and the prompt, and can be stopped and started again.

Passes:
- pairs: do two nodes name one object (merge candidates and the ALTERNATIVE_NAME / ABBREVIATION relations), ten
  pairs to a call;
- direction: does a relation inside a cycle of a type that cannot hold both ways go the stated way;
- person_chunk: is a source chunk of a person node about that person (chunks that state other initials, nodes
  with two profiles of one site);
- abbreviation: which long form an abbreviation has in a chunk, among the ones the dictionary and the corpus give.
"""
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def around(text: str, names: list, width: int = 300) -> str:
    """The part of a chunk around the first of the names it has (the start of the chunk when it has none)."""
    folded = text.casefold()
    for name in names:
        for form in (name.casefold(), name.casefold()[:max(5, len(name) * 2 // 3)]):
            position = folded.find(form) if len(form) >= 3 else -1
            if position >= 0:
                start = max(0, position - width // 3)
                return ' '.join(text[start:start + width].split())
    return ' '.join(text[:width].split())


def _side(letter: str, row: dict) -> str:
    return (f'{letter}: «{row["name"]}» (тип {row["type"]}). Описание: {row["description"][:300]}\n'
            f'   Из источника {letter}: {row["source"][:300]}')


PAIRS_HEAD = ('Ты проверяешь граф знаний об университете. Ниже пары записей графа. Для каждой пары реши, называют ли '
              'записи A и B один и тот же объект: одного человека, одну организацию, одну публикацию, одно понятие.\n'
              'same — один объект: другое написание, сокращение, перевод названия, другой порядок слов имени.\n'
              'different — разные объекты: часть и целое, вид и род, пример и класс, соавторы, однофамильцы, '
              'соседние номера или годы, объект и документ о нём.\n'
              'unsure — по данным решить нельзя (например, совпадают только фамилия и инициалы, а сведения не '
              'пересекаются).\n'
              'Опирайся на описания и фрагменты источников, не на сходство названий.\n')
PAIRS_TAIL = ('\nОтветь только JSON-списком по одной записи на пару, в том же порядке: '
              '[{"n": 1, "verdict": "same|different|unsure", "why": "до 12 слов"}, ...]')


def pairs_prompt(pairs: list) -> str:
    """pairs: {a: {name, type, description, source}, b: {...}, stated: text of an alias relation or ''}."""
    parts = [PAIRS_HEAD]
    for number, pair in enumerate(pairs, 1):
        stated = f'\n   Граф утверждает: {pair["stated"][:250]}' if pair.get('stated') else ''
        parts.append(f'\nПара {number}.\n{_side("A", pair["a"])}\n{_side("B", pair["b"])}{stated}')
    parts.append(PAIRS_TAIL)
    return '\n'.join(parts)


def direction_prompt(kind: str, meaning: str, subject: str, obj: str, description: str, source: str) -> str:
    return ('Ты проверяешь направление связи в графе знаний.\n'
            f'Тип связи {kind}: {meaning}\n'
            f'Связь записана так: «{subject}» —{kind}→ «{obj}».\n'
            f'Описание связи: {description[:500]}\n'
            f'Фрагмент источника: {source[:700]}\n\n'
            'as_is — по описанию и источнику связь этого типа идёт именно так.\n'
            'reversed — связь этого типа верна в обратную сторону.\n'
            'other — между ними нет связи этого типа ни в какую сторону (другое отношение или один и тот же объект).\n'
            'Ответь только JSON: {"verdict": "as_is|reversed|other", "why": "до 12 слов"}')


def person_chunk_prompt(name: str, description: str, source: str) -> str:
    return ('Ты проверяешь, к тому ли человеку отнесён фрагмент источника в графе знаний.\n'
            f'Человек в графе: «{name}». Что о нём известно: {description[:500]}\n'
            f'Фрагмент источника: {source[:900]}\n\n'
            'this — во фрагменте упомянут именно этот человек.\n'
            'other — во фрагменте упомянут другой человек с той же или похожей фамилией (другие имя, отчество, '
            'инициалы, пол, эпоха или место работы), а этого человека нет.\n'
            'unsure — по фрагменту решить нельзя.\n'
            'Ответь только JSON: {"verdict": "this|other|unsure", "other_name": "имя другого человека, как в '
            'источнике, или пусто", "why": "до 12 слов"}')


def abbreviation_prompt(abbreviation: str, options: list, node_name: str, source: str) -> str:
    listed = '\n'.join(f'{number}. {option}' for number, option in enumerate(options, 1))
    return (f'Во фрагменте источника встречается сокращение «{abbreviation}». Реши, что оно означает именно здесь.\n'
            f'Варианты:\n{listed}\n0. Ни один из вариантов или решить нельзя.\n'
            f'Запись графа, к которой отнесён фрагмент: «{node_name}».\n'
            f'Фрагмент источника: {source[:900]}\n\n'
            'Ответь только JSON: {"choice": номер варианта, "why": "до 12 слов"}')


def batches(rows: list, size: int) -> list:
    return [rows[start:start + size] for start in range(0, len(rows), size)]


# ---------------------------------------------------------------------------------------------------------- runner

def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256((model + '\0' + prompt).encode()).hexdigest()


def parse_answer(text: str):
    """The JSON value of an answer, with or without a code fence or words around it; None when it has none."""
    text = re.sub(r'^```(?:json)?|```$', '', (text or '').strip(), flags=re.M).strip()
    for opening, closing in (('[', ']'), ('{', '}')):
        start, end = text.find(opening), text.rfind(closing)
        if 0 <= start < end and (opening == '[' or text.find('[') < 0 or text.find('[') > start):
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    return None


def load_cache(path) -> dict:
    cache = {}
    try:
        with open(path, encoding='utf-8') as rows:
            for row in rows:
                if row.strip():
                    row = json.loads(row)
                    cache[row['key']] = row['answer']
    except FileNotFoundError:
        pass
    return cache


def ask(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float = 600) -> str:
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                       'max_tokens': max_tokens}).encode()
    request = urllib.request.Request(base_url.rstrip('/') + '/chat/completions', data=body,
                                     headers={'content-type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())['choices'][0]['message']['content']


def run_jobs(jobs: list, cache_path, out_path, model: str, call, concurrency: int = 2, retries: int = 2,
             progress=None) -> dict:
    """Asks every job once (an answer in the cache is not asked again), appends answers to the cache as they come and
    writes {job fields, answer, parsed} rows to out_path. call: prompt -> answer text. Returns the counts."""
    cache = load_cache(cache_path)
    counts = {'jobs': len(jobs), 'cached': 0, 'asked': 0, 'failed': 0, 'unparsed': 0}
    waiting = {}
    for job in jobs:
        key = cache_key(model, job['prompt'])
        if key in cache:
            counts['cached'] += 1
        else:
            waiting.setdefault(key, job['prompt'])

    def attempt(prompt):
        error = None
        for _ in range(retries + 1):
            try:
                return call(prompt)
            except Exception as failure:  # the endpoint may drop a request under load; the job is asked on the next run
                error = failure
        raise error

    with open(cache_path, 'a', encoding='utf-8') as cache_rows, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(attempt, prompt): key for key, prompt in waiting.items()}
        for done, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                cache[key] = future.result()
            except Exception:
                counts['failed'] += 1
                continue
            counts['asked'] += 1
            cache_rows.write(json.dumps({'key': key, 'answer': cache[key]}, ensure_ascii=False) + '\n')
            cache_rows.flush()
            if progress and done % 100 == 0:
                progress(done, len(futures))
    with open(out_path, 'w', encoding='utf-8') as out:
        for job in jobs:
            answer = cache.get(cache_key(model, job['prompt']))
            if answer is None:
                continue
            parsed = parse_answer(answer)
            counts['unparsed'] += parsed is None
            row = {k: v for k, v in job.items() if k != 'prompt'}
            out.write(json.dumps({**row, 'answer': answer, 'parsed': parsed}, ensure_ascii=False) + '\n')
    return counts


# ------------------------------------------------------------------------------------------- merge candidates

def _key(name: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]|_', ' ', name.casefold().replace('ё', 'е')).split())


def bracket_pairs(names: dict) -> set:
    """"X (Y)" against a node named X or named Y. names: node id -> name."""
    by_key = {}
    for node_id, name in names.items():
        by_key.setdefault(_key(name), []).append(node_id)
    pairs = set()
    for node_id, name in names.items():
        match = re.match(r'^(.{3,}?)\s*\(([^()]{2,})\)\s*$', name)
        if match:
            for part in match.groups():
                pairs |= {tuple(sorted((node_id, other))) for other in by_key.get(_key(part), []) if other != node_id}
    return pairs


def same_key_pairs(names: dict) -> set:
    """One name key under several ids (several types, or spellings that differ in case and punctuation)."""
    by_key = {}
    for node_id, name in names.items():
        by_key.setdefault(_key(name), []).append(node_id)
    return {tuple(sorted((a, b))) for members in by_key.values() if 1 < len(members) <= 6
            for i, a in enumerate(members) for b in members[i + 1:]}


def stem_pairs(names: dict, max_group: int = 6) -> set:
    """Names of at least two words that have the same words up to their endings and order."""
    groups = {}
    for node_id, name in names.items():
        words = _key(name).split()
        if len(words) >= 2:
            groups.setdefault(' '.join(sorted(word[:5] for word in words)), []).append(node_id)
    return {tuple(sorted((a, b))) for members in groups.values() if 1 < len(members) <= max_group
            for i, a in enumerate(members) for b in members[i + 1:]}


def person_pairs(names: dict, readings, fits, detail) -> set:
    """Two forms of a person name where one fits the other and every fuller form the shorter one fits in the graph
    fits the others: the cases a document did not settle."""
    found = {node_id: readings(name) for node_id, name in names.items()}
    by_surname = {}
    for node_id, forms in found.items():
        for surname in {form[0] for form in forms}:
            by_surname.setdefault(surname, []).append(node_id)

    def best(a, b):
        fitting = [(x, y) for x in found[a] for y in found[b] if fits(x, y)]
        return max(fitting, key=lambda pair: detail(pair[0]) + detail(pair[1])) if fitting else None

    fuller = {}
    for surname, members in by_surname.items():
        if len(members) > 2000:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pair = best(a, b)
                if pair and pair[0][0] == surname and detail(pair[0]) + detail(pair[1]) > 0:
                    short, long = (a, b) if detail(pair[0]) <= detail(pair[1]) else (b, a)
                    fuller.setdefault(short, set()).add(long)
    pairs = set()
    for short, others in fuller.items():
        others = sorted(others)
        if all(best(a, b) for i, a in enumerate(others) for b in others[i + 1:]):
            pairs |= {tuple(sorted((short, other))) for other in others}
    return pairs
