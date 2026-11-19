"""Registry of units, persons and appointments from structured NSU pages (proposal 3 of the navigation audit).
PREPARED, NOT RUN: `plan` only counts the inputs and estimates the calls; `run` is the extraction itself.

A source is described by where its pages are and which records they hold; the model fills the records, and a record
is kept only when its evidence is a verbatim fragment of the page and every value stands in that fragment. There
are no rules about particular units, persons or positions.

usage: registry_extract.py plan CORPUS.jsonl
       registry_extract.py run CORPUS.jsonl OUT.jsonl MODEL BASE_URL [CONCURRENCY]
"""
import asyncio, json, re, sys
from collections import Counter

PROMPT_VERSION = 'registry-v1'
# where the pages are (URL), what identifies the described object (group 1 of `id`), which records a page holds
SOURCES = {
    'staff_card': {'url': r'^https?://www\.nsu\.ru/n/sveden/employees/#', 'id': r'#(.+)$',
                   'records': ['Appointment', 'Teaching', 'Qualification', 'Contact']},
    'structure': {'url': r'^https?://www\.nsu\.ru/n/sveden/struktura-ngu/?$', 'id': None, 'records': ['Unit']},
    'pure_person': {'url': r'^https?://pure\.nsu\.ru/portal/ru/persons/[^/]*\(([0-9a-f-]{36})\)\.html$',
                    'id': r'\(([0-9a-f-]{36})\)', 'records': ['Appointment', 'Contact']},
    'pure_unit': {'url': r'^https?://pure\.nsu\.ru/portal/ru/organisations/[^/]*\(([0-9a-f-]{36})\)(\.html|/persons\.html)',
                  'id': r'\(([0-9a-f-]{36})\)', 'records': ['Unit', 'Appointment']},
}
RECORDS = {
    'Appointment': 'person, position, unit, parent_unit, employment_condition, is_head_of_unit (true only if the page '
                   'says the person heads this unit), valid_from, valid_to',
    'Unit': 'name, short_name, kind, parent_unit, valid_from, valid_to',
    'Teaching': 'person, discipline, program_code, program_name, academic_year',
    'Qualification': 'person, degree_or_title, speciality',
    'Contact': 'owner, kind (email, phone, room, address), value',
}
SYSTEM = (
    'Ты извлекаешь записи из страницы структурированного источника НГУ. Верни JSON-массив одной строкой. '
    'Каждая запись: {"type": тип, поля типа, "evidence": фрагмент}. evidence — непрерывный фрагмент страницы, '
    'скопированный дословно, без пропусков и многоточий; все значения записи должны стоять в этом фрагменте. '
    'Чего на странице нет — null. valid_from и valid_to заполняй только когда на странице написана дата или период '
    'этой записи; дату страницы и текущий год не подставляй. Если запись нельзя подтвердить фрагментом, не выводи её.'
)


def squeeze(text):
    return re.sub(r'\s+', ' ', str(text).replace('ё', 'е').lower()).strip()


def checked(records, page):
    """Records whose evidence is a verbatim fragment of the page; a value that is not in the evidence is dropped
    (booleans and nulls aside). Returns (kept records, counts)."""
    page_text, kept, counts = squeeze(page), [], Counter()
    for record in records if isinstance(records, list) else []:
        evidence = squeeze(record.get('evidence') or '') if isinstance(record, dict) else ''
        if not evidence or evidence not in page_text or record.get('type') not in RECORDS:
            counts['records_without_verbatim_evidence'] += 1
            continue
        clean = {'type': record['type'], 'evidence': record['evidence']}
        for field, value in record.items():
            if field in clean or value is None or isinstance(value, bool):
                if isinstance(value, bool):
                    clean[field] = value
                continue
            if squeeze(value) in evidence:
                clean[field] = value
            else:
                counts['values_not_in_evidence'] += 1
        kept.append(clean)
        counts['records_kept'] += 1
    return kept, counts


def pages(corpus_path):
    for line in open(corpus_path, encoding='utf-8'):
        if 'nsu.ru' not in line:
            continue
        doc = json.loads(line)
        urls = doc['url'] if isinstance(doc['url'], list) else [doc['url']]
        for kind, source in SOURCES.items():
            url = next((u for u in urls if isinstance(u, str) and re.search(source['url'], u)), None)
            if url:
                found = re.search(source['id'], url) if source['id'] else None
                yield kind, url, found.group(1) if found else None, doc
                break


def plan(corpus_path, chars_per_call=6000):
    stats = {kind: Counter() for kind in SOURCES}
    objects = {kind: set() for kind in SOURCES}
    for kind, url, object_id, doc in pages(corpus_path):
        text = doc['doc_full_text']
        stats[kind]['pages'] += 1
        stats[kind]['chars'] += len(text)
        stats[kind]['calls'] += max(1, -(-len(text) // chars_per_call))
        objects[kind].add(object_id or url)
    for kind in SOURCES:
        stats[kind]['objects'] = len(objects[kind])
    print(json.dumps({kind: dict(value) for kind, value in stats.items()}, ensure_ascii=False, indent=1))
    print('calls', sum(value['calls'] for value in stats.values()), 'chars', sum(value['chars'] for value in stats.values()))


async def run(corpus_path, out_path, model, base_url, concurrency=2, chars_per_call=6000):
    import httpx
    gate = asyncio.Semaphore(int(concurrency))
    totals = Counter()
    async with httpx.AsyncClient(timeout=600) as http:
        async def one(kind, url, object_id, doc, part, text, out):
            schema = '; '.join(f'{name}{{{RECORDS[name]}}}' for name in SOURCES[kind]['records'])
            body = {'model': model, 'temperature': 0, 'max_tokens': 4000, 'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f'Тип страницы: {kind}\nСхема записей: {schema}\nURL: {url}\n\nСтраница:\n{text}'}]}
            async with gate:
                answer = (await http.post(f'{base_url}/chat/completions', json=body)).json()['choices'][0]['message']['content']
            try:
                records = json.loads(answer[answer.index('['):answer.rindex(']') + 1])
            except ValueError:
                records, totals['unparsed'] = [], totals['unparsed'] + 1
            kept, counts = checked(records, text)
            totals.update(counts)
            out.write(json.dumps({'source': kind, 'url': url, 'object_id': object_id, 'part': part,
                                  'source_version': doc.get('kb_source_version_id'), 'as_of': doc.get('kb_release'),
                                  'generator': {'model': model, 'prompt_version': PROMPT_VERSION},
                                  'records': kept}, ensure_ascii=False) + '\n')
        with open(out_path, 'w', encoding='utf-8') as out:
            tasks = []
            for kind, url, object_id, doc in pages(corpus_path):
                text = doc['doc_full_text']
                for part, start in enumerate(range(0, max(len(text), 1), chars_per_call)):
                    tasks.append(one(kind, url, object_id, doc, part, text[start:start + chars_per_call], out))
            await asyncio.gather(*tasks)
    print(json.dumps(dict(totals), ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1] == 'plan':
        plan(sys.argv[2])
    else:
        asyncio.run(run(*sys.argv[2:7]))
