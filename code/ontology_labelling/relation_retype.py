"""gemma retypes graph relations with the NSU ontology: from the relation's own text it picks, among the types the
ontology allows between the (new) types of its two ends — in either direction — the one that states the fact, or
RELATED_TO when none does. The type the extraction gave stays in the output next to the new one.

usage: python -I relation_retype.py ONTOLOGY.yaml RELATIONS.jsonl ENTITY_LABELS.jsonl OUT.jsonl
  RELATIONS.jsonl: pilot_sample.py's format; ENTITY_LABELS.jsonl: entity_subtypes.py's output (name, type -> subtype)
"""
import asyncio, json, os, sys, time

from openai import AsyncOpenAI
from ragu.triplet.ontology import Ontology

ontology_path, relations_path, labels_path, out_path = sys.argv[1:5]
MODEL = os.environ.get('LABEL_MODEL', 'gemma-4-12B-it')
URL = os.environ.get('LABEL_URL', 'http://everest.nsu.ru:9022/v1')
BATCH, CONCURRENCY, PROMPT_VERSION = 15, int(os.environ.get('LABEL_CONCURRENCY', '16')), 'nsu-retype-v4'
GENERIC = {'ABBREVIATION', 'ALTERNATIVE_NAME'}
ontology = Ontology.from_yaml(ontology_path)
subtype = {(r['name'], r['type']): r['subtype'] for r in map(json.loads, open(labels_path, encoding='utf-8'))}
SYSTEM = (
    'Ты уточняешь типы связей графа знаний о Новосибирском государственном университете и Академгородке. Для каждой '
    'связи даны два узла A и B с типами, тип, который поставила модель при извлечении, текст связи из документа и '
    'допустимые типы: «A→B» — A субъект, B объект; «B→A» — B субъект, A объект. Выбери тип, который точно называет '
    'факт из текста, и направление (reverse=true, если тип из списка «B→A»).\n'
    'Правила:\n'
    '- Если текст прямо называет связь, бери конкретный тип, а не RELATED_TO: соавторство — KNOWS, научное '
    'руководство — SUPERVISES, работа в организации — WORKPLACE, членство — MEMBER_OF, руководство подразделением — '
    'HEAD_OF, учёба — STUDIES_AT.\n'
    '- Место: объект находится или найден в месте — LOCATED_IN; мероприятие, занятия, процесс проходят в месте или '
    'организации — TAKES_PLACE_IN.\n'
    '- AUTHOR_OF — только автор статьи, книги или диссертации; публикация в журнале или сборнике (и в сборнике '
    'конференции) — PUBLISHED_IN; доклад на мероприятии — PRESENTED_AT; редактор — EDITOR_OF.\n'
    '- PART_OF — только если A — составная часть B того же рода (кафедра — часть факультета, глава — часть книги, '
    'район — часть города). Не PART_OF: место — LOCATED_IN; мероприятие — TAKES_PLACE_IN или ORGANIZES; статья в '
    'сборнике — PUBLISHED_IN; дисциплина в программе — IN_PROGRAM; доклад — PRESENTED_AT; тема — RESEARCHES или '
    'RELATED_TO; причина — HAS_CAUSE.\n'
    '- ALTERNATIVE_NAME — только другое имя того же самого объекта (сокращение, перевод, прежнее название); '
    '«другой подход», «разновидность», «профиль кафедры» — не другое имя.\n'
    '- RESEARCHES — только если субъект изучает объект как тему исследования; инструмент или метод, которым '
    'пользуются, — USES (только если в тексте сказано, что используют); авторство — AUTHOR_OF.\n'
    '- Не выбирай тип по одним типам узлов: «разработал курс» — не TEACHES.\n'
    '- Если тип, поставленный при извлечении, есть среди допустимых и верно называет факт, оставь его.\n'
    '- RELATED_TO — когда текст говорит о связи, но ни один допустимый тип её не называет (текст сохранится).\n'
    'Верни JSON со списком labels: номер связи i, type, reverse.'
)
SCHEMA = {'type': 'object', 'properties': {'labels': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'i': {'type': 'integer'}, 'type': {'type': 'string'}, 'reverse': {'type': 'boolean'}},
    'required': ['i', 'type', 'reverse']}}}, 'required': ['labels']}


def options(st, ot, original):
    # the two name relations are allowed between any types, so they are offered only when the extraction chose them
    keep = lambda t: t != 'RELATED_TO' and (t not in GENERIC or t == original)
    fwd = [t for t in ontology.allowed_relations(st, ot) if keep(t)]
    rev = [t for t in ontology.allowed_relations(ot, st) if keep(t)]
    return fwd, rev


rows = [json.loads(l) for l in open(relations_path, encoding='utf-8') if l.strip()]
done = set()
if os.path.exists(out_path):
    done = {r['id'] for r in map(json.loads, open(out_path, encoding='utf-8'))}
todo = [r for r in rows if r['id'] not in done]
for r in todo:
    r['st'] = subtype.get((r['subject']['name'], r['subject']['type']), r['subject']['type'])
    r['ot'] = subtype.get((r['object']['name'], r['object']['type']), r['object']['type'])
    r['fwd'], r['rev'] = options(r['st'], r['ot'], ontology.normalize_type_name(r['relation_type'].removeprefix('UNFIT:')))
batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
print(len(rows), 'relations,', len(todo), 'to retype in', len(batches), 'batches', flush=True)
client = AsyncOpenAI(base_url=URL, api_key='EMPTY', timeout=300)
out = open(out_path, 'a', encoding='utf-8')
gate = asyncio.Semaphore(CONCURRENCY)
started = time.monotonic()


async def retype(batch):
    lines = '\n'.join(
        f'{i}. A = «{r["subject"]["name"]}» [{r["st"]}]; B = «{r["object"]["name"]}» [{r["ot"]}]; извлечено как '
        f'A —{r["relation_type"].removeprefix("UNFIT:")}→ B; текст: {r["description"][:240]}; допустимые: '
        f'A→B: {", ".join(r["fwd"]) or "—"}; B→A: {", ".join(r["rev"]) or "—"}; RELATED_TO'
        for i, r in enumerate(batch))
    got = None
    async with gate:
        for _ in range(3):
            try:
                reply = await client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=2000,
                    messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': lines}],
                    response_format={'type': 'json_schema', 'json_schema': {'name': 'labels', 'schema': SCHEMA}},
                    extra_body={'chat_template_kwargs': {'enable_thinking': False}} if 'qwen' in MODEL else None)
                got = {x['i']: x for x in json.loads(reply.choices[0].message.content)['labels']}
                break
            except Exception:  # noqa: BLE001 - retried; a batch that keeps failing stays unlabeled
                got = None
    for i, r in enumerate(batch):
        if not got or i not in got:
            continue
        t, rev = ontology.normalize_type_name(got[i]['type']), bool(got[i]['reverse'])
        allowed = (r['rev'] if rev else r['fwd']) + ['RELATED_TO']
        valid = t in allowed
        out.write(json.dumps({'id': r['id'], 'subject': [r['subject']['name'], r['subject']['type']],
                              'object': [r['object']['name'], r['object']['type']],
                              'original': r['relation_type'], 'subject_type': r['st'], 'object_type': r['ot'],
                              'type': t if valid else 'RELATED_TO', 'reverse': rev and valid and t != 'RELATED_TO',
                              'model_choice_valid': valid, 'model_choice': [got[i]['type'], bool(got[i]['reverse'])],
                              'model': MODEL, 'prompt': PROMPT_VERSION},
                             ensure_ascii=False) + '\n')


async def main():
    await asyncio.gather(*(retype(b) for b in batches))
    out.close()

asyncio.run(main())
print('retyped in', round(time.monotonic() - started), 's', flush=True)
