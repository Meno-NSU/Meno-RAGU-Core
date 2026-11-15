"""gemma labels graph entities with the NSU ontology: the most specific type that fits (a subtype of its NEREL type,
or another type when the NEREL one is wrong) and what kind of node it is (a named object, a role or status, a
generic concept, an identifier, a garbled name). Resumable; keeps model and prompt version per label.

usage: python -I entity_subtypes.py ONTOLOGY.yaml ENTITIES.jsonl LABELS.jsonl [--limit N]
  ENTITIES.jsonl: one entity per line with name, type, description (the review sample's format or graph_sample.py's)
"""
import asyncio, json, os, sys, time
from pathlib import Path

from openai import AsyncOpenAI
from ragu.triplet.ontology import Ontology

ontology_path, entities_path, labels_path = sys.argv[1:4]
limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
MODEL = os.environ.get('LABEL_MODEL', 'gemma-4-12B-it')
URL = os.environ.get('LABEL_URL', 'http://everest.nsu.ru:9022/v1')
BATCH, CONCURRENCY, PROMPT_VERSION = 20, int(os.environ.get('LABEL_CONCURRENCY', '16')), 'nsu-subtype-v3'
KINDS = ['named', 'concept', 'vague', 'role', 'identifier', 'garbled']
VALUES = {'DATE', 'TIME', 'AGE', 'MONEY', 'NUMBER', 'PERCENT', 'ORDINAL'}
ontology = Ontology.from_yaml(ontology_path)
TYPES = sorted(t for t in ontology.entity_types if t not in VALUES)
type_lines = '\n'.join(f'- {name}: {spec.description}' + (f' (подтип {spec.parent})' if spec.parent else '')
                       for name, spec in ontology.entity_types.items() if name not in VALUES)
SYSTEM = (
    'Ты размечаешь узлы графа знаний о Новосибирском государственном университете и Академгородке. Для каждого узла '
    '(имя, нынешний тип, описание) верни:\n'
    'type — самый конкретный тип из списка ниже, который подходит объекту (подтип, если он есть; другой тип, если '
    'нынешний неверен). PRODUCT, WORK_OF_ART и CONCEPT — только если не подходит ни один более точный тип; '
    'LANGUAGE — только естественный язык (язык программирования — SOFTWARE).\n'
    'kind — что это за узел:\n'
    '- named — один конкретный объект со своим именем: человек, организация, публикация, программа, документ, '
    'мероприятие, конкретная геологическая или анатомическая единица («Сибирский кратон»);\n'
    '- concept — понятие, которое может быть темой вопроса: вещество, минерал, метод, технология, процедура, '
    'область науки, язык, явление, болезнь, общественный строй («Рутил», «Рентгеновская дифракция», «Технологический '
    'аудит», «Революционная бюрократия»);\n'
    '- vague — фраза без конкретного объекта, которая и не тема: безымянный объект или общее слово («Аудиторская '
    'организация», «рабочая секция», «соседи», «германское начало»);\n'
    '- role — роль, статус, профессия или степень человека, а не сам человек («Претендент», «Магистрант», «Студенты '
    'бакалавриата», «SMM-менеджер», «Кандидат физико-математических наук», «Заказчик»);\n'
    '- identifier — адрес почты, URL, номер кабинета, рисунка или группы, код («Группа 24503.1»);\n'
    '- garbled — обрезанное, склеенное или испорченное имя: слово или формула оборваны, два имени склеены, скобка '
    'или кавычка не закрыта.\n'
    'Типы:\n' + type_lines + '\nВерни JSON со списком labels: номер узла i, type, kind.'
)
SCHEMA = {'type': 'object', 'properties': {'labels': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'i': {'type': 'integer'}, 'type': {'type': 'string', 'enum': TYPES}, 'kind': {'type': 'string', 'enum': KINDS}},
    'required': ['i', 'type', 'kind']}}}, 'required': ['labels']}

rows = [json.loads(l) for l in open(entities_path, encoding='utf-8') if l.strip()]
rows = rows[:limit] if limit else rows
key = lambda r: (r['name'], r['type'])
done = set()
if os.path.exists(labels_path):
    done = {(r['name'], r['type']) for r in map(json.loads, open(labels_path, encoding='utf-8'))}
todo = [r for r in rows if key(r) not in done]
batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
print(len(rows), 'entities,', len(todo), 'to label in', len(batches), 'batches', flush=True)
client = AsyncOpenAI(base_url=URL, api_key='EMPTY', timeout=300)
out = open(labels_path, 'a', encoding='utf-8')
gate = asyncio.Semaphore(CONCURRENCY)
started = time.monotonic()


async def label(batch):
    lines = '\n'.join(f'{i}. {r["name"]} [{r["type"]}] — {str(r.get("description") or "")[:220]}' for i, r in enumerate(batch))
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
        if got and i in got:
            out.write(json.dumps({'name': r['name'], 'type': r['type'], 'subtype': got[i]['type'], 'kind': got[i]['kind'],
                                  'model': MODEL, 'prompt': PROMPT_VERSION}, ensure_ascii=False) + '\n')


async def main():
    await asyncio.gather(*(label(b) for b in batches))
    out.close()

asyncio.run(main())
print('labeled in', round(time.monotonic() - started), 's', flush=True)
