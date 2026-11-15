"""Check files for a pilot run, with the items of an earlier check (same order, same n), so two prompt versions are
judged on the same entities and relations.

usage: python3 -I make_check.py RUN_DIR PREVIOUS_CHECK_DIR OUT_DIR
  RUN_DIR: entities.jsonl, relations.jsonl, entity_labels.jsonl, relation_labels.jsonl (label_run.sh's output)
  PREVIOUS_CHECK_DIR: check_entities.jsonl, check_relations.jsonl of the earlier run
"""
import json, re, sys
from pathlib import Path

run, previous, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
read = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
norm = lambda name: ' '.join(re.sub(r'[^\w\s]|_', ' ', str(name).casefold().replace('ё', 'е')).split())

entities = {(norm(e['name']), e['type']): e for e in read(run / 'entities.jsonl')}
labels = {(norm(r['name']), r['type']): r for r in read(run / 'entity_labels.jsonl')}
missing = 0
with open(out / 'check_entities.jsonl', 'w', encoding='utf-8') as w:
    for old in read(previous / 'check_entities.jsonl'):
        key = (norm(old['name']), old['old_type'])
        e, lab = entities.get(key), labels.get(key)
        if not e or not lab:
            missing += 1
            continue
        w.write(json.dumps({'n': old['n'], 'name': e['name'], 'old_type': e['type'], 'new_type': lab['subtype'],
                            'kind': lab['kind'], 'description': e['description']}, ensure_ascii=False) + '\n')

relations = {(r['description'], r['relation_type']): r for r in read(run / 'relations.jsonl')}
relabels = {r['id']: r for r in read(run / 'relation_labels.jsonl')}
with open(out / 'check_relations.jsonl', 'w', encoding='utf-8') as w:
    for old in read(previous / 'check_relations.jsonl'):
        r = relations.get((old['text'], old['extracted_as']))
        lab = relabels.get(r['id']) if r else None
        if not lab:
            missing += 1
            continue
        s, o = r['subject'], r['object']
        st, ot = lab['subject_type'], lab['object_type']
        if lab['reverse']:
            s, o, st, ot = o, s, ot, st
        w.write(json.dumps({'n': old['n'], 'new': f'{s["name"]} [{st}] —{lab["type"]}→ {o["name"]} [{ot}]',
                            'extracted_as': r['relation_type'], 'text': r['description']}, ensure_ascii=False) + '\n')
print('missing', missing)
