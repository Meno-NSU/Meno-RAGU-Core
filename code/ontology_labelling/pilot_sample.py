"""A pilot of the NSU ontology on a built graph: random relations (half of them UNFIT) with their two ends.
usage: python -I pilot_sample.py CHECKPOINT_DIR OUT_DIR N_UNFIT N_PLAIN
  OUT_DIR/relations.jsonl: id, subject/object {name, type, description}, relation_type, description
  OUT_DIR/entities.jsonl: the distinct ends {name, type, description}"""
import json, random, sys
from pathlib import Path

d, out, n_unfit, n_plain = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
entities, relations = {}, []
with open(d / manifest['stages']['11-graph-ready']['file'], encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row['group'] == 0:
            entities[row['value']['id']] = row['value']
        elif row['group'] == 1:
            relations.append(row['value'])
clip = lambda s, n: ' '.join(str(s or '').split())[:n]
random.seed(23)
unfit = [r for r in relations if r['relation_type'].startswith('UNFIT:') and r['subject_id'] in entities and r['object_id'] in entities]
plain = [r for r in relations if not r['relation_type'].startswith('UNFIT:') and r['subject_id'] in entities and r['object_id'] in entities]
picked = random.sample(unfit, n_unfit) + random.sample(plain, n_plain)
ends = {}


def end(eid):
    e = entities[eid]
    ends[(e['entity_name'], e['entity_type'])] = {'name': e['entity_name'], 'type': e['entity_type'],
                                                  'description': clip(e.get('description'), 300)}
    return {'name': e['entity_name'], 'type': e['entity_type']}


with open(out / 'relations.jsonl', 'w', encoding='utf-8') as f:
    for r in picked:
        f.write(json.dumps({'id': r['id'], 'subject': end(r['subject_id']), 'object': end(r['object_id']),
                            'relation_type': r['relation_type'], 'description': clip(r['description'], 300)},
                           ensure_ascii=False) + '\n')
with open(out / 'entities.jsonl', 'w', encoding='utf-8') as f:
    for e in ends.values():
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
print(len(picked), 'relations,', len(ends), 'ends')
