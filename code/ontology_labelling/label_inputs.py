"""Inputs of the NSU-ontology labelling for a whole built graph: every node that is not a value, every relation between
two such nodes. Names are shown whole: graph v4's name trimming cut closing brackets and quotes ('Дексаметазон (DEX'),
so each name takes the most frequent spelling the extraction (stage 03) gave its key — the name up to punctuation and
case, the key the graph build and the cleanup inputs match names by — with the fixed trimming.

usage: python -I label_inputs.py CHECKPOINT_DIR OUT_DIR [CODE_DIR] [--repair SAMPLE_DIR]
  CODE_DIR: the graph code with the fixed trim_edge_punctuation (default: ./gv5)
  --repair: instead of the whole graph, write SAMPLE_DIR's entities.jsonl and relations.jsonl (pilot_sample.py's
  output) to OUT_DIR with whole names
  OUT_DIR/entities.jsonl: name, type, description; OUT_DIR/relations.jsonl: id, subject/object {name, type},
  relation_type, description (pilot_sample.py's format)
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

args = sys.argv[1:]
sample = None
if '--repair' in args:
    i = args.index('--repair')
    sample = Path(args[i + 1])
    del args[i:i + 2]
d, out = Path(args[0]), Path(args[1])
sys.path.insert(0, str(Path(args[2] if len(args) > 2 else 'gv5') / 'code'))
from graph_utils.graph_utils import trim_edge_punctuation  # noqa: E402

VALUES = {'NUMBER', 'DATE', 'TIME', 'AGE', 'MONEY', 'PERCENT', 'ORDINAL'}
out.mkdir(parents=True, exist_ok=True)
stages = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))['stages']
norm = lambda name: ' '.join(re.sub(r'[^\w\s]|_', ' ', str(name).casefold().replace('ё', 'е')).split())
clip = lambda s, n: ' '.join(str(s or '').split())[:n]

spellings: dict[str, Counter] = defaultdict(Counter)
with open(d / stages['03-entity-validation']['file'], encoding='utf-8') as f:
    for line in f:
        value = json.loads(line).get('value') or {}
        for e in value.get('entities') or []:
            spellings[norm(e['entity_name'])][e['entity_name']] += 1
whole = {key: trim_edge_punctuation(max(c.items(), key=lambda it: (it[1], it[0]))[0]) for key, c in spellings.items()}
del spellings


def made_whole(name):
    fixed = whole.get(norm(name)) or name
    return fixed if norm(fixed) == norm(name) else name


if sample is not None:
    n = 0
    for fname in ('entities.jsonl', 'relations.jsonl'):
        with open(sample / fname, encoding='utf-8') as f, open(out / fname, 'w', encoding='utf-8') as w:
            for line in f:
                row = json.loads(line)
                for end in (row, row.get('subject'), row.get('object')):
                    if isinstance(end, dict) and 'name' in end:
                        fixed = made_whole(end['name'])
                        n += fixed != end['name']
                        end['name'] = fixed
                w.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(n, 'names made whole in', sample)
    sys.exit(0)

ready = d / stages['11-graph-ready']['file']
ends, repaired = {}, 0
with open(ready, encoding='utf-8') as f, open(out / 'entities.jsonl', 'w', encoding='utf-8') as w:
    for line in f:
        row = json.loads(line)
        if row['group'] != 0:
            continue
        e = row['value']
        if e['entity_type'] in VALUES:
            continue
        name = e['entity_name']
        fixed = made_whole(name)
        repaired += fixed != name
        ends[e['id']] = (fixed, e['entity_type'])
        w.write(json.dumps({'name': fixed, 'type': e['entity_type'], 'description': clip(e.get('description'), 300)},
                           ensure_ascii=False) + '\n')
n_rel = 0
with open(ready, encoding='utf-8') as f, open(out / 'relations.jsonl', 'w', encoding='utf-8') as w:
    for line in f:
        row = json.loads(line)
        if row['group'] != 1:
            continue
        r = row['value']
        if r['subject_id'] not in ends or r['object_id'] not in ends:
            continue
        (sn, st), (on, ot) = ends[r['subject_id']], ends[r['object_id']]
        w.write(json.dumps({'id': r['id'], 'subject': {'name': sn, 'type': st}, 'object': {'name': on, 'type': ot},
                            'relation_type': r['relation_type'], 'description': clip(r['description'], 300)},
                           ensure_ascii=False) + '\n')
        n_rel += 1
print(len(ends), 'entities (', repaired, 'names made whole ),', n_rel, 'relations')
