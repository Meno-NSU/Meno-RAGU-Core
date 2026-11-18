"""Core stratum of the 18.11 benchmark: which answers contradict the reference (qwen judge), for RAGU v6k2, RAGU v5k
(16.11) and Menon 1.1, and, for RAGU v6k2, which prompt block each contradicting or unsupported claim resembles most
(word and number overlap): a graph element, a document, or none (the model's own)."""
import collections, json, re, os
J = os.path.expanduser('~/meno-v2/final1118/judge')
F = os.path.expanduser('~/meno-v2/final1118')
keys = json.load(open(f'{J}/keys/point_keys.json'))
core = set(json.load(open(f'{F}/ids/core.json')))
items = {json.loads(l)['id']: json.loads(l) for l in open(os.path.expanduser('~/meno-v2/ragu-probe/v3new/items_v3new_ref.jsonl'))}
verdict = {}
import glob
for p in glob.glob(f'{J}/out/*.qwen.jsonl'):
    for l in open(p):
        r = json.loads(l); k = keys.get(r['id'])
        if k and k['set'] == 'V3' and k['qid'] in core:
            verdict[(k['system'], k['qid'])] = r
def lst(v):
    if isinstance(v, list): return v
    try: return json.loads(v)
    except Exception:
        try: import ast; return ast.literal_eval(v)
        except Exception: return []
systems = ('ragu6k', 'ragu', 'menon')
contra = {s: {q for (sy, q), r in verdict.items() if sy == s and lst(r['contradicts_reference'])} for s in systems}
judged = {s: {q for (sy, q) in verdict if sy == s} for s in systems}
print('core judged:', {s: len(judged[s]) for s in systems})
print('answers contradicting the reference:', {s: len(contra[s]) for s in systems})
print('v6k2 only:', len(contra['ragu6k'] - contra['menon']), '| Menon only:', len(contra['menon'] - contra['ragu6k']),
      '| both:', len(contra['ragu6k'] & contra['menon']), '| v6k2 new vs v5k:', len(contra['ragu6k'] - contra['ragu']),
      '| fixed vs v5k:', len(contra['ragu'] - contra['ragu6k']))
recs = {}
for l in open(f'{F}/answers/ragu6k_v3.records.jsonl'):
    r = json.loads(l); recs[r['item_id']] = r
W = re.compile(r'[A-Za-zА-Яа-яЁё]{4,}|\d+')
def toks(t): return set(w.casefold() for w in W.findall(t or ''))
def best_block(claim, ctx):
    c = toks(claim)
    if not c: return None, 0.0
    best, score = None, 0.0
    for b in ctx:
        s = len(c & toks(b.get('text'))) / len(c)
        if s > score: best, score = b, s
    return best, score
kinds = collections.Counter(); kinds_uns = collections.Counter(); examples = []
cat = collections.Counter(); cat_all = collections.Counter()
for q in judged['ragu6k']:
    cat_all[items.get(q, {}).get('category')] += 1
for q in sorted(contra['ragu6k']):
    r = recs.get(q); ctx = r['contexts'] if r and isinstance(r['contexts'], list) else json.loads((r or {}).get('contexts') or '[]')
    cat[items.get(q, {}).get('category')] += 1
    for claim in lst(verdict[('ragu6k', q)]['contradicts_reference']):
        b, s = best_block(claim, ctx)
        k = 'none (<0.5)' if s < 0.5 or b is None else b.get('kind')
        kinds[k] += 1
        if len(examples) < 40:
            examples.append((q, items.get(q, {}).get('category'), claim[:160], k, round(s, 2), (b or {}).get('title', '')[:70] if s >= 0.5 else '',
                             (b or {}).get('url', '')[:90] if s >= 0.5 else '', (items.get(q, {}).get('ground_truth') or '')[:160], q in contra['menon']))
for q in judged['ragu6k']:
    r = recs.get(q); ctx = r['contexts'] if r and isinstance(r['contexts'], list) else json.loads((r or {}).get('contexts') or '[]')
    for claim in lst(verdict[('ragu6k', q)]['unsupported']):
        b, s = best_block(claim, ctx)
        kinds_uns['none (<0.5)' if s < 0.5 or b is None else b.get('kind')] += 1
print('contradicting claims of v6k2 resemble:', dict(kinds))
print('unsupported claims of v6k2 resemble:', dict(kinds_uns))
print('contradicting answers by category (v6k2):', {c: f'{cat[c]}/{cat_all[c]}' for c in cat_all})
gb = collections.Counter()
for q in judged['ragu6k']:
    r = recs.get(q); ctx = r['contexts'] if r and isinstance(r['contexts'], list) else json.loads((r or {}).get('contexts') or '[]')
    has = any(b.get('kind') in ('entity', 'relation') for b in ctx)
    gb[('graph block' if has else 'no graph block', q in contra['ragu6k'])] += 1
print('contradiction rate by presence of graph blocks:', {k: v for k, v in sorted(gb.items())})
json.dump(examples, open(f'{F}/contra_examples.json', 'w'), ensure_ascii=False, indent=1)
for e in examples[:40]:
    print(json.dumps(e, ensure_ascii=False))
