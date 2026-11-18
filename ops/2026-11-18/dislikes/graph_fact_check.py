"""Graph elements in the prompts of the 136 dislike replays (RAGU v6k2): do their descriptions' numbers and years occur
in their source chunks, and do both ends of a relation occur there? Mechanical, judge-free."""
import collections, json, os, re
D = os.path.expanduser('~/meno-v2/dislikes1118')
S = os.path.expanduser('~/meno-v2/ragu-v2/stand-v6k2')
recs = [json.loads(l) for l in open(f'{D}/ragu6k2.records.jsonl', encoding='utf-8')]
def L(v):
    if isinstance(v, list): return v
    try: return json.loads(v or '[]')
    except Exception: return []
blocks = []
for r in recs:
    for b in L(r.get('contexts')):
        if b.get('kind') in ('entity', 'relation'):
            blocks.append((r['item_id'], b))
need = {int(c) for _, b in blocks for c in L(b.get('chunk_ids')) if str(c).isdigit()}
cmap = json.load(open(f'{S}/chunk_mapping_to_texts.json'))
want_docs = collections.defaultdict(set)
for g in need:
    m = cmap.get(str(g))
    if m: want_docs[m['doc_index']].add((m['local_chunk_index'], g))
text = {}
with open(f'{S}/chunked_texts_about_nsu_with_metadata_and_scores.jsonl', encoding='utf-8') as f:
    for di, line in enumerate(f):
        if di in want_docs:
            d = json.loads(line); full = d.get('doc_full_text') or ''
            for li, g in want_docs[di]:
                c = d['chunks'][li]; text[g] = full[c['start_char']:c['end_char']]
NUM = re.compile(r'(?<![\d.,])(?:1[89]\d\d|20\d\d|\d{3,})(?![\d])')
def norm(t): return ' '.join(re.sub(r'[^\w\s]', ' ', t.casefold().replace('ё', 'е')).split())
res = collections.Counter(); bad = []
for iid, b in blocks:
    t = b.get('text') or ''
    desc = t.split('**Описание:**', 1)[-1] if '**Описание:**' in t else t
    src = ' '.join(text.get(int(c), '') for c in L(b.get('chunk_ids')) if str(c).isdigit())
    if not src:
        res['no source text'] += 1; continue
    nums = set(NUM.findall(desc)); missing = {n for n in nums if n not in src}
    res['blocks'] += 1; res[f"{b['kind']}s"] += 1
    if nums: res['with numbers'] += 1
    if missing:
        res['numbers not in sources'] += 1
        bad.append(('numbers', iid, b['kind'], t.split('**Название:**')[-1].split('\n')[0][:90] if '**Название:**' in t else t[:90], sorted(missing)[:6]))
    if b['kind'] == 'relation':
        head = t.split('**Описание:**')[0]
        ends = re.findall(r'\*\*(?:Субъект|Объект|От|К|Источник связи|Цель связи)[^*]*:\*\*\s*([^\n(]+)', head)
        if len(ends) >= 2:
            ns = norm(src)
            hit = [any(w in ns for w in norm(e).split() if len(w) > 3) for e in ends[:2]]
            if not all(hit):
                res['relation end not in sources'] += 1
                bad.append(('end', iid, 'relation', ' — '.join(e.strip()[:40] for e in ends[:2]), hit))
print(json.dumps(res, ensure_ascii=False))
print('first relation header sample:', json.dumps(next((b.get('text')[:300] for _, b in blocks if b.get('kind') == 'relation'), ''), ensure_ascii=False))
for x in bad[:30]: print(json.dumps(x, ensure_ascii=False))
json.dump({'counts': res, 'flagged': bad}, open(f'{D}/graph_fact_check.json', 'w'), ensure_ascii=False, indent=1)
