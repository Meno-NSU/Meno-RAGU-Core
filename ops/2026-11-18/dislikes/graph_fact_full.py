"""For the graph elements flagged by graph_fact_check.py: are the description's numbers in ANY source chunk of the
element (its full source list in the GML), or in none (merged or invented)?"""
import collections, html, json, os, re
D = os.path.expanduser('~/meno-v2/dislikes1118')
S = os.path.expanduser('~/meno-v2/ragu-v2/stand-v6k2')
G = f'{S}/knowledge/ragu_graph/knowledge_graph.gml'
chk = json.load(open(f'{D}/graph_fact_check.json'))
recs = [json.loads(l) for l in open(f'{D}/ragu6k2.records.jsonl', encoding='utf-8')]
def L(v):
    if isinstance(v, list): return v
    try: return json.loads(v or '[]')
    except Exception: return []
# the flagged blocks, with their full text
NUM = re.compile(r'(?<![\d.,])(?:1[89]\d\d|20\d\d|\d{3,})(?![\d])')
flag = []
for r in recs:
    for b in L(r.get('contexts')):
        t = b.get('text') or ''
        if b.get('kind') == 'entity' and '**Название:**' in t:
            name, typ = re.match(r'\s*(.*) \(тип: ([A-Z_]+)\)', t.split('**Название:**')[1].split('\n')[0]).groups()
            flag.append((r['item_id'], 'entity', (name.strip(), typ), t))
        elif b.get('kind') == 'relation' and '**Связь:**' in t:
            m = re.match(r'\s*(.*?) —\[([A-Z_:]+)\]→ (.*?)\.?\s*$', t.split('**Связь:**')[1].split('\n')[0])
            if m: flag.append((r['item_id'], 'relation', (m.group(1).strip(), m.group(2), m.group(3).strip().rstrip('.')), t))
names = {}; ent_src = collections.defaultdict(list); rel_src = collections.defaultdict(list)
kv = re.compile(r'^\s+(\w+) (.*)$'); cur = None; kind = None; chunks = []
with open(G, encoding='utf-8') as f:
    for line in f:
        s = line.strip()
        if s in ('node [', 'edge ['): kind, cur, chunks = s[:4], {}, []; continue
        if s == ']' and kind:
            if kind == 'node':
                i = int(cur['id']); nm = html.unescape(cur.get('entity_name', '').strip('"')); tp = cur.get('entity_type', '').strip('"')
                names[i] = nm; ent_src[(nm, tp)].extend(chunks)
            else:
                rel_src[(int(cur['source']), cur.get('relation_type', '').strip('"'), int(cur['target']))].extend(chunks)
            kind = None; continue
        if kind:
            m = kv.match(line)
            if m:
                k, v = m.group(1), m.group(2)
                if k == 'source_chunk_id' and v.strip('"').startswith('chunk-'): chunks.append(v.strip('"'))
                elif k in ('id', 'entity_name', 'entity_type', 'source', 'target', 'relation_type'): cur[k] = v
by_name = collections.defaultdict(list)
for i, n in names.items(): by_name[n].append(i)
menon = json.load(open(f'{S}/knowledge/ragu_graph/menon_chunk_mapping.json'))
kvchunks = json.load(open(f'{S}/knowledge/ragu_graph/kv_chunks.json'))
res = collections.Counter(); out = []
seen = set()
for iid, kind, key, t in flag:
    desc = t.split('**Описание:**', 1)[-1].split('**Встречается во фрагментах')[0]
    nums = set(NUM.findall(desc))
    if not nums or (kind, key) in seen: continue
    seen.add((kind, key))
    if kind == 'entity':
        src = ent_src.get(key, [])
    else:
        src = [c for a in by_name.get(key[0], []) for b in by_name.get(key[2], []) for c in rel_src.get((a, key[1], b), [])]
    texts = ' '.join((kvchunks.get(c) or {}).get('content', '') for c in src)
    missing = sorted(n for n in nums if n not in texts)
    res['elements with numbers'] += 1
    res['all sources found' if src else 'element not found in GML'] += 1
    if src and missing:
        res['numbers in NO source chunk'] += 1
        out.append((iid, kind, ' / '.join(key), len(src), missing[:8], ' '.join(desc.split())[:260]))
    elif src:
        res['numbers all supported by some source'] += 1
print(json.dumps(res, ensure_ascii=False))
for o in out[:40]: print(json.dumps(o, ensure_ascii=False))
json.dump(out, open(f'{D}/graph_fact_unsupported.json', 'w'), ensure_ascii=False, indent=1)
