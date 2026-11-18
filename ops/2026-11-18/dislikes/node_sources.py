import collections, html, json, os, re, sys
S = os.path.expanduser('~/meno-v2/ragu-v2/stand-v6k2')
want = sys.argv[1:]
kv = re.compile(r'^\s+(\w+) (.*)$'); cur = None; kind = None; chunks = []; found = []
with open(f'{S}/knowledge/ragu_graph/knowledge_graph.gml', encoding='utf-8') as f:
    for line in f:
        s = line.strip()
        if s == 'node [': kind, cur, chunks = 'node', {}, []; continue
        if s == 'edge [': kind = 'edge'; continue
        if s == ']' and kind == 'node':
            nm = html.unescape(cur.get('entity_name', '').strip('"'))
            if nm in want: found.append((nm, cur.get('entity_type', '').strip('"'), html.unescape(cur.get('description', '').strip('"')), list(chunks)))
            kind = None; continue
        if s == ']': kind = None; continue
        if kind == 'node':
            m = kv.match(line)
            if m:
                k, v = m.group(1), m.group(2)
                if k == 'source_chunk_id' and v.strip('"').startswith('chunk-'): chunks.append(v.strip('"'))
                elif k in ('entity_name', 'entity_type', 'description'): cur[k] = v
menon = json.load(open(f'{S}/knowledge/ragu_graph/menon_chunk_mapping.json'))
cmap = json.load(open(f'{S}/chunk_mapping_to_texts.json'))
need = collections.defaultdict(list)
for nm, tp, desc, ch in found:
    for c in ch:
        g = menon.get(c)
        if g is not None: need[cmap[str(g)]['doc_index']].append((nm, c))
docs = {}
with open(f'{S}/chunked_texts_about_nsu_with_metadata_and_scores.jsonl', encoding='utf-8') as f:
    for di, line in enumerate(f):
        if di in need:
            d = json.loads(line); docs[di] = (str(d.get('url'))[:100], (d.get('doc_title') or '')[:80])
for nm, tp, desc, ch in found:
    print(f'== {nm} ({tp}), {len(ch)} source chunks')
    print('DESC:', ' '.join(desc.split())[:700])
    seen = collections.Counter(docs[di] for di, v in need.items() for n, c in v if n == nm)
    for (u, t), k in seen.most_common(12): print(f'  {k} × {t} <{u}>')
