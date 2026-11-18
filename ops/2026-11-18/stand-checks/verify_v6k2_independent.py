"""Independent full check of stand-v6k2's KB side and chunk store (read-only)."""
import collections, difflib, glob, gzip, hashlib, json, pathlib, re, sys, time
home = pathlib.Path.home()
S = home / 'meno-v2/ragu-v2/stand-v6k2'
KB = home / 'meno-v2/kb-v39b/kb/chunks/v1'
t0 = time.time()
h = lambda s: hashlib.md5(s.encode('utf-8')).hexdigest()
out = collections.Counter()
examples = collections.defaultdict(list)
def ex(key, val):
    out[key] += 1
    if len(examples[key]) < 3: examples[key].append(val)

# 1. KB v3.9b chunks: what the stand should hold per document
expect = collections.defaultdict(list)   # doc_id -> [(chunk_index, md5 expected, md5 raw, repaired, raw, index_text)]
not_indexed = collections.defaultdict(set)  # doc_id -> md5 of raw texts of chunks not indexed
src_ver = collections.defaultdict(set)
for p in sorted(glob.glob(str(KB / 'chunks/part-*.jsonl.gz'))):
    for line in gzip.open(p, 'rt', encoding='utf-8'):
        c = json.loads(line)
        out['kb chunks'] += 1
        if not c.get('indexed'):
            out['kb chunks not indexed'] += 1
            not_indexed[c['doc_id']].add(h(c['text']))
            continue
        rep = bool(c.get('index_repairs'))
        expect[c['doc_id']].append((int(c['chunk_index']), rep, c['text'], c['index_text'] if rep else None))
        src_ver[c['doc_id']].add(c.get('source_version_id'))
out['kb docs with indexed chunks'] = len(expect)
print(f'kb chunks read {time.time() - t0:.0f} s', file=sys.stderr)
docs_kb = {}
for line in gzip.open(KB / 'docs.jsonl.gz', 'rt', encoding='utf-8'):
    d = json.loads(line)
    docs_kb[d['doc_id']] = d
first = next(iter(docs_kb.values()))
print('kb doc fields:', sorted(first)[:40])
title_key = next((k for k in ('title', 'doc_title') if k in first), None)
ann_key = next((k for k in ('annotation', 'doc_annotation', 'summary') if k in first), None)
print('title key', title_key, 'annotation key', ann_key)

# 2. stand corpus: KB rows against the release; slice md5 per global chunk
slice_md5 = {}   # (doc_index, local) -> md5 of the corpus slice
seen_kb = set()
gidx = 0
with open(S / 'chunked_texts_about_nsu_with_metadata_and_scores.jsonl', encoding='utf-8') as f:
    for di, line in enumerate(f):
        d = json.loads(line)
        full = d.get('doc_full_text') or ''
        texts = [full[c['start_char']:c['end_char']] for c in d.get('chunks', [])]
        for li, t in enumerate(texts):
            slice_md5[(di, li)] = h(t)
        out['stand docs'] += 1
        out[f"stand docs: {d.get('content_source')}"] += 1
        if d.get('content_source') != 'menon-kb':
            continue
        k = d.get('kb_doc_id')
        if k in seen_kb: ex('kb doc twice in the stand', k)
        seen_kb.add(k)
        want = sorted(expect.get(k, []))
        if len(texts) != len(want):
            ex('kb doc: chunk count differs', (k, len(texts), len(want))); continue
        for t, (ci, rep, raw, itext) in zip(texts, want):
            if not rep:
                if t != raw: ex('indexed chunk text differs from KB text', (k, ci, t[:80], raw[:80]))
                else: out['indexed chunks equal to KB text'] += 1
            else:
                ratio = difflib.SequenceMatcher(None, t, raw, autojunk=False).ratio() if len(t) < 20000 else None
                if t == raw: ex('repaired chunk kept the raw text', (k, ci))
                elif t not in (itext or ''): ex('repaired chunk text not inside index_text', (k, ci, t[:80]))
                elif ratio is not None and ratio < 0.9: ex('repaired chunk far from raw', (k, ci, round(ratio, 3)))
                else: out['repaired chunks ok (inside index_text, close to raw)'] += 1
        stand_md5 = {h(t) for t in texts}
        want_md5 = {h(raw) for _, _, raw, _ in want}
        leaked = (not_indexed.get(k, set()) & stand_md5) - want_md5
        if leaked: ex('non-indexed chunk text present', (k, len(leaked)))
        kd = docs_kb.get(k)
        if kd is None: ex('kb doc missing in docs.jsonl', k); continue
        if title_key and (d.get('doc_title') or '') != (kd.get(title_key) or ''):
            ex('title differs from KB doc title', (k, (d.get('doc_title') or '')[:70], (kd.get(title_key) or '')[:70]))
        if ann_key:
            a_s, a_k = (d.get('doc_annotation') or ''), (kd.get(ann_key) or '')
            if not a_s and a_k: ex('annotation lost', k)
            elif a_s != a_k and a_k: ex('annotation differs from KB', (k, a_s[:60], a_k[:60]))
            elif a_s: out['annotation equal to KB'] += 1
        if d.get('kb_source_version_id') not in src_ver.get(k, set()): ex('source version differs', k)
missing_docs = set(expect) - seen_kb
out['kb docs with indexed chunks missing from the stand'] = len(missing_docs)
for k in list(missing_docs)[:3]: examples['kb docs with indexed chunks missing from the stand'].append(k)
print(f'corpus read {time.time() - t0:.0f} s', file=sys.stderr)

# 3. chunk store: every channel id resolves to the corpus slice; graph source ids resolve
kv = json.loads((S / 'knowledge/ragu_graph/kv_chunks.json').read_text(encoding='utf-8'))
cmap = json.loads((S / 'chunk_mapping_to_texts.json').read_text(encoding='utf-8'))
menon = json.loads((S / 'knowledge/ragu_graph/menon_chunk_mapping.json').read_text(encoding='utf-8'))
out['kv chunks'] = len(kv); out['channel ids'] = len(menon); out['corpus chunks'] = len(slice_md5)
for cid, gi in menon.items():
    v = kv.get(cid)
    if v is None: ex('channel id without kv entry', cid); continue
    m = cmap.get(str(gi))
    if m is None: ex('channel index without corpus mapping', (cid, gi)); continue
    body = v['content'].split('\n\n', 1)[1] if v['content'].startswith('**Документ:**') else v['content']
    if h(body) != slice_md5.get((m['doc_index'], m['local_chunk_index'])): ex('kv text differs from corpus slice', cid)
    if v.get('doc_id') != f"doc_{m['doc_index']}": ex('kv doc_id differs from mapping', cid)
print(f'kv checked {time.time() - t0:.0f} s', file=sys.stderr)
graph_ids = set()
pat = re.compile(r'chunk-[0-9a-f]{32}')
with open(S / 'knowledge/ragu_graph/knowledge_graph.gml', encoding='utf-8') as f:
    for line in f:
        if 'source_chunk_id' in line:
            graph_ids.update(pat.findall(line))
out['graph source chunk ids'] = len(graph_ids)
out['graph source ids without kv entry'] = len(graph_ids - set(kv))
out['graph source ids in the chunk channel'] = len(graph_ids & set(menon))
examples['graph source ids without kv entry'] = list(graph_ids - set(kv))[:3]
print(f'gml scanned {time.time() - t0:.0f} s', file=sys.stderr)
print(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in examples.items() if v}, ensure_ascii=False, indent=1)[:4000])
