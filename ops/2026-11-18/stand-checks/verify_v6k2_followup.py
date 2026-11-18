"""Follow-up: split the KB rows into v3.9b rows and the v3.8 rows kept for graph pages; explain kv-vs-corpus diffs."""
import collections, gzip, glob, hashlib, json, pathlib, sys
home = pathlib.Path.home()
KB = home / 'meno-v2/kb-v39b/kb/chunks/v1'
h = lambda s: hashlib.md5(s.encode('utf-8')).hexdigest()
def corpus(S):
    rows = []
    with open(S / 'chunked_texts_about_nsu_with_metadata_and_scores.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            full = d.pop('doc_full_text', '') or ''
            d['_texts'] = [full[c['start_char']:c['end_char']] for c in d.get('chunks', [])]
            rows.append(d)
    return rows
def kv_check(S, rows, label):
    kv = json.loads((S / 'knowledge/ragu_graph/kv_chunks.json').read_text(encoding='utf-8'))
    cmap = json.loads((S / 'chunk_mapping_to_texts.json').read_text(encoding='utf-8'))
    menon = json.loads((S / 'knowledge/ragu_graph/menon_chunk_mapping.json').read_text(encoding='utf-8'))
    diff = collections.Counter(); total = collections.Counter(); shown = []
    for cid, gi in menon.items():
        m = cmap[str(gi)]; d = rows[m['doc_index']]; src = d.get('content_source')
        t = d['_texts'][m['local_chunk_index']]
        c = kv[cid]['content']
        body = c.split('\n\n', 1)[1] if c.startswith('**Документ:**') else c
        total[src] += 1
        if body != t:
            kind = 'strip-equal' if body.strip() == t.strip() else ('kv contains slice' if t in c else 'other')
            diff[(src, kind)] += 1
            if kind == 'other' and len(shown) < 3: shown.append((cid, src, c[:160], t[:160]))
    print(label, 'channel chunks by source', dict(total))
    print(label, 'kv body != corpus slice', sorted(diff.items(), key=lambda x: -x[1]))
    for s in shown: print(label, 'EXAMPLE', json.dumps(s, ensure_ascii=False))
S2 = home / 'meno-v2/ragu-v2/stand-v6k2'
rows = corpus(S2)
expect = collections.defaultdict(list); ver = collections.defaultdict(set)
for p in sorted(glob.glob(str(KB / 'chunks/part-*.jsonl.gz'))):
    for line in gzip.open(p, 'rt', encoding='utf-8'):
        c = json.loads(line)
        if c.get('indexed'):
            expect[c['doc_id']].append((int(c['chunk_index']), c['text'], bool(c.get('index_repairs'))))
            ver[c['doc_id']].add(c.get('source_version_id'))
by = collections.defaultdict(list)
for i, d in enumerate(rows):
    if d.get('content_source') == 'menon-kb': by[d['kb_doc_id']].append(i)
rel = collections.Counter((rows[i].get('kb_release'), len(v)) for v in by.values() for i in v)
print('rows by (kb_release, rows for the doc)', sorted(rel.items(), key=lambda x: -x[1]))
bad = collections.Counter()
for k, idx in by.items():
    want = sorted(expect.get(k, []))
    new = [i for i in idx if rows[i].get('kb_release') != 'v3.8'] or idx
    for i in new:
        d = rows[i]
        ok_count = len(d['_texts']) == len(want)
        ok_text = ok_count and all(t == raw or rep for t, (_, raw, rep) in zip(d['_texts'], want))
        ok_ver = d.get('kb_source_version_id') in ver.get(k, set())
        if not (ok_count and ok_text and ok_ver): bad[('v3.9b row', ok_count, ok_text, ok_ver)] += 1
print('v3.9b rows failing (count ok, text ok, version ok)', dict(bad))
olds = [i for v in by.values() for i in v if rows[i].get('kb_release') == 'v3.8']
print('v3.8 rows kept', len(olds), 'their chunks', sum(len(rows[i]['_texts']) for i in olds),
      'urls also in a v3.9b row', sum(1 for i in olds if len(by[rows[i]['kb_doc_id']]) > 1))
kv_check(S2, rows, 'v6k2')
del rows
S5 = home / 'meno-v2/ragu-v2/stand-v5k'
kv_check(S5, corpus(S5), 'v5k')
