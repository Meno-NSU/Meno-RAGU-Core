"""stand-v6k2: graph v6 + the corrected KB v3.9b side of the chunk channel (owner, 2026-11-18, after an external audit).

Corrections against stand-v6k (kb39b_side.py):
1. only the KB chunks the KB indexes (`indexed`; Menon searches those): no duplicate_chunk, low_content or
   doc_chunk_cap chunks; a document without an indexed chunk is left out;
2. a chunk with `index_repairs` takes its repaired text: the body of `index_text` (after its header lines: title,
   "Источник: ...", "Раздел: ..."), which equals `text` for every chunk without repairs (checked here);
3. `doc_annotation` from the KB's annotation (lost in stand-v6k by an edit of kb39b_side.py); also kb_source_version_id.
Unchanged: graph-corpus rows of stand-v3k (with the teachers' section sources) first; every KB document, the pages whose
URL is in the graph corpus included (KB text beside the graph copy); graph v6's own chunks stay graph chunks (a v3.8 row
for each official or Pure page whose graph chunks are not KB chunks as they are); chunk format "**Документ:** <title>.\\n\\n
<text>", so every unchanged vector stays.

usage: PY -I kb39b_v2.py build  STAND_OUT WORK_DIR          new corpus, mappings, kv, dates; graph files by hard link
       PY -I kb39b_v2.py points-plan STAND_OUT WORK_DIR COLLECTION  read-only: ids to add (embedded now with bge-m3
                                                             :8001, 4 batches at a time, RAGU BM25) and ids to drop
       PY -I kb39b_v2.py points-apply WORK_DIR COLLECTION     writes the shared chunk collection in place: the new
                                                             points first, then the drops (wait=true batches)
       PY -I kb39b_v2.py check  STAND_OUT COLLECTION        mapping ids = points; KB texts; annotations
"""
import collections, glob, gzip, json, os, random, shutil, sys, time, uuid
from pathlib import Path

HOME = Path.home()
CODE = HOME / 'meno-v2/ragu-probe/gv5/code'
sys.path.insert(0, str(CODE))
from ragu.chunker.types import Chunk  # noqa: E402
from graph_utils.graph_utils import collect_knowledge_base_chunks  # noqa: E402

B = HOME / 'meno-v2/ragu-v2'
KB = HOME / 'meno-v2/kb-v39b/kb/chunks/v1'
KB_REVISION = (HOME / 'meno-v2/kb-v39b/REVISION').read_text().strip()
V3K, V6K, FULL = B / 'stand-v3k', B / 'stand-v6k', B / 'bundle-v6/full'
CORPUS, MAPPING, GRAPH = 'chunked_texts_about_nsu_with_metadata_and_scores.jsonl', 'chunk_mapping_to_texts.json', 'knowledge/ragu_graph'
SOURCES = {'summary', 'teachers', 'news', 'base', 'menon', 'teaching', 'menon-kb'}
HEADER = {'base', 'news', 'menon', 'teaching', 'menon-kb'}
Q = 'http://127.0.0.1:6433'
started = time.monotonic()


def log(*a):
    print(f'{time.monotonic() - started:7.0f} s', *a, flush=True)


def free_gb():
    return shutil.disk_usage('/').free / 2**30


def body(index_text):
    """The chunk text of an index_text: what follows its header lines (title, Источник, Раздел) and a blank line."""
    head, sep, rest = index_text.partition('\n\n')
    return rest if sep else index_text


def kb_rows(stats):
    docs = [json.loads(l) for l in gzip.open(KB / 'docs.jsonl.gz', 'rt', encoding='utf-8')]
    parts, dates = collections.defaultdict(list), collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(str(KB / 'chunks/part-*.jsonl.gz'))):
        for line in gzip.open(path, 'rt', encoding='utf-8'):
            c = json.loads(line)
            stats['kb chunks'] += 1
            if c.get('date'):
                dates[c['doc_id']][str(c['date'])] += 1
            if not c.get('indexed'):
                stats[f'kb chunks not indexed: {c.get("not_indexed_reason")}'] += 1
                continue
            if c.get('index_repairs'):
                text = body(c['index_text'])
                stats['kb indexed chunks with index_repairs (repaired text)'] += 1
                stats['  of them with a repaired text unlike text'] += text != c['text']
            else:
                text = c['text']
                ok = body(c['index_text']) == text
                stats['kb indexed chunks without repairs: body(index_text) == text'] += ok
                stats['kb indexed chunks without repairs: body(index_text) != text'] += not ok
            parts[c['doc_id']].append((int(c['chunk_index']), text))
    out = []
    for doc in docs:
        texts = [t for _, t in sorted(parts.get(doc['doc_id'], [])) if t and t.strip()]
        if not texts:
            stats['kb documents without an indexed chunk (left out)'] += 1
            continue
        spans, full, title = [], '', ' '.join(str(doc.get('title') or '').split())
        for text in texts:
            if full:
                full += '\n\n'
            spans.append({'start_char': len(full), 'end_char': len(full) + len(text)})
            full += text
        urls = [u for u in doc.get('urls') or [] if u]
        row = {'doc_full_text': full, 'doc_title': title, 'doc_annotation': doc.get('annotation') or '',
               'url': urls[0] if len(urls) == 1 else urls, 'quality_score': float(doc.get('quality_score') or 0.75),
               'content_source': 'menon-kb', 'kb_doc_id': doc['doc_id'], 'kb_tier': doc.get('tier'), 'chunks': spans,
               'kb_release': KB_REVISION, 'kb_source_version_id': doc.get('source_version_id')}
        chunks = [Chunk(f'**Документ:** {title.rstrip(".")}.\n\n{t}' if title else t, local, f'kb_{doc["doc_id"]}')
                  for local, t in enumerate(texts)]
        date = dates[doc['doc_id']].most_common(1)[0][0] if dates.get(doc['doc_id']) else None
        out.append((row, chunks, date))
    stats['kb documents'] = len(out)
    stats['kb document chunks'] = sum(len(c) for _, c, _ in out)
    stats['kb documents with an annotation'] = sum(bool(r['doc_annotation'].strip()) for r, _, _ in out)
    return out


def graph_chunks():
    docs = [json.loads(l) for l in open(FULL / CORPUS, encoding='utf-8') if l.strip()]
    mapping = json.loads((FULL / MAPPING).read_text())
    texts, gi = collect_knowledge_base_chunks(docs, mapping, content_sources=SOURCES, enrich_mode='title', enrich_sources=HEADER)
    seen, out = set(), []
    for i, (t, g) in enumerate(zip(texts, gi)):
        c = Chunk(t, i, f'doc_{i}')
        if c.id not in seen:
            seen.add(c.id)
            out.append((c.id, t, g, i))
    return out, docs, mapping


def build(stand, work):
    stats = collections.Counter()
    stand.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=True)
    (stand / GRAPH).mkdir(parents=True)
    v3k_rows = []
    with open(V3K / CORPUS, encoding='utf-8') as f:
        for line in f:
            if json.loads(line).get('content_source') == 'menon-kb':
                break
            v3k_rows.append(line if line.endswith('\n') else line + '\n')
    n_graph_docs = len(v3k_rows)
    v3k_map = json.loads((V3K / MAPPING).read_text())
    mapping = {g: v for g, v in v3k_map.items() if v['doc_index'] < n_graph_docs}
    n_graph_globals = len(mapping)
    assert set(mapping) == {str(i) for i in range(n_graph_globals)}
    gchunks, full_docs, full_mapping = graph_chunks()
    for g in range(n_graph_globals):
        assert full_mapping[str(g)] == mapping[str(g)]
    log(n_graph_docs, 'graph-corpus rows,', n_graph_globals, 'chunks;', len(gchunks), 'graph v6 chunks')
    kb = kb_rows(stats)
    log(len(kb), 'KB documents with indexed chunks')
    bad = stats['kb indexed chunks without repairs: body(index_text) != text']
    assert bad <= 0.001 * stats['kb indexed chunks without repairs: body(index_text) == text'], f'{bad} chunks: index_text body unlike text'
    kb_ids = {c.id for _, chunks, _ in kb for c in chunks}
    rows, menon, kv, dates = list(v3k_rows), {}, {}, {}
    for cid, text, g, i in gchunks:
        kv[cid] = {'content': text, 'chunk_order_idx': i, 'doc_id': f'doc_{i}', 'num_tokens': None}
        if g < n_graph_globals:
            menon[cid] = g
    nxt = n_graph_globals
    for row, chunks, date in kb:
        di = len(rows)
        rows.append(json.dumps(row, ensure_ascii=False) + '\n')
        if date:
            dates[str(di)] = date
        for local, c in enumerate(chunks):
            mapping[str(nxt)] = {'doc_index': di, 'local_chunk_index': local}
            menon[c.id] = nxt  # the last of identical texts wins, as in kb_extend.py
            kv.setdefault(c.id, {'content': c.content, 'chunk_order_idx': local, 'doc_id': c.doc_id, 'num_tokens': None})
            nxt += 1
    by_doc = collections.defaultdict(list)
    for cid, text, g, _ in gchunks:
        if g >= n_graph_globals:
            by_doc[full_mapping[str(g)]['doc_index']].append((cid, full_mapping[str(g)]['local_chunk_index']))
    fallback = 0
    for d, items in sorted(by_doc.items()):
        if all(cid in kb_ids for cid, _ in items):
            continue
        fallback += 1
        di = len(rows)
        row = dict(full_docs[d])
        row['kb_release'] = 'v3.8 (the text graph v6 was built on)'
        rows.append(json.dumps(row, ensure_ascii=False) + '\n')
        first = nxt
        for local in range(len(row['chunks'])):
            mapping[str(nxt)] = {'doc_index': di, 'local_chunk_index': local}
            nxt += 1
        for cid, local in items:
            if cid not in kb_ids:
                menon[cid] = first + local
    gids = {cid for cid, _, _, _ in gchunks}
    assert all(cid in menon for cid in gids), 'a graph chunk without a corpus chunk'
    assert all(cid in kv for cid in menon), 'a mapped chunk without kv text'
    stats['v3.8 rows of graph v6 pages'] = fallback
    stats['stand documents'] = len(rows)
    stats['stand corpus chunks'] = nxt
    stats['chunk ids (mapping = chunk channel)'] = len(menon)
    stats['graph v6 chunk ids'] = len(gids)
    stats['KB chunk ids that are graph chunks'] = len(kb_ids & gids)
    stats['dated documents'] = len(dates)
    log(f'{free_gb():.1f} GB free before writing')
    with open(stand / CORPUS, 'w', encoding='utf-8') as w:
        w.writelines(rows)
    del rows
    (stand / MAPPING).write_text(json.dumps(mapping))
    (stand / GRAPH / 'menon_chunk_mapping.json').write_text(json.dumps(menon))
    (stand / GRAPH / 'kv_chunks.json').write_text(json.dumps(kv, ensure_ascii=False))
    (stand / 'document_dates.json').write_text(json.dumps(dates, ensure_ascii=False))
    os.link(V6K / 'knowledge' / 'abbreviations.json', stand / 'knowledge' / 'abbreviations.json')
    for f in ('knowledge_graph.gml', 'kv_community.json', 'kv_community_summary.json', 'qdrant_collections.json'):
        os.link(V6K / GRAPH / f, stand / GRAPH / f)
    shutil.copytree(V6K / GRAPH / 'llm_debug', stand / GRAPH / 'llm_debug', copy_function=os.link)
    manifest = json.loads((V6K / 'build_manifest.json').read_text())
    manifest['chunks'] = (f'KB side: KB {KB_REVISION} indexed chunks only (repaired texts, annotations), all its documents '
                          'incl. the pages whose URL is in the graph corpus; chunk collection '
                          f'{json.loads((V6K / GRAPH / "qdrant_collections.json").read_text())["vdb_chunk"]} with every '
                          'chunk of the stand corpus')
    manifest['kb_side'] = dict(stats)
    (stand / 'build_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (stand / 'kb_side.json').write_text(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    (work / 'build_stats.json').write_text(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    log(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    log(f'{free_gb():.1f} GB free after writing')


def collection_ids(h, name):
    ids, offset = set(), None
    while True:
        body_ = {'limit': 5000, 'with_vector': False, 'with_payload': {'include': ['__ragu_id__']}}
        if offset is not None:
            body_['offset'] = offset
        page = h.post(f'/collections/{name}/points/scroll', json=body_).json()['result']
        ids.update(p['payload']['__ragu_id__'] for p in page['points'])
        offset = page.get('next_page_offset')
        if offset is None:
            return ids


def guard_disk():
    while free_gb() < 15.5:
        log(f'only {free_gb():.1f} GB free: waiting (floor 15)')
        time.sleep(60)


def points_plan(stand, work, collection):
    """Read-only: the points to add (embedded here into WORK/points_v2.jsonl.gz) and the ids to drop (WORK/drop_ids.json)."""
    import asyncio
    import base64
    import httpx
    import numpy as np
    menon = json.loads((stand / GRAPH / 'menon_chunk_mapping.json').read_text())
    gids = {cid for cid, _, _, _ in graph_chunks()[0]}
    with httpx.Client(base_url=Q, timeout=600) as h:
        have = collection_ids(h, collection)
    add = sorted(set(menon) - have)
    drop = sorted(have - set(menon))
    assert not set(drop) & gids, 'a graph chunk would be dropped'
    log(f'{len(have)} points in {collection}; {len(add)} to add, {len(drop)} to drop')
    json.dump(drop, open(work / 'drop_ids.json', 'w'))
    kv = json.loads((stand / GRAPH / 'kv_chunks.json').read_text())
    from ragu import Settings
    from ragu.models.embedder import EmbedderOpenAI
    from ragu.models.openai import CachedAsyncOpenAI
    from ragu.models.sparse_embedder import BM25 as RagUBM25
    from graph_utils.compact_embeddings import CompactEmbeddingClient
    Settings.tokenizer_embedder_backend = 'local'
    Settings.tokenizer_embedder_name = 'BAAI/bge-m3'
    Settings.embedder_token_limit = 8192
    embedder = EmbedderOpenAI(client=CompactEmbeddingClient(CachedAsyncOpenAI(base_url='http://127.0.0.1:8001/v1', api_key='EMPTY')),
                              model_name='BAAI/bge-m3', max_concurrent_batches=4)
    sparse = RagUBM25(language='russian')
    loop = asyncio.new_event_loop()
    with gzip.open(work / 'points_v2.jsonl.gz', 'wt', encoding='utf-8') as w:
        for start in range(0, len(add), 2000):
            batch = add[start:start + 2000]
            texts = [kv[cid]['content'] for cid in batch]
            dense = np.asarray(loop.run_until_complete(embedder.batch_embed_text(texts)), dtype=np.float32)
            for cid, d, s in zip(batch, dense, sparse.embed_document(texts)):
                w.write(json.dumps({'id': cid, 'dense': base64.b64encode(d.tobytes()).decode(),
                                    'bm25': {'indices': list(s.indices), 'values': list(s.values)},
                                    'payload': {'content': kv[cid]['content'], 'doc_id': kv[cid]['doc_id']}}, ensure_ascii=False) + '\n')
            log(f'{start + len(batch)}/{len(add)} embedded')
    plan = {'collection': collection, 'points_before': len(have), 'add': len(add), 'drop': len(drop),
            'points_after': len(have) + len(add) - len(drop), 'mapping_ids': len(menon)}
    (work / 'points_plan.json').write_text(json.dumps(plan, indent=1))
    log(json.dumps(plan))


def points_apply(work, collection):
    """Writes to the shared chunk collection: the planned points first (batches of 250, wait=true), then the drops
    (batches of 1000, wait=true); pauses while less than 15.5 GB are free."""
    import base64
    import httpx
    import numpy as np
    key = lambda cid: str(uuid.uuid5(uuid.NAMESPACE_URL, f'{collection}:{cid}'))
    plan = json.loads((work / 'points_plan.json').read_text())
    assert plan['collection'] == collection
    drop = json.load(open(work / 'drop_ids.json'))
    with httpx.Client(base_url=Q, timeout=600) as h:
        batch, added = [], 0
        for line in gzip.open(work / 'points_v2.jsonl.gz', 'rt', encoding='utf-8'):
            p = json.loads(line)
            dense = np.frombuffer(base64.b64decode(p['dense']), dtype=np.float32).tolist()
            batch.append({'id': key(p['id']), 'vector': {'dense': dense, 'bm25': p['bm25']},
                          'payload': {**p['payload'], '__ragu_id__': p['id']}})
            if len(batch) == 250:
                guard_disk()
                h.put(f'/collections/{collection}/points?wait=true', json={'points': batch}).raise_for_status()
                added += len(batch)
                batch = []
        if batch:
            h.put(f'/collections/{collection}/points?wait=true', json={'points': batch}).raise_for_status()
            added += len(batch)
        log(f'{added} points added; {free_gb():.1f} GB free')
        for start in range(0, len(drop), 1000):
            guard_disk()
            h.post(f'/collections/{collection}/points/delete?wait=true',
                   json={'points': [key(cid) for cid in drop[start:start + 1000]]}).raise_for_status()
        log(f'{len(drop)} points dropped; {free_gb():.1f} GB free')


def check(stand, collection):
    import httpx
    menon = json.loads((stand / GRAPH / 'menon_chunk_mapping.json').read_text())
    with httpx.Client(base_url=Q, timeout=600) as h:
        ids = collection_ids(h, collection)
        count = h.post(f'/collections/{collection}/points/count', json={'exact': True}).json()['result']['count']
    res = {'mapping_ids': len(menon), 'points': count, 'distinct_point_ids': len(ids),
           'mapping_ids_without_point': len(set(menon) - ids), 'points_outside_mapping': len(ids - set(menon))}
    # KB texts: a sample of KB rows against the KB release
    kb_chunks = {}
    for path in sorted(glob.glob(str(KB / 'chunks/part-*.jsonl.gz'))):
        for line in gzip.open(path, 'rt', encoding='utf-8'):
            c = json.loads(line)
            if c.get('indexed'):
                kb_chunks.setdefault(c['doc_id'], []).append((int(c['chunk_index']), body(c['index_text']) if c.get('index_repairs') else c['text'], bool(c.get('index_repairs'))))
    docs_kb = {json.loads(l)['doc_id']: json.loads(l) for l in gzip.open(KB / 'docs.jsonl.gz', 'rt', encoding='utf-8')}
    kv = json.loads((stand / GRAPH / 'kv_chunks.json').read_text())
    mapping = json.loads((stand / MAPPING).read_text())
    rows, ann = [], collections.Counter()
    with open(stand / CORPUS, encoding='utf-8') as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if d.get('content_source') == 'menon-kb':
                ann['menon-kb documents'] += 1
                ann['with doc_annotation'] += bool((d.get('doc_annotation') or '').strip())
                if d.get('kb_release') == KB_REVISION:
                    rows.append((i, d))
    rnd = random.Random(7)
    sample = rnd.sample(rows, 300) + [r for r in rows if any(rep for _, _, rep in kb_chunks.get(r[1]['kb_doc_id'], []))][:100]
    glob_of = collections.defaultdict(dict)
    for g, v in mapping.items():
        glob_of[v['doc_index']][v['local_chunk_index']] = int(g)
    inv = {}
    for cid, g in menon.items():
        inv.setdefault(g, cid)
    good = bad = ann_ok = 0
    for i, d in sample:
        want = [t for _, t, _ in sorted(kb_chunks[d['kb_doc_id']]) if t and t.strip()]
        got = [d['doc_full_text'][s['start_char']:s['end_char']] for s in d['chunks']]
        title = d['doc_title']
        kv_ok = all(kv.get(inv.get(glob_of[i][j], ''), {}).get('content') in (f'**Документ:** {title.rstrip(".")}.\n\n{t}' if title else t,)
                    for j, t in enumerate(got) if glob_of[i].get(j) in inv)
        if want == got and kv_ok:
            good += 1
        else:
            bad += 1
        ann_ok += d.get('doc_annotation', '') == (docs_kb[d['kb_doc_id']].get('annotation') or '')
    res.update({'kb_text_sample': len(sample), 'kb_text_sample_match': good, 'kb_text_sample_mismatch': bad,
                'annotation_sample_match': ann_ok, **dict(ann)})
    v5k = collections.Counter()
    for line in open(B / 'stand-v5k' / CORPUS, encoding='utf-8'):
        d = json.loads(line)
        if d.get('content_source') == 'menon-kb':
            v5k['stand-v5k menon-kb documents'] += 1
            v5k['stand-v5k with doc_annotation'] += bool((d.get('doc_annotation') or '').strip())
    res.update(v5k)
    (stand / 'knowledge/ragu_graph/chunk_collection_check.json').write_text(json.dumps(res, indent=1))
    log(json.dumps(res, ensure_ascii=False, indent=1))
    assert not bad, res  # the point checks pass once points-apply has run


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build':
        build(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == 'points-plan':
        points_plan(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    elif cmd == 'points-apply':
        points_apply(Path(sys.argv[2]), sys.argv[3])
    elif cmd == 'check':
        check(Path(sys.argv[2]), sys.argv[3])
    else:
        sys.exit(__doc__)
    os._exit(0)
