"""Review digest of the dislike replay: per case the question (+ last turns), the owner's comment, the disliked answer
and its sources, and the new RAGU v6k2 answer with its sources and prompt blocks (documents with date, graph elements)."""
import json, os
D = os.path.expanduser('~/meno-v2/dislikes1118')
S = os.path.expanduser('~/meno-v2/ragu-v2/stand-v6k2')
meta = {json.loads(l)['id']: json.loads(l) for l in open(f'{D}/meta.jsonl', encoding='utf-8')}
items = {json.loads(l)['id']: json.loads(l) for l in open(f'{D}/items.jsonl', encoding='utf-8')}
recs = {json.loads(l)['item_id']: json.loads(l) for l in open(f'{D}/ragu6k2.records.jsonl', encoding='utf-8')}
dates = json.load(open(f'{S}/document_dates.json'))
cmap = json.load(open(f'{S}/chunk_mapping_to_texts.json'))
def L(v):
    if isinstance(v, list): return v
    try: return json.loads(v or '[]')
    except Exception: return []
def clip(t, n): t = ' '.join(str(t or '').split()); return t if len(t) <= n else t[:n] + '…'
def src_line(s):
    if isinstance(s, dict):
        return f"{clip(s.get('document_title') or s.get('title'), 70)} <{s.get('source_url') or s.get('url')}>"
    return clip(s, 100)
out = []
for iid in sorted(meta):
    m, it, r = meta[iid], items[iid], recs.get(iid, {})
    lines = [f"## {iid} | disliked {m['fb_at'][:16]} | was {m['model']}", f"Q: {clip(m['question'], 300)}"]
    if it['history']:
        lines.append('History: ' + ' / '.join(f"{h['role'][0]}: {clip(h['content'], 140)}" for h in it['history'][-4:]))
    lines.append(f"OWNER: {clip(m['comment'], 500)}")
    lines.append(f"OLD answer: {clip(m['old_answer'], 500)}")
    lines.append('OLD sources: ' + ' | '.join(src_line(s) for s in L(m['old_sources'])[:5]))
    if not r.get('ok'):
        lines.append(f"NEW: FAILED {clip(r.get('error'), 200)}")
    else:
        lines.append(f"NEW answer: {clip(r.get('response'), 900)}")
        lines.append('NEW sources: ' + ' | '.join(src_line(s) for s in L(r.get('sources'))[:6]))
        ctx = L(r.get('contexts'))
        docs, graph = [], []
        for b in ctx:
            if b.get('kind') == 'document':
                ids = L(b.get('chunk_ids')) or [b.get('chunk_id')]
                di = (cmap.get(str(ids[0])) or {}).get('doc_index') if ids and ids[0] is not None else None
                docs.append(f"{clip(b.get('title'), 60)} [{dates.get(str(di), 'no date')}] <{clip(b.get('url'), 80)}>")
            else:
                name = (b.get('text') or '').split('**Название:**')[-1].split('\n')[0] if '**Название:**' in (b.get('text') or '') else clip(b.get('text'), 80)
                graph.append(f"{b.get('kind')}: {clip(name, 90)}")
        lines.append(f"NEW context: {len(docs)} documents, {len(graph)} graph elements; queries: {clip(' || '.join(L(r.get('search_queries'))), 250)}")
        lines.extend('  D ' + d for d in docs[:8])
        lines.extend('  G ' + g for g in graph[:3])
    out.append('\n'.join(lines))
open(f'{D}/review.md', 'w', encoding='utf-8').write('\n\n'.join(out) + '\n')
print(len(out), 'cases;', sum(1 for i in meta if recs.get(i, {}).get('ok')), 'answered;', os.path.getsize(f'{D}/review.md'), 'bytes')
