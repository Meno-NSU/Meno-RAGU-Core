import json, os
D = os.path.expanduser('~/meno-v2/dislikes1118')
rows = [json.loads(l) for l in open(f'{D}/dislikes_all.jsonl', encoding='utf-8')]
items, meta = [], []
for i, r in enumerate(rows):
    iid = f'dl-{i:03d}'
    hist = [{'role': h['role'], 'content': h['content'], 'model': None} for h in r.get('history') or [] if h.get('content')]
    items.append({'id': iid, 'question': r['question'], 'history': hist, 'category': 'dislike', 'topic': 'dislike'})
    meta.append({'id': iid, 'fb_at': r['fb_at'], 'model': r['model'], 'comment': r['comment'], 'old_answer': r['answer'],
                 'old_sources': r['sources'], 'question': r['question'], 'history_turns': len(hist)})
with open(f'{D}/items.jsonl', 'w', encoding='utf-8') as f:
    for x in items: f.write(json.dumps(x, ensure_ascii=False) + '\n')
with open(f'{D}/meta.jsonl', 'w', encoding='utf-8') as f:
    for x in meta: f.write(json.dumps(x, ensure_ascii=False, default=str) + '\n')
json.dump([x['id'] for x in items], open(f'{D}/ids.json', 'w'))
json.dump({'jobs': [{'name': 'dislikes', 'items': f'{D}/items.jsonl', 'ids': f'{D}/ids.json', 'out': f'{D}/ragu6k2.records.jsonl'}]},
          open(f'{D}/jobs.json', 'w'), indent=1)
print(len(items), 'items;', sum(1 for x in meta if '@ragu' in x['model']), 'were RAGU answers')
