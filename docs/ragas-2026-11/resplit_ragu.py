"""RAGU records with one context block per prompt document and per graph element. ragu_ragas_run.py (before
2026-11-12 10:00) split the prompt at paragraph breaks: a bare "DOCUMENT 1" header came first, every document ended
with the next one's header, and each graph element fell into 3-4 pieces. Only context precision (rank-weighted, per
block) depends on the blocks; the other metrics read the contexts as one text.
usage: resplit_ragu.py IN.jsonl OUT.jsonl"""
import json, re, sys

ELEMENT = re.compile(r'^\*\*Элемент графа знаний: ', re.M)
HEADER = re.compile(r'^=+\nDOCUMENT \d+\n=+$', re.M)


def blocks(pieces):
    first = next((i for i, p in enumerate(pieces) if ELEMENT.match(p)), len(pieces))
    documents, graph = '\n\n'.join(pieces[:first]), '\n\n'.join(pieces[first:])
    out = [b.strip() for b in re.split(r'(?=^=+\nDOCUMENT \d+\n=+$)', documents, flags=re.M) if b.strip()]
    out += [b.strip() for b in re.split(r'(?=^\*\*Элемент графа знаний: )', graph, flags=re.M) if b.strip()]
    assert not any(HEADER.fullmatch(b) for b in out), 'a bare header is left'
    return out


n = docs = elements = 0
with open(sys.argv[2], 'w', encoding='utf-8') as out:
    for line in open(sys.argv[1], encoding='utf-8'):
        r = json.loads(line)
        pieces = [c['text'] for c in r.get('contexts') or [] if c.get('text')]
        if pieces:
            b = blocks(pieces)
            assert '\n\n'.join(pieces).replace('\n', '').replace(' ', '') == ''.join(b).replace('\n', '').replace(' ', ''), r['item_id']
            r['contexts'] = [{'text': t} for t in b]
            n += 1
            docs += sum(t.startswith('=') for t in b)
            elements += sum(t.startswith('**Элемент') for t in b)
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'{sys.argv[1]}: {n} items, {docs / n:.1f} documents + {elements / n:.1f} graph elements per item, {docs + elements} blocks')
