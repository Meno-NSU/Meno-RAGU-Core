"""Does a RAGU stand's chunk search match its Qdrant chunk collection? Every id of the stand's chunk channel
(knowledge/ragu_graph/menon_chunk_mapping.json) must be a point of the collection its qdrant_collections.json names
(vdb_chunk), and the collection must hold no other point. Neither shows in the health check: the engine drops a hit
whose id the stand does not know, and a mapped id without a point is never found (18.11.2026: stand-v6k's map against
the collection updated in place for stand-v6k2 — 54,678 mapped ids without a point, 3,307 points the map lacks).
Points carry the chunk id in the payload field __ragu_id__.

usage: python3 -I stand_parity.py STAND_DIR [QDRANT_URL]    exit 0: they match; 1: they do not; 2: cannot check
"""
import json, sys, urllib.request

stand = sys.argv[1].rstrip('/')
qdrant = sys.argv[2] if len(sys.argv) > 2 else 'http://127.0.0.1:6433'
try:
    graph = f'{stand}/knowledge/ragu_graph'
    collection = json.load(open(f'{graph}/qdrant_collections.json'))['vdb_chunk']
    mapped = set(json.load(open(f'{graph}/menon_chunk_mapping.json')))
    points, offset = set(), None
    while True:
        body = {'limit': 20000, 'with_vector': False, 'with_payload': {'include': ['__ragu_id__']}}
        if offset is not None:
            body['offset'] = offset
        request = urllib.request.Request(f'{qdrant}/collections/{collection}/points/scroll',
                                         data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        result = json.load(urllib.request.urlopen(request, timeout=120))['result']
        points.update((p.get('payload') or {}).get('__ragu_id__') for p in result['points'])
        offset = result.get('next_page_offset')
        if offset is None:
            break
except Exception as exc:  # noqa: BLE001
    print(f'stand parity: cannot check {stand}: {exc!r}')
    sys.exit(2)
points.discard(None)
missing, unknown = len(mapped - points), len(points - mapped)
print(f'stand parity {stand}: collection {collection}: {len(mapped)} mapped ids, {len(points)} points, '
      f'{missing} mapped ids without a point, {unknown} points the stand does not map')
sys.exit(0 if missing == 0 and unknown == 0 else 1)
