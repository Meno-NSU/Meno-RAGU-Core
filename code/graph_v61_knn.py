"""
Nearest neighbours of every entity in the stand's entity collection (read-only queries by point id), the input of
the merge candidates: rows {id, neighbours: [[id, score], ...]}.

    python graph_v61_knn.py --collections .../ragu_graph/qdrant_collections.json --out llm/knn.jsonl
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collections', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:6433')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--min_score', type=float, default=0.86)
    args = parser.parse_args()
    base = f'{args.url}/collections/' + json.loads(args.collections.read_text())['vdb_entity']

    def post(path, body):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
        return json.loads(urllib.request.urlopen(request, timeout=300).read())['result']

    ids, offset = {}, None
    while True:
        body = {'limit': 5000, 'with_payload': ['__ragu_id__'], 'with_vector': False}
        if offset:
            body['offset'] = offset
        page = post('/points/scroll', body)
        ids.update({point['id']: point['payload']['__ragu_id__'] for point in page['points']})
        offset = page.get('next_page_offset')
        if not offset:
            break
    points, started = sorted(ids), time.time()
    with open(args.out, 'w', encoding='utf-8') as out:
        for start in range(0, len(points), 100):
            part = points[start:start + 100]
            found = post('/points/query/batch', {'searches': [{'query': point, 'using': 'dense', 'limit': args.limit + 1,
                                                               'with_payload': False} for point in part]})
            for point, result in zip(part, found):
                near = [[ids[p['id']], round(p['score'], 4)] for p in result['points']
                        if p['id'] != point and p['score'] >= args.min_score]
                if near:
                    out.write(json.dumps({'id': ids[point], 'neighbours': near}) + '\n')
            if start % 20000 == 0:
                print(start, len(points), round(time.time() - started), flush=True)
            time.sleep(0.02)


if __name__ == '__main__':
    main()
