"""
The entity and relation collections of graph v6.1, built from the v6 ones: the vectors of the items whose embedded
text did not change are copied (also under a new id), the changed texts are embedded with the embedder of the v6 build
(bge-m3 through ragu's EmbedderOpenAI and RAGU BM25), the ids that are gone are not carried over. Point ids follow
ragu's rule, uuid5(URL, "<collection>:<ragu id>"), so every point of the new collection has a new point id.

A dry run by default: counts, estimates and guards, nothing written. --apply writes the two new collections (never
an existing one) and <graph_dir>/qdrant_collections.json for the stand; it pauses under the disk floor.

    python graph_v61_collections.py --stand ~/meno-v2/ragu-v2/stand-v6k2 --graph_dir ~/meno-v2/ragu-v2/graph-v61
    ... --apply --suffix _v61_trial --limit 300      a trial on a few hundred points in a throw-away collection
"""
import argparse
import json
import pickle
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DISK_FLOOR_GB = 15.5


def point_id(collection: str, ragu_id: str) -> str:
    """ragu's QdrantVDB point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{collection}:{ragu_id}'))


def classify(new_texts: dict, old_texts: dict) -> tuple[dict, list]:
    """new id -> the old id whose vector it takes (the same id with the same text, else any old id with that text),
    and the new ids to embed."""
    by_text = {}
    for old_id, text in old_texts.items():
        by_text.setdefault(text, old_id)
    reuse, embed = {}, []
    for new_id, text in new_texts.items():
        if old_texts.get(new_id) == text:
            reuse[new_id] = new_id
        elif text in by_text:
            reuse[new_id] = by_text[text]
        else:
            embed.append(new_id)
    return reuse, embed


def entity_item(node, chunks_as_text: bool):
    chunks = str(list(node.chunks)) if chunks_as_text else list(node.chunks)
    return f'{node.name} - {node.description}', {
        'entity_name': node.name, 'entity_type': node.type, 'description': node.description, 'source_chunk_id': chunks,
        'documents_id': [], 'clusters': [], 'id': node.id, '__ragu_id__': node.id}


def relation_item(edge, nodes):
    return edge.description, {
        'subject_id': edge.source, 'object_id': edge.target, 'subject_name': nodes[edge.source].name,
        'object_name': nodes[edge.target].name, 'relation_type': edge.type, 'description': edge.description,
        'relation_strength': edge.strength, 'source_chunk_id': list(edge.chunks), 'id': edge.id, '__ragu_id__': edge.id}


class Qdrant:
    def __init__(self, url):
        self.url = url.rstrip('/')

    def call(self, method, path, body=None):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={'content-type': 'application/json'}, method=method)
        return json.loads(urllib.request.urlopen(request, timeout=600).read())['result']

    def exists(self, collection):
        return collection in {c['name'] for c in self.call('GET', '/collections')['collections']}


def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stand', type=Path, required=True)
    parser.add_argument('--graph_dir', type=Path, required=True)
    parser.add_argument('--qdrant', default='http://127.0.0.1:6433')
    parser.add_argument('--embedder', default='http://127.0.0.1:8001/v1')
    parser.add_argument('--suffix', default='_v61')
    parser.add_argument('--limit', type=int, default=0, help='a trial: this many items of each collection, half of them to embed')
    parser.add_argument('--embed_batches', type=int, default=4, help='concurrent batches to the embedder (the v6 build used 4)')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    log = lambda *parts: print(time.strftime('%H:%M:%S'), *parts, flush=True)
    names = json.loads((args.stand / 'knowledge/ragu_graph/qdrant_collections.json').read_text())
    qdrant = Qdrant(args.qdrant)
    old = pickle.loads((args.graph_dir / 'work/graph-0-stand.pkl').read_bytes())
    new = pickle.loads((args.graph_dir / 'work/graph-3-relations.pkl').read_bytes())
    sample = qdrant.call('POST', f'/collections/{names["vdb_entity"]}/points/scroll', {'limit': 200, 'with_payload': ['source_chunk_id']})
    chunks_as_text = sum(isinstance(p['payload']['source_chunk_id'], str) for p in sample['points']) > 100
    plan, summary = {}, {'free_gb': round(free_gb(args.graph_dir), 1), 'entity source_chunk_id stored as text': chunks_as_text}
    for kind in ('vdb_entity', 'vdb_relation'):
        source, target = names[kind], names[kind] + args.suffix
        if kind == 'vdb_entity':
            items = {node.id: entity_item(node, chunks_as_text) for node in new.nodes.values()}
            old_texts = {node.id: f'{node.name} - {node.description}' for node in old.nodes.values()}
        else:
            items = {edge.id: relation_item(edge, new.nodes) for edge in new.edges}
            old_texts = {edge.id: edge.description for edge in old.edges}
        reuse, embed = classify({key: text for key, (text, _) in items.items()}, old_texts)
        if args.limit:
            embed = embed[:args.limit // 2]
            keep = set(embed) | set(list(reuse)[:args.limit - len(embed)])
            items = {key: value for key, value in items.items() if key in keep}
            reuse = {key: value for key, value in reuse.items() if key in keep}
        info = qdrant.call('GET', f'/collections/{source}')
        payload_keys = set(qdrant.call('POST', f'/collections/{source}/points/scroll', {'limit': 1, 'with_payload': True})['points'][0]['payload'])
        assert payload_keys == set(next(iter(items.values()))[1]), (kind, payload_keys)
        plan[kind] = (source, target, items, reuse, embed, info)
        summary[kind] = {'source': source, 'target': target, 'target exists': qdrant.exists(target), 'source points': info['points_count'],
                         'points to write': len(items), 'vectors copied': len(reuse), 'of them under a new id': sum(a != b for a, b in reuse.items()),
                         'texts to embed': len(embed), 'old ids not carried over': len(set(old_texts) - set(items)) if not args.limit else None,
                         'characters to embed': sum(len(items[key][0]) for key in embed)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not args.apply:
        log('dry run: nothing written')
        return
    for kind, (source, target, *_rest) in plan.items():
        if qdrant.exists(target):
            raise SystemExit(f'{target} exists: refusing to write to an existing collection')
        if target in names.values() or not args.suffix:
            raise SystemExit('the target must be a new collection')
    if free_gb(args.graph_dir) < DISK_FLOOR_GB:
        raise SystemExit(f'only {free_gb(args.graph_dir):.1f} GB free (floor {DISK_FLOOR_GB})')

    import asyncio
    import numpy as np
    from ragu import Settings
    from ragu.models.embedder import EmbedderOpenAI
    from ragu.models.openai import CachedAsyncOpenAI
    from ragu.models.sparse_embedder import BM25
    from graph_utils.compact_embeddings import CompactEmbeddingClient
    Settings.tokenizer_embedder_backend = 'local'
    Settings.tokenizer_embedder_name = 'BAAI/bge-m3'
    Settings.embedder_token_limit = 8192
    embedder = EmbedderOpenAI(client=CompactEmbeddingClient(CachedAsyncOpenAI(base_url=args.embedder, api_key='EMPTY')),
                              model_name='BAAI/bge-m3', max_concurrent_batches=args.embed_batches)
    sparse, loop = BM25(language='russian'), asyncio.new_event_loop()

    def guard():
        while free_gb(args.graph_dir) < DISK_FLOOR_GB:
            log(f'only {free_gb(args.graph_dir):.1f} GB free: waiting (floor {DISK_FLOOR_GB})')
            time.sleep(60)

    timing = {}
    for kind, (source, target, items, reuse, embed, info) in plan.items():
        params = info['config']['params']
        qdrant.call('PUT', f'/collections/{target}', {'vectors': params['vectors'], 'sparse_vectors': params['sparse_vectors'],
                                                      'on_disk_payload': params.get('on_disk_payload', True),
                                                      'hnsw_config': {'m': info['config']['hnsw_config']['m'],
                                                                      'ef_construct': info['config']['hnsw_config']['ef_construct']}})
        started, copied = time.time(), 0
        keys = list(reuse)
        for start in range(0, len(keys), 500):
            part = keys[start:start + 500]
            guard()
            found = qdrant.call('POST', f'/collections/{source}/points', {'ids': [point_id(source, reuse[key]) for key in part],
                                                                         'with_vector': True, 'with_payload': ['__ragu_id__']})
            vectors = {point['payload']['__ragu_id__']: point['vector'] for point in found}
            points = [{'id': point_id(target, key), 'vector': vectors[reuse[key]], 'payload': items[key][1]} for key in part if reuse[key] in vectors]
            assert len(points) == len(part), f'{len(part) - len(points)} source points missing'
            qdrant.call('PUT', f'/collections/{target}/points?wait=true', {'points': points})
            copied += len(points)
            if start % 50000 == 0:
                log(kind, f'{copied}/{len(keys)} copied')
        timing[kind + ' copy seconds'] = round(time.time() - started, 1)
        started = time.time()
        for start in range(0, len(embed), 1000):
            part = embed[start:start + 1000]
            texts = [items[key][0] for key in part]
            guard()
            dense = np.asarray(loop.run_until_complete(embedder.batch_embed_text(texts)), dtype=np.float32)
            points = [{'id': point_id(target, key), 'payload': items[key][1],
                       'vector': {'dense': vector.tolist(), 'bm25': {'indices': [int(i) for i in s.indices], 'values': [float(v) for v in s.values]}}}
                      for key, vector, s in zip(part, dense, sparse.embed_document(texts))]
            for at in range(0, len(points), 250):
                qdrant.call('PUT', f'/collections/{target}/points?wait=true', {'points': points[at:at + 250]})
            log(kind, f'{start + len(part)}/{len(embed)} embedded')
        timing[kind + ' embed seconds'] = round(time.time() - started, 1)
        count = qdrant.call('POST', f'/collections/{target}/points/count', {'exact': True})['count']
        assert count == len(items), (count, len(items))
        timing[kind + ' points'] = count
    if not args.limit:
        stand_names = {**names, 'vdb_entity': plan['vdb_entity'][1], 'vdb_relation': plan['vdb_relation'][1]}
        (args.graph_dir / 'qdrant_collections.json').write_text(json.dumps(stand_names, ensure_ascii=False, indent=1))
    print(json.dumps(timing, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
