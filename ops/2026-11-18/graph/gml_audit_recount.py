"""Recount the v6 graph audit's structural claims from the GML (read-only, streaming)."""
import collections, hashlib, html, json, pathlib, re, sys, time
G = pathlib.Path.home() / 'meno-v2/ragu-v2/stand-v6k2/knowledge/ragu_graph/knowledge_graph.gml'
t0 = time.time()
names, types, nchunks = {}, {}, {}
edges = []  # (src, tgt, type, desc md5, chunk tuple)
kv = re.compile(r'^\s+(\w+) (.*)$')
cur, kind, chunks = None, None, []
with open(G, encoding='utf-8') as f:
    for line in f:
        s = line.strip()
        if s == 'node [' or s == 'edge [':
            kind, cur, chunks = s[:4], {}, []
            continue
        if s == ']' and kind:
            if kind == 'node':
                i = int(cur['id']); names[i] = html.unescape(cur.get('entity_name', '').strip('"'))
                types[i] = cur.get('entity_type', '').strip('"'); nchunks[i] = len(chunks)
            else:
                edges.append((int(cur['source']), int(cur['target']), cur.get('relation_type', '').strip('"'),
                              hashlib.md5(cur.get('description', '').encode()).hexdigest()[:12], tuple(chunks)))
            kind = None
            continue
        if kind:
            m = kv.match(line)
            if not m: continue
            k, v = m.group(1), m.group(2)
            if k == 'source_chunk_id':
                v = v.strip('"')
                if v.startswith('chunk-'): chunks.append(sys.intern(v))
            elif k in ('id', 'entity_name', 'entity_type', 'source', 'target', 'relation_type', 'description'):
                cur[k] = v
print(f'parsed {len(names)} nodes, {len(edges)} edges in {time.time() - t0:.0f} s', file=sys.stderr)
r = collections.OrderedDict()
r['nodes'] = len(names); r['edges'] = len(edges)
deg = collections.Counter(); nb = collections.defaultdict(set); etype = collections.Counter()
for a, b, t, d, c in edges:
    deg[a] += 1; deg[b] += 1; nb[a].add(b); nb[b].add(a); etype[t] += 1
r['RELATED_TO share'] = round(etype['RELATED_TO'] / len(edges), 4)
r['top edge types'] = etype.most_common(8)
by_name = collections.defaultdict(list)
for i, n in names.items(): by_name[n].append(i)
def node_info(n):
    out = []
    for i in by_name.get(n, []):
        inc = collections.Counter(); inc_from = collections.Counter(); allt = collections.Counter()
        for a, b, t, d, c in edges:
            pass
        out.append({'id': i, 'type': types[i], 'edges': deg[i], 'neighbours': len(nb[i]), 'source_chunks': nchunks[i]})
    return out
top = deg.most_common(12)
r['top hubs'] = [(names[i], types[i], k, len(nb[i])) for i, k in top]
watch = ['Новосибирский государственный университет', 'Новосибирский государственный университет (НГУ)', 'НГУ',
         'Pure', 'Россия', 'Новосибирск', 'Английский', 'Бакалавриат', 'ISSC-2025', 'Кафедра высшей математики',
         'Дмитрий Пышный', 'Пышный Дмитрий Владимирович', 'Михаил Петрович Федорук', 'Михаил Федорук',
         'Федорук Максим Петрович', 'Новосибирский государственный университет Pure', '1234']
r['named nodes'] = {n: node_info(n) for n in watch}
ids_of = {n: set(by_name.get(n, [])) for n in watch}
ngu = ids_of['Новосибирский государственный университет']; pure = ids_of['Pure']
inc_ngu = collections.Counter(t for a, b, t, d, c in edges if b in ngu)
r['NGU incoming types'] = inc_ngu.most_common(6)
r['NGU RELATED_TO (any direction)'] = sum(1 for a, b, t, d, c in edges if t == 'RELATED_TO' and (a in ngu or b in ngu))
r['Pure edges by type'] = collections.Counter(t for a, b, t, d, c in edges if a in pure or b in pure).most_common(6)
r['Pure incoming USES from PERSON'] = sum(1 for a, b, t, d, c in edges if b in pure and t == 'USES' and types[a] == 'PERSON')
r['Pure incoming OWNER_OF from PERSON'] = sum(1 for a, b, t, d, c in edges if b in pure and t == 'OWNER_OF' and types[a] == 'PERSON')
top4 = {i for i, _ in deg.most_common(4)}
r['edges touching the top-4 hubs'] = sum(1 for a, b, t, d, c in edges if a in top4 or b in top4)
r['nodes with exactly one neighbour'] = sum(1 for i in names if len(nb[i]) == 1)
r['nodes without edges'] = sum(1 for i in names if len(nb[i]) == 0)
alt = [(a, b) for a, b, t, d, c in edges if t == 'ALTERNATIVE_NAME']
r['ALTERNATIVE_NAME'] = len(alt); r['ALTERNATIVE_NAME between different types'] = sum(1 for a, b in alt if types[a] != types[b])
od = collections.defaultdict(set)
for a, b, t, d, c in edges: od[(a, b)].add(t)
r['ordered pairs with several types'] = sum(1 for v in od.values() if len(v) > 1)
ud = collections.defaultdict(list)
for e in edges: ud[(min(e[0], e[1]), max(e[0], e[1]))].append(e)
rel_pairs = shared = same_desc = 0
for es in ud.values():
    rel = [e for e in es if e[2] == 'RELATED_TO']; other = [e for e in es if e[2] != 'RELATED_TO']
    if rel and other:
        rel_pairs += 1
        if any(set(x[4]) & set(y[4]) for x in rel for y in other): shared += 1
        if any(x[3] == y[3] for x in rel for y in other): same_desc += 1
r['undirected pairs with RELATED_TO and a typed edge'] = rel_pairs
r['  of them sharing a source chunk'] = shared; r['  of them with an identical description'] = same_desc
def both_ways(kind):
    s = {(a, b): c for a, b, t, d, c in edges if t == kind and a != b}
    pairs = [(a, b) for (a, b) in s if a < b and (b, a) in s]
    return len(pairs), sum(1 for a, b in pairs if set(s[(a, b)]) & set(s[(b, a)]))
r['KNOWS'] = etype['KNOWS']; r['KNOWS both ways (pairs, sharing a chunk)'] = both_ways('KNOWS')
r['PART_OF both ways (pairs, sharing a chunk)'] = both_ways('PART_OF')
r['LOCATED_IN both ways (pairs, sharing a chunk)'] = both_ways('LOCATED_IN')
for kind in ('PLACE_OF_BIRTH', 'PLACE_OF_DEATH'):
    objs = collections.defaultdict(set)
    for a, b, t, d, c in edges:
        if t == kind: objs[a].add(b)
    r[f'{kind}: subjects with several objects'] = sum(1 for v in objs.values() if len(v) > 1)
kaf = ids_of['Кафедра высшей математики']
r['Кафедра высшей математики PART_OF targets'] = sorted({names[b] for a, b, t, d, c in edges if a in kaf and t == 'PART_OF'})[:12]
fed = ids_of['Федорук Максим Петрович']
r['Федорук Максим Петрович edges'] = collections.Counter(t for a, b, t, d, c in edges if a in fed or b in fed).most_common(5)
print(json.dumps(r, ensure_ascii=False, indent=1, default=list))
print(f'done in {time.time() - t0:.0f} s', file=sys.stderr)
