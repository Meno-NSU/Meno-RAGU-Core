"""
Prepares the model passes of graph v6.1 (graph_utils/graph_patch_jobs.py) from the patched graph; asks nothing.

    python graph_v61_llm_jobs.py --stand ~/meno-v2/ragu-v2/stand-v6k2 --graph_dir ~/meno-v2/ragu-v2/graph-v61 \
        --abbreviations .../abbreviations.json --ontology graph_utils/ontology/nsu.yaml \
        [--knn llm/knn.jsonl] [--other_initials .../namesakes.jsonl]

Writes <graph_dir>/llm/{pairs,direction,person_chunk,abbreviation}.jobs.jsonl and llm/jobs.json (the counts).
--knn: rows {id, neighbours: [[id, score], ...]} of the stand's entity collection (ids of the stand graph).
--other_initials: rows {label, conflicting: [{chunk}]}: person nodes of the stand graph with source chunks that state
other initials (the audit of 19.11).
"""
import argparse
import collections
import json
import pickle
import re
from pathlib import Path

from graph_utils import graph_patch as gp
from graph_utils import graph_patch_jobs as jobs

ONE_WAY = ('PART_OF', 'LOCATED_IN', 'SUBORDINATE_OF', 'SUPERVISES', 'SUBEVENT_OF', 'SUBFIELD_OF', 'PARENT_OF')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stand', type=Path, required=True)
    parser.add_argument('--graph_dir', type=Path, required=True)
    parser.add_argument('--abbreviations', type=Path, required=True)
    parser.add_argument('--ontology', type=Path, required=True)
    parser.add_argument('--knn', type=Path)
    parser.add_argument('--knn_score', type=float, default=0.88)
    parser.add_argument('--other_initials', type=Path)
    parser.add_argument('--batch', type=int, default=10)
    args = parser.parse_args()
    work, out = args.graph_dir / 'work', args.graph_dir / 'llm'
    out.mkdir(exist_ok=True)
    graph = pickle.loads((work / 'graph-3-relations.pkl').read_bytes())
    changes = [json.loads(row) for number, step in enumerate(('source_form', 'people', 'relations'), 1)
               for row in open(work / f'changes-{number}-{step}.jsonl', encoding='utf-8')]
    moved = gp.id_map(changes)
    now = lambda node_id: moved.get(node_id, node_id)
    chunks = json.loads((args.stand / 'knowledge/ragu_graph/kv_chunks.json').read_text())
    text = lambda chunk: (chunks.get(chunk) or {}).get('content') or ''
    dictionary = json.loads(args.abbreviations.read_text())
    pairs_of_dictionary = gp.expansions(dictionary)
    nodes = graph.nodes
    counts = {}

    def side(node_id):
        node = nodes[node_id]
        forms = gp.name_forms(node.name, pairs_of_dictionary)
        if node.type == 'PERSON':
            forms += [reading[0] for reading in gp.readings(node.name)]
        source = ''
        for chunk in node.chunks[:8]:
            source = jobs.around(text(chunk), forms)
            if any(form.casefold()[:5] in source.casefold() for form in forms):
                break
        return {'id': node_id, 'name': node.name, 'type': node.type, 'description': node.description, 'source': source}

    # ---- pairs: merge candidates and stated aliases
    names = {node_id: node.name for node_id, node in nodes.items()}
    people = {node_id: name for node_id, name in names.items() if nodes[node_id].type == 'PERSON'}
    things = {node_id: name for node_id, name in names.items() if nodes[node_id].type != 'PERSON'}
    found = collections.OrderedDict()
    aliases = [edge for edge in graph.edges if edge.type in ('ALTERNATIVE_NAME', 'ABBREVIATION')]
    for edge in aliases:
        found[tuple(sorted((edge.source, edge.target)))] = ('alias:' + edge.type, edge.description, edge.id)
    sources = [('bracket', jobs.bracket_pairs(names)), ('same_key', jobs.same_key_pairs(names)),
               ('stems', jobs.stem_pairs(things)), ('person', jobs.person_pairs(people, gp.readings, gp.fits, gp.detail))]
    if args.knn:
        near = set()
        for row in open(args.knn, encoding='utf-8'):
            row = json.loads(row)
            a = now(row['id'])
            for other, score in row['neighbours']:
                b = now(other)
                if score >= args.knn_score and a != b and a in nodes and b in nodes \
                        and 'PERSON' not in (nodes[a].type, nodes[b].type):
                    near.add(tuple(sorted((a, b))))
        sources.append(('neighbour', near))
    for kind, pairs in sources:
        new = 0
        for pair in sorted(pairs):
            if pair not in found:
                found[pair] = (kind, '', '')
                new += 1
        counts['pairs: ' + kind] = new
    counts['pairs: alias relations'] = len(aliases)
    rows = [{'a': side(a), 'b': side(b), 'kind': kind, 'stated': stated, 'edge': edge}
            for (a, b), (kind, stated, edge) in found.items()]
    rows.sort(key=lambda row: row['kind'])
    with open(out / 'pairs.jobs.jsonl', 'w', encoding='utf-8') as file:
        for number, batch in enumerate(jobs.batches(rows, args.batch)):
            file.write(json.dumps({'id': f'pairs-{number}', 'kind': 'pairs',
                                   'pairs': [{'a': r['a']['id'], 'b': r['b']['id'], 'kind': r['kind'], 'edge': r['edge']} for r in batch],
                                   'prompt': jobs.pairs_prompt(batch)}, ensure_ascii=False) + '\n')
    counts['pairs'] = len(rows)
    counts['pairs: calls'] = -(-len(rows) // args.batch)

    # ---- direction: relations inside cycles of one-way types
    import networkx as nx
    from ragu.triplet.ontology.ontology import Ontology
    ontology = Ontology.from_yaml(args.ontology)
    calls = 0
    with open(out / 'direction.jobs.jsonl', 'w', encoding='utf-8') as file:
        for kind in ONE_WAY:
            edges = [edge for edge in graph.edges if edge.type == kind]
            component = {}
            for index, members in enumerate(nx.strongly_connected_components(nx.DiGraph([(e.source, e.target) for e in edges]))):
                if len(members) > 1:
                    component.update({member: index for member in members})
            spec = ontology.relation_types.get(kind)
            meaning = (getattr(spec, 'description', '') or '') if spec else ''
            inside = [e for e in edges if e.source in component and component.get(e.target) == component[e.source]]
            counts[f'direction: {kind}'] = len(inside)
            for edge in inside:
                subject, obj = nodes[edge.source], nodes[edge.target]
                source = jobs.around(text(edge.chunks[0]) if edge.chunks else '', [subject.name, obj.name], 700)
                file.write(json.dumps({'id': 'direction-' + edge.id, 'kind': 'direction', 'edge': edge.id, 'type': kind,
                                       'subject': edge.source, 'object': edge.target,
                                       'prompt': jobs.direction_prompt(kind, meaning, subject.name, obj.name, edge.description, source)},
                                      ensure_ascii=False) + '\n')
                calls += 1
    counts['direction: calls'] = calls

    # ---- person_chunk: chunks that state other initials; nodes with two records of one site
    calls = 0
    with open(out / 'person_chunk.jobs.jsonl', 'w', encoding='utf-8') as file:
        def write(node_id, chunk, why):
            nonlocal calls
            node = nodes[node_id]
            surnames = [reading[0] for reading in gp.readings(node.name)] or [node.name]
            file.write(json.dumps({'id': f'person-{node_id}-{chunk}', 'kind': 'person_chunk', 'node': node_id, 'chunk': chunk,
                                   'why': why, 'prompt': jobs.person_chunk_prompt(node.name, node.description,
                                                                                  jobs.around(text(chunk), surnames, 900))},
                                  ensure_ascii=False) + '\n')
            calls += 1
        if args.other_initials:
            for row in open(args.other_initials, encoding='utf-8'):
                row = json.loads(row)
                node_id = now(row.get('label', ''))
                if node_id in nodes and nodes[node_id].type == 'PERSON':
                    for item in row.get('conflicting', []):
                        if item['chunk'] in nodes[node_id].chunks:
                            write(node_id, item['chunk'], 'other initials in the chunk')
        profiles = args.graph_dir / 'node_profiles.jsonl'
        several = 0
        for row in open(profiles, encoding='utf-8'):
            row = json.loads(row)
            if row['several_profiles_of_one_site'] and row['node'] in nodes:
                several += 1
                for profile in row['profiles']:
                    for chunk in profile['evidence_chunks'][:2]:
                        write(row['node'], chunk, 'the node has several records of one site: ' + profile['key'])
        counts['person_chunk: nodes with several records of one site'] = several
    counts['person_chunk: calls'] = calls

    # ---- abbreviation: which long form in a chunk, for the abbreviations with several
    in_dictionary = {key: ([value] if isinstance(value, str) else list(value)) for key, value in dictionary.items()
                     if not isinstance(value, dict)}
    documents = json.loads((args.stand / 'knowledge/ragu_graph/menon_chunk_mapping.json').read_text())
    stated = gp.stated_long_forms(((documents.get(chunk), value.get('content') or '') for chunk, value in chunks.items()),
                                  list(in_dictionary))
    options = {}
    for abbreviation, expansions in in_dictionary.items():
        forms = list(expansions)
        keys = {' '.join(w[:max(4, len(w) - 2)].casefold() for w in form.split()) for form in forms}
        for key, record in sorted(stated.get(abbreviation, {}).items(), key=lambda item: -item[1]['documents']):
            if key not in keys and not any(key[:12] == other[:12] for other in keys):
                forms.append(record['example'])
                keys.add(key)
        if len(forms) > 1:
            options[abbreviation] = forms
    counts['abbreviation: abbreviations with several long forms'] = len(options)
    (out / 'abbreviation_options.json').write_text(json.dumps(options, ensure_ascii=False, indent=1))
    replaced = json.loads((work / 'replaced.json').read_text())
    restored = collections.defaultdict(set)
    for node_id, found_ in replaced.items():
        for abbreviation, _ in found_:
            restored[now(node_id)].add(abbreviation)
    tokens = {abbreviation: gp._token(abbreviation) for abbreviation in options}
    calls, split_nodes = 0, 0
    with open(out / 'abbreviation.jobs.jsonl', 'w', encoding='utf-8') as file:
        for node_id, node in nodes.items():
            folded = node.name.casefold()
            mine = {a for a in restored.get(node_id, ()) if a in options}
            mine |= {a for a, forms in options.items() if any(form.casefold() in folded for form in forms)}
            wrote = False
            for abbreviation in sorted(mine):
                long_forms = [form.casefold()[:max(8, len(form) * 2 // 3)] for form in options[abbreviation]]
                for chunk in node.chunks:
                    content = text(chunk)
                    if tokens[abbreviation].search(content) and not any(form in content.casefold() for form in long_forms):
                        match = tokens[abbreviation].search(content)
                        source = ' '.join(content[max(0, match.start() - 400):match.start() + 500].split())
                        file.write(json.dumps({'id': f'abbr-{node_id}-{abbreviation}-{chunk}', 'kind': 'abbreviation',
                                               'node': node_id, 'chunk': chunk, 'abbreviation': abbreviation,
                                               'options': options[abbreviation],
                                               'prompt': jobs.abbreviation_prompt(abbreviation, options[abbreviation], node.name, source)},
                                              ensure_ascii=False) + '\n')
                        calls += 1
                        wrote = True
            split_nodes += wrote
    counts['abbreviation: nodes'] = split_nodes
    counts['abbreviation: calls'] = calls
    (out / 'jobs.json').write_text(json.dumps(counts, ensure_ascii=False, indent=1))
    print(json.dumps(counts, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
