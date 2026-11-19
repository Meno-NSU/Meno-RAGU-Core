"""
Graph v6.1 without a model: patches the stand's knowledge_graph.gml (graph_utils/graph_patch.py) and writes the
patched graph, every change with its reason and evidence, the profiles and template facts of the nodes, the checks
before and after each step and the texts the vector collections must embed again. Reads the stand, writes --out only.

    python patch_graph_v61.py --stand ~/meno-v2/ragu-v2/stand-v6k2 --abbreviations .../abbreviations.json \
        --ontology graph_utils/ontology/nsu.yaml --out ~/meno-v2/ragu-v2/graph-v61
"""
import argparse
import collections
import json
import pickle
import random
import re
import time
from pathlib import Path

from graph_utils import graph_patch as gp

STEPS = ('source_form', 'people', 'relations')


def log(*parts):
    print(time.strftime('%H:%M:%S'), *parts, flush=True)


def load_documents(stand: Path):
    """chunk id -> document index, and (url, title) of each document of the stand corpus."""
    numbers = json.loads((stand / 'knowledge/ragu_graph/menon_chunk_mapping.json').read_text())
    mapping = json.loads((stand / 'chunk_mapping_to_texts.json').read_text())
    chunk_document = {chunk: mapping[str(number)]['doc_index'] for chunk, number in numbers.items() if str(number) in mapping}
    documents = []
    with open(stand / 'chunked_texts_about_nsu_with_metadata_and_scores.jsonl', encoding='utf-8') as lines:
        for line in lines:
            row = json.loads(line)
            url = row.get('url')
            documents.append(((url[0] if url else '') if isinstance(url, list) else (url or ''), row.get('doc_title') or ''))
    return chunk_document, documents


def symmetric_types(ontology_path) -> frozenset:
    from ragu.triplet.ontology.ontology import Ontology
    ontology = Ontology.from_yaml(ontology_path)
    return frozenset(name for name, spec in ontology.relation_types.items() if spec.symmetric)


def counts(graph):
    return {'nodes': len(graph.nodes), 'relations': len(graph.edges)}


def sample(changes, ops, size, seed):
    rows = [c for c in changes if c['op'] in ops]
    random.Random(seed).shuffle(rows)
    return rows[:size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stand', type=Path, required=True)
    parser.add_argument('--abbreviations', type=Path, required=True)
    parser.add_argument('--ontology', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', default=','.join(STEPS))
    parser.add_argument('--hub_degree', type=int, default=500)
    parser.add_argument('--template_others', type=int, default=50,
                        help='a line that alone ties a hub to at least this many nodes is a line of a page template')
    parser.add_argument('--write_gml', action='store_true')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    steps = args.steps.split(',')

    source = args.stand / 'knowledge/ragu_graph/knowledge_graph.gml'
    cached = work / 'graph-0-stand.pkl'
    if cached.exists():
        original = pickle.loads(cached.read_bytes())
    else:
        log('reading', source)
        original = gp.read_gml(source)
        cached.write_bytes(pickle.dumps(original, protocol=4))
    log('stand graph', counts(original))
    log('reading chunks and documents')
    chunks = json.loads((args.stand / 'knowledge/ragu_graph/kv_chunks.json').read_text())
    chunk_text = lambda chunk: (chunks.get(chunk) or {}).get('content')
    chunk_document_index, documents = load_documents(args.stand)
    chunk_document = chunk_document_index.get
    pairs = gp.expansions(json.loads(args.abbreviations.read_text()))
    stats_path = out / 'stats.json'
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats['stand'] = counts(original)

    graph, previous = original, 'graph-0-stand.pkl'
    for number, step in enumerate(STEPS, 1):
        result = work / f'graph-{number}-{step}.pkl'
        changes_path = work / f'changes-{number}-{step}.jsonl'
        if step not in steps:
            if result.exists():
                graph = pickle.loads(result.read_bytes())
            continue
        log('step', step, 'on', previous, counts(graph))
        before, changes, record = counts(graph), [], {}
        if step == 'source_form':
            renames, replaced = gp.source_form_renames(graph, chunk_text, pairs)
            record['names with an expansion no source states, before'] = len(renames)
            record['expansions replaced'] = collections.Counter(a for found in replaced.values() for a, _ in found).most_common(40)
            record['restored name found word for word in a source chunk'] = sum(
                1 for node_id, (name, evidence) in renames.items()
                if any(name.casefold() in ' '.join((chunk_text(c) or '').casefold().split()) for c in evidence[:2]))
            graph, changes = gp.rename_nodes(graph, renames, step, 'the name holds an expansion no source chunk states; '
                                             'the sources have the abbreviation')
            record['names with an expansion no source states, after'] = len(gp.unstated_expansions(graph, chunk_text, pairs))
            (work / 'replaced.json').write_text(json.dumps(replaced, ensure_ascii=False))
        elif step == 'people':
            profile_of_document = {}
            for index, (url, title) in enumerate(documents):
                key = gp.profile_key(url, title)
                if key:
                    profile_of_document[index] = (key, gp.title_person(title))
            record['profile documents, titled'] = len(profile_of_document)
            profile_of_document = gp.consistent_profiles(profile_of_document)
            record['profile documents'] = len(profile_of_document)
            record['profiles'] = len({key for key, _ in profile_of_document.values()})
            merges = gp.full_name_merges(graph)
            record['full names in another order, merges'] = len(merges)
            graph, changes = gp.merge_nodes(graph, merges, step)
            owners, misspelled = gp.profile_owners(graph, chunk_document, profile_of_document.get)
            record['profiles with an owner node, before'] = len(owners)
            record['profiles with several owner nodes, before'] = sum(1 for nodes in owners.values() if len(nodes) > 1)
            record['nodes with a misspelled surname in a profile'] = sum(len(nodes) for nodes in misspelled.values())
            merges = gp.profile_merges(graph, owners, misspelled, chunk_text)
            graph, more = gp.merge_nodes(graph, merges, step)
            changes += more
            owners, misspelled = gp.profile_owners(graph, chunk_document, profile_of_document.get)
            profiles_of = collections.defaultdict(list)
            for key, nodes in owners.items():
                for node_id in nodes:
                    profiles_of[node_id].append(key)
            merges = gp.short_name_merges(graph, chunk_document, profiles_of)
            record['short names joined through a shared document or profile'] = len(merges)
            graph, more = gp.merge_nodes(graph, merges, step)
            changes += more
            # the profiles are attributes of the nodes that remain; a second round catches the owners the joins made
            for _ in range(3):
                owners, misspelled = gp.profile_owners(graph, chunk_document, profile_of_document.get)
                merges = gp.profile_merges(graph, {k: v for k, v in owners.items() if len(v) > 1}, {}, chunk_text)
                if not merges:
                    break
                graph, more = gp.merge_nodes(graph, merges, step)
                changes += more
            record['profiles with an owner node, after'] = len(owners)
            record['profiles with several owner nodes, after'] = sum(1 for nodes in owners.values() if len(nodes) > 1)
            profiles_of = collections.defaultdict(list)
            for key, nodes in owners.items():
                for node_id, evidence in nodes.items():
                    profiles_of[node_id].append((key, evidence))
            url_of = {}
            for index, (key, _) in profile_of_document.items():
                url_of.setdefault(key, documents[index][0])
            several = 0
            with open(out / 'node_profiles.jsonl', 'w', encoding='utf-8') as rows:
                for node_id, found in profiles_of.items():
                    # two UUIDs of one site are two records of that site; several pages of a site about one person are not
                    sites = collections.Counter(key.split(':')[0] for key, _ in found if not key.startswith('http'))
                    sites[''] = 1
                    several += max(sites.values()) > 1
                    rows.write(json.dumps({'node': node_id, 'name': graph.nodes[node_id].name,
                                           'profiles': [{'key': key, 'url': url_of[key], 'evidence_chunks': evidence[:3]}
                                                        for key, evidence in sorted(found)],
                                           'several_profiles_of_one_site': max(sites.values()) > 1}, ensure_ascii=False) + '\n')
                    changes.append({'step': step, 'op': 'attribute', 'reason': 'the title of the profile page names this person',
                                    'node': node_id, 'name': graph.nodes[node_id].name, 'profiles': [key for key, _ in sorted(found)],
                                    'evidence_chunks': [c for _, evidence in sorted(found) for c in evidence[:1]][:5]})
            record['nodes with a profile'] = len(profiles_of)
            record['nodes with several profiles of one site (for the namesake pass)'] = several
        elif step == 'relations':
            symmetric = symmetric_types(args.ontology)
            record['symmetric types'] = sorted(symmetric)
            two_way = lambda g: {kind: sum(1 for a, b in found if a < b and (b, a) in found) for kind, found in (
                (kind, {(e.source, e.target) for e in g.edges if e.type == kind}) for kind in sorted(symmetric))}
            record['two-way pairs of symmetric types, before'] = {k: v for k, v in two_way(graph).items() if v}
            record['cycles, before'] = gp.cycle_counts(graph)
            graph, changes = gp.fold_relations(graph, symmetric, step)
            record['two-way pairs of symmetric types, after'] = {k: v for k, v in two_way(graph).items() if v}
            record['fold'] = dict(collections.Counter((c['op'], c['reason']) for c in changes).most_common())
            record['fold'] = {f'{op}: {reason}': n for (op, reason), n in record['fold'].items()}
            degree = collections.Counter()
            for edge in graph.edges:
                degree[edge.source] += 1
                degree[edge.target] += 1
            hubs = {node_id for node_id, n in degree.items() if n >= args.hub_degree}
            log('hubs', len(hubs))
            forms = {hub: gp.name_forms(graph.nodes[hub].name, pairs) for hub in hubs}

            def other_forms(node_id):
                node = graph.nodes[node_id]
                found_ = gp.name_forms(node.name, pairs)
                if node.type == 'PERSON':
                    found_ += [reading[0] for reading in gp.readings(node.name)]
                return found_
            found = gp.template_relations(graph, hubs, chunk_text, forms, other_forms, args.template_others)
            by_hub = collections.Counter(hub for _, hub, _ in found)
            record['hubs (degree >= %d)' % args.hub_degree] = len(hubs)
            record['relations of the hubs'] = len([1 for e in graph.edges if e.source in hubs or e.target in hubs])
            record['template relations'] = len(found)
            record['template relations by hub'] = [(graph.nodes[hub].name, n, degree[hub]) for hub, n in by_hub.most_common(40)]
            graph, more, metadata = gp.relations_to_metadata(graph, found, step)
            changes += more
            with open(out / 'node_metadata.jsonl', 'w', encoding='utf-8') as rows:
                for node_id, facts in metadata.items():
                    rows.write(json.dumps({'node': node_id, 'name': graph.nodes[node_id].name, 'template_facts': facts},
                                          ensure_ascii=False) + '\n')
            record['nodes with template facts'] = len(metadata)
            record['cycles, after'] = gp.cycle_counts(graph)
            linked = {e.source for e in graph.edges} | {e.target for e in graph.edges}
            record['nodes left without relations'] = len(set(graph.nodes) - linked)
        record['before'], record['after'] = before, counts(graph)
        record['changes'] = dict(collections.Counter(c['op'] for c in changes))
        stats[step] = record
        with open(changes_path, 'w', encoding='utf-8') as rows:
            gp.write_changes(changes, rows)
        with open(work / f'review-{number}-{step}.jsonl', 'w', encoding='utf-8') as rows:
            gp.write_changes(sample(changes, ('rename', 'merge', 'edge_drop', 'edge_to_metadata', 'edge_merge'), 400, number), rows)
        result.write_bytes(pickle.dumps(graph, protocol=4))
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
        log('done', step, json.dumps(record, ensure_ascii=False)[:3000])
        previous = result.name

    if args.write_gml:
        log('writing the patched graph', counts(graph))
        gp.write_gml(graph, out / 'knowledge_graph.gml')
        with open(out / 'changes.jsonl', 'w', encoding='utf-8') as rows:
            for number, step in enumerate(STEPS, 1):
                path = work / f'changes-{number}-{step}.jsonl'
                if path.exists():
                    rows.write(path.read_text(encoding='utf-8'))
        embed = gp.embedded_text_changes(original, graph)
        stats['embedding'] = {key: len(value) for key, value in embed.items()}
        edges = {edge.id: edge for edge in graph.edges}
        with open(out / 'reembed_entities.jsonl', 'w', encoding='utf-8') as rows:
            for node_id in embed['entities_to_embed']:
                node = graph.nodes[node_id]
                rows.write(json.dumps({'id': node_id, 'entity_name': node.name, 'entity_type': node.type,
                                       'text': f'{node.name} - {node.description}', 'new_id': node_id not in original.nodes},
                                      ensure_ascii=False) + '\n')
        with open(out / 'reembed_relations.jsonl', 'w', encoding='utf-8') as rows:
            for key in ('relations_to_embed', 'relations_rekeyed'):
                for edge_id in embed[key]:
                    edge = edges[edge_id]
                    rows.write(json.dumps({'id': edge_id, 'subject_id': edge.source, 'object_id': edge.target,
                                           'relation_type': edge.type, 'text': edge.description,
                                           'text_changed': key == 'relations_to_embed'}, ensure_ascii=False) + '\n')
        (out / 'vector_ids_gone.json').write_text(json.dumps({'entities': embed['entities_gone'],
                                                              'relations': embed['relations_gone']}))
        stats['final'] = counts(graph)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
        log('embedding', stats['embedding'])


if __name__ == '__main__':
    main()
