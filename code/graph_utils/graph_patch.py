"""
Patches of an aggregated graph (graph v6.1): the fixes of the v6 audit that need no model, applied to the stand's
knowledge_graph.gml rather than to the extraction checkpoints, so nothing is extracted or summarized again.

Every step is a general mechanism over the build's own data (its abbreviations, the source chunks, the documents of
the chunks, the ontology); none names an entity. Every change is a row for changes.jsonl with the reason and the
chunks that show it.

1. Source form of a name. The build rewrites a chunk with the expansions of its abbreviations before extraction, so
   a node may carry a name the source never states ("Спорткомплекс Новосибирский государственный университет" for
   "Спорткомплекс НГУ"; a wrong expansion names another institute). A name holding an expansion that none of the
   node's source chunks states, while one of them has the abbreviation, takes the abbreviation back; the nodes that
   then share a name key and a type merge.
2. People. Full names in another word order are one node. A shorter form of a name (initials, no patronymic) joins a
   fuller one only when a document or a profile holds both and the fuller forms it fits there do not contradict each
   other. A profile (a page whose title names one person, keyed by the UUID of its URL or by the URL) is an attribute
   of the node of that person, and the nodes of one profile are one node.
3. Relations. A relation of a type the ontology calls symmetric keeps one order of its ends; a RELATED_TO that states
   what a typed relation of the same two nodes states goes. A relation of a hub whose only support is a line that
   many documents repeat word for word (a page template: the link to a profile, the language of an article) becomes
   metadata of its other end; a hub's relations stated in lines of their own (the participants of a conference) stay.
"""
import dataclasses
import functools
import hashlib
import html
import html.entities
import json
import re
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------------------------------------- graph

@dataclasses.dataclass
class Node:
    id: str
    name: str
    type: str
    description: str
    chunks: list
    extra: dict = dataclasses.field(default_factory=dict)  # other GML attributes, as written


@dataclasses.dataclass
class Edge:
    id: str
    source: str
    target: str
    type: str
    description: str
    strength: float
    chunks: list


@dataclasses.dataclass
class Graph:
    nodes: dict  # id -> Node, in file order
    edges: list


def entity_id(name: str, kind: str) -> str:
    """The id ragu gives an entity (ragu.graph.types.Entity)."""
    return 'ent-' + hashlib.md5(('\0' + name + ' - ' + kind).encode()).hexdigest()


def relation_id(source: str, target: str, kind: str) -> str:
    """The id ragu gives a relation (ragu.graph.types.Relation)."""
    return 'rel-' + hashlib.md5(('\0' + source + ' -> ' + target + kind).encode()).hexdigest()


_ATTRIBUTE = re.compile(r'^\s+(\w+) (.*)$')
_LIST_START = '_networkx_list_start'
_NOT_ASCII = re.compile(r'[^ -~]|[&"]')


_REFERENCE = re.compile(r'&(?:#(\d+)|#x([0-9a-fA-F]+)|(\w+));')


def _reference(match) -> str:
    # as networkx reads it: every numeric reference is its character (html.unescape drops control characters)
    if match.group(1) or match.group(2):
        return chr(int(match.group(1)) if match.group(1) else int(match.group(2), 16))
    code = html.entities.name2codepoint.get(match.group(3))
    return chr(code) if code else match.group(0)


def _unquote(value: str) -> str:
    return _REFERENCE.sub(_reference, value[1:-1]) if value.startswith('"') else value


def _quote(value: str) -> str:
    return '"' + _NOT_ASCII.sub(lambda m: '&#' + str(ord(m.group(0))) + ';', value) + '"'


def read_gml(path) -> Graph:
    """The stand's knowledge_graph.gml (networkx.write_gml of ragu's NetworkXStorage), read as a stream."""
    nodes, edges, label_of = {}, [], {}
    kind, current, chunks = None, None, None
    with open(path, encoding='utf-8') as lines:
        for line in lines:
            text = line.strip()
            if text in ('node [', 'edge ['):
                kind, current, chunks = text[:4], {}, []
            elif text == ']' and kind:
                if kind == 'node':
                    label = _unquote(current.pop('label'))
                    label_of[current.pop('id')] = label
                    nodes[label] = Node(label, _unquote(current.pop('entity_name')), _unquote(current.pop('entity_type')),
                                        _unquote(current.pop('description', '""')), chunks, current)
                else:
                    edges.append(Edge(_unquote(current['key']), label_of[current['source']], label_of[current['target']],
                                      _unquote(current['relation_type']), _unquote(current.get('description', '""')),
                                      float(current.get('relation_strength', 1.0)), chunks))
                kind = None
            elif kind:
                match = _ATTRIBUTE.match(line)
                if match:
                    key, value = match.group(1), match.group(2)
                    if key == 'source_chunk_id':
                        value = _unquote(value)
                        if value != _LIST_START:
                            chunks.append(value)
                    else:
                        current[key] = value
    return Graph(nodes, edges)


def _chunk_lines(chunks: list) -> list:
    values = ([_LIST_START] if len(chunks) == 1 else []) + list(chunks)  # networkx marks a list of one item
    return [f'    source_chunk_id {_quote(value)}' for value in values]


def write_gml(graph: Graph, path) -> None:
    """The same file format, so the stand loads it as it loads its own."""
    number = {label: index for index, label in enumerate(graph.nodes)}
    with open(path, 'w', encoding='utf-8') as out:
        out.write('graph [\n  directed 1\n  multigraph 1\n')
        for label, node in graph.nodes.items():
            rows = [f'    id {number[label]}', f'    label {_quote(label)}', f'    entity_name {_quote(node.name)}',
                    f'    entity_type {_quote(node.type)}', f'    description {_quote(node.description)}']
            rows += _chunk_lines(node.chunks)
            rows += [f'    {key} {value}' for key, value in node.extra.items()]
            out.write('  node [\n' + '\n'.join(rows) + '\n  ]\n')
        for edge in graph.edges:
            rows = [f'    source {number[edge.source]}', f'    target {number[edge.target]}', f'    key {_quote(edge.id)}',
                    f'    subject_name {_quote(graph.nodes[edge.source].name)}',
                    f'    object_name {_quote(graph.nodes[edge.target].name)}',
                    f'    relation_type {_quote(edge.type)}', f'    description {_quote(edge.description)}',
                    f'    relation_strength {float(edge.strength)!r}']
            rows += _chunk_lines(edge.chunks)
            out.write('  edge [\n' + '\n'.join(rows) + '\n  ]\n')
        out.write(']\n')


# ---------------------------------------------------------------------------------------------------------- merges

def name_key(name: str) -> str:
    """The key the build unifies names by (graph_cleanup._normalize)."""
    return ' '.join(re.sub(r'[^\w\s]|_', ' ', str(name).casefold().replace('ё', 'е')).split())


def _statement(description: str) -> str:
    return ' '.join(str(description).casefold().split()).rstrip(' .;')


def _join_descriptions(first: str, second: str, separator: str) -> str:
    """The second text after the first, without the sentences the first already has."""
    have = {_statement(part) for part in re.split(r'(?<=[.!?])\s+|\n+', first)}
    new = [part.strip() for part in re.split(r'(?<=[.!?])\s+|\n+', second)
           if part.strip() and _statement(part) not in have]
    return (first.rstrip() + separator + ' '.join(new)) if new else first


def _union(first: list, second: list) -> list:
    seen = set(first)
    return list(first) + [item for item in second if not (item in seen or seen.add(item))]


def merge_nodes(graph: Graph, merges: list, step: str) -> tuple[Graph, list]:
    """merges: (duplicate id, target id, reason, evidence chunks). The target keeps its name and takes the sources and
    the new sentences of the duplicate; relations follow, two relations that become one (same ends and type) join
    their sources and texts, a relation between the two merged nodes goes."""
    parent = {}

    def root(node_id):
        while node_id in parent:
            node_id = parent[node_id]
        return node_id

    changes, applied = [], []
    for duplicate, target, reason, evidence in merges:
        duplicate, target = root(duplicate), root(target)
        if duplicate == target or duplicate not in graph.nodes or target not in graph.nodes:
            continue
        parent[duplicate] = target
        applied.append((duplicate, target, reason, evidence))
    if not applied:
        return graph, changes
    nodes = dict(graph.nodes)
    for duplicate, _, reason, evidence in applied:
        target = root(duplicate)
        gone, kept = nodes.pop(duplicate), nodes[target]
        nodes[target] = dataclasses.replace(kept, chunks=_union(kept.chunks, gone.chunks),
                                            description=_join_descriptions(kept.description, gone.description, ' '))
        changes.append({'step': step, 'op': 'merge', 'reason': reason, 'from': duplicate, 'from_name': gone.name,
                        'from_type': gone.type, 'to': target, 'to_name': kept.name, 'to_type': kept.type,
                        'from_chunks': len(gone.chunks), 'description_changed': nodes[target].description != kept.description,
                        'evidence_chunks': list(evidence)[:5]})
    edges, edge_changes = _follow(graph.edges, {duplicate: root(duplicate) for duplicate in parent}, step)
    return Graph(nodes, edges), changes + edge_changes


def _follow(edges: list, moved: dict, step: str) -> tuple[list, list]:
    by_id, order, changes = {}, [], []
    for edge in edges:
        source, target = moved.get(edge.source, edge.source), moved.get(edge.target, edge.target)
        if (source, target) != (edge.source, edge.target):
            if source == target:
                changes.append({'step': step, 'op': 'edge_drop', 'reason': 'both ends became one node', 'edge': edge.id,
                                'type': edge.type, 'evidence_chunks': edge.chunks[:5]})
                continue
            new_id = relation_id(source, target, edge.type)
            changes.append({'step': step, 'op': 'edge_rekey', 'reason': 'an end was merged or renamed', 'edge': edge.id,
                            'to': new_id, 'evidence_chunks': []})
            edge = dataclasses.replace(edge, id=new_id, source=source, target=target)
        if edge.id in by_id:
            kept = by_id[edge.id]
            by_id[edge.id] = dataclasses.replace(
                kept, chunks=_union(kept.chunks, edge.chunks), strength=max(kept.strength, edge.strength),
                description=_join_descriptions(kept.description, edge.description, '\n'))
            changes.append({'step': step, 'op': 'edge_merge', 'reason': 'same ends and type after a merge',
                            'edge': edge.id, 'description_changed': by_id[edge.id].description != kept.description,
                            'evidence_chunks': edge.chunks[:5]})
        else:
            by_id[edge.id] = edge
            order.append(edge.id)
    return [by_id[edge_id] for edge_id in order], changes


def rename_nodes(graph: Graph, renames: dict, step: str, reason: str) -> tuple[Graph, list]:
    """renames: id -> (new name, evidence chunks). A renamed node takes the id of its new name; when a node of that
    name key and type exists (or several renamed nodes share one), they merge into the one with more sources."""
    changes, nodes, new_id_of = [], {}, {}
    for node_id, node in graph.nodes.items():
        if node_id in renames and renames[node_id][0] != node.name:
            name, evidence = renames[node_id]
            changes.append({'step': step, 'op': 'rename', 'reason': reason, 'node': node_id, 'type': node.type,
                            'from_name': node.name, 'to_name': name, 'evidence_chunks': list(evidence)[:5]})
            new_id_of[node_id] = entity_id(name, node.type)
            node = dataclasses.replace(node, name=name)
        nodes[node_id] = node
    if not changes:
        return graph, changes
    groups = defaultdict(list)
    for node_id, node in nodes.items():
        groups[(name_key(node.name), node.type)].append(node_id)
    final, merges = {}, []
    renamed_keys = {(name_key(nodes[node_id].name), nodes[node_id].type) for node_id in new_id_of}
    for key, members in groups.items():
        if key in renamed_keys and len(members) > 1:
            # an existing spelling wins over a restored one, then the node with more sources
            target = max(members, key=lambda m: (m not in new_id_of, len(nodes[m].chunks), m))
            for member in members:
                if member != target:
                    merges.append((member, target, 'same name key and type after: ' + reason, renames.get(member, ('', []))[1]))
    absorbed = {duplicate for duplicate, *_ in merges}
    moved = {node_id: (new_id_of.get(node_id, node_id) if node_id not in absorbed else node_id) for node_id in nodes}
    taken = Counter(moved.values())
    for node_id, node in nodes.items():
        if taken[moved[node_id]] > 1 and moved[node_id] != node_id:
            moved[node_id] = node_id  # never two nodes under one id
        final[moved[node_id]] = dataclasses.replace(node, id=moved[node_id])
    for change in changes:
        change['to'] = moved[change['node']]  # the id of the new name; a node that merges away keeps its id till then
    edges, edge_changes = _follow(graph.edges, {old: new for old, new in moved.items() if old != new}, step)
    graph = Graph(final, edges)
    merges = [(moved[duplicate], moved[target], reason_, evidence) for duplicate, target, reason_, evidence in merges]
    graph, merge_changes = merge_nodes(graph, merges, step)
    return graph, changes + edge_changes + merge_changes


# ----------------------------------------------------------------------------------------- source form of a name

def expansions(abbreviations: dict, min_length: int = 4) -> list:
    """(expansion, abbreviation) of the build's abbreviations.json, the longest expansion first."""
    pairs = set()
    for key, record in abbreviations.items():
        if isinstance(record, dict):
            key, record = record.get('abbreviation', key), record.get('explanation', [])
        for expansion in ([record] if isinstance(record, str) else record):
            if len(expansion) >= min_length and expansion.casefold() != key.casefold():
                pairs.add((expansion, key))
    return sorted(pairs, key=lambda pair: (-len(pair[0]), pair))


@functools.lru_cache(maxsize=None)
def _token(abbreviation: str) -> 're.Pattern':
    return re.compile(r'(?<![^\W_])' + re.escape(abbreviation) + r'(?![^\W_])')


def source_form(name: str, texts: dict, pairs: list, tokens: dict | None = None):
    """The name with every expansion that no source chunk states replaced by its abbreviation, when a source chunk has
    that abbreviation. texts: chunk id -> text as crawled (before the expansion). Returns (name, evidence chunks,
    replaced (abbreviation, expansion) list) or None when the name stays."""
    folded_texts = None
    current, evidence, replaced = name, [], []
    for expansion, abbreviation in pairs:
        position = current.casefold().find(expansion.casefold())
        if position < 0:
            continue
        if folded_texts is None:
            folded_texts = {chunk: text.casefold() for chunk, text in texts.items()}
        if any(expansion.casefold() in text for text in folded_texts.values()):
            continue  # a source states the expansion itself
        pattern = (tokens or {}).get(abbreviation) or _token(abbreviation)
        stated = [chunk for chunk, text in texts.items() if pattern.search(text)]
        if not stated:
            continue  # the expansion is the model's wording, not a rewritten abbreviation
        while position >= 0:
            current = current[:position] + abbreviation + current[position + len(expansion):]
            position = current.casefold().find(expansion.casefold())
        evidence = _union(evidence, stated[:2])
        replaced.append((abbreviation, expansion))
    if current == name:
        return None
    current = ' '.join(current.split())
    exact = [chunk for chunk, text in (folded_texts or {}).items() if current.casefold() in ' '.join(text.split())]
    return current, _union(exact[:2], evidence), replaced


def source_form_renames(graph: Graph, chunk_text, pairs: list) -> tuple[dict, dict]:
    """id -> (name in the form of the source, evidence chunks); and id -> the (abbreviation, expansion) replaced."""
    tokens = {abbreviation: _token(abbreviation) for _, abbreviation in pairs}
    index = defaultdict(list)  # the first word of an expansion -> its pairs: few names hold an expansion
    for pair in pairs:
        index[pair[0].casefold().split()[0]].append(pair)
    renames, replaced = {}, {}
    for node_id, node in graph.nodes.items():
        folded = node.name.casefold()
        words = set(re.findall(r'[^\W_]+', folded))
        candidates = [pair for word in words if word in index for pair in index[word] if pair[0].casefold() in folded]
        if not candidates:
            continue
        candidates.sort(key=lambda pair: (-len(pair[0]), pair))
        texts = {chunk: chunk_text(chunk) or '' for chunk in node.chunks}
        result = source_form(node.name, texts, candidates, tokens)
        if result:
            renames[node_id] = (result[0], result[1])
            replaced[node_id] = result[2]
    return renames, replaced


def unstated_expansions(graph: Graph, chunk_text, pairs: list) -> list:
    """The check of step 1: nodes whose name holds an expansion none of their sources states while one has the
    abbreviation. Empty after the renames."""
    return sorted(source_form_renames(graph, chunk_text, pairs)[0])


def stated_long_forms(texts, abbreviations: list, min_documents: int = 3) -> dict:
    """abbreviation -> {long form: documents} as the corpus itself spells it out: "Институт физики полупроводников
    (ИФП)". texts: (document, text) pairs. An abbreviation with several long forms is ambiguous whatever the
    dictionary says."""
    wanted = set(abbreviations)
    pattern = re.compile(r'((?:[^\W\d_][\w\-]*\.?[  ]+){1,8}?)\(([^\W_][\w\-]{1,11})\)')
    found = defaultdict(lambda: defaultdict(set))
    examples = defaultdict(Counter)
    for document, text in texts:
        for match in pattern.finditer(text):
            abbreviation = match.group(2)
            if abbreviation not in wanted:
                continue
            words = match.group(1).split()
            letters = [c for c in abbreviation if c.isalpha()]
            # the long form: the last words before the bracket whose initials spell the abbreviation
            for size in range(len(letters), len(words) + 1):
                tail = words[-size:]
                initials = [w[0].casefold() for w in tail if len(w) > 2 or w[0].isupper()]
                if initials == [c.casefold() for c in letters] and tail[0][0].isupper():
                    form = ' '.join(w[:max(4, len(w) - 2)].casefold() for w in tail)
                    found[abbreviation][form].add(document)
                    examples[(abbreviation, form)][' '.join(tail)] += 1
                    break
    return {abbreviation: {form: {'documents': len(documents), 'example': examples[(abbreviation, form)].most_common(1)[0][0]}
                           for form, documents in forms.items() if len(documents) >= min_documents}
            for abbreviation, forms in found.items()}


def id_map(changes: list) -> dict:
    """old node id -> the id the node has after the changes (renames and merges), for the nodes that moved."""
    moved = {}
    for change in changes:
        if change['op'] == 'rename' and change['to'] != change['node']:
            moved[change['node']] = change['to']
        elif change['op'] == 'merge':
            moved[change['from']] = change['to']

    def final(node_id):
        seen = set()
        while node_id in moved and node_id not in seen:
            seen.add(node_id)
            node_id = moved[node_id]
        return node_id
    return {node_id: final(node_id) for node_id in moved}


# ---------------------------------------------------------------------------------------------------------- people

_PATRONYMIC = re.compile(r'(ович|евич|ьич|овна|евна|ична|инична)$')
_LETTER = r'A-Za-zА-Яа-яЁё'


def _fold(word: str) -> str:
    return word.casefold().replace('ё', 'е').rstrip('.')


def readings(name: str) -> list:
    """The ways to read a person name: (surname, given name, patronymic); the given name and the patronymic are '',
    an initial or a word. A name of two words reads both ways; a patronymic fixes the order."""
    text = re.sub(r'\([^)]*\)', ' ', name)
    comma = bool(re.match(r'^\s*[^\s,]+,', text))
    text = re.sub(rf'([{_LETTER}])\.', r'\1. ', text)
    words = [w for w in re.split(r'[\s,]+', text) if w and re.search(rf'[{_LETTER}]', w)]
    if not 2 <= len(words) <= 4:
        return []
    folded = [(_fold(w), len(w.rstrip('.')) == 1) for w in words]
    full = [i for i, (_, initial) in enumerate(folded) if not initial]
    initials = [w for w, initial in folded if initial]
    if len(full) == 1:
        if len(initials) > 2:
            return []
        return [(folded[full[0]][0], initials[0] if initials else '', initials[1] if len(initials) > 1 else '')]
    patronymic = [i for i in full if _PATRONYMIC.search(folded[i][0])]
    if len(full) == 3 and not initials:
        if len(patronymic) == 1 and patronymic[0] in (1, 2):
            p = patronymic[0]
            return [(folded[0][0] if p == 2 else folded[2][0], folded[p - 1][0], folded[p][0])]
        if comma:
            return [(folded[0][0], folded[1][0], folded[2][0])]
        return [(folded[2][0], folded[0][0], folded[1][0]), (folded[0][0], folded[1][0], folded[2][0])]
    if len(full) == 2:
        if len(patronymic) == 1 and not initials:
            return []  # a given name and a patronymic, no surname
        if len(initials) == 1:
            at = [i for i, (_, initial) in enumerate(folded) if initial][0]
            if at == 1:
                return [(folded[2][0], folded[0][0], folded[1][0])]
            if at == 2:
                return [(folded[0][0], folded[1][0], folded[2][0])]
            return []
        if initials:
            return []
        first, second = full
        if comma:
            return [(folded[first][0], folded[second][0], '')]
        return [(folded[second][0], folded[first][0], ''), (folded[first][0], folded[second][0], '')]
    return []


def _part_fits(one: str, other: str) -> bool:
    if not one or not other:
        return True
    if len(one) == 1 or len(other) == 1:
        return one[0] == other[0]
    return one == other


def fits(one: tuple, other: tuple) -> bool:
    return one[0] == other[0] and _part_fits(one[1], other[1]) and _part_fits(one[2], other[2])


def detail(reading: tuple) -> int:
    return sum(2 if len(part) > 1 else 1 if part else 0 for part in reading[1:])


def full_name(name: str):
    """(surname, given name, patronymic) of a full name whose patronymic fixes the order, else None."""
    found = readings(name)
    if len(found) == 1 and len(found[0][1]) > 1 and _PATRONYMIC.search(found[0][2]):
        return found[0]
    return None


def full_name_merges(graph: Graph) -> list:
    """Full names of the same three words in another order are one person node (the one with more sources)."""
    groups = defaultdict(list)
    for node_id, node in graph.nodes.items():
        if node.type == 'PERSON':
            key = full_name(node.name)
            if key:
                groups[key].append(node_id)
    merges = []
    for members in groups.values():
        if len(members) > 1:
            target = max(members, key=lambda m: (len(graph.nodes[m].chunks), m))
            shared = set(graph.nodes[target].chunks)
            merges += [(member, target, 'the same full name in another word order',
                        [c for c in graph.nodes[member].chunks if c in shared] or graph.nodes[member].chunks[:1])
                       for member in members if member != target]
    return merges


def short_name_merges(graph: Graph, chunk_document, profiles_of=None) -> list:
    """A shorter form of a person's name joins a fuller one only through a document (or a profile) that holds both:
    among the person nodes of that document the fuller forms it fits must fit each other, and the most detailed one
    with the most sources is the target. Names a document never brings together stay apart."""
    people = {node_id: readings(node.name) for node_id, node in graph.nodes.items() if node.type == 'PERSON'}
    people = {node_id: found for node_id, found in people.items() if found}
    by_place = defaultdict(lambda: defaultdict(list))  # document or profile -> surname -> nodes
    places = defaultdict(set)
    for node_id, found in people.items():
        node_places = {('doc', chunk_document(chunk)) for chunk in graph.nodes[node_id].chunks} - {('doc', None)}
        node_places |= {('profile', key) for key in (profiles_of or {}).get(node_id, ())}
        places[node_id] = node_places
        for place in node_places:
            for surname in {reading[0] for reading in found}:
                by_place[place][surname].append(node_id)

    def best(one, other):
        pairs = [(a, b) for a in people[one] for b in people[other] if fits(a, b)]
        return max(pairs, key=lambda pair: detail(pair[0]) + detail(pair[1])) if pairs else None

    merges = []
    for node_id, found in people.items():
        if (profiles_of or {}).get(node_id):
            continue  # a node with a profile of its own is that person already
        fuller, equal = {}, {}
        for place in places[node_id]:
            for surname in {reading[0] for reading in found}:
                for other in by_place[place][surname]:
                    if other == node_id or other in fuller or other in equal:
                        continue
                    pair = best(node_id, other)
                    if not pair or pair[0][0] != surname or detail(pair[1]) == 0:
                        continue
                    if detail(pair[1]) > detail(pair[0]):
                        fuller[other] = (pair, place)
                    elif pair[0] == pair[1]:  # the same words or initials, written in another order
                        equal[other] = (pair, place)
        candidates = {}
        if fuller and all(best(a, b) for i, a in enumerate(fuller) for b in list(fuller)[i + 1:]):
            candidates = fuller  # else the document has two different people this name fits
        elif equal:
            size = (len(graph.nodes[node_id].chunks), node_id)
            candidates = {other: found_ for other, found_ in equal.items() if (len(graph.nodes[other].chunks), other) > size}
        if not candidates:
            continue
        target = max(candidates, key=lambda o: (detail(candidates[o][0][1]), len(graph.nodes[o].chunks), o))
        place = candidates[target][1]
        shared = [c for c in graph.nodes[node_id].chunks if ('doc', chunk_document(c)) == place]
        merges.append((node_id, target, 'a shorter form of the name; both are in one ' +
                       ('document' if place[0] == 'doc' else 'profile'), shared or graph.nodes[node_id].chunks[:1]))
    return merges


_UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
_CASE = r'(?:вич|вича|вичу|вичем|виче|вна|вны|вне|вну|вной|ична|ичны|ичне|ичну|ичной|ьич|ьича|ьичу|ьичем)'
_TITLE_FULL = re.compile(rf'(?<![{_LETTER}])([А-ЯЁ][а-яё\-]+)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+{_CASE})(?![{_LETTER}])|'
                         rf'(?<![{_LETTER}])([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+{_CASE})\s+([А-ЯЁ][а-яё\-]+)(?![{_LETTER}])')


def title_person(title: str):
    """(surname, given name, patronymic) as the title writes them (any case form) when it names exactly one person in
    full, else None: such a page is a profile of that person."""
    found = set()
    for match in _TITLE_FULL.finditer(title or ''):
        groups = match.groups()
        surname, given, patronymic = groups[:3] if groups[0] else (groups[5], groups[3], groups[4])
        found.add((_fold(surname), _fold(given), _fold(patronymic)))
    return next(iter(found)) if len(found) == 1 else None


def profile_key(url: str, title: str):
    """The stable key of a person profile page: the UUID of its URL (Pure, the timetable) with the host, else the URL
    without its query. None when the title names no single person."""
    if not url or title_person(title) is None:
        return None
    host = re.sub(r'^https?://([^/]+).*$', r'\1', url)
    uuid = _UUID.search(url)
    return f'{host}:{uuid.group(0)}' if uuid else re.sub(r'[?].*$', '', url)


def consistent_profiles(document_profiles: dict) -> dict:
    """document -> (key, person) with one person to a key: when the pages of one key are titled with different people
    (a generated title may name a co-author), the key keeps the pages of its most frequent person."""
    people = defaultdict(Counter)
    for key, person in document_profiles.values():
        people[key][(person[0][:4], person[1][:1], person[2][:1])] += 1
    main = {key: found.most_common(1)[0][0] for key, found in people.items()}
    return {document: (key, person) for document, (key, person) in document_profiles.items()
            if (person[0][:4], person[1][:1], person[2][:1]) == main[key]}


def _same_surname(surname: str, titled: str) -> bool:
    """A surname against the surname of a title in any case form: they differ in the ending only."""
    common = 0
    while common < min(len(surname), len(titled)) and surname[common] == titled[common]:
        common += 1
    if len(surname) < 4:  # "Ян" and "Яна"
        return common == len(surname) and len(titled) - common <= 2
    return common >= 4 and common >= len(surname) - 2 and common >= len(titled) - 3


def _same_word(word: str, titled: str, ending: int) -> bool:
    """A given name or patronymic against its case form in a title: after their common start at most `ending` letters
    are left in each ("Павел" and "Павла", not "Николай" and "Никиты", not "Алексей" and "Александра")."""
    common = 0
    while common < min(len(word), len(titled)) and word[common] == titled[common]:
        common += 1
    return common >= 3 and len(word) - common <= ending and len(titled) - common <= ending


def owns(reading: tuple, person: tuple) -> bool:
    """A node's reading against the person of a profile title, up to the case form of the title."""
    surname, given, patronymic = person
    if not _same_surname(reading[0], surname):
        return False
    for part, titled, ending in ((reading[1], given, 3), (reading[2], patronymic, 5)):
        if part and (part[0] != titled[0] or (len(part) > 1 and not _same_word(part, titled, ending))):
            return False
    if len(reading[2]) > 1 and ('вн' in reading[2][-4:]) != ('вн' in patronymic[-5:]):
        return False  # a patronymic of a man and of a woman
    return bool(reading[1])


def _distance(one: str, other: str, limit: int = 2) -> int:
    if abs(len(one) - len(other)) > limit:
        return limit + 1
    previous = list(range(len(other) + 1))
    for i, a in enumerate(one, 1):
        row = [i]
        for j, b in enumerate(other, 1):
            row.append(min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = row
    return previous[-1]


def _misspelled(surname: str, titled: str) -> bool:
    """One or two letters away from the titled surname in one of its case forms, and not that surname itself."""
    if len(surname) < 5 or surname[0] != titled[0] or _same_surname(surname, titled):
        return False
    return min(_distance(surname, titled[:len(titled) - cut]) for cut in range(4)) <= 2


def profile_owners(graph: Graph, chunk_document, document_profile) -> tuple[dict, dict]:
    """profile key -> {node: its chunks of the profile's pages} for the person nodes with a full name (given name and
    patronymic) the profile's title names (a shorter name may be another person: it joins through short_name_merges), and
    profile key -> {node: chunks} for the nodes there whose surname is a misspelling of the titled one (same given name
    and patronymic in full, the surname at most two letters away). document_profile: document -> (key, person)."""
    owners, misspelled = defaultdict(dict), defaultdict(dict)
    for node_id, node in graph.nodes.items():
        if node.type != 'PERSON':
            continue
        found = None
        for chunk in node.chunks:
            profile = document_profile(chunk_document(chunk))
            if not profile:
                continue
            if found is None:
                found = readings(node.name)
            key, person = profile
            if any(detail(reading) == 4 and owns(reading, person) for reading in found):  # a full name only
                owners[key].setdefault(node_id, []).append(chunk)
            elif any(len(r[1]) > 1 and len(r[2]) > 1 and _same_word(r[1], person[1], 3) and _same_word(r[2], person[2], 5)
                     and _misspelled(r[0], person[0]) for r in found):
                misspelled[key].setdefault(node_id, []).append(chunk)
    return owners, misspelled


def _surname_count(graph: Graph, node_id: str, among: list, chunk_text, limit: int = 300) -> int:
    """How often the surname of the node stands, as written, in the source chunks of the nodes it is compared with."""
    surnames = {reading[0] for reading in readings(graph.nodes[node_id].name)}
    patterns = [re.compile(r'(?<![^\W\d_])' + re.escape(surname).replace('е', '[её]') + r'(?![^\W\d_])', re.I) for surname in surnames]
    chunks = [chunk for other in among for chunk in graph.nodes[other].chunks][:limit]
    return max((sum(len(pattern.findall(chunk_text(chunk) or '')) for chunk in chunks) for pattern in patterns), default=0)


def profile_merges(graph: Graph, owners: dict, misspelled: dict, chunk_text=None) -> list:
    """One profile, one node: the full-name nodes a profile's title names (case forms and spellings of one name) merge
    into the one with the most sources; a node of the profile whose surname is a misspelling of the titled one joins."""
    merges = []
    for key in sorted(owners):
        nodes = owners[key]
        among = sorted(nodes)
        # the spelling the sources themselves use most stays ("Медных", not its mistaken case form "Медный")
        target = max(nodes, key=lambda n: (_surname_count(graph, n, among, chunk_text) if chunk_text and len(nodes) > 1 else 0,
                                           len(graph.nodes[n].chunks), len(nodes[n]), n))
        for node_id, chunks in sorted(nodes.items()):
            if node_id != target:
                merges.append((node_id, target, f'one profile ({key}) names both nodes', chunks))
        for node_id, chunks in sorted(misspelled.get(key, {}).items()):
            # the spelling the sources use more often stays, then the one with more sources
            pair = sorted((node_id, target), key=lambda n: (
                _surname_count(graph, n, [node_id, target], chunk_text) if chunk_text else 0, len(graph.nodes[n].chunks), n == target))
            merges.append((pair[0], pair[1], f'two spellings of one surname in the pages of the profile ({key})', chunks))
    return merges


# ------------------------------------------------------------------------------------------------------- relations

def fold_relations(graph: Graph, symmetric: frozenset, step: str = 'fold') -> tuple[Graph, list]:
    """One statement, one relation, on the aggregated graph: the two directions of a symmetric type become one relation
    (ends ordered by id) with the sources and texts of both; a RELATED_TO whose text a typed relation between the same
    two nodes states goes, its sources joining the typed one."""
    changes, moved = [], []
    for edge in graph.edges:
        if edge.type in symmetric and edge.source > edge.target:
            new_id = relation_id(edge.target, edge.source, edge.type)
            changes.append({'step': step, 'op': 'edge_reorder', 'reason': 'a symmetric type keeps one order of its ends',
                            'edge': edge.id, 'to': new_id, 'type': edge.type, 'evidence_chunks': edge.chunks[:5]})
            edge = dataclasses.replace(edge, id=new_id, source=edge.target, target=edge.source)
        moved.append(edge)
    by_id, order = {}, []
    for edge in moved:
        if edge.id in by_id:
            kept = by_id[edge.id]
            by_id[edge.id] = dataclasses.replace(
                kept, chunks=_union(kept.chunks, edge.chunks), strength=max(kept.strength, edge.strength),
                description=_join_descriptions(kept.description, edge.description, '\n'))
            changes.append({'step': step, 'op': 'edge_merge', 'reason': 'the two directions of a symmetric relation',
                            'edge': edge.id, 'type': edge.type,
                            'description_changed': by_id[edge.id].description != kept.description,
                            'evidence_chunks': edge.chunks[:5]})
        else:
            by_id[edge.id] = edge
            order.append(edge.id)
    typed = {}
    for edge_id in order:
        edge = by_id[edge_id]
        if edge.type != 'RELATED_TO' and not edge.type.startswith('UNFIT:'):
            for part in edge.description.split('\n'):
                typed.setdefault((frozenset((edge.source, edge.target)), _statement(part)), edge_id)
    result = []
    for edge_id in order:
        edge = by_id[edge_id]
        if edge.type == 'RELATED_TO':
            pair = frozenset((edge.source, edge.target))
            parts = [part for part in edge.description.split('\n') if part.strip()]
            repeated = [typed.get((pair, _statement(part))) for part in parts]
            if parts and all(repeated):
                keeper = by_id[repeated[0]]
                by_id[repeated[0]] = dataclasses.replace(keeper, chunks=_union(keeper.chunks, edge.chunks))
                changes.append({'step': step, 'op': 'edge_drop', 'reason': 'RELATED_TO states what the typed relation '
                                'of the same two nodes states', 'edge': edge.id, 'type': edge.type, 'kept': repeated[0],
                                'evidence_chunks': edge.chunks[:5]})
                continue
        result.append(edge_id)
    return Graph(graph.nodes, [by_id[edge_id] for edge_id in result]), changes


def _plain(text: str) -> str:
    """A text up to case, markup and punctuation, to find a name in a line that writes it with tags or other dashes."""
    return name_key(re.sub(r'<[^>]{0,20}>', '', text))


def line_key(line: str) -> str:
    """A line up to case, spacing and digits: the lines a site repeats differ by an id or a year at most."""
    return re.sub(r'\d+', '#', ' '.join(line.casefold().split()))


def template_relations(graph: Graph, hubs: set, chunk_text, forms: dict, other_forms, min_others: int = 50) -> list:
    """(edge index, hub id, evidence chunks) for the relations of a hub that only a page template supports.

    A line of a source chunk that names the hub while neither it nor the lines next to it name the other end of the
    relation says nothing about that other end by itself (a title with its authors on the next line is a statement of
    its own); when the same line (up to digits) is all that ties the hub to at least min_others different nodes
    ("Вход в Pure", "Язык оригинала: английский", the header of a list), it is a line of a template. A relation is a
    template relation when, in every one of its source chunks, every line that names the hub is such a line. A
    relation with a line of its own (the hub and the other end together: a participant of a conference), a relation
    whose sources do not name the hub in a way we can find, and the relations of a hub with few other ends to a line
    (a person and the header of their own page) stay.
    forms: hub id -> spellings of the hub; other_forms: node id -> spellings of a node to look for in a line."""
    spellings = {hub: [form.casefold() for form in forms[hub] if len(form) >= 2] for hub in hubs}
    others_of = defaultdict(set)  # (hub, line key) -> other ends
    hub_lines = {}
    lines_of = {}  # (edge index, hub) -> per chunk, the keys of the lines that name the hub (None: one names the other end too)
    for index, edge in enumerate(graph.edges):
        for hub, other in ((edge.source, edge.target), (edge.target, edge.source)):
            if hub not in hubs:
                continue
            named = [key[:30] for key in (_plain(form) for form in other_forms(other)) if len(key) >= 3]
            per_chunk = []
            for chunk in edge.chunks:
                if (chunk, hub) not in hub_lines:  # one chunk gives a hub many relations
                    folded_lines = [line.casefold() for line in (chunk_text(chunk) or '').split('\n') if line.strip()]
                    hub_lines[(chunk, hub)] = [(_plain(' \n '.join(folded_lines[max(0, at - 1):at + 2])), line_key(line))
                                               for at, line in enumerate(folded_lines)
                                               if any(form in line for form in spellings[hub])]
                keys = []
                for nearby, key in hub_lines[(chunk, hub)]:
                    if any(form in nearby for form in named):
                        keys = None  # the hub and the other end stand together: a statement of its own
                        break
                    keys.append(key)
                per_chunk.append(keys)
                if not keys:
                    break
            if len(per_chunk) == len(edge.chunks) and all(per_chunk):
                lines_of[(index, hub)] = per_chunk
                for keys in per_chunk:
                    for key in keys:
                        others_of[(hub, key)].add(other)
            break
    found = []
    for (index, hub), per_chunk in lines_of.items():
        if all(len(others_of[(hub, key)]) >= min_others for keys in per_chunk for key in keys):
            found.append((index, hub, graph.edges[index].chunks[:5]))
    return sorted(found)


def statement_types(relations: dict, parents: dict) -> frozenset:
    """The entity types whose relations state something about the other end even from a header or a list: a person,
    what the ontology lets only a person relate to (the place of work or study, a position, a degree, a publication, a
    discipline taught: the ranges of the relation types whose only domain is PERSON), a law (the only domain of a
    relation type with an open range), and their subtypes. A hub of another type (a language, a software platform, a
    field of science a classifier assigns) is where a page template can be the whole support of a relation.
    relations: type -> (domain or None, range or None); parents: entity type -> parent or None."""
    base = set()
    for domain, range_ in relations.values():
        if domain and len(domain) == 1 and (tuple(domain) == ('PERSON',) or not range_):
            base.add(domain[0])
            if tuple(domain) == ('PERSON',):
                base.update(range_ or ())
    result = set()
    for kind in set(parents) | base:
        seen, current = set(), kind
        while current and current not in seen:
            if current in base:
                result.add(kind)
                break
            seen.add(current)
            current = parents.get(current)
    return frozenset(result)


def name_forms(name: str, pairs: list) -> list:
    """A name and its spellings with an abbreviation expanded or an expansion abbreviated (the build's dictionary)."""
    forms = [name]
    for expansion, abbreviation in pairs:
        if expansion.casefold() in name.casefold():
            forms.append(re.sub(re.escape(expansion), abbreviation, name, flags=re.I))
        elif abbreviation in name and _token(abbreviation).search(name):
            forms.append(_token(abbreviation).sub(lambda _: expansion, name))
    return forms


def relations_to_metadata(graph: Graph, found: list, step: str = 'template') -> tuple[Graph, list, dict]:
    """The template relations leave the graph; what they said stays as metadata of the other end:
    node id -> [{hub, hub_name, relation_type, direction, description, chunks}]."""
    gone = {index for index, _, _ in found}
    metadata, changes = defaultdict(list), []
    for index, hub, evidence in found:
        edge = graph.edges[index]
        other = edge.target if edge.source == hub else edge.source
        metadata[other].append({'hub': hub, 'hub_name': graph.nodes[hub].name, 'relation_type': edge.type,
                                'direction': 'to_hub' if edge.target == hub else 'from_hub',
                                'description': edge.description, 'chunks': list(edge.chunks)})
        changes.append({'step': step, 'op': 'edge_to_metadata', 'reason': 'every source line that names the hub is a line '
                        'many documents repeat (a page template)', 'edge': edge.id, 'type': edge.type, 'hub': hub,
                        'hub_name': graph.nodes[hub].name, 'node': other, 'evidence_chunks': list(evidence)})
    return Graph(graph.nodes, [edge for index, edge in enumerate(graph.edges) if index not in gone]), changes, dict(metadata)


# ---------------------------------------------------------------------------------------------------------- checks

def cycle_counts(graph: Graph, kinds=('PART_OF', 'LOCATED_IN', 'SUBORDINATE_OF', 'SUPERVISES')) -> dict:
    """Two-way pairs and relations inside cycles of the types that cannot hold both ways."""
    import networkx as nx
    result = {}
    for kind in kinds:
        pairs = {(edge.source, edge.target) for edge in graph.edges if edge.type == kind}
        directed = nx.DiGraph(list(pairs))
        component = {}
        for index, members in enumerate(nx.strongly_connected_components(directed)):
            if len(members) > 1:
                for member in members:
                    component[member] = index
        result[kind] = {'relations': len(pairs),
                        'two_way_pairs': sum(1 for a, b in pairs if a < b and (b, a) in pairs),
                        'relations_inside_cycles': sum(1 for a, b in pairs if a in component and component.get(b) == component[a])}
    return result


def embedded_text_changes(before: Graph, after: Graph) -> dict:
    """What the vector collections must take: the entities whose embedded text (name and description) is new or
    changed, the relations whose text (description) is new or changed, the relations that only changed their id, and
    the ids that are gone."""
    old_nodes = {node_id: (node.name, node.description) for node_id, node in before.nodes.items()}
    old_edges = {edge.id: edge.description for edge in before.edges}
    old_texts = Counter(old_edges.values())
    result = {'entities_to_embed': [], 'relations_to_embed': [], 'relations_rekeyed': [],
              'entities_gone': sorted(set(old_nodes) - set(after.nodes)), 'relations_gone': []}
    for node_id, node in after.nodes.items():
        if old_nodes.get(node_id) != (node.name, node.description):
            result['entities_to_embed'].append(node_id)
    new_ids = set()
    for edge in after.edges:
        new_ids.add(edge.id)
        if old_edges.get(edge.id) == edge.description:
            continue
        key = 'relations_rekeyed' if edge.id not in old_edges and old_texts.get(edge.description) else 'relations_to_embed'
        result[key].append(edge.id)
    result['relations_gone'] = sorted(set(old_edges) - new_ids)
    return result


def write_changes(changes: list, out) -> None:
    for change in changes:
        out.write(json.dumps(change, ensure_ascii=False) + '\n')
