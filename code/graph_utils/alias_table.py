"""Names of the graph's entities as a lookup table for the mentions of a question.

The vector search of entities gets the whole question, so a unit named in it ("ФИТ", "кафедра общей физики") competes
with every entity near the topic. The table maps a normalized name to the nodes that carry it, and a question is
scanned for the longest names it contains. Everything in the table comes from the graph:

* the name of every node (word order, case and inflection ignored: the stems of its words, sorted);
* an abbreviation of a node, when its letters are the initials of the node's name. The abbreviation is taken from an
  ``ALTERNATIVE_NAME``/``ABBREVIATION`` edge or from brackets in the node's description, and is kept only after
  this check: 38-75% of the alias edges of graph v6 join different things (audit of 18.11.2026).

A one-word name that many nodes share as a word of their names says little about which node is meant; such names are
left out unless they are checked abbreviations (``single_word_max_names``).

usage: python -m graph_utils.alias_table GRAPH.gml OUT.json
"""
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from py_rust_stemmers import SnowballStemmer

ALIAS_RELATIONS = ('ALTERNATIVE_NAME', 'ABBREVIATION')
MAX_MENTION_WORDS = 12
_WORD = re.compile(r'[0-9a-zа-я]+')
_BRACKETS = re.compile(r'\(([^()]{2,24})\)')
_STEMMER = SnowballStemmer('russian')
_STEMS: dict[str, str] = {}


def words(text):
    return _WORD.findall((text or '').lower().replace('ё', 'е'))


def stems(text):
    found = []
    for word in words(text):
        stem = _STEMS.get(word)
        if stem is None:
            stem = _STEMS[word] = _STEMMER.stem_word(word)
        found.append(stem)
    return found


def name_key(text):
    """The words of a name as sorted stems: «Кафедра общей физики» and «кафедру физики общей» give one key."""
    return ' '.join(sorted(set(stems(text))))


def is_abbreviation_of(short, full):
    """Whether the letters of ``short`` are the initials of the words of ``full`` (all its words, or the words
    longer than two letters; a hyphen separates words). Further words of ``full`` after the initials (the owner
    of a unit, as in «Факультет информационных технологий НГУ») are allowed for abbreviations of three letters up."""
    letters = ''.join(words(short))
    if len(letters) < 2 or ' ' in (short or '').strip() and len(words(short)) > 3:
        return False
    full_words = words(full)
    if len(full_words) < 2 or letters == ''.join(full_words):
        return False
    for chosen in (full_words, [word for word in full_words if len(word) > 2]):
        initials = ''.join(word[0] for word in chosen)
        if letters == initials or (len(letters) >= 3 and initials.startswith(letters)):
            return True
    return False


def build_alias_table(nodes, edges, single_word_max_names=50, max_nodes_per_name=5):
    """``nodes``: (id, name, type, description); ``edges``: (subject id, object id, relation type).
    Returns {'names': {key: [node ids, most connected first]}, 'meta': {...}}."""
    names, descriptions = {}, {}
    for node_id, name, _, description in nodes:
        names[node_id] = name
        descriptions[node_id] = description or ''
    degree = Counter()
    alias_pairs = []
    for subject, target, kind in edges:
        degree[subject] += 1
        degree[target] += 1
        if kind in ALIAS_RELATIONS and subject in names and target in names:
            alias_pairs.append((subject, target))
    by_key = defaultdict(set)
    word_names = Counter()  # in how many names a stem occurs
    for node_id, name in names.items():
        key = name_key(name)
        if key:
            by_key[key].add(node_id)
            word_names.update(key.split(' '))
    checked = defaultdict(set)  # abbreviation key -> nodes it names
    stats = Counter(nodes=len(names), alias_edges=len(alias_pairs))
    for first, second in alias_pairs:
        for short, full in ((first, second), (second, first)):
            if is_abbreviation_of(names[short], names[full]):
                checked[name_key(names[short])].add(full)
                stats['alias_edges_checked'] += 1
                break
    for node_id, description in descriptions.items():
        for found in _BRACKETS.findall(description + ' ' + names[node_id]):
            if is_abbreviation_of(found, names[node_id]):
                key = name_key(found)
                if key and node_id not in checked[key]:
                    checked[key].add(node_id)
                    stats['bracket_abbreviations'] += 1
    table = {}
    for key in set(by_key) | set(checked):
        if not key:
            continue
        targets = by_key.get(key, set()) | checked.get(key, set())
        if ' ' not in key and key not in checked and word_names[key] > single_word_max_names:
            stats['single_words_left_out'] += 1
            continue
        table[key] = sorted(targets, key=lambda node_id: (-degree[node_id], node_id))[:max_nodes_per_name]
    stats['names'] = len(table)
    stats['abbreviations'] = sum(1 for key in checked if key in table)
    return {'names': table, 'meta': {**stats, 'single_word_max_names': single_word_max_names,
                                     'max_nodes_per_name': max_nodes_per_name}}


class AliasTable:
    def __init__(self, names):
        self.names = names
        self.longest = max((key.count(' ') + 1 for key in names), default=0)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8'))['names'])

    def link(self, text):
        """The names found in the text, longest first and without overlaps: [(words of the mention, node ids)]."""
        tokens = stems(text)
        taken = [False] * len(tokens)
        found = []
        for size in range(min(self.longest, MAX_MENTION_WORDS, len(tokens)), 0, -1):
            for start in range(len(tokens) - size + 1):
                if any(taken[start:start + size]):
                    continue
                span = tokens[start:start + size]
                if len(set(span)) != size:
                    continue
                node_ids = self.names.get(' '.join(sorted(span)))
                if node_ids:
                    found.append((start, span, node_ids))
                    taken[start:start + size] = [True] * size
        return [(span, node_ids) for _, span, node_ids in sorted(found, key=lambda item: item[0])]

    def node_ids(self, text):
        return list(dict.fromkeys(node_id for _, node_ids in self.link(text) for node_id in node_ids))


def read_gml(path):
    """Nodes (label, name, type, description) and edges (subject label, object label, type) of a RAGU graph GML,
    streamed (networkx needs tens of minutes and gigabytes for graph v6)."""
    field = re.compile(r'^\s+(\w+) (.*)$')
    nodes, edges, labels = [], [], {}
    kind, current = None, {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if stripped in ('node [', 'edge ['):
                kind, current = stripped[:4], {}
            elif stripped == ']' and kind:
                if kind == 'node':
                    labels[current['id']] = current['label']
                    nodes.append((current['label'], current.get('entity_name', ''), current.get('entity_type', ''),
                                  current.get('description', '')))
                else:
                    edges.append((labels[current['source']], labels[current['target']],
                                  current.get('relation_type', '')))
                kind = None
            elif kind:
                match = field.match(line)
                if match and match.group(1) in ('id', 'label', 'entity_name', 'entity_type', 'description', 'source',
                                                'target', 'relation_type'):
                    current[match.group(1)] = html.unescape(match.group(2).strip('"'))
    return nodes, edges


if __name__ == '__main__':
    graph_nodes, graph_edges = read_gml(sys.argv[1])
    built = build_alias_table(graph_nodes, graph_edges)
    built['meta']['graph'] = str(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(built, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(built['meta'], ensure_ascii=False))
