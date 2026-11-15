"""
Graph cleanup after extraction (--graph_cleanup): values become descriptions, and variants of one entity one node.

Values. NEREL types values (NUMBER, DATE, TIME, AGE, MONEY, PERCENT, ORDINAL) as entities, so "11 стр.", "2015" or a
publication id become nodes, and a year links thousands of unrelated things (about a quarter of the entity mentions of
graph v2). A value is a property of what it describes: the description of the relation that states it ("Статья
опубликована в 2015 году") is added to the description of the other end, in the same chunk, and the value node and
its relations go. A relation between two values says nothing about an entity and goes as well.

Variants. Entities merge by name and type, so one name under two types stays two nodes ("Дрозофила" PRODUCT and
ORGANIZATION), and a person is split over the forms of the name ("Мулляджанов Р.И.", "Рустам Мулляджанов",
"Рустам Илхамович Мулляджанов"). One name under several types takes its most frequent type; people are left out,
since an institute, a prize or a street is named after a person. A person name of two or more words joins the full
name (three words) it fits, word for word or by initials, when exactly one full name fits; full names written in
another word order join their most frequent spelling. One-word names and names more than one full name fits stay.
Relations follow their endpoints; a relation that becomes a loop (an alias of the same person) goes.

Inputs (CleanupInputs, files of a graph build): chunks whose artifacts stay out of the graph (pages that repeat one
record in several citation formats; the chunks stay in the chunk index), entities a model labeled as values or junk
(a DOI typed PRODUCT, an e-mail typed LOCATION), folded like values after the name variants are merged, merges of
duplicate entities a model confirmed (name and type to name and type), and relations a model refuted (an
ALTERNATIVE_NAME between two different objects, such as «Новосибирск» and «Москва»), which keep their text with the type
UNFIT:<type> after the merges.

Ontology labels (graph v5, the NSU ontology): relations a model retyped from their own text (a type among those the
ontology allows between the types of the two ends, maybe the other way round, or RELATED_TO; the text stays), nodes a
model found not to be objects (a role or status, an identifier, a vague phrase, a garbled name), folded like values, and
the type each other node takes (the most specific type of the ontology that fits it). They name nodes and relations as
the graph they were made on did, so they apply after the merges, in this order.

An input finds its node by name and type up to punctuation and case, the key the name canonicalization unifies names
by: inputs made on graph v4, whose name trimming cut closing brackets ('Дексаметазон (DEX'), name the same nodes of a
build whose names are whole ('Дексаметазон (DEX)'), and a merge target keeps the spelling of this build.
"""
import dataclasses
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ragu.graph.types import Entity, Relation

VALUE_TYPES = frozenset({'NUMBER', 'DATE', 'TIME', 'AGE', 'MONEY', 'PERCENT', 'ORDINAL'})
NOT_OBJECT_KINDS = frozenset({'role', 'identifier', 'vague', 'garbled'})
_WORD = re.compile(r'[^\W\d_]+')


@dataclasses.dataclass(frozen=True)
class CleanupInputs:
    excluded_chunks: frozenset = frozenset()
    folded: frozenset = frozenset()  # (name, type) labeled as a value or junk
    merges: tuple = ()  # ((name, type), (name, type)): duplicate -> target
    # (frozenset of the two ends' (name, type), relation type): relations of that type between the two, either way,
    # that a model refuted (an ALTERNATIVE_NAME between two different objects); kept as UNFIT:<type>
    marked_relations: frozenset = frozenset()
    # (((name, type), (name, type), relation type), (new type, reverse)): a relation retyped from its text
    retypes: tuple = ()
    not_objects: frozenset = frozenset()  # (name, type) of nodes that are no objects, folded like values
    subtypes: tuple = ()  # ((name, type), type): the type of the ontology a node takes


def _rows(path):
    if path:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                yield json.loads(line)


def load_cleanup_inputs(excluded_path=None, labels_path=None, merges_path=None, marks_path=None, subtypes_path=None,
                        retypes_path=None, not_object_kinds=NOT_OBJECT_KINDS) -> CleanupInputs:
    excluded = frozenset(json.loads(Path(excluded_path).read_text(encoding='utf-8'))) if excluded_path else frozenset()
    folded = set()
    if labels_path:
        for line in Path(labels_path).read_text(encoding='utf-8').splitlines():
            row = json.loads(line) if line.strip() else None
            if row and row['label'] in ('value', 'junk'):
                folded.add((row['name'], row['type']))
    merges = []
    if merges_path:
        for line in Path(merges_path).read_text(encoding='utf-8').splitlines():
            row = json.loads(line) if line.strip() else None
            if row:
                merges.append((tuple(row['from']), tuple(row['to'])))
    target = {_key(*source): to for source, to in merges}

    def node(end):  # an input names the nodes of the graph it was made on; the merges may rename them
        return target.get(_key(*end), tuple(end))

    marks = {(frozenset(node(end) for end in (row['subject'], row['object'])), row['relation_type'])
             for row in _rows(marks_path)}
    subtypes, not_objects = [], set()
    for row in _rows(subtypes_path):
        if row['kind'] in not_object_kinds:
            not_objects.add(node((row['name'], row['type'])))
        else:
            subtypes.append((node((row['name'], row['type'])), row['subtype']))
    retypes = tuple(((node(row['subject']), node(row['object']), row['original']), (row['type'], bool(row['reverse'])))
                    for row in _rows(retypes_path))
    return CleanupInputs(excluded, frozenset(folded), tuple(merges), frozenset(marks), retypes, frozenset(not_objects),
                         tuple(subtypes))


def exclude_chunks(entities: list[Entity], relations: list[Relation], excluded: frozenset
                   ) -> tuple[list[Entity], list[Relation], dict[str, int]]:
    stats = {'excluded_entities': 0, 'excluded_relations': 0, 'dangling_relations': 0}

    def kept(items, key):
        result = []
        for item in items:
            chunks = [chunk for chunk in item.source_chunk_id if chunk not in excluded]
            if not chunks:
                stats[key] += 1
            else:
                result.append(item if len(chunks) == len(item.source_chunk_id)
                              else dataclasses.replace(item, source_chunk_id=chunks))
        return result

    entities = kept(entities, 'excluded_entities')
    ids = {entity.id for entity in entities}
    result = []
    for relation in kept(relations, 'excluded_relations'):
        if relation.subject_id in ids and relation.object_id in ids:
            result.append(relation)
        else:
            stats['dangling_relations'] += 1
    return entities, result, stats


def _with_facts(description: str, facts: list[str]) -> str:
    parts = [description.strip()] if description.strip() else []
    for fact in facts:
        fact = fact.strip()
        if fact and all(fact not in part for part in parts):
            parts.append(fact)
    return ' '.join(parts)


def _terms(text: str) -> list[str]:
    return [term.casefold().replace('ё', 'е') for term in re.findall(r'[^\W_]+', str(text))]


def _names_value(description: str, value: str) -> bool:
    """Every word or number of the value starts one of the description ("11 стр." in "Объём статьи 11 страниц")."""
    terms = _terms(description)
    return all(any(term.startswith(part) for term in terms) for part in _terms(value))


def _fact(description: str, value: str) -> str:
    # "Дата публикации работы." names no date: the value itself was the name of the node that goes
    return description if _names_value(description, value) else f"{description.strip().rstrip('.')}: {value}."


def fold_values(entities: list[Entity], relations: list[Relation], folded: frozenset = frozenset()
                ) -> tuple[list[Entity], list[Relation], dict[str, int]]:
    folded_keys = {_key(*end) for end in folded}
    value_names = {entity.id: entity.entity_name for entity in entities
                   if entity.entity_type in VALUE_TYPES or _key(entity.entity_name, entity.entity_type) in folded_keys}
    value_ids = set(value_names)
    facts: dict[tuple[str, str], list[str]] = defaultdict(list)
    kept_relations = []
    stats = {'value_entities': 0, 'folded_relations': 0, 'value_to_value_relations': 0}
    for relation in relations:
        subject_value, object_value = relation.subject_id in value_ids, relation.object_id in value_ids
        if not subject_value and not object_value:
            kept_relations.append(relation)
        elif subject_value and object_value:
            stats['value_to_value_relations'] += 1
        else:
            target, value = ((relation.object_id, relation.subject_id) if subject_value
                             else (relation.subject_id, relation.object_id))
            for chunk_id in relation.source_chunk_id:
                facts[(target, chunk_id)].append(_fact(relation.description, value_names[value]))
            stats['folded_relations'] += 1
    kept_entities = []
    for entity in entities:
        if entity.id in value_ids:
            stats['value_entities'] += 1
            continue
        extra = []
        for chunk_id in entity.source_chunk_id:
            extra.extend(facts.pop((entity.id, chunk_id), []))  # pop: only the first mention of a chunk gets them
        if extra:
            entity = dataclasses.replace(entity, description=_with_facts(entity.description, extra))
        kept_entities.append(entity)
    return kept_entities, kept_relations, stats


def _normalize(name: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]|_', ' ', str(name).casefold().replace('ё', 'е')).split())


def _key(name: str, kind: str) -> tuple[str, str]:
    return _normalize(name), kind


def _follow(relations: list[Relation], replacement: dict[str, str], name_by_id: dict[str, str]
            ) -> tuple[list[Relation], int]:
    """Relations follow their renamed ends; a relation whose two ends became one node goes (returned: how many)."""
    kept, loops = [], 0
    for relation in relations:
        subject_id = replacement.get(relation.subject_id, relation.subject_id)
        object_id = replacement.get(relation.object_id, relation.object_id)
        if subject_id == object_id:
            loops += 1
            continue
        if (subject_id, object_id) != (relation.subject_id, relation.object_id):
            relation = Relation(subject_id=subject_id, object_id=object_id,
                                subject_name=name_by_id.get(subject_id, relation.subject_name),
                                object_name=name_by_id.get(object_id, relation.object_name),
                                relation_type=relation.relation_type, description=relation.description,
                                relation_strength=relation.relation_strength, source_chunk_id=relation.source_chunk_id)
        kept.append(relation)
    return kept, loops


def _renamed(entity: Entity, name: str, kind: str) -> Entity:
    return Entity(entity_name=name, entity_type=kind, description=entity.description,
                  source_chunk_id=entity.source_chunk_id, documents_id=entity.documents_id, clusters=entity.clusters)


def _words(name: str) -> list[str]:
    return [word.casefold().replace('ё', 'е') for word in _WORD.findall(str(name))]


def _word_fits(word: str, full_word: str) -> bool:
    return full_word == word if len(word) > 1 else full_word.startswith(word)


def _fits(short: list[str], full: list[str]) -> bool:
    """The short name has the surname of the full one (its first or its last word), and its other words are the given
    name and then the patronymic, whole or as initials: "S. Smirnov" is not "Evgenii Sergeevich Smirnov"."""
    for surname, given, patronymic in ((0, 1, 2), (2, 0, 1)):
        if full[surname] not in short:
            continue
        rest = list(short)
        rest.remove(full[surname])
        order = [full[given], full[patronymic]]
        if len(rest) <= 2 and all(_word_fits(word, order[i]) for i, word in enumerate(rest)):
            return True
    return False


def _person_targets(entities: list[Entity]) -> dict[str, str]:
    """Person name -> the full name it is a variant of."""
    counts = Counter(entity.entity_name for entity in entities if entity.entity_type == 'PERSON')
    full = {name for name in counts if len(_words(name)) == 3 and all(len(w) > 1 for w in _words(name))}
    by_words: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for name in full:
        by_words[tuple(sorted(_words(name)))].append(name)
    targets = {}
    canonical = {}
    for spellings in by_words.values():
        best = max(spellings, key=lambda name: (counts[name], name))
        for name in spellings:
            canonical[name] = best
            if name != best:
                targets[name] = best
    index: dict[str, set] = defaultdict(set)
    for name in full:
        for word in _words(name):
            index[word].add(canonical[name])
    for name in counts:
        words = _words(name)
        if name in full or len(words) < 2 or len(words) > 3 or all(len(w) == 1 for w in words):
            continue
        whole = [w for w in words if len(w) > 1]
        candidates = set.intersection(*(index.get(w, set()) for w in whole))
        fitting = [candidate for candidate in candidates if _fits(words, _words(candidate))]
        if len(fitting) == 1:
            targets[name] = fitting[0]
    return targets


def merge_variants(entities: list[Entity], relations: list[Relation]) -> tuple[list[Entity], list[Relation], dict[str, int]]:
    stats = {'retyped_entities': 0, 'renamed_entities': 0, 'self_loops_dropped': 0}
    mentions = Counter((_normalize(entity.entity_name), entity.entity_type) for entity in entities)
    types_by_name: dict[str, Counter] = defaultdict(Counter)
    spelling: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for entity in entities:
        key = (_normalize(entity.entity_name), entity.entity_type)
        types_by_name[key[0]][entity.entity_type] = mentions[key]
        spelling[key][entity.entity_name] += 1
    retype = {}
    for name, types in types_by_name.items():
        if len(types) > 1 and 'PERSON' not in types:
            best = max(types.items(), key=lambda item: (item[1], item[0]))[0]
            retype[name] = (max(spelling[(name, best)].items(), key=lambda item: (item[1], item[0]))[0], best)
    person = _person_targets(entities)
    replacement: dict[str, str] = {}
    name_by_id: dict[str, str] = {}
    result = []
    for entity in entities:
        new_name, new_type = entity.entity_name, entity.entity_type
        if entity.entity_type == 'PERSON' and entity.entity_name in person:
            new_name = person[entity.entity_name]
            stats['renamed_entities'] += 1
        elif entity.entity_type != 'PERSON' and _normalize(entity.entity_name) in retype:
            target_name, target_type = retype[_normalize(entity.entity_name)]
            if target_type != entity.entity_type:
                stats['retyped_entities'] += 1
            new_name, new_type = target_name, target_type
        if (new_name, new_type) != (entity.entity_name, entity.entity_type):
            renamed = _renamed(entity, new_name, new_type)
            replacement[entity.id] = renamed.id
            entity = renamed
        name_by_id[entity.id] = entity.entity_name
        result.append(entity)
    kept, stats['self_loops_dropped'] = _follow(relations, replacement, name_by_id)
    return result, kept, stats


def apply_merges(entities: list[Entity], relations: list[Relation], merges: tuple
                 ) -> tuple[list[Entity], list[Relation], dict[str, int]]:
    target_of = {_key(*source): target for source, target in merges}
    spelling: dict[tuple[str, str], str] = {}
    for entity in entities:
        spelling.setdefault(_key(entity.entity_name, entity.entity_type), entity.entity_name)
    stats = {'merged_entities': 0}
    replacement: dict[str, str] = {}
    name_by_id: dict[str, str] = {}
    result = []
    for entity in entities:
        target = target_of.get(_key(entity.entity_name, entity.entity_type))
        if target is not None:
            target = (spelling.get(_key(*target), target[0]), target[1])  # the spelling of this build
        if target is not None and target != (entity.entity_name, entity.entity_type):
            renamed = _renamed(entity, *target)
            replacement[entity.id] = renamed.id
            stats['merged_entities'] += 1
            entity = renamed
        name_by_id[entity.id] = entity.entity_name
        result.append(entity)
    kept, stats['merge_self_loops_dropped'] = _follow(relations, replacement, name_by_id)
    return result, kept, stats


def clean_graph(entities: list[Entity], relations: list[Relation], inputs: CleanupInputs | None = None
                ) -> tuple[list[Entity], list[Relation], dict[str, int]]:
    inputs = inputs or CleanupInputs()
    stats: dict[str, int] = {}
    if inputs.excluded_chunks:
        entities, relations, excluded = exclude_chunks(entities, relations, inputs.excluded_chunks)
        stats.update(excluded)
    entities, relations, folded = fold_values(entities, relations)
    entities, relations, merged = merge_variants(entities, relations)
    stats.update({**folded, **merged})
    if inputs.folded:
        entities, relations, labeled = fold_values(entities, relations, inputs.folded)
        stats.update({f'labeled_{key}': value for key, value in labeled.items()})
    if inputs.merges:
        entities, relations, applied = apply_merges(entities, relations, inputs.merges)
        stats.update(applied)
    if inputs.marked_relations:
        relations, marked = mark_relations(entities, relations, inputs.marked_relations)
        stats['marked_relations'] = marked
    if inputs.retypes:
        relations, retyped = retype_relations(entities, relations, inputs.retypes)
        stats.update(retyped)
    if inputs.not_objects:
        entities, relations, folded = fold_values(entities, relations, inputs.not_objects)
        stats.update({f'not_object_{key}': value for key, value in folded.items()})
    if inputs.subtypes:
        entities, relations, subtyped = apply_subtypes(entities, relations, inputs.subtypes)
        stats.update(subtyped)
    return entities, relations, stats


def mark_relations(entities: list[Entity], relations: list[Relation], marks: frozenset
                   ) -> tuple[list[Relation], int]:
    """Give UNFIT:<type> to the relations of a marked type between a marked pair of nodes; the text stays."""
    node = {entity.id: _key(entity.entity_name, entity.entity_type) for entity in entities}
    marks = {(frozenset(_key(*end) for end in ends), kind) for ends, kind in marks}
    result, marked = [], 0
    for relation in relations:
        ends = frozenset((node.get(relation.subject_id), node.get(relation.object_id)))
        if not relation.relation_type.startswith('UNFIT:') and (ends, relation.relation_type) in marks:
            relation = dataclasses.replace(relation, relation_type='UNFIT:' + relation.relation_type)
            marked += 1
        result.append(relation)
    return result, marked


def retype_relations(entities: list[Entity], relations: list[Relation], retypes: tuple
                     ) -> tuple[list[Relation], dict[str, int]]:
    """A retyped relation takes its new type, from the other end when reversed; its text stays."""
    node = {entity.id: _key(entity.entity_name, entity.entity_type) for entity in entities}
    name = {entity.id: entity.entity_name for entity in entities}
    new = {(_key(*subject), _key(*obj), kind): label for (subject, obj, kind), label in retypes}
    stats = {'retyped_relations': 0, 'reversed_relations': 0}
    result = []
    for relation in relations:
        label = new.get((node.get(relation.subject_id), node.get(relation.object_id), relation.relation_type))
        if label is None or label == (relation.relation_type, False):
            result.append(relation)
            continue
        kind, reverse = label
        subject_id, object_id = (relation.object_id, relation.subject_id) if reverse else (relation.subject_id,
                                                                                            relation.object_id)
        subject_name, object_name = ((relation.object_name, relation.subject_name) if reverse
                                     else (relation.subject_name, relation.object_name))
        result.append(Relation(subject_id=subject_id, object_id=object_id,
                               subject_name=name.get(subject_id, subject_name), object_name=name.get(object_id, object_name),
                               relation_type=kind, description=relation.description,
                               relation_strength=relation.relation_strength, source_chunk_id=relation.source_chunk_id))
        stats['retyped_relations'] += 1
        stats['reversed_relations'] += int(reverse)
    return result, stats


def apply_subtypes(entities: list[Entity], relations: list[Relation], subtypes: tuple
                   ) -> tuple[list[Entity], list[Relation], dict[str, int]]:
    """A node takes the type of the ontology it was labeled with; mentions of one name that get one type are one node."""
    new = {_key(*end): kind for end, kind in subtypes}
    stats = {'subtyped_entities': 0}
    replacement: dict[str, str] = {}
    name_by_id: dict[str, str] = {}
    result = []
    for entity in entities:
        kind = new.get(_key(entity.entity_name, entity.entity_type))
        if kind is not None and kind != entity.entity_type:
            renamed = _renamed(entity, entity.entity_name, kind)
            replacement[entity.id] = renamed.id
            stats['subtyped_entities'] += 1
            entity = renamed
        name_by_id[entity.id] = entity.entity_name
        result.append(entity)
    kept, stats['subtype_self_loops_dropped'] = _follow(relations, replacement, name_by_id)
    return result, kept, stats
