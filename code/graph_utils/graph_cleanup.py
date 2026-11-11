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
(a DOI typed PRODUCT, an e-mail typed LOCATION), folded like values after the name variants are merged, and merges of
duplicate entities a model confirmed (name and type to name and type), applied last.
"""
import dataclasses
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ragu.graph.types import Entity, Relation

VALUE_TYPES = frozenset({'NUMBER', 'DATE', 'TIME', 'AGE', 'MONEY', 'PERCENT', 'ORDINAL'})
_WORD = re.compile(r'[^\W\d_]+')


@dataclasses.dataclass(frozen=True)
class CleanupInputs:
    excluded_chunks: frozenset = frozenset()
    folded: frozenset = frozenset()  # (name, type) labeled as a value or junk
    merges: tuple = ()  # ((name, type), (name, type)): duplicate -> target


def load_cleanup_inputs(excluded_path=None, labels_path=None, merges_path=None) -> CleanupInputs:
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
    return CleanupInputs(excluded, frozenset(folded), tuple(merges))


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
    value_names = {entity.id: entity.entity_name for entity in entities
                   if entity.entity_type in VALUE_TYPES or (entity.entity_name, entity.entity_type) in folded}
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
            renamed = Entity(entity_name=new_name, entity_type=new_type, description=entity.description,
                             source_chunk_id=entity.source_chunk_id, documents_id=entity.documents_id,
                             clusters=entity.clusters)
            replacement[entity.id] = renamed.id
            entity = renamed
        name_by_id[entity.id] = entity.entity_name
        result.append(entity)
    kept = []
    for relation in relations:
        subject_id = replacement.get(relation.subject_id, relation.subject_id)
        object_id = replacement.get(relation.object_id, relation.object_id)
        if subject_id == object_id:
            stats['self_loops_dropped'] += 1
            continue
        if (subject_id, object_id) != (relation.subject_id, relation.object_id):
            relation = Relation(subject_id=subject_id, object_id=object_id,
                                subject_name=name_by_id.get(subject_id, relation.subject_name),
                                object_name=name_by_id.get(object_id, relation.object_name),
                                relation_type=relation.relation_type, description=relation.description,
                                relation_strength=relation.relation_strength, source_chunk_id=relation.source_chunk_id)
        kept.append(relation)
    return result, kept, stats


def apply_merges(entities: list[Entity], relations: list[Relation], merges: tuple
                 ) -> tuple[list[Entity], list[Relation], dict[str, int]]:
    target_of = dict(merges)
    stats = {'merged_entities': 0, 'merge_self_loops_dropped': 0}
    replacement: dict[str, str] = {}
    name_by_id: dict[str, str] = {}
    result = []
    for entity in entities:
        target = target_of.get((entity.entity_name, entity.entity_type))
        if target is not None and target != (entity.entity_name, entity.entity_type):
            renamed = Entity(entity_name=target[0], entity_type=target[1], description=entity.description,
                             source_chunk_id=entity.source_chunk_id, documents_id=entity.documents_id,
                             clusters=entity.clusters)
            replacement[entity.id] = renamed.id
            stats['merged_entities'] += 1
            entity = renamed
        name_by_id[entity.id] = entity.entity_name
        result.append(entity)
    kept = []
    for relation in relations:
        subject_id = replacement.get(relation.subject_id, relation.subject_id)
        object_id = replacement.get(relation.object_id, relation.object_id)
        if subject_id == object_id:
            stats['merge_self_loops_dropped'] += 1
            continue
        if (subject_id, object_id) != (relation.subject_id, relation.object_id):
            relation = Relation(subject_id=subject_id, object_id=object_id,
                                subject_name=name_by_id.get(subject_id, relation.subject_name),
                                object_name=name_by_id.get(object_id, relation.object_name),
                                relation_type=relation.relation_type, description=relation.description,
                                relation_strength=relation.relation_strength, source_chunk_id=relation.source_chunk_id)
        kept.append(relation)
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
    return entities, relations, stats
