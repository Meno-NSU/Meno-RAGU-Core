"""
Relation review: the stage that replaces RAGU's rewriting relation validation.

RAGU's validation asks the model to return the whole relation list again. That costs as much generation as the
extraction itself, lets the model rewrite correct relations, and never shows it the domain → range signatures the
ontology later enforces, so a third of the relations were dropped after it for type pairs the model never saw.

The review shows the numbered relations with the types of their endpoints and the admissible relation types with
their signatures, flags the relations the ontology would reject together with the types that accept their pair, and
asks only for changes of type, drops with a reason, and missing relations. Every
field of a structured answer is mandatory, so listing only the changes keeps the answer short. Code applies them:

* a relation the review does not list stays exactly as extracted, and a chunk whose review failed keeps all of its
  relations, so the review removes only what it names with a reason;
* a change alters only the type and is applied only when the ontology keeps the result (turning it around itself
  when the reversed pair fits); a change to a label outside the ontology means that no type fits: the relation stays
  as it is;
* a relation the ontology would drop for its type (no fitting type for the pair, or a type outside the vocabulary) is
  kept with the type ``UNFIT:<type>``. Dangling endpoints and self-loops are left to the ontology, which drops them.
"""
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from jinja2 import Template
from pydantic import BaseModel, Field
from ragu.common.logger import logger
from ragu.common.prompts.default_models import RelationModel, RelationsExtractionModel
from ragu.triplet.ontology import OntologyValidator, ValidationReport

UNFIT_PREFIX = 'UNFIT:'
REVIEW_STAGE = '05-relation-review'
_UNFIT_VIOLATIONS = {'constraint_violation', 'unknown_relation_type'}


class RelationChange(BaseModel):
    id: int = Field(..., description='Number of the relation in the list')
    relation_type: str = Field(..., description=(
        'An allowed type that fits the two entity types and states what the text says; '
        'if none does, your own short UPPER_SNAKE_CASE label'))


class RelationDrop(BaseModel):
    id: int = Field(..., description='Number of the relation in the list')
    reason: Literal['unsupported', 'different_rows', 'boilerplate']


class RelationReviewModel(BaseModel):
    change: list[RelationChange] = Field(default_factory=list, description='Relations to change')
    drop: list[RelationDrop] = Field(default_factory=list, description='Relations to drop')
    added: list[RelationModel] = Field(default_factory=list, description='Relations the list misses')


REVIEW_SYSTEM = """You check relations that were extracted from a text, against that text and against the allowed relation types.
List only the relations that need a change, by their numbers, each at most once. Every relation you do not list is kept exactly as it is.

- "change": the text supports the relation, but its type is wrong. Give "relation_type": an allowed type whose signature fits the types of the two entities and that states what the text says about them. Do not pick a type only because it accepts any pair. When no allowed type states it, give a short UPPER_SNAKE_CASE label of your own: the relation is then kept as it is and marked.
- "drop": the relation has to go. Give the "reason":
  - "unsupported": the text gives no basis for this connection at all;
  - "different_rows": the two entities come from different rows of a table or different items of a list, and nothing in the text connects them;
  - "boilerplate": the relation comes from the furniture of the page (navigation, copyright notices, identifiers, counters) and says nothing about the entities themselves.
  A weak or indirect connection that the text supports is not a reason to drop. When in doubt, do not list the relation.

A relation followed by "!" will be discarded as it is. Its note names the allowed types for its pair of entity types: change the relation to one of them when one states its meaning, otherwise give your own label.
{% if relation_types %}
Allowed relation types. A type shown as `TYPE [SUBJECT -> OBJECT] (meaning)` may only go from an entity of a SUBJECT type to an entity of an OBJECT type; `|` separates alternatives, `<->` marks a relation that holds in both directions, and a type shown without brackets accepts any pair. The source entity is the SUBJECT: follow the arrow.
{{ relation_types }}
{% endif %}
In "added", list the relations between the listed entities that the text states clearly but the list misses: "source_entity" and "target_entity" (exact names from the entity list), "relation_type" (an allowed type that fits), "description" (grounded in the text, in {{ language }}), "relationship_strength" (1 to 5: how explicitly the text states it). Do not repeat relations that are already in the list.

Judge only by the text, not by outside knowledge.
Return the result as valid JSON matching the provided schema."""


_SYSTEM_TEMPLATE = Template(REVIEW_SYSTEM)


def review_messages(
    context: str,
    entities: Sequence[dict],
    relations: Sequence[dict],
    relation_types: str | None,
    notes: Sequence[str | None],
    language: str,
) -> list[dict]:
    """Build the chat messages that ask for the changes to the numbered relations of one chunk."""
    types = {entity['entity_name']: entity['entity_type'] for entity in entities}
    lines = ['Entities:']
    lines.extend(f'- {entity["entity_name"]} ({entity["entity_type"]})' for entity in entities)
    lines.append('')
    lines.append('Relations:')
    for number, (item, note) in enumerate(zip(relations, notes, strict=True), 1):
        source, target = item['source_entity'], item['target_entity']
        line = (f'{number}. {source} -> {target} | {item["relation_type"]} | '
                f'{types.get(source, "?")} -> {types.get(target, "?")} | {item["description"]}')
        lines.append(f'{line} ! {note}' if note else line)
    lines.append('')
    lines.append('Text:')
    lines.append(context)
    return [
        {'role': 'system', 'content': _SYSTEM_TEMPLATE.render(relation_types=relation_types, language=language)},
        {'role': 'user', 'content': '\n'.join(lines) + '\n'},
    ]


def _rejection(relation: RelationModel, types: dict, validator: OntologyValidator) -> str | None:
    """The check that would drop the relation, or ``None`` when the ontology keeps or repairs it."""
    report = ValidationReport()
    if validator._validate_relation(relation, types, report) is not None:
        return None
    return next(iter(report.dropped), 'dropped')


def _accepting(subject_type: str, object_type: str, validator: OntologyValidator) -> str:
    names = [name for name in validator.ontology.relation_type_names
             if validator.ontology.allows(name, subject_type, object_type)]
    return (f'allowed for {subject_type} -> {object_type}: {", ".join(names)}' if names
            else f'no allowed type accepts {subject_type} -> {object_type}')


def relation_note(relation: RelationModel, types: dict, validator: OntologyValidator | None) -> str | None:
    """Why the ontology would discard the relation and which types accept its pair, or ``None`` if it would not."""
    if validator is None:
        return None
    reason = _rejection(relation, types, validator)
    if reason == 'dangling_endpoint':
        missing = relation.source_entity if relation.source_entity not in types else relation.target_entity
        return f'{missing} is not in the entity list'
    if reason not in _UNFIT_VIOLATIONS:
        return None
    subject_type, object_type = types[relation.source_entity], types[relation.target_entity]
    problem = (f'{relation.relation_type} does not accept {subject_type} -> {object_type}'
               if reason == 'constraint_violation' else f'{relation.relation_type} is not an allowed type')
    return f'{problem}; {_accepting(subject_type, object_type, validator)}'


def _changes(review: RelationReviewModel, count: int, stats: Counter) -> dict[int, tuple[str, BaseModel]]:
    """The change of each listed relation; a number listed twice or out of range changes nothing."""
    listed: dict[int, list[tuple[str, BaseModel]]] = {}
    for kind, items in (('change', review.change), ('drop', review.drop)):
        for item in items:
            if 1 <= item.id <= count:
                listed.setdefault(item.id, []).append((kind, item))
            else:
                stats['verdict_ignored'] += 1
    changes = {}
    for number, items in listed.items():
        if len(items) == 1:
            changes[number] = items[0]
        else:
            stats['verdict_conflict'] += 1
    return changes


def apply_review(
    relations: Sequence[RelationModel],
    review: RelationReviewModel | None,
    entities: Sequence[dict],
    validator: OntologyValidator | None,
) -> tuple[list[RelationModel], Counter]:
    """Apply the review of one chunk and mark the relations the ontology has no fitting type for."""
    stats: Counter = Counter()
    types = {entity['entity_name']: entity['entity_type'] for entity in entities}
    if review is None:
        stats['review_missing'] += 1
        result = list(relations)
    else:
        changes = _changes(review, len(relations), stats)
        result = []
        for number, item in enumerate(relations, 1):
            kind, change = changes.get(number, ('keep', None))
            if kind == 'drop':
                stats[f'drop:{change.reason}'] += 1
                continue
            if kind == 'change':
                label = change.relation_type.strip()
                fixed = item.model_copy(update={'relation_type': label})
                if not label or label == item.relation_type:
                    stats['change_unchanged'] += 1
                elif validator is None or _rejection(fixed, types, validator) is None:
                    stats['fix'] += 1
                    item = fixed
                elif validator.ontology.resolve_relation_type(label) is None:
                    stats['no_fit'] += 1
                else:
                    stats['fix_rejected'] += 1
            else:
                stats['keep'] += 1
            result.append(item)
        stats['added'] += len(review.added)
        result.extend(review.added)
    if validator is None:
        return result, stats
    marked = []
    for item in result:
        if _rejection(item, types, validator) in _UNFIT_VIOLATIONS:
            stats['marked_unfit'] += 1
            item = item.model_copy(update={'relation_type': UNFIT_PREFIX + item.relation_type})
        marked.append(item)
    return marked, stats


def is_unfit(relation: RelationModel) -> bool:
    return relation.relation_type.startswith(UNFIT_PREFIX)


class RelationReviewMixin:
    """Extractor mixin: with ``relation_review=True`` the relation validation pass becomes the review."""

    def __init__(self, *args, relation_review: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.relation_review = relation_review
        self.review_stats: Counter = Counter()

    def _review_relation_types(self, entities: Sequence[dict]) -> str | None:
        if self.ontology is None:
            return None
        present = {str(entity.get('entity_type', '')) for entity in entities}
        pairs = [(subject, obj) for subject in present for obj in present]
        return (self.ontology.render_relation_types(with_signatures=True, for_pairs=pairs)
                or self.ontology.render_relation_types(with_signatures=True))

    async def _review_relations(
        self,
        context: list[str],
        entities_payload: list[list[dict]],
        relations: list[RelationsExtractionModel | None],
    ) -> list[RelationReviewModel | None]:
        conversations = []
        for text, entities, extracted in zip(context, entities_payload, relations, strict=True):
            types = {entity['entity_name']: entity['entity_type'] for entity in entities}
            items = list(extracted.relations) if extracted is not None else []
            conversations.append(review_messages(
                context=text, entities=entities, relations=[item.model_dump() for item in items],
                relation_types=self._review_relation_types(entities),
                notes=[relation_note(item, types, self.validator) for item in items],
                language=self.language,
            ))
        return await self.llm.batch_chat_completion(
            conversations, output_schema=RelationReviewModel, continue_on_error=True,
            desc='Reviewing extracted relations',
        )

    async def _validate_relations(self, context, entities_payload, relations):
        if not self.relation_review:
            return await super()._validate_relations(context, entities_payload, relations)
        reviews = await self._review_relations(context, entities_payload, relations)
        result = []
        stats: Counter = Counter()
        for entities, extracted, review in zip(entities_payload, relations, reviews, strict=True):
            items = list(extracted.relations) if extracted is not None else []
            applied, chunk_stats = apply_review(items, review, entities, self.validator)
            stats.update(chunk_stats)
            result.append(RelationsExtractionModel(relations=applied))
        self.review_stats.update(stats)
        logger.info('Relation review: {}', ', '.join(f'{key}={value}' for key, value in sorted(stats.items())))
        return result

    def _apply_ontology(self, entities_per_chunk, relations_per_chunk, stage):
        unfit = [[item for item in relations if is_unfit(item)] for relations in relations_per_chunk]
        others = [[item for item in relations if not is_unfit(item)] for relations in relations_per_chunk]
        entities, relations = super()._apply_ontology(entities_per_chunk, others, stage)
        kept = sum(len(items) for items in unfit)
        if kept:
            logger.info('Ontology validation ({}): {} relations without a fitting type kept as {}<type>.',
                        stage, kept, UNFIT_PREFIX)
        return entities, [checked + extra for checked, extra in zip(relations, unfit, strict=True)]
