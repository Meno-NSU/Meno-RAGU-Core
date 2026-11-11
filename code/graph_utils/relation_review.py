"""
Relation review: the stage that replaces RAGU's rewriting relation validation.

RAGU's validation asks the model to return the whole relation list again. That costs as much generation as the
extraction itself, lets the model rewrite correct relations, and never shows it the domain → range signatures the
ontology later enforces, so a third of the relations were dropped after it for type pairs the model never saw.

The review shows the numbered relations with the types of their endpoints and the admissible relation types with
their signatures, flags the relations the ontology would reject together with the types that accept their pair, and
asks only for changes: fixes, relations without a fitting type, drops with a reason, and missing relations. Every
field of a structured answer is mandatory, so listing only the changes keeps the answer short. Code applies them:

* a relation the review does not list stays exactly as extracted, and a chunk whose review failed keeps all of its
  relations, so the review removes only what it names with a reason;
* a fix may change only the type, the direction or an endpoint name, and is accepted only when the result fits the
  ontology exactly; a "no fit" whose proposed type is an admissible type that fits is applied as a fix;
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


class RelationFix(BaseModel):
    id: int = Field(..., description='Number of the relation in the list')
    relation_type: str = Field(..., description='The new type from the allowed list, or "" to keep the type')
    reverse: bool = Field(..., description='true to swap the source and the target')
    source_entity: str = Field(..., description='The exact source name from the entity list, or "" to keep it')
    target_entity: str = Field(..., description='The exact target name from the entity list, or "" to keep it')


class RelationNoFit(BaseModel):
    id: int = Field(..., description='Number of the relation in the list')
    proposed_type: str = Field(..., description='Your own short UPPER_SNAKE_CASE label for the relation')


class RelationDrop(BaseModel):
    id: int = Field(..., description='Number of the relation in the list')
    reason: Literal['unsupported', 'different_rows', 'boilerplate', 'duplicate']


class RelationReviewModel(BaseModel):
    fix: list[RelationFix] = Field(default_factory=list, description='Relations to fix')
    no_fit: list[RelationNoFit] = Field(default_factory=list, description='Relations no allowed type fits')
    drop: list[RelationDrop] = Field(default_factory=list, description='Relations to drop')
    added: list[RelationModel] = Field(default_factory=list, description='Relations the list misses')


REVIEW_SYSTEM = """You check relations that were extracted from a text, against that text and against the allowed relation types.
List only the relations that need a change, by their numbers. Every relation you do not list is kept exactly as it is.

- "fix": the text supports the relation, but its type, its direction or an endpoint name is wrong. Give "relation_type": an allowed type whose signature fits the types of the two entities, or "" to keep the type; "reverse": true to swap the source and the target; "source_entity" / "target_entity": the exact name from the entity list when an endpoint is a misspelling or a variant of a listed entity, or "" to keep it.
- "no_fit": the text supports the relation, but no allowed type fits these two entities. Give "proposed_type": a short UPPER_SNAKE_CASE label of your own that names the relation. The relation is kept.
- "drop": the relation has to go. Give the "reason":
  - "unsupported": the text gives no basis for this connection at all;
  - "different_rows": the two entities come from different rows of a table or different items of a list, and nothing in the text connects them;
  - "boilerplate": the relation comes from the furniture of the page (navigation, copyright notices, identifiers, counters) and says nothing about the entities themselves;
  - "duplicate": another relation in the list states the same connection between the same entities.
  A weak or indirect connection that the text supports is not a reason to drop. When in doubt, do not list the relation.

A relation followed by "!" will be discarded as it is. Its note names the allowed types for its pair of entity types: fix the relation with one of them when one matches its meaning, otherwise answer "no_fit".
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
        line = (f'{number}. {source} ({types.get(source, "?")}) -> {target} ({types.get(target, "?")}) | '
                f'{item["relation_type"]} | {item["description"]}')
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


def _fits(relation: RelationModel, types: dict, validator: OntologyValidator | None) -> bool:
    """Whether a fixed relation can be accepted as it is: listed endpoints, no loop, a type that fits exactly."""
    if relation.source_entity not in types or relation.target_entity not in types:
        return False
    if relation.source_entity == relation.target_entity:
        return False
    if validator is None:
        return True
    ontology = validator.ontology
    canonical = ontology.resolve_relation_type(relation.relation_type)
    return canonical is not None and ontology.allows(
        canonical, types[relation.source_entity], types[relation.target_entity])


def _fixed(relation: RelationModel, change: RelationFix) -> RelationModel:
    update = {field: getattr(change, field).strip()
              for field in ('relation_type', 'source_entity', 'target_entity') if getattr(change, field).strip()}
    result = relation.model_copy(update=update)
    if change.reverse:
        result = result.model_copy(update={
            'source_entity': result.target_entity, 'target_entity': result.source_entity,
        })
    return result


def _changes(review: RelationReviewModel, count: int, stats: Counter) -> dict[int, tuple[str, BaseModel]]:
    """The change of each listed relation; a number listed twice or out of range changes nothing."""
    listed: dict[int, list[tuple[str, BaseModel]]] = {}
    for kind, items in (('fix', review.fix), ('no_fit', review.no_fit), ('drop', review.drop)):
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
            if kind == 'no_fit':
                proposed = item.model_copy(update={'relation_type': change.proposed_type.strip()})
                if proposed.relation_type and _fits(proposed, types, validator):
                    stats['no_fit_fixed'] += 1
                    item = proposed
                else:
                    stats['no_fit'] += 1
            elif kind == 'fix':
                fixed = _fixed(item, change)
                if fixed == item:
                    stats['fix_unchanged'] += 1
                elif _fits(fixed, types, validator):
                    stats['fix'] += 1
                    item = fixed
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
