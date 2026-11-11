import asyncio

import pytest
from graph_utils.checkpoints import CheckpointedExtractor, GraphCheckpoints
from graph_utils.relation_review import (
    UNFIT_PREFIX,
    RelationReviewModel,
    apply_review,
    review_messages,
)
from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import EntitiesExtractionModel, RelationModel, RelationsExtractionModel
from ragu.models.llm import LLM
from ragu.triplet.ontology import OntologyValidator

ENTITIES = [
    {'entity_name': 'Иванов', 'entity_type': 'PERSON', 'description': 'Преподаватель'},
    {'entity_name': 'НГУ', 'entity_type': 'ORGANIZATION', 'description': 'Университет'},
    {'entity_name': 'Генетика дрозофилы', 'entity_type': 'WORK_OF_ART', 'description': 'Курс'},
]


def relation(source, target, relation_type, description='Описание из текста', strength=4):
    return RelationModel(source_entity=source, target_entity=target, relation_type=relation_type,
                         description=description, relationship_strength=strength)


WORKS = relation('Иванов', 'НГУ', 'WORKPLACE', 'Иванов работает в НГУ', 5)
TEACHES = relation('Иванов', 'Генетика дрозофилы', 'AGENT', 'Иванов читает курс', 4)
PART = relation('Генетика дрозофилы', 'НГУ', 'PART_OF', 'Курс читается в НГУ', 3)


def review(*verdicts, added=()):
    return RelationReviewModel.model_validate({'verdicts': list(verdicts), 'added': list(added)})


@pytest.fixture(scope='module')
def validator():
    return OntologyValidator('nerel')


def dumps(relations):
    return [item.model_dump() for item in relations]


def test_kept_relations_stay_exactly_as_extracted(validator):
    result, stats = apply_review([WORKS, PART], review({'id': 1, 'verdict': 'keep'}, {'id': 2, 'verdict': 'keep'}),
                                 ENTITIES, validator)
    assert dumps(result) == dumps([WORKS, PART])
    assert stats['keep'] == 2


def test_missing_verdict_and_failed_review_keep_the_relation(validator):
    result, stats = apply_review([WORKS, PART], review({'id': 2, 'verdict': 'keep'}), ENTITIES, validator)
    assert dumps(result) == dumps([WORKS, PART])
    assert stats['no_verdict'] == 1
    result, stats = apply_review([WORKS, PART], None, ENTITIES, validator)
    assert dumps(result) == dumps([WORKS, PART])
    assert stats['review_missing'] == 1


def test_drop_needs_a_reason(validator):
    result, stats = apply_review(
        [WORKS, PART],
        review({'id': 1, 'verdict': 'drop'}, {'id': 2, 'verdict': 'drop', 'reason': 'different_rows'}),
        ENTITIES, validator,
    )
    assert dumps(result) == dumps([WORKS])
    assert stats['drop_without_reason'] == 1 and stats['drop:different_rows'] == 1


def test_verdicts_with_unknown_or_repeated_numbers_are_ignored(validator):
    result, stats = apply_review(
        [WORKS],
        review({'id': 1, 'verdict': 'keep'}, {'id': 1, 'verdict': 'drop', 'reason': 'unsupported'},
               {'id': 7, 'verdict': 'drop', 'reason': 'unsupported'}),
        ENTITIES, validator,
    )
    assert dumps(result) == dumps([WORKS])
    assert stats['verdict_ignored'] == 2


def test_fix_changes_only_type_direction_and_endpoint_names(validator):
    backwards = relation('НГУ', 'Иванов', 'PART_OF', 'Иванов работает в НГУ', 5)
    misspelt = relation('Иванов', 'Нгу', 'MEMBER_OF', 'Иванов состоит в НГУ', 2)
    result, stats = apply_review(
        [backwards, misspelt],
        review({'id': 1, 'verdict': 'fix', 'relation_type': 'WORKPLACE', 'reverse': True},
               {'id': 2, 'verdict': 'fix', 'target_entity': 'НГУ'}),
        ENTITIES, validator,
    )
    assert dumps(result) == dumps([
        relation('Иванов', 'НГУ', 'WORKPLACE', 'Иванов работает в НГУ', 5),
        relation('Иванов', 'НГУ', 'MEMBER_OF', 'Иванов состоит в НГУ', 2),
    ])
    assert stats['fix'] == 2


def test_fix_that_breaks_the_relation_is_rejected(validator):
    result, stats = apply_review(
        [WORKS, PART],
        review({'id': 1, 'verdict': 'fix', 'relation_type': 'DATE_OF_BIRTH'},
               {'id': 2, 'verdict': 'fix', 'target_entity': 'Новосибирск'}),
        ENTITIES, validator,
    )
    assert dumps(result) == dumps([WORKS, PART])
    assert stats['fix_rejected'] == 2


def test_relation_without_a_fitting_type_is_kept_with_a_mark(validator):
    invented = relation('Иванов', 'Генетика дрозофилы', 'TEACHES', 'Иванов читает курс', 4)
    result, stats = apply_review(
        [TEACHES, invented],
        review({'id': 1, 'verdict': 'no_fit', 'proposed_type': 'TEACHES'}, {'id': 2, 'verdict': 'keep'}),
        ENTITIES, validator,
    )
    assert [item.relation_type for item in result] == [UNFIT_PREFIX + 'AGENT', UNFIT_PREFIX + 'TEACHES']
    assert [item.description for item in result] == ['Иванов читает курс', 'Иванов читает курс']
    assert stats['no_fit'] == 1 and stats['marked_unfit'] == 2


def test_relations_the_ontology_repairs_or_drops_are_not_marked(validator):
    reversed_workplace = relation('НГУ', 'Иванов', 'WORKPLACE')
    dangling = relation('Иванов', 'Петров', 'KNOWS')
    loop = relation('НГУ', 'НГУ', 'PART_OF')
    result, stats = apply_review([reversed_workplace, dangling, loop], None, ENTITIES, validator)
    assert dumps(result) == dumps([reversed_workplace, dangling, loop])
    assert stats['marked_unfit'] == 0


def test_added_relations_are_checked_like_the_rest(validator):
    result, stats = apply_review(
        [WORKS],
        review({'id': 1, 'verdict': 'keep'}, added=[
            {'source_entity': 'Генетика дрозофилы', 'target_entity': 'НГУ', 'relation_type': 'PART_OF',
             'description': 'Курс НГУ', 'relationship_strength': 3},
            {'source_entity': 'Иванов', 'target_entity': 'Генетика дрозофилы', 'relation_type': 'AGENT',
             'description': 'Иванов читает курс', 'relationship_strength': 4},
        ]),
        ENTITIES, validator,
    )
    assert [item.relation_type for item in result] == ['WORKPLACE', 'PART_OF', UNFIT_PREFIX + 'AGENT']
    assert stats['added'] == 2


def test_without_an_ontology_nothing_is_marked():
    result, _ = apply_review([TEACHES], None, ENTITIES, None)
    assert dumps(result) == dumps([TEACHES])


def test_review_prompt_numbers_relations_shows_signatures_and_flags_misfits(validator):
    messages = review_messages(
        context='Иванов работает в НГУ и читает курс «Генетика дрозофилы».',
        entities=ENTITIES,
        relations=[WORKS.model_dump(), TEACHES.model_dump(), relation('Иванов', 'Петров', 'KNOWS').model_dump()],
        relation_types=validator.ontology.render_relation_types(
            with_signatures=True, for_pairs=[('PERSON', 'ORGANIZATION'), ('PERSON', 'WORK_OF_ART')]),
        notes=[None, 'AGENT does not accept PERSON -> WORK_OF_ART', 'Петров is not in the entity list'],
        language='russian',
    )
    assert [message['role'] for message in messages] == ['system', 'user']
    system, user = messages[0]['content'], messages[1]['content']
    assert 'WORKPLACE [PERSON -> ORGANIZATION|FACILITY]' in system
    assert 'DATE_OF_BIRTH' not in system
    assert '1. Иванов (PERSON) -> НГУ (ORGANIZATION) | WORKPLACE | Иванов работает в НГУ\n' in user
    assert ('2. Иванов (PERSON) -> Генетика дрозофилы (WORK_OF_ART) | AGENT | Иванов читает курс '
            '! AGENT does not accept PERSON -> WORK_OF_ART\n') in user
    assert '3. Иванов (PERSON) -> Петров (?) | KNOWS' in user
    assert user.rstrip().endswith('Иванов работает в НГУ и читает курс «Генетика дрозофилы».')


class ReviewModel(LLM):
    def __init__(self, verdicts, stop=None):
        self.verdicts = verdicts
        self.stop = stop
        self.calls = []
        self.conversations = {}

    async def chat_completion(self, conversation, **kwargs):
        raise AssertionError('Unexpected unbatched LLM call')

    async def batch_chat_completion(self, conversations, output_schema, desc=None, **kwargs):
        self.calls.append(desc)
        self.conversations[desc] = conversations
        if desc == self.stop:
            raise asyncio.CancelledError()
        if output_schema is EntitiesExtractionModel:
            return [EntitiesExtractionModel(entities=ENTITIES) for _ in conversations]
        if output_schema is RelationsExtractionModel:
            return [RelationsExtractionModel(relations=[WORKS, TEACHES]) for _ in conversations]
        assert output_schema is RelationReviewModel
        return [RelationReviewModel.model_validate({'verdicts': self.verdicts}) for _ in conversations]


def chunks():
    return [Chunk(f'Иванов работает в НГУ и читает курс «Генетика дрозофилы». {i}', i, f'doc_{i}') for i in range(2)]


def review_extractor(model, store=None):
    result = CheckpointedExtractor(llm=model, ontology='nerel', do_entity_validation=True,
                                   do_relation_validation=True, relation_review=True)
    result.checkpoints = store
    return result


REVIEW_PASSES = ['Extracting entities from chunks', 'Validating extracted entities',
                 'Extracting relations from chunks', 'Reviewing extracted relations']


def test_review_replaces_the_rewriting_validation_and_unfit_relations_reach_the_graph(tmp_path):
    async def run():
        model = ReviewModel([{'id': 1, 'verdict': 'keep'}, {'id': 2, 'verdict': 'no_fit', 'proposed_type': 'TEACHES'}])
        with GraphCheckpoints(tmp_path, {}) as store:
            _, relations = await review_extractor(model, store).extract(chunks())
            assert '05-relation-review' in store.manifest['stages']
            assert '05-relation-validation' not in store.manifest['stages']
        assert model.calls == REVIEW_PASSES
        assert sorted({item.relation_type for item in relations}) == [UNFIT_PREFIX + 'AGENT', 'WORKPLACE']
        assert len(relations) == 4
        prompt = model.conversations['Reviewing extracted relations'][0][1]['content']
        assert '! AGENT does not accept PERSON -> WORK_OF_ART' in prompt
    asyncio.run(run())


@pytest.mark.parametrize('stop', REVIEW_PASSES)
def test_interrupted_review_build_resumes_without_repeating_completed_llm_passes(tmp_path, stop):
    async def run():
        verdicts = [{'id': 1, 'verdict': 'keep'}, {'id': 2, 'verdict': 'keep'}]
        with GraphCheckpoints(tmp_path, {}) as store, pytest.raises(asyncio.CancelledError):
            await review_extractor(ReviewModel(verdicts, stop=stop), store).extract(chunks())
        resumed = ReviewModel(verdicts)
        with GraphCheckpoints(tmp_path, {}) as store:
            _, relations = await review_extractor(resumed, store).extract(chunks())
        assert resumed.calls == REVIEW_PASSES[REVIEW_PASSES.index(stop):]
        assert len(relations) == 4
    asyncio.run(run())
