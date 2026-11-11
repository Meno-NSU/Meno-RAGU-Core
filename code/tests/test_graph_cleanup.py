import asyncio

from graph_utils.checkpoints import CheckpointedExtractor, GraphCheckpoints
from graph_utils.graph_cleanup import clean_graph, fold_values, merge_variants
from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import EntitiesExtractionModel, RelationsExtractionModel
from ragu.graph.types import Entity, Relation
from ragu.models.llm import LLM


def entity(name, kind, description='', chunk='c1'):
    return Entity(entity_name=name, entity_type=kind, description=description or f'{name} в тексте',
                  source_chunk_id=[chunk])


def relation(subject, obj, kind, description, chunk='c1'):
    return Relation(subject_id=subject.id, object_id=obj.id, subject_name=subject.entity_name,
                    object_name=obj.entity_name, relation_type=kind, description=description,
                    relation_strength=5, source_chunk_id=[chunk])


def test_values_become_part_of_the_description_of_what_they_describe():
    article = entity('Эволюция волновых пакетов', 'WORK_OF_ART', 'Статья о волновых пакетах.')
    year = entity('2015', 'DATE')
    pages = entity('11 стр.', 'NUMBER')
    other = entity('Иванов Иван Иванович', 'PERSON')
    relations = [
        relation(article, year, 'DATE_OF_CREATION', 'Статья опубликована в 2015 году.'),
        relation(article, pages, 'UNFIT:INANIMATE_INVOLVED', 'Объём статьи 11 страниц.'),
        relation(year, pages, 'UNFIT:PART_OF', 'Связь числа и даты.'),
        relation(other, article, 'PRODUCES', 'Иванов написал статью.'),
    ]
    entities, kept, stats = fold_values([article, year, pages, other], relations)
    assert [e.entity_name for e in entities] == ['Эволюция волновых пакетов', 'Иванов Иван Иванович']
    assert entities[0].description == 'Статья о волновых пакетах. Статья опубликована в 2015 году. Объём статьи 11 страниц.'
    assert entities[0].id == article.id
    assert [r.description for r in kept] == ['Иванов написал статью.']
    assert stats == {'value_entities': 2, 'folded_relations': 2, 'value_to_value_relations': 1}


def test_a_value_fact_goes_only_to_the_mention_of_the_same_chunk():
    first = entity('НГУ', 'ORGANIZATION', 'Университет.', chunk='c1')
    second = entity('НГУ', 'ORGANIZATION', 'Университет в Академгородке.', chunk='c2')
    year = entity('1959', 'DATE', chunk='c2')
    entities, _, _ = fold_values([first, second, year], [relation(second, year, 'DATE_FOUNDED_IN', 'НГУ основан в 1959 году.', chunk='c2')])
    assert [e.description for e in entities] == ['Университет.', 'Университет в Академгородке. НГУ основан в 1959 году.']


def test_one_name_under_several_types_takes_the_most_frequent_type_but_people_keep_theirs():
    flies = [entity('Дрозофила', 'PRODUCT', chunk=f'c{i}') for i in range(3)] + [entity('Дрозофила', 'ORGANIZATION', chunk='c9')]
    person = [entity('Лаврентьев', 'PERSON')] + [entity('Лаврентьев', 'ORGANIZATION', chunk=f'd{i}') for i in range(2)]
    course = entity('Генетика', 'WORK_OF_ART')
    relations = [relation(course, flies[3], 'UNFIT:INANIMATE_INVOLVED', 'Курс изучает дрозофилу.', chunk='c9')]
    entities, kept, stats = merge_variants(flies + person + [course], relations)
    assert sorted({(e.entity_name, e.entity_type) for e in entities}) == [
        ('Генетика', 'WORK_OF_ART'), ('Дрозофила', 'PRODUCT'), ('Лаврентьев', 'ORGANIZATION'), ('Лаврентьев', 'PERSON')]
    assert kept[0].object_id == flies[0].id and kept[0].object_name == 'Дрозофила'
    assert stats['retyped_entities'] == 1


def test_variants_of_a_person_name_join_the_only_full_name_they_fit():
    names = ['Мулляджанов Рустам Илхамович', 'Мулляджанов Рустам Илхамович', 'Рустам Илхамович Мулляджанов',
             'Мулляджанов Р.И.', 'Р. И. Мулляджанов', 'Рустам Мулляджанов', 'Мулляджанов',
             'Иванов Иван Иванович', 'Иванов Илья Игоревич', 'Иванов И.И.', 'Иванов И.']
    people = [entity(name, 'PERSON', chunk=f'c{i}') for i, name in enumerate(names)]
    alias = relation(people[3], people[0], 'ALTERNATIVE_NAME', 'Сокращение имени.', chunk='c3')
    lab = entity('Лаборатория прикладных цифровых технологий', 'ORGANIZATION', chunk='c5')
    works = relation(people[5], lab, 'WORKPLACE', 'Мулляджанов заведует лабораторией.', chunk='c5')
    entities, kept, stats = merge_variants([*people, lab], [alias, works])
    assert [e.entity_name for e in entities if e.entity_type == 'PERSON'] == [
        'Мулляджанов Рустам Илхамович'] * 6 + ['Мулляджанов', 'Иванов Иван Иванович', 'Иванов Илья Игоревич',
                                               'Иванов И.И.', 'Иванов И.']
    assert [(r.subject_name, r.object_name) for r in kept] == [('Мулляджанов Рустам Илхамович',
                                                                 'Лаборатория прикладных цифровых технологий')]
    assert kept[0].subject_id == people[0].id
    assert stats['renamed_entities'] == 4 and stats['self_loops_dropped'] == 1


def test_clean_graph_folds_values_before_merging_names():
    article = entity('Статья', 'WORK_OF_ART')
    year = entity('2015', 'DATE')
    as_number = entity('2015', 'NUMBER', chunk='c2')
    entities, relations, stats = clean_graph([article, year, as_number], [relation(article, year, 'DATE_OF_CREATION', 'Опубликована в 2015.')])
    assert [e.entity_name for e in entities] == ['Статья'] and relations == []
    assert stats['value_entities'] == 2


class Model(LLM):
    async def chat_completion(self, conversation, **kwargs):
        raise AssertionError('Unexpected unbatched LLM call')

    async def batch_chat_completion(self, conversations, output_schema, desc=None, **kwargs):
        if output_schema is EntitiesExtractionModel:
            return [EntitiesExtractionModel(entities=[
                {'entity_name': 'Иванов Иван Иванович', 'entity_type': 'PERSON', 'description': 'Преподаватель НГУ.'},
                {'entity_name': '2019', 'entity_type': 'DATE', 'description': 'Год.'},
            ]) for _ in conversations]
        return [RelationsExtractionModel(relations=[{
            'source_entity': 'Иванов Иван Иванович', 'target_entity': '2019', 'relation_type': 'START_TIME',
            'description': 'Иванов работает в НГУ с 2019 года.', 'relationship_strength': 5,
        }]) for _ in conversations]


def test_the_cleanup_shapes_the_saved_artifacts_when_asked(tmp_path):
    async def run(cleanup):
        extractor = CheckpointedExtractor(llm=Model(), ontology=None, do_entity_validation=False,
                                          do_relation_validation=False, graph_cleanup=cleanup)
        with GraphCheckpoints(tmp_path / str(cleanup), {}) as store:
            extractor.checkpoints = store
            return await extractor.extract([Chunk('Иванов работает в НГУ с 2019 года.', 0, 'doc_0')])

    entities, relations = asyncio.run(run(True))
    assert [(e.entity_name, e.description) for e in entities] == [
        ('Иванов Иван Иванович', 'Преподаватель НГУ. Иванов работает в НГУ с 2019 года.')]
    assert relations == []
    entities, relations = asyncio.run(run(False))
    assert len(entities) == 2 and len(relations) == 1
