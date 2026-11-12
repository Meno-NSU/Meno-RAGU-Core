import asyncio
import json

from graph_utils.checkpoints import CheckpointedExtractor, GraphCheckpoints
from graph_utils.graph_cleanup import CleanupInputs, clean_graph, fold_values, merge_variants
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


def test_a_value_the_relation_does_not_name_is_written_into_the_fact():
    work = entity('Control of pulse regime generation', 'WORK_OF_ART', 'Доклад о генерации импульсов.')
    year = entity('2019', 'DATE')
    entities, _, _ = fold_values([work, year], [relation(work, year, 'DATE_OF_CREATION', 'Дата публикации работы.')])
    assert entities[0].description == 'Доклад о генерации импульсов. Дата публикации работы: 2019.'


def test_an_initial_stands_for_the_given_name_then_the_patronymic():
    names = ['Evgenii Sergeevich Smirnov', 'S. Smirnov', 'E. S. Smirnov', 'Smirnov, E', 'Смирнов Сергей Захарович',
             'Захар Смирнов', 'С. З. Смирнов']
    people = [entity(name, 'PERSON', chunk=f'c{i}') for i, name in enumerate(names)]
    entities, _, _ = merge_variants(people, [])
    assert [e.entity_name for e in entities] == [
        'Evgenii Sergeevich Smirnov', 'S. Smirnov', 'Evgenii Sergeevich Smirnov', 'Evgenii Sergeevich Smirnov',
        'Смирнов Сергей Захарович', 'Захар Смирнов', 'Смирнов Сергей Захарович']


def test_artifacts_of_excluded_chunks_leave_the_graph_and_shared_ones_keep_their_other_chunks():
    from graph_utils.graph_cleanup import exclude_chunks
    shared = Entity('НГУ', 'ORGANIZATION', 'Университет.', ['export', 'page'])
    only_export = entity('ANPCyT', 'ORGANIZATION', chunk='export')
    funding = relation(only_export, shared, 'PARTICIPANT_IN', 'Фонд поддержал работу.', chunk='export')
    entities, relations, stats = exclude_chunks([shared, only_export], [funding], frozenset({'export'}))
    assert [(e.entity_name, e.source_chunk_id) for e in entities] == [('НГУ', ['page'])]
    assert entities[0].id == shared.id and relations == []
    assert stats == {'excluded_entities': 1, 'excluded_relations': 1, 'dangling_relations': 0}


def test_entities_labeled_as_values_or_junk_are_folded_like_values():
    article = entity('Статья о волновых пакетах', 'WORK_OF_ART', 'Статья.')
    doi = entity('10.1134/S0021364022330013', 'PRODUCT')
    email = entity('fit_dek@nsu.ru', 'LOCATION')
    dean = entity('Деканат ФИТ', 'ORGANIZATION', 'Деканат.')
    relations = [relation(article, doi, 'ALTERNATIVE_NAME', 'DOI статьи.'),
                 relation(dean, email, 'LOCATED_IN', 'Почта деканата fit_dek@nsu.ru.')]
    inputs = CleanupInputs(folded=frozenset({('10.1134/S0021364022330013', 'PRODUCT'), ('fit_dek@nsu.ru', 'LOCATION')}))
    entities, kept, stats = clean_graph([article, doi, email, dean], relations, inputs)
    assert [(e.entity_name, e.description) for e in entities] == [
        ('Статья о волновых пакетах', 'Статья. DOI статьи: 10.1134/S0021364022330013.'),
        ('Деканат ФИТ', 'Деканат. Почта деканата fit_dek@nsu.ru.')]
    assert kept == [] and stats['labeled_value_entities'] == 2


def test_merges_join_duplicates_into_the_target_and_retarget_relations():
    from graph_utils.graph_cleanup import apply_merges
    full = entity('Новосибирский государственный университет', 'ORGANIZATION', 'Университет.', chunk='c1')
    other = entity('Новосибирский национальный исследовательский государственный университет', 'ORGANIZATION', chunk='c2')
    person = entity('Иванов Иван Иванович', 'PERSON', chunk='c2')
    works = relation(person, other, 'WORKPLACE', 'Иванов работает в НГУ.', chunk='c2')
    alias = relation(other, full, 'ALTERNATIVE_NAME', 'Полное название.', chunk='c2')
    merges = ((('Новосибирский национальный исследовательский государственный университет', 'ORGANIZATION'),
               ('Новосибирский государственный университет', 'ORGANIZATION')),)
    entities, kept, stats = apply_merges([full, other, person], [works, alias], merges)
    assert {e.id for e in entities if e.entity_type == 'ORGANIZATION'} == {full.id}
    assert [(r.subject_name, r.object_name, r.object_id) for r in kept] == [
        ('Иванов Иван Иванович', 'Новосибирский государственный университет', full.id)]
    assert stats == {'merged_entities': 1, 'merge_self_loops_dropped': 1}


def test_cleanup_inputs_load_from_their_files(tmp_path):
    from graph_utils.graph_cleanup import load_cleanup_inputs
    (tmp_path / 'excluded.json').write_text('["c1", "c2"]', encoding='utf-8')
    (tmp_path / 'labels.jsonl').write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in [
        {'name': 'НГУ', 'type': 'ORGANIZATION', 'label': 'entity'},
        {'name': '1742-6588', 'type': 'PRODUCT', 'label': 'value'},
        {'name': 'Подробнее', 'type': 'PRODUCT', 'label': 'junk'}]), encoding='utf-8')
    (tmp_path / 'merges.jsonl').write_text(json.dumps({'from': ['ФФ НГУ', 'ORGANIZATION'],
                                                       'to': ['Физический факультет', 'ORGANIZATION']}, ensure_ascii=False),
                                           encoding='utf-8')
    inputs = load_cleanup_inputs(tmp_path / 'excluded.json', tmp_path / 'labels.jsonl', tmp_path / 'merges.jsonl')
    assert inputs.excluded_chunks == frozenset({'c1', 'c2'})
    assert inputs.folded == frozenset({('1742-6588', 'PRODUCT'), ('Подробнее', 'PRODUCT')})
    assert inputs.merges == ((('ФФ НГУ', 'ORGANIZATION'), ('Физический факультет', 'ORGANIZATION')),)
    assert load_cleanup_inputs(None, None, None) == CleanupInputs()


def test_a_refuted_alias_relation_keeps_its_text_with_the_unfit_mark():
    """ALTERNATIVE_NAME relations of graph v3 join e.g. «Новосибирск» and «Москва»; the pairs a model refuted are kept
    as UNFIT:ALTERNATIVE_NAME (their text stays), in either direction; other relations of the pair are left alone."""
    city = entity('Новосибирск', 'CITY')
    capital = entity('Москва', 'CITY')
    alias = relation(capital, city, 'ALTERNATIVE_NAME', 'Конференция прошла в Москве и Новосибирске.')
    near = relation(city, capital, 'LOCATED_IN', 'Связь городов.')
    marks = frozenset({(frozenset({('Новосибирск', 'CITY'), ('Москва', 'CITY')}), 'ALTERNATIVE_NAME')})
    _, kept, stats = clean_graph([city, capital], [alias, near], CleanupInputs(marked_relations=marks))
    assert [(r.relation_type, r.description) for r in kept] == [
        ('UNFIT:ALTERNATIVE_NAME', 'Конференция прошла в Москве и Новосибирске.'), ('LOCATED_IN', 'Связь городов.')]
    assert stats['marked_relations'] == 1


def test_relation_marks_name_the_nodes_the_merges_leave(tmp_path):
    from graph_utils.graph_cleanup import load_cleanup_inputs
    (tmp_path / 'merges.jsonl').write_text(json.dumps({'from': ['НГУ', 'ORGANIZATION'], 'to': [
        'Новосибирский государственный университет', 'ORGANIZATION']}, ensure_ascii=False) + '\n', encoding='utf-8')
    (tmp_path / 'marks.jsonl').write_text(json.dumps({'subject': ['НГУ', 'ORGANIZATION'], 'object': ['Москва', 'CITY'],
                                                      'relation_type': 'ALTERNATIVE_NAME'}, ensure_ascii=False) + '\n',
                                          encoding='utf-8')
    inputs = load_cleanup_inputs(None, None, tmp_path / 'merges.jsonl', tmp_path / 'marks.jsonl')
    assert inputs.marked_relations == frozenset({(frozenset({
        ('Новосибирский государственный университет', 'ORGANIZATION'), ('Москва', 'CITY')}), 'ALTERNATIVE_NAME')})


def test_inputs_find_nodes_whose_spelling_differs_only_in_punctuation():
    """Inputs of graph v4 name nodes as v4 spelled them ('Дексаметазон (DEX', its closing bracket cut by the old name
    trimming); the node of this build ('Дексаметазон (DEX)') is the same one, and keeps this build's spelling."""
    whole = entity('Дексаметазон (DEX)', 'PRODUCT', 'Глюкокортикоид.')
    short = entity('DEX', 'PRODUCT', 'Сокращение.')
    rat = entity('Крысы', 'ORGANISM', 'Животные.')
    given = relation(rat, short, 'UNFIT:AGENT', 'Крысам вводили DEX.')
    merges = ((('DEX', 'PRODUCT'), ('Дексаметазон (DEX', 'PRODUCT')),)
    entities, kept, _ = clean_graph([whole, short, rat], [given], CleanupInputs(merges=merges))
    assert {(e.entity_name, e.id) for e in entities} == {('Дексаметазон (DEX)', whole.id), ('Крысы', rat.id)}
    assert [(r.subject_id, r.object_id) for r in kept] == [(rat.id, whole.id)]
    folded = CleanupInputs(folded=frozenset({('Дексаметазон (DEX', 'PRODUCT')}))
    entities, _, _ = clean_graph([whole, rat], [relation(rat, whole, 'UNFIT:AGENT', 'Крысам вводили DEX.')], folded)
    assert [e.entity_name for e in entities] == ['Крысы']
    _, kept, _ = clean_graph([whole, rat], [relation(rat, whole, 'AGENT', 'Крысам вводили DEX.')],
                             CleanupInputs(marked_relations=frozenset({(frozenset({
                                 ('Крысы', 'ORGANISM'), ('Дексаметазон (DEX', 'PRODUCT')}), 'AGENT')})))
    assert [r.relation_type for r in kept] == ['UNFIT:AGENT']


def test_labeled_entities_take_their_subtype_and_relations_follow():
    chair = entity('Кафедра общей физики', 'ORGANIZATION', 'Кафедра ФФ.')
    person = entity('Иванов Иван Иванович', 'PERSON', 'Доцент.')
    fly = entity('Дрозофила', 'PRODUCT', 'Муха.', chunk='c1')
    fly_too = entity('Дрозофила', 'ORGANIZATION', 'Модельный организм.', chunk='c2')
    works = relation(person, chair, 'WORKPLACE', 'Иванов работает на кафедре общей физики.')
    subtypes = ((('Кафедра общей физики', 'ORGANIZATION'), 'DEPARTMENT'), (('Дрозофила', 'PRODUCT'), 'ORGANISM'),
                (('Дрозофила', 'ORGANIZATION'), 'ORGANISM'), (('Иванов Иван Иванович', 'PERSON'), 'PERSON'))
    entities, kept, stats = clean_graph([chair, person, fly, fly_too], [works], CleanupInputs(subtypes=subtypes))
    assert [(e.entity_name, e.entity_type) for e in entities] == [
        ('Кафедра общей физики', 'DEPARTMENT'), ('Иванов Иван Иванович', 'PERSON'), ('Дрозофила', 'ORGANISM'),
        ('Дрозофила', 'ORGANISM')]
    assert entities[2].id == entities[3].id  # one node now: the build joins mentions of one id
    assert [(r.object_name, r.object_id) for r in kept] == [('Кафедра общей физики', entities[0].id)]
    assert stats['subtyped_entities'] == 3


def test_relations_take_their_new_type_and_direction_and_keep_their_text():
    """Retypes name the relation by its two ends and its type in the graph they were made on (before the subtypes):
    'UNFIT:AGENT' from an article to its author becomes AUTHOR_OF from the author to the article."""
    article = entity('Эволюция волновых пакетов', 'WORK_OF_ART', 'Статья.')
    journal = entity('Журнал ПМТФ', 'WORK_OF_ART', 'Журнал.')
    author = entity('Иванов Иван Иванович', 'PERSON', 'Физик.')
    rector = entity('Ректор', 'PROFESSION', 'Должность.')
    relations = [relation(article, journal, 'PART_OF', 'Статья опубликована в журнале ПМТФ.'),
                 relation(article, author, 'UNFIT:AGENT', 'Иванов — автор статьи.'),
                 relation(author, rector, 'UNFIT:WORKS_AS', 'Иванов подписал приказ ректора.')]
    retypes = (((('Эволюция волновых пакетов', 'WORK_OF_ART'), ('Журнал ПМТФ', 'WORK_OF_ART'), 'PART_OF'),
                ('PUBLISHED_IN', False)),
               ((('Эволюция волновых пакетов', 'WORK_OF_ART'), ('Иванов Иван Иванович', 'PERSON'), 'UNFIT:AGENT'),
                ('AUTHOR_OF', True)),
               ((('Иванов Иван Иванович', 'PERSON'), ('Ректор', 'PROFESSION'), 'UNFIT:WORKS_AS'), ('RELATED_TO', False)))
    _, kept, stats = clean_graph([article, journal, author, rector], relations, CleanupInputs(retypes=retypes))
    assert [(r.subject_name, r.relation_type, r.object_name, r.description) for r in kept] == [
        ('Эволюция волновых пакетов', 'PUBLISHED_IN', 'Журнал ПМТФ', 'Статья опубликована в журнале ПМТФ.'),
        ('Иванов Иван Иванович', 'AUTHOR_OF', 'Эволюция волновых пакетов', 'Иванов — автор статьи.'),
        ('Иванов Иван Иванович', 'RELATED_TO', 'Ректор', 'Иванов подписал приказ ректора.')]
    assert [(r.subject_id, r.object_id) for r in kept][1] == (author.id, article.id)
    assert stats['retyped_relations'] == 3 and stats['reversed_relations'] == 1


def test_nodes_that_are_not_objects_fold_into_their_neighbours():
    """A role ('Ректор'), an identifier, a vague phrase or a garbled name is not an object to find: what its relations
    say goes into the description of the other end, as values do, and the node goes."""
    person = entity('Федорук Михаил Петрович', 'PERSON', 'Учёный.')
    rector = entity('Ректор', 'PROFESSION', 'Руководитель вуза.')
    room = entity('Ауд. 313', 'FACILITY', 'Аудитория.')
    field = entity('Нелинейная оптика', 'CONCEPT', 'Раздел физики.')
    relations = [relation(person, rector, 'WORKS_AS', 'Михаил Федорук — ректор НГУ.'),
                 relation(person, room, 'LOCATED_IN', 'Федорук принимает в ауд. 313.'),
                 relation(person, field, 'RESEARCHES', 'Федорук занимается нелинейной оптикой.')]
    inputs = CleanupInputs(not_objects=frozenset({('Ректор', 'PROFESSION'), ('Ауд. 313', 'FACILITY')}))
    entities, kept, stats = clean_graph([person, rector, room, field], relations, inputs)
    assert [(e.entity_name, e.description) for e in entities] == [
        ('Федорук Михаил Петрович', 'Учёный. Михаил Федорук — ректор НГУ. Федорук принимает в ауд. 313.'),
        ('Нелинейная оптика', 'Раздел физики.')]
    assert [r.relation_type for r in kept] == ['RESEARCHES']
    assert stats['not_object_value_entities'] == 2


def test_v5_inputs_load_from_their_files(tmp_path):
    from graph_utils.graph_cleanup import load_cleanup_inputs
    (tmp_path / 'merges.jsonl').write_text(json.dumps({'from': ['НГУ', 'ORGANIZATION'], 'to': [
        'Новосибирский государственный университет', 'ORGANIZATION']}, ensure_ascii=False) + '\n', encoding='utf-8')
    (tmp_path / 'subtypes.jsonl').write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in [
        {'name': 'НГУ', 'type': 'ORGANIZATION', 'subtype': 'UNIVERSITY', 'kind': 'named', 'model': 'm', 'prompt': 'p'},
        {'name': 'Ректор', 'type': 'PROFESSION', 'subtype': 'POSITION', 'kind': 'role', 'model': 'm', 'prompt': 'p'},
        {'name': 'Рутил', 'type': 'PRODUCT', 'subtype': 'MATERIAL', 'kind': 'concept', 'model': 'm', 'prompt': 'p'},
    ]) + '\n', encoding='utf-8')
    (tmp_path / 'retypes.jsonl').write_text(json.dumps({
        'subject': ['Статья', 'WORK_OF_ART'], 'object': ['НГУ', 'ORGANIZATION'], 'original': 'UNFIT:AGENT',
        'type': 'RELATED_TO', 'reverse': False, 'model': 'm', 'prompt': 'p'}, ensure_ascii=False) + '\n', encoding='utf-8')
    inputs = load_cleanup_inputs(merges_path=tmp_path / 'merges.jsonl', subtypes_path=tmp_path / 'subtypes.jsonl',
                                 retypes_path=tmp_path / 'retypes.jsonl')
    university = ('Новосибирский государственный университет', 'ORGANIZATION')
    assert inputs.subtypes == ((university, 'UNIVERSITY'), (('Рутил', 'PRODUCT'), 'MATERIAL'))
    assert inputs.not_objects == frozenset({('Ректор', 'PROFESSION')})
    assert inputs.retypes == (((('Статья', 'WORK_OF_ART'), university, 'UNFIT:AGENT'), ('RELATED_TO', False)),)
