import io

from graph_utils import graph_patch as gp


def node(name, kind='ORGANIZATION', chunks=('c1',), description=''):
    return gp.Node(gp.entity_id(name, kind), name, kind, description or f'{name}.', list(chunks),
                   {'documents_id': '"[]"', 'clusters': '"[]"'})


def edge(source, target, kind, description, chunks=('c1',), strength=5.0):
    return gp.Edge(gp.relation_id(source.id, target.id, kind), source.id, target.id, kind, description, strength, list(chunks))


def graph(nodes, edges=()):
    return gp.Graph({n.id: n for n in nodes}, list(edges))


def names(g):
    return sorted(n.name for n in g.nodes.values())


def test_gml_round_trip_keeps_the_stand_format(tmp_path):
    a, b = node('НГУ «Наука»', 'UNIVERSITY', ['c1']), node('Иванов & Co', 'ENTERPRISE', ['c1', 'c2'])
    g = graph([a, b], [edge(b, a, 'PART_OF', 'Часть "вуза".', ['c2'], 4.75)])
    path = tmp_path / 'g.gml'
    gp.write_gml(g, path)
    text = path.read_text()
    assert text.isascii() and '&#34;' in text and '&#38;' in text  # networkx escapes quotes, & and non-ASCII
    assert 'source_chunk_id "_networkx_list_start"\n    source_chunk_id "c1"' in text  # a list of one item
    assert 'relation_strength 4.75' in text and 'documents_id "[]"' in text
    assert gp.read_gml(path) == g
    import networkx as nx
    loaded = nx.read_gml(path)
    assert loaded.nodes[a.id]['source_chunk_id'] == ['c1'] and loaded.nodes[b.id]['entity_name'] == 'Иванов & Co'
    assert list(loaded.edges(keys=True)) == [(b.id, a.id, g.edges[0].id)]


def test_control_characters_survive_the_file(tmp_path):
    g = graph([node('Мезоны $D^0\x08ar{D}^0$ &amp; <b>', 'CONCEPT')])
    gp.write_gml(g, tmp_path / 'g.gml')
    assert gp.read_gml(tmp_path / 'g.gml') == g


def test_a_rename_may_take_the_id_another_renamed_node_leaves():
    a, b = node('Институт Новосибирский государственный университет', 'FACULTY'), node('Институт НГУ', 'FACULTY')
    patched, _ = gp.rename_nodes(graph([a, b]), {a.id: ('Институт НГУ', []), b.id: ('Институт Новосибирского университета', [])}, 's', 'r')
    assert all(n.id == gp.entity_id(n.name, n.type) for n in patched.nodes.values()) and len(patched.nodes) == 2


def test_ids_are_the_ids_ragu_gives():
    assert gp.entity_id('"Circassian" helmets with plated mailaventails of the late middle ages and early Modern times',
                        'PUBLICATION') == 'ent-7d7ddc1f0306ec8275e646144c5b46fb'


PAIRS = gp.expansions({'НГУ': 'Новосибирский государственный университет', 'ИФП': 'Институт философии и права',
                       'СО': 'Сибирское отделение', 'ИК': ['Институт Конфуция', 'Институт катализа'], 'ИИ': 'ИИ'})


def test_a_name_takes_back_the_abbreviation_its_sources_have():
    texts = {'c1': 'Запись в Спорткомплекс НГУ открыта.'}
    name, evidence, replaced = gp.source_form('Спорткомплекс Новосибирский государственный университет', texts, PAIRS)
    assert name == 'Спорткомплекс НГУ' and evidence == ['c1']
    assert replaced == [('НГУ', 'Новосибирский государственный университет')]


def test_a_name_whose_source_states_the_expansion_stays():
    texts = {'c1': 'ИФП СО РАН.', 'c2': 'Институт философии и права основан в 1990 году.'}
    name, evidence, _ = gp.source_form('Институт философии и права Сибирское отделение РАН', texts, PAIRS)
    assert name == 'Институт философии и права СО РАН'  # only the expansion no source states goes
    assert gp.source_form('Институт философии и права', texts, PAIRS) is None


def test_an_expansion_that_rewrites_no_abbreviation_of_the_sources_stays():
    assert gp.source_form('Институт Конфуция', {'c1': 'Курсы китайского языка.'}, PAIRS) is None
    assert gp.source_form('Институт Конфуция', {'c1': 'Работает в ИКТ.'}, PAIRS) is None  # ИК inside another word


def test_renamed_nodes_of_one_key_merge_and_relations_follow():
    wrong = node('Столовая Новосибирский государственный университет', 'FACILITY', ['c1'], 'Работает до 17:00.')
    right = node('Столовая НГУ', 'FACILITY', ['c2', 'c3'], 'Столовая университета.')
    nsu = node('НГУ', 'UNIVERSITY', ['c1', 'c2'])
    g = graph([wrong, right, nsu], [edge(wrong, nsu, 'PART_OF', 'Столовая входит в НГУ.', ['c1']),
                                    edge(right, nsu, 'PART_OF', 'Столовая находится в университете.', ['c2'])])
    texts = {'c1': 'Столовая НГУ работает до 17:00.', 'c2': 'Столовая НГУ.', 'c3': ''}
    renames, _ = gp.source_form_renames(g, texts.get, PAIRS)
    assert renames == {wrong.id: ('Столовая НГУ', ['c1'])}
    patched, changes = gp.rename_nodes(g, renames, 'source_form', 'the source form of the name')
    assert names(patched) == ['НГУ', 'Столовая НГУ']
    canteen = patched.nodes[right.id]
    assert canteen.chunks == ['c2', 'c3', 'c1'] and canteen.description == 'Столовая университета. Работает до 17:00.'
    (relation,) = patched.edges
    assert relation.chunks == ['c1', 'c2'] and relation.description == 'Столовая входит в НГУ.\nСтоловая находится в университете.'
    assert sorted(c['op'] for c in changes) == ['edge_merge', 'edge_rekey', 'merge', 'rename']
    assert changes[0]['evidence_chunks'] == ['c1']
    assert gp.unstated_expansions(patched, texts.get, PAIRS) == []
    assert gp.id_map(changes) == {wrong.id: right.id}


def test_a_renamed_node_alone_takes_the_id_of_its_new_name():
    wrong, nsu = node('Музей Новосибирский государственный университет', 'FACILITY'), node('НГУ', 'UNIVERSITY')
    g = graph([wrong, nsu], [edge(wrong, nsu, 'PART_OF', 'Музей НГУ.')])
    patched, _ = gp.rename_nodes(g, {wrong.id: ('Музей НГУ', ['c1'])}, 's', 'r')
    new_id = gp.entity_id('Музей НГУ', 'FACILITY')
    assert list(patched.nodes) == [new_id, nsu.id]
    patched, changes = gp.rename_nodes(g, {wrong.id: ('Музей НГУ', ['c1'])}, 's', 'r')
    assert patched.edges[0].source == new_id and patched.edges[0].id == gp.relation_id(new_id, nsu.id, 'PART_OF')
    assert gp.id_map(changes) == {wrong.id: new_id}


def test_long_forms_the_corpus_states_show_an_ambiguous_abbreviation():
    texts = [(i, 'Сотрудник Института физики полупроводников (ИФП) СО РАН.') for i in range(3)]
    texts += [(i, 'Студент Института философии и права (ИФП) НГУ.') for i in range(3, 6)]
    texts += [(9, 'Институт физических проблем (ИФП).'), (10, 'по словам Ивана Фёдоровича Петрова (ИФП)')]
    forms = gp.stated_long_forms(texts, ['ИФП', 'НГУ'])
    assert forms == {'ИФП': {'институ физи полупроводник': {'documents': 3, 'example': 'Института физики полупроводников'},
                             'институ философ и прав': {'documents': 3, 'example': 'Института философии и права'}}}


def test_readings_and_full_names():
    assert gp.readings('Федорук Михаил Петрович') == gp.readings('Михаил Петрович Федорук') == [('федорук', 'михаил', 'петрович')]
    assert gp.readings('М. П. Федорук') == gp.readings('Федорук М.П.') == [('федорук', 'м', 'п')]
    assert gp.readings('Михаил Федорук') == [('федорук', 'михаил', ''), ('михаил', 'федорук', '')]
    assert gp.readings('Жанна Владимировна') == [] and gp.readings('Федорук') == []
    assert gp.full_name('Михаил Федорук') is None and gp.full_name('Федорук М. П.') is None


def test_full_names_in_another_order_merge():
    one = node('Лапин Василий Николаевич', 'PERSON', ['c1', 'c2', 'c3'])
    two = node('Василий Николаевич Лапин', 'PERSON', ['c4'])
    other = node('Лапин Василий Петрович', 'PERSON', ['c5'])
    g = graph([one, two, other])
    merges = gp.full_name_merges(g)
    assert [(m[0], m[1]) for m in merges] == [(two.id, one.id)]
    patched, changes = gp.merge_nodes(g, merges, 'people')
    assert names(patched) == ['Лапин Василий Николаевич', 'Лапин Василий Петрович']
    assert changes[0]['op'] == 'merge' and changes[0]['from_name'] == 'Василий Николаевич Лапин'


def test_a_short_name_joins_a_full_one_only_through_a_shared_document():
    full = node('Михаил Петрович Федорук', 'PERSON', ['a1', 'a2'])
    short = node('Михаил Федорук', 'PERSON', ['a3'])
    elsewhere = node('М. Федорук', 'PERSON', ['z1'])
    document = {'a1': 'A', 'a2': 'A', 'a3': 'A', 'z1': 'Z'}.get
    g = graph([full, short, elsewhere])
    merges = gp.short_name_merges(g, document)
    assert [(m[0], m[1], m[3]) for m in merges] == [(short.id, full.id, ['a3'])]


def test_a_short_name_two_people_of_the_document_fit_stays():
    one, two = node('Кузнецов Владимир Васильевич', 'PERSON', ['a1']), node('Кузнецов Виктор Владимирович', 'PERSON', ['a2'])
    short, same = node('Кузнецов В. В.', 'PERSON', ['a3']), node('В. В. Кузнецов', 'PERSON', ['a4'])
    g = graph([one, two, short, same])
    merges = gp.short_name_merges(g, lambda chunk: 'A')
    assert all(m[1] not in (one.id, two.id) for m in merges)  # neither full name takes them
    assert len(merges) == 1 and {merges[0][0], merges[0][1]} == {short.id, same.id}  # the same initials in another order


def test_a_short_name_and_initials_of_another_kind_do_not_merge():
    g = graph([node('Иван Петров', 'PERSON', ['a1']), node('Петров И. С.', 'PERSON', ['a2'])])
    assert gp.short_name_merges(g, lambda chunk: 'A') == []


def test_profile_title_and_key():
    assert gp.title_person('Профиль исследователя Сычёва Александра Анатольевича') == ('сычева', 'александра', 'анатольевича')
    assert gp.title_person('Список научных публикаций Ивана Викторовича Литвинова') == ('литвинова', 'ивана', 'викторовича')
    assert gp.title_person('Кафедра высшей математики') is None
    assert gp.title_person('Иванов Иван Иванович и Петров Пётр Петрович') is None
    url = 'https://pure.nsu.ru/portal/ru/persons/--(7f45bfae-123d-4356-9861-340a279a7ede)/publications.html?page=0'
    assert gp.profile_key(url, 'Список научных публикаций Ивана Викторовича Литвинова') == 'pure.nsu.ru:7f45bfae-123d-4356-9861-340a279a7ede'
    assert gp.profile_key('https://www.nsu.ru/n/sveden/employees/#Иванов_Иван', 'Иванов Иван Иванович — педагогический состав НГУ') \
        == 'https://www.nsu.ru/n/sveden/employees/#Иванов_Иван'
    assert gp.profile_key(url, 'Новости портала') is None


def test_one_profile_is_one_node_and_a_misspelled_surname_joins_it():
    owner = node('Павел Игоревич Кадильников', 'PERSON', ['p1', 'p2', 'x1'])
    again = node('Кадильникова Павел Игоревич', 'PERSON', ['p3'])  # the case form of a title taken for a name
    short = node('Кадильников П. И.', 'PERSON', ['p5'])
    typo = node('Павел Игоревич Кадиников', 'PERSON', ['p4'])
    coauthor = node('Иван Юрьевич Кулаков', 'PERSON', ['p1'])
    namesake = node('Павел Игоревич Кадильников', 'PERSON', ['q1'])
    g = graph([owner, again, typo, coauthor, short])
    document = lambda chunk: 'profile' if chunk.startswith('p') else 'other'
    profile = lambda doc: ('pure:1', ('кадильникова', 'павла', 'игоревича')) if doc == 'profile' else None
    owners, misspelled = gp.profile_owners(g, document, profile)
    assert owners == {'pure:1': {owner.id: ['p1', 'p2'], again.id: ['p3']}} and misspelled == {'pure:1': {typo.id: ['p4']}}
    merges = gp.profile_merges(g, owners, misspelled)
    texts = {'p3': 'Кадильникова П. И.', 'p1': 'Список публикаций П. И. Кадильникова', 'p4': 'Кадиников', 'p2': 'Кадильников П. И., Кадильников'}
    assert {m[1] for m in gp.profile_merges(g, owners, misspelled, texts.get)} == {owner.id}  # the spelling the sources use most
    assert sorted((m[0], m[1]) for m in merges) == sorted([(again.id, owner.id), (typo.id, owner.id)])
    patched, _ = gp.merge_nodes(g, merges, 'profiles')
    assert names(patched) == ['Иван Юрьевич Кулаков', 'Кадильников П. И.', 'Павел Игоревич Кадильников']
    assert [(m[0], m[1]) for m in gp.short_name_merges(patched, document)] == [(short.id, owner.id)]
    owners, misspelled = gp.profile_owners(patched, document, profile)
    assert [len(nodes) for nodes in owners.values()] == [1] and not misspelled and namesake.id == owner.id


def test_a_woman_of_the_same_surname_does_not_own_a_mans_profile():
    assert not gp.owns(('вишневская', 'ирина', 'александровна'), ('вишневского', 'андрея', 'владиславовича'))
    assert gp.owns(('вишневский', 'а', 'в'), ('вишневского', 'андрея', 'владиславовича'))
    assert not gp.owns(('вишневский', 'а', 'е'), ('вишневского', 'андрея', 'владиславовича'))
    assert not gp.owns(('верниковский', 'антон', 'евгеньевич'), ('верниковская', 'антонина', 'евгеньевна'))
    assert not gp.owns(('кузнецов', 'николай', 'александрович'), ('кузнецова', 'никиты', 'александровича'))
    assert not gp.owns(('иванов', 'алексей', 'владимирович'), ('иванова', 'александра', 'владиславовича'))
    assert gp.owns(('кравчук', 'артем', 'виталевич'), ('кравчук', 'артем', 'витальевич')) and gp.owns(('ян', 'павел', 'а'), ('яна', 'павла', 'александровича'))
    assert gp.owns(('верниковская', 'антонина', 'евгеньевна'), ('верниковской', 'антонины', 'евгеньевны'))


def test_symmetric_relations_fold_and_a_repeating_related_to_goes():
    a, b, c = node('А А', 'PERSON'), node('Б Б', 'PERSON'), node('НГУ', 'UNIVERSITY')
    first, second = sorted([a, b], key=lambda n: n.id)
    g = graph([a, b, c], [edge(first, second, 'KNOWS', 'Соавторы статьи.', ['c1']),
                          edge(second, first, 'KNOWS', 'Соавторы статьи.', ['c2'], 7.0),
                          edge(second, first, 'SUPERVISES', 'Руководитель.', ['c2']),
                          edge(a, c, 'WORKPLACE', 'Работает в НГУ.', ['c1']),
                          edge(a, c, 'RELATED_TO', 'работает в НГУ', ['c3']),
                          edge(c, a, 'RELATED_TO', 'Получил благодарность ректора.', ['c1'])])
    patched, changes = gp.fold_relations(g, frozenset({'KNOWS'}))
    assert sorted(e.type for e in patched.edges) == ['KNOWS', 'RELATED_TO', 'SUPERVISES', 'WORKPLACE']
    assert next(e for e in patched.edges if e.type == 'SUPERVISES').source == second.id  # not a symmetric type
    knows = next(e for e in patched.edges if e.type == 'KNOWS')
    assert (knows.source, knows.target) == (first.id, second.id)
    assert knows.chunks == ['c1', 'c2'] and knows.strength == 7.0 and knows.description == 'Соавторы статьи.'
    assert next(e for e in patched.edges if e.type == 'WORKPLACE').chunks == ['c1', 'c3']
    assert next(e for e in patched.edges if e.type == 'RELATED_TO').description == 'Получил благодарность ректора.'
    assert sorted(c['op'] for c in changes) == ['edge_drop', 'edge_merge', 'edge_reorder']


def test_only_relations_supported_by_a_template_line_become_metadata():
    pure, conference, author = node('Pure', 'SOFTWARE'), node('МНСК', 'CONFERENCE'), node('Ачасов М. Н.', 'PERSON')
    people = [node(f'{name} И. И.', 'PERSON') for name in ('Адамов', 'Белов', 'Волков', 'Гусев', 'Дроздов')]
    paper, department = node('Статья о мезонах', 'PUBLICATION'), node('Кафедра физики', 'DEPARTMENT')
    texts = {f'p{i}': f'Профиль: {p.name}\nID: {i}00\nВход в Pure' for i, p in enumerate(people)}
    texts['own'] = 'Адамов И. И. настроил Pure для факультета.'
    texts.update({f'm{i}': f'Доклад на МНСК: {p.name}' for i, p in enumerate(people)})
    # the header of a person's own page is in every part of it, yet ties the person to one department only
    texts.update({f'a{i}': 'Ачасов М. Н., место работы\n\nСведения\nКафедра физики\nID 5\nСтатья о мезонах' for i in range(5)})
    # the same list of authors under many titles: each title with its authors is a statement of its own
    papers = [node(f'Распад мезона {name}', 'PUBLICATION') for name in ('альфа', 'бета', 'гамма', 'дельта', 'эпсилон')]
    texts.update({f't{i}': f'Список\n## {p.name.replace("мезона", "<i>мезона</i>")}:\nАблиkим М., Ачасов М. Н., Ахмед С.\nЖурнал' for i, p in enumerate(papers)})
    edges = [edge(p, pure, 'USES', 'Имеет профиль в Pure.', [f'p{i}']) for i, p in enumerate(people)]
    edges += [edge(p, conference, 'PARTICIPANT_IN', 'Участник МНСК.', [f'm{i}']) for i, p in enumerate(people)]
    edges += [edge(people[0], pure, 'PRODUCES', 'Настроил Pure.', ['own', 'p0']),
              edge(author, department, 'WORKPLACE', 'Работает на кафедре.', [f'a{i}' for i in range(5)]),
              edge(author, paper, 'AUTHOR_OF', 'Автор статьи.', ['a0']),
              edge(paper, pure, 'RELATED_TO', 'Нет в источнике.', ['a0'])]
    edges += [edge(author, p, 'AUTHOR_OF', 'Соавтор статьи.', [f't{i}']) for i, p in enumerate(papers)]
    g = graph([pure, conference, author, paper, department] + people + papers, edges)
    hubs = {pure.id, conference.id, author.id}
    forms = {pure.id: ['Pure'], conference.id: ['МНСК'], author.id: ['Ачасов М. Н.', 'Ачасов']}
    other_forms = lambda node_id: [g.nodes[node_id].name]
    found = gp.template_relations(g, hubs, texts.get, forms, other_forms, min_others=5)
    assert [(g.edges[index].type, hub) for index, hub, _ in found] == [('USES', pure.id)] * 5
    assert gp.template_relations(g, hubs, texts.get, forms, other_forms, min_others=6) == []
    patched, changes, metadata = gp.relations_to_metadata(g, found)
    assert collections_counter(e.type for e in patched.edges) == {'PARTICIPANT_IN': 5, 'PRODUCES': 1, 'WORKPLACE': 1,
                                                                  'AUTHOR_OF': 6, 'RELATED_TO': 1}
    assert metadata[people[1].id] == [{'hub': pure.id, 'hub_name': 'Pure', 'relation_type': 'USES', 'direction': 'to_hub',
                                       'description': 'Имеет профиль в Pure.', 'chunks': ['p1']}]
    assert changes[0]['op'] == 'edge_to_metadata' and changes[0]['evidence_chunks'] == ['p0']


def collections_counter(items):
    import collections
    return dict(collections.Counter(items))


def test_the_pages_of_one_profile_key_keep_one_person():
    profiles = {1: ('k', ('хлуднева', 'александра', 'михайловича')), 2: ('k', ('хлуднев', 'александр', 'михайлович')),
                3: ('k', ('аннина', 'бориса', 'дмитриевича')), 4: ('z', ('аннин', 'борис', 'дмитриевич'))}
    assert sorted(gp.consistent_profiles(profiles)) == [1, 2, 4]


def test_a_node_with_a_profile_of_its_own_joins_no_other_name():
    full, short = node('Андрей Викторович Иванов', 'PERSON', ['a1']), node('А. Иванов', 'PERSON', ['a2'])
    g = graph([full, short])
    assert len(gp.short_name_merges(g, lambda chunk: 'A')) == 1
    assert gp.short_name_merges(g, lambda chunk: 'A', {short.id: ['pure:9']}) == []


def test_name_forms_use_the_dictionary_both_ways():
    assert gp.name_forms('Новосибирский государственный университет', PAIRS) == ['Новосибирский государственный университет', 'НГУ']
    assert gp.name_forms('НГУ', PAIRS) == ['НГУ', 'Новосибирский государственный университет']


def test_cycles_and_embedded_text_changes():
    a, b, c = node('А'), node('Б'), node('В')
    before = graph([a, b, c], [edge(a, b, 'PART_OF', 'А в Б.'), edge(b, a, 'PART_OF', 'Б в А.'), edge(b, c, 'PART_OF', 'Б в В.')])
    assert gp.cycle_counts(before, ('PART_OF',)) == {'PART_OF': {'relations': 3, 'two_way_pairs': 1, 'relations_inside_cycles': 2}}
    after, _ = gp.rename_nodes(before, {c.id: ('Г', [])}, 's', 'r')
    changes = gp.embedded_text_changes(before, after)
    new_id = gp.entity_id('Г', 'ORGANIZATION')
    assert changes['entities_to_embed'] == [new_id] and changes['entities_gone'] == [c.id]
    assert changes['relations_to_embed'] == [] and changes['relations_rekeyed'] == [gp.relation_id(b.id, new_id, 'PART_OF')]
    assert changes['relations_gone'] == [before.edges[2].id]
    out = io.StringIO()
    gp.write_changes([{'op': 'rename', 'to_name': 'Г'}], out)
    assert out.getvalue() == '{"op": "rename", "to_name": "Г"}\n'
