from graph_utils.alias_table import AliasTable, build_alias_table, is_abbreviation_of, name_key


def test_abbreviation_is_checked_against_the_initials():
    assert is_abbreviation_of('ФИТ', 'Факультет информационных технологий')
    assert is_abbreviation_of('ММФ', 'Механико-математический факультет')
    assert is_abbreviation_of('ИФП', 'Институт философии и права')
    assert is_abbreviation_of('ФИТ', 'Факультет информационных технологий НГУ')
    # false aliases of graph v6 (audit of 18.11.2026)
    assert not is_abbreviation_of('ИО РАН', 'Институт гидродинамики')
    assert not is_abbreviation_of('РПД', 'Деловая информатика')
    assert not is_abbreviation_of('English', 'Люминесценция')
    assert not is_abbreviation_of('Штраф', 'Депортация')


def test_table_keeps_names_and_checked_abbreviations_only():
    nodes = [('f', 'Факультет информационных технологий', 'FACULTY', 'Факультет (ФИТ) университета'),
             ('a', 'ФИТ', 'FACULTY', ''), ('wrong', 'Институт гидродинамики', 'RESEARCH_INSTITUTE', ''),
             ('io', 'ИО РАН', 'RESEARCH_INSTITUTE', ''), ('m', 'Механико-математический факультет', 'FACULTY', ''),
             ('mm', 'ММФ', 'FACULTY', ''), ('v', 'факультета информационных технологий', 'FACULTY', '')]
    edges = [('io', 'wrong', 'ALTERNATIVE_NAME'), ('mm', 'm', 'ABBREVIATION'), ('f', 'v', 'PART_OF'), ('f', 'm', 'KNOWS')]
    built = build_alias_table(nodes, edges)
    names = built['names']
    assert names[name_key('ФИТ')] == ['f', 'a']  # from the brackets of the description; the most connected first
    assert names[name_key('ММФ')] == ['m', 'mm']  # the alias edge passes the check
    assert names[name_key('ИО РАН')] == ['io']  # the false alias edge adds nothing
    assert names[name_key('Факультет информационных технологий')] == ['f', 'v']  # inflected variant, one key
    assert built['meta']['alias_edges_checked'] == 1


def test_common_single_words_are_left_out_and_longest_mention_wins():
    nodes = [(f'd{i}', f'Договор {i}', 'DOCUMENT', '') for i in range(4)]
    nodes += [('d', 'Договор', 'DOCUMENT', ''), ('y', 'YADRO', 'ENTERPRISE', ''),
              ('k', 'Кафедра общей физики', 'DEPARTMENT', ''), ('p', 'Физика', 'RESEARCH_FIELD', '')]
    table = AliasTable(build_alias_table(nodes, [], single_word_max_names=3)['names'])
    assert name_key('Договор') not in table.names
    assert table.node_ids('типовой договор с YADRO') == ['y']
    assert table.node_ids('Кто заведует кафедрой общей физики?') == ['k']
    assert table.node_ids('физика и кафедра общей физики') == ['p', 'k']
