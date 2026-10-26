import pytest

from context_utils.context_utils import (
    build_reverse_chunk_index,
    format_document_reference,
    global_chunk_index_to_local,
    global_chunk_index_to_text,
    graph_element_to_text,
    normalize_urls,
    prepare_context,
    prepare_graph_elements_context,
    prepare_relevant_documents,
)


def _make_documents():
    text_1 = 'Иванов Иван Иванович читает лекции по математическому анализу в НГУ.'
    text_2 = 'Институт информационных технологий готовит специалистов по машинному обучению.'
    documents = [
        {
            'text': text_1,
            'doc_full_text': text_1,
            'doc_title': 'Расписание занятий Иванова И.И.',
            'doc_annotation': 'Расписание лекций преподавателя.',
            'quality_score': 0.9,
            'url': 'https://table.nsu.ru/teacher/ivanov',
            'chunks': [{'start_char': 0, 'end_char': len(text_1)}]
        },
        {
            'text': text_2,
            'doc_full_text': text_2,
            'doc_title': 'Институт информационных технологий',
            'doc_annotation': 'Описание института.',
            'quality_score': 0.5,
            'url': ['https://nsu.ru/fit', 'https://nsu.ru/fit/about'],
            'chunks': [{'start_char': 0, 'end_char': len(text_2)}]
        }
    ]
    chunk_mapping = {'0': {'doc_index': 0, 'local_chunk_index': 0},
                     '1': {'doc_index': 1, 'local_chunk_index': 0}}
    return documents, chunk_mapping


def _make_multi_chunk_documents():
    text_1 = 'Первый документ состоит из двух фрагментов. Второй фрагмент первого документа.'
    sep_1 = text_1.index('. ') + 1
    text_2 = 'Второй документ. Он состоит из трёх фрагментов. Третий фрагмент короткий.'
    sep_2a = text_2.index('. ') + 1
    sep_2b = text_2.index('. ', sep_2a) + 1
    documents = [
        {
            'text': text_1,
            'chunks': [
                {'start_char': 0, 'end_char': sep_1},
                {'start_char': sep_1, 'end_char': len(text_1)},
            ]
        },
        {
            'text': text_2,
            'chunks': [
                {'start_char': 0, 'end_char': sep_2a},
                {'start_char': sep_2a, 'end_char': sep_2b},
                {'start_char': sep_2b, 'end_char': len(text_2)},
            ]
        },
    ]
    return documents


class TestNormalizeUrls:
    def test_none_gives_empty_list(self):
        assert normalize_urls(None) == []

    def test_string_gives_single_element_list(self):
        assert normalize_urls('  https://nsu.ru  ') == ['https://nsu.ru']

    def test_list_deduplicates_and_strips(self):
        value = ['https://a.ru', ' https://b.ru ', 'https://a.ru', '']
        assert normalize_urls(value) == ['https://a.ru', 'https://b.ru']


class TestBuildReverseChunkIndex:
    def test_sequential_numbering_over_documents_in_order(self):
        documents = _make_multi_chunk_documents()
        mapping = build_reverse_chunk_index(documents)
        assert set(mapping.keys()) == {'0', '1', '2', '3', '4'}
        assert mapping['0'] == {'doc_index': 0, 'local_chunk_index': 0}
        assert mapping['1'] == {'doc_index': 0, 'local_chunk_index': 1}
        assert mapping['2'] == {'doc_index': 1, 'local_chunk_index': 0}
        assert mapping['4'] == {'doc_index': 1, 'local_chunk_index': 2}

    def test_round_trip_with_index_to_local_and_text(self):
        documents = _make_multi_chunk_documents()
        mapping = build_reverse_chunk_index(documents)
        for global_index in range(5):
            document_index, local_chunk_index = global_chunk_index_to_local(global_index, documents, mapping)
            assert 0 <= document_index < len(documents)
            assert 0 <= local_chunk_index < len(documents[document_index]['chunks'])
            chunk_text = global_chunk_index_to_text(global_index, documents, mapping)
            assert len(chunk_text) > 0

    def test_doc_full_text_is_preferred_over_text(self):
        documents = _make_multi_chunk_documents()
        documents[0]['doc_full_text'] = documents[0]['text']
        mapping = build_reverse_chunk_index(documents)
        assert mapping['0']['doc_index'] == 0

    def test_empty_chunk_raises(self):
        documents = [
            {'text': '   ', 'chunks': [{'start_char': 0, 'end_char': 3}]},
        ]
        with pytest.raises(ValueError):
            build_reverse_chunk_index(documents)


class TestGlobalChunkIndex:
    def test_index_to_local(self):
        documents, chunk_mapping = _make_documents()
        assert global_chunk_index_to_local(1, documents, chunk_mapping) == (1, 0)

    def test_unknown_index_raises(self):
        documents, chunk_mapping = _make_documents()
        with pytest.raises(ValueError):
            global_chunk_index_to_local(42, documents, chunk_mapping)

    def test_index_to_text(self):
        documents, chunk_mapping = _make_documents()
        text = global_chunk_index_to_text(0, documents, chunk_mapping)
        assert 'Иванов' in text


class TestPrepareContext:
    def test_documents_are_grouped_and_filtered_by_quality(self):
        documents, chunk_mapping = _make_documents()
        document_texts, references = prepare_context(
            indices_of_relevant_chunks=[0, 1],
            scores_of_relevant_chunks=[0.8, 0.2],
            documents=documents,
            chunk_mapping=chunk_mapping,
            min_document_quality=0.7
        )
        assert len(document_texts) == 1
        assert 'Иванов' in document_texts[0]
        assert len(references) == 1
        assert 'https://table.nsu.ru/teacher/ivanov' in references[0]

    def test_multi_url_reference_lists_all_urls(self):
        documents, chunk_mapping = _make_documents()
        document_texts, references = prepare_context(
            indices_of_relevant_chunks=[1],
            scores_of_relevant_chunks=[1.0],
            documents=documents,
            chunk_mapping=chunk_mapping,
            min_document_quality=0.0
        )
        assert 'https://nsu.ru/fit' in references[0]
        assert 'https://nsu.ru/fit/about' in references[0]

    def test_format_document_reference(self):
        documents, _ = _make_documents()
        reference = format_document_reference(0, '1. ', 4, documents)
        assert reference.startswith('1. Расписание занятий Иванова И.И.')
        assert 'https://table.nsu.ru/teacher/ivanov' in reference


class TestGraphElements:
    def test_entity_rendering_contains_name_type_and_sources(self):
        documents, chunk_mapping = _make_documents()
        item = {'key': 'e1', 'kind': 'entity', 'name': 'Иванов Иван', 'type': 'PER',
                'description': 'преподаватель НГУ', 'score': 0.5, 'sources': [0]}
        text = graph_element_to_text(item, documents, chunk_mapping)
        assert 'Элемент графа знаний: сущность' in text
        assert 'Иванов Иван' in text
        assert 'PER' in text
        assert 'преподаватель НГУ' in text
        assert 'Расписание занятий Иванова И.И.' in text

    def test_relation_rendering_contains_subject_object_and_type(self):
        documents, chunk_mapping = _make_documents()
        item = {'key': 'r1', 'kind': 'relation', 'subject': 'Иванов Иван', 'object': 'НГУ',
                'relation_type': 'работает в', 'description': 'место работы', 'score': 0.4,
                'sources': [0]}
        text = graph_element_to_text(item, documents, chunk_mapping)
        assert 'Элемент графа знаний: связь' in text
        assert 'Иванов Иван' in text
        assert 'работает в' in text
        assert 'НГУ' in text

    def test_element_without_sources_has_no_source_block(self):
        documents, chunk_mapping = _make_documents()
        item = {'key': 'e1', 'kind': 'entity', 'name': 'X', 'type': 'PER',
                'description': '', 'score': 1.0, 'sources': []}
        text = graph_element_to_text(item, documents, chunk_mapping)
        assert 'Встречается во фрагментах документов' not in text

    def test_prepare_graph_elements_context_collects_document_indices(self):
        documents, chunk_mapping = _make_documents()
        items = [
            {'key': 'e1', 'kind': 'entity', 'name': 'Иванов Иван', 'type': 'PER',
             'description': 'преподаватель', 'score': 0.5, 'sources': [0]},
            {'key': 'r1', 'kind': 'relation', 'subject': 'Иванов Иван', 'object': 'ИИТ',
             'relation_type': 'работает в', 'description': '', 'score': 0.3, 'sources': [1]},
        ]
        texts, referenced_indices = prepare_graph_elements_context(items, documents, chunk_mapping)
        assert len(texts) == 2
        assert referenced_indices == {0, 1}

    def test_prepare_relevant_documents_groups_chunks(self):
        documents, chunk_mapping = _make_documents()
        selected = prepare_relevant_documents(
            indices_of_relevant_chunks=[0, 1],
            scores_of_relevant_chunks=[0.7, 0.3],
            documents=documents,
            chunk_mapping=chunk_mapping,
            min_document_quality=0.0
        )
        assert set(selected.keys()) == {0, 1}
        assert selected[0]['chunks'] == [0]
