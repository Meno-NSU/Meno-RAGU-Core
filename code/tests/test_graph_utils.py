import asyncio
from types import SimpleNamespace

import pytest

from graph_utils.graph_utils import (
    AbbreviationExpandingExtractor,
    GraphSearchEngines,
    _normalize_scores,
    _rank_scores,
    build_chunk_mapping,
    canonicalize_artifacts,
    collect_knowledge_base_chunks,
    count_abbreviation_occurrences,
    enrich_chunk_text,
    find_abbreviation_candidates,
    load_chunk_mapping,
    save_chunk_mapping,
)
from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation
from ragu.utils.ragu_utils import compute_mdhash_id


CHUNK_ID_PREFIX = 'chunk-'


def _make_chunk(chunk_id: str, content: str):
    return SimpleNamespace(id=chunk_id, content=content)


def _make_retrieval(chunks, scores=None, entities=None, relations=None, summaries=None,
                    entity_metrics=None):
    result = SimpleNamespace(chunks=chunks, scores=scores if scores is not None else [],
                             entities=entities or [], relations=relations or [],
                             summaries=summaries or [])
    metrics = {'entities': entity_metrics or []}
    return SimpleNamespace(result=result, metrics=metrics)


class TestBuildChunkMapping:
    def test_distinct_chunks_map_to_distinct_indices(self):
        texts = ['первый текст', 'второй текст', 'третий текст']
        mapping = build_chunk_mapping(texts, [10, 20, 30])
        assert len(mapping) == 3
        for cur_text, cur_index in zip(texts, [10, 20, 30]):
            assert mapping[compute_mdhash_id(cur_text, prefix=CHUNK_ID_PREFIX)] == cur_index

    def test_duplicated_text_keeps_the_first_sorted_index(self):
        # The chunks come sorted by the global index (see collect_knowledge_base_chunks),
        # so the first occurrence of a duplicated text is the smallest global index.
        texts = ['одинаковый текст', 'другой текст', 'одинаковый текст']
        mapping = build_chunk_mapping(texts, [2, 5, 7])
        chunk_id = compute_mdhash_id('одинаковый текст', prefix=CHUNK_ID_PREFIX)
        assert mapping[chunk_id] == 2


class TestChunkMappingSerialization:
    def test_save_and_load_round_trip(self, tmp_path):
        mapping = {'chunk-aaa': 1, 'chunk-bbb': 2}
        save_chunk_mapping(mapping, str(tmp_path))
        loaded = load_chunk_mapping(str(tmp_path))
        assert loaded == mapping


class TestEnrichChunkText:
    def test_title_mode_prepends_frozen_header(self):
        document = {'doc_title': '  Новосибирский   государственный университет  '}
        result = enrich_chunk_text('текст фрагмента', document, 'title')
        assert result == ('**Документ:** Новосибирский государственный университет.\n\n'
                          'текст фрагмента')

    def test_final_punctuation_is_not_duplicated(self):
        document = {'doc_title': 'НГУ.'}
        result = enrich_chunk_text('текст', document, 'title')
        assert result == '**Документ:** НГУ.\n\nтекст'

    def test_title_annotation_mode_adds_annotation_line(self):
        document = {'doc_title': 'НГУ', 'doc_annotation': 'Справочник  для абитуриентов'}
        result = enrich_chunk_text('текст', document, 'title_annotation')
        assert result == '**Документ:** НГУ.\n\n**Аннотация:** Справочник для абитуриентов.\n\nтекст'

    def test_empty_annotation_falls_back_to_title(self):
        document = {'doc_title': 'НГУ', 'doc_annotation': '   '}
        result = enrich_chunk_text('текст', document, 'title_annotation')
        assert result == '**Документ:** НГУ.\n\nтекст'

    def test_empty_title_degrades_to_plain_chunk(self):
        assert enrich_chunk_text('текст', {'doc_title': ''}, 'title') == 'текст'
        assert enrich_chunk_text('текст', {'doc_title': ''}, 'title_annotation') == 'текст'
        assert enrich_chunk_text('текст', {}, 'title') == 'текст'

    def test_none_mode_returns_chunk_as_is(self):
        document = {'doc_title': 'НГУ'}
        assert enrich_chunk_text('текст', document, 'none') == 'текст'
        assert enrich_chunk_text('текст', document, 'unknown') == 'текст'


class TestCollectKnowledgeBaseChunksWithEnrichment:
    @staticmethod
    def _make_document(text, title, annotation, source):
        return {'doc_full_text': text, 'chunks': [{'start_char': 0, 'end_char': len(text)}],
                'doc_title': title, 'doc_annotation': annotation, 'content_source': source}

    def _make_corpus(self):
        documents = [
            self._make_document('текст новости', 'Название новости', '', 'news'),
            self._make_document('текст справочника', 'Название справочника', '', 'summary'),
        ]
        mapping = {
            '0': {'doc_index': 0, 'local_chunk_index': 0},
            '1': {'doc_index': 1, 'local_chunk_index': 0},
        }
        return documents, mapping

    def test_only_requested_sources_are_enriched(self):
        documents, mapping = self._make_corpus()
        stats = dict()
        texts, indices = collect_knowledge_base_chunks(
            documents, mapping, enrich_mode='title', enrich_sources={'news'}, stats=stats)
        assert indices == [0, 1]
        assert texts[0] == '**Документ:** Название новости.\n\nтекст новости'
        assert texts[1] == 'текст справочника'
        assert stats == {'total': 2, 'enriched': 1}

    def test_enrichment_without_source_filter_covers_all_chunks(self):
        documents, mapping = self._make_corpus()
        stats = dict()
        texts, indices = collect_knowledge_base_chunks(
            documents, mapping, enrich_mode='title', stats=stats)
        assert indices == [0, 1]
        assert texts[0] == '**Документ:** Название новости.\n\nтекст новости'
        assert texts[1] == '**Документ:** Название справочника.\n\nтекст справочника'
        assert stats == {'total': 2, 'enriched': 2}

    def test_disabled_enrichment_keeps_texts_intact(self):
        documents, mapping = self._make_corpus()
        texts, indices = collect_knowledge_base_chunks(documents, mapping)
        assert texts == ['текст новости', 'текст справочника']
        assert indices == [0, 1]


class TestScoreNormalization:
    def test_positive_scores_are_normalized_to_sum_one(self):
        scores = _normalize_scores([1.0, 3.0])
        assert abs(sum(scores) - 1.0) < 1e-9
        assert scores[1] > scores[0]

    def test_non_positive_scores_fall_back_to_rank_based(self):
        scores = _normalize_scores([-1.0, -2.0, -3.0])
        expected = [s / sum(_rank_scores(3)) for s in _rank_scores(3)]
        assert scores == pytest.approx(expected)

    def test_partial_scores_fall_back_to_rank_based(self):
        scores = _normalize_scores([0.5, None, 0.5])
        expected = [s / sum(_rank_scores(3)) for s in _rank_scores(3)]
        assert scores == pytest.approx(expected)


def _make_engines(mapping):
    return GraphSearchEngines(local_engine=None, naive_engine=None, chunk_to_global=mapping)


class TestChunksFromRetrieval:
    def test_chunks_are_mapped_and_normalized(self):
        texts = ['текст про Иванова', 'текст про НГУ']
        mapping = build_chunk_mapping(texts, [0, 1])
        engines = _make_engines(mapping)
        chunks = [_make_chunk(compute_mdhash_id(t, prefix=CHUNK_ID_PREFIX), t) for t in texts]
        retrieval = _make_retrieval(chunks, scores=[3.0, 1.0])
        result = engines._chunks_from_retrieval(retrieval, rank_based=False)
        assert [it[0] for it in result] == [0, 1]
        assert result[0][1] == pytest.approx(0.75)
        assert result[1][1] == pytest.approx(0.25)

    def test_unknown_chunk_id_falls_back_to_content_hash(self):
        texts = ['уникальный текст фрагмента']
        mapping = build_chunk_mapping(texts, [4])
        engines = _make_engines(mapping)
        chunks = [_make_chunk('wrong-id', texts[0])]
        retrieval = _make_retrieval(chunks, scores=[1.0])
        result = engines._chunks_from_retrieval(retrieval, rank_based=False)
        assert result == [(4, 1.0)]

    def test_unknown_chunks_are_dropped(self):
        engines = _make_engines({})
        chunks = [_make_chunk('unknown-id', 'никому не известный текст')]
        retrieval = _make_retrieval(chunks, scores=[1.0])
        result = engines._chunks_from_retrieval(retrieval, rank_based=False)
        assert result == []

    def test_empty_scores_fall_back_to_rank_based(self):
        texts = ['а', 'б']
        mapping = build_chunk_mapping(texts, [0, 1])
        engines = _make_engines(mapping)
        chunks = [_make_chunk(compute_mdhash_id(t, prefix=CHUNK_ID_PREFIX), t) for t in texts]
        retrieval = _make_retrieval(chunks, scores=[])
        result = engines._chunks_from_retrieval(retrieval, rank_based=False)
        assert [it[0] for it in result] == [0, 1]
        assert abs(sum(it[1] for it in result) - 1.0) < 1e-9


class TestEntitiesFromLocal:
    def test_entities_are_scored_and_mapped(self):
        texts = ['Иванов читает лекции по математике', 'текст про НГУ']
        mapping = build_chunk_mapping(texts, [0, 1])
        engines = _make_engines(mapping)
        entities = [
            SimpleNamespace(id='e1', entity_name='Иванов Иван', entity_type='PER',
                            description='преподаватель', source_chunk_id=[
                                compute_mdhash_id(texts[0], prefix=CHUNK_ID_PREFIX)]),
            SimpleNamespace(id='e2', entity_name='НГУ', entity_type='ORG',
                            description='университет', source_chunk_id=[
                                compute_mdhash_id(texts[0], prefix=CHUNK_ID_PREFIX),
                                compute_mdhash_id(texts[1], prefix=CHUNK_ID_PREFIX)]),
        ]
        retrieval = _make_retrieval([], entities=entities,
                                    entity_metrics=[{'relevance_score': 3.0}, {'relevance_score': 1.0}])
        items = engines._entities_from_local(retrieval)
        assert len(items) == 2
        assert items[0]['key'] == 'e1'
        assert items[0]['name'] == 'Иванов Иван'
        assert items[0]['type'] == 'PER'
        assert items[0]['score'] == pytest.approx(0.75)
        assert items[0]['sources'] == [0]
        assert items[1]['sources'] == [0, 1]
        assert abs(sum(it['score'] for it in items) - 1.0) < 1e-9

    def test_entities_without_scores_use_rank_based_scores(self):
        engines = _make_engines({})
        entities = [
            SimpleNamespace(id='e1', entity_name='А', entity_type='X', description='',
                            source_chunk_id=[]),
            SimpleNamespace(id='e2', entity_name='Б', entity_type='X', description='',
                            source_chunk_id=[]),
        ]
        retrieval = _make_retrieval([], entities=entities)
        items = engines._entities_from_local(retrieval)
        assert items[0]['score'] > items[1]['score']
        assert abs(sum(it['score'] for it in items) - 1.0) < 1e-9


class TestRelationsFromLocal:
    def test_relations_get_rank_based_scores(self):
        engines = _make_engines({})
        relations = [
            SimpleNamespace(id='r1', subject_name='Иванов', object_name='НГУ',
                            relation_type='работает в', description='', source_chunk_id=[]),
            SimpleNamespace(id='r2', subject_name='Петров', object_name='НГУ',
                            relation_type='работает в', description='', source_chunk_id=[]),
        ]
        retrieval = _make_retrieval([], relations=relations)
        items = engines._relations_from_local(retrieval)
        assert items[0]['key'] == 'r1'
        assert items[0]['subject'] == 'Иванов'
        assert items[0]['relation_type'] == 'работает в'
        assert items[0]['score'] > items[1]['score']
        assert abs(sum(it['score'] for it in items) - 1.0) < 1e-9


def _make_abbr_dict():
    return {
        'нгу': {'abbreviation': 'НГУ', 'explanation': ['Новосибирский государственный университет']},
        'ммф': {'abbreviation': 'ММФ', 'explanation': ['механико-математический факультет']},
        'егэ': {'abbreviation': 'ЕГЭ', 'explanation': ['единый государственный экзамен']},
    }


class TestFindAbbreviationCandidates:
    def test_word_boundaries_are_respected(self):
        found = find_abbreviation_candidates('Обучение в НГУ и в НГУшка', _make_abbr_dict())
        assert found == ['нгу']

    def test_matching_is_case_sensitive(self):
        found = find_abbreviation_candidates('обучение в нгу', _make_abbr_dict())
        assert found == []

    def test_no_candidates_gives_empty_list(self):
        found = find_abbreviation_candidates('Простой текст без аббревиатур', _make_abbr_dict())
        assert found == []

    def test_several_abbreviations_are_found(self):
        found = find_abbreviation_candidates('ММФ — один из факультетов НГУ', _make_abbr_dict())
        assert set(found) == {'ммф', 'нгу'}

    def test_occurrences_are_counted(self):
        counts = count_abbreviation_occurrences('НГУ, НГУ и ещё раз НГУ, а также ММФ.', _make_abbr_dict())
        assert counts == {'нгу': 3, 'ммф': 1}

    def test_occurrence_counting_respects_word_boundaries(self):
        counts = count_abbreviation_occurrences('НГУ и НГУшка, но не нгу.', _make_abbr_dict())
        assert counts == {'нгу': 1}


class TestCanonicalizeArtifacts:
    def test_case_variants_are_unified_to_the_most_frequent_spelling(self):
        entities = [
            Entity(entity_name='Новосибирский государственный университет', entity_type='ORGANIZATION',
                   description='первое упоминание', source_chunk_id=['chunk-a']),
            Entity(entity_name='Новосибирский Государственный Университет', entity_type='ORGANIZATION',
                   description='второе упоминание', source_chunk_id=['chunk-b']),
            Entity(entity_name='Новосибирский государственный университет', entity_type='ORGANIZATION',
                   description='третье упоминание', source_chunk_id=['chunk-c']),
        ]
        canonical_entities, canonical_relations = canonicalize_artifacts(entities, [])
        assert len(canonical_entities) == 3
        assert all(e.entity_name == 'Новосибирский государственный университет' for e in canonical_entities)
        assert len(set(e.id for e in canonical_entities)) == 1
        assert canonical_relations == []

    def test_punctuation_variants_are_unified_and_trimmed(self):
        entities = [
            Entity(entity_name='НГУ.', entity_type='ORGANIZATION', description='', source_chunk_id=['chunk-a']),
            Entity(entity_name='НГУ', entity_type='ORGANIZATION', description='', source_chunk_id=['chunk-b']),
            Entity(entity_name='НГУ.', entity_type='ORGANIZATION', description='', source_chunk_id=['chunk-c']),
        ]
        canonical_entities, _ = canonicalize_artifacts(entities, [])
        assert len(canonical_entities) == 3
        assert all(e.entity_name == 'НГУ' for e in canonical_entities)
        assert len(set(e.id for e in canonical_entities)) == 1

    def test_internal_punctuation_is_preserved(self):
        entities = [
            Entity(entity_name='А. В. Бессонов', entity_type='PERSON', description='', source_chunk_id=[]),
            Entity(entity_name='А.В. Бессонов', entity_type='PERSON', description='', source_chunk_id=[]),
        ]
        canonical_entities, _ = canonicalize_artifacts(entities, [])
        assert len(canonical_entities) == 2
        assert len(set(e.entity_name for e in canonical_entities)) == 1
        assert '.' in canonical_entities[0].entity_name
        assert canonical_entities[0].entity_name.startswith('А')
        assert canonical_entities[0].entity_name.endswith('Бессонов')

    def test_relation_endpoints_are_remapped_to_canonical_ids(self):
        variant = Entity(entity_name='НГУ.', entity_type='ORGANIZATION', description='', source_chunk_id=[])
        canonical_name = 'НГУ'
        canonical_first = Entity(entity_name=canonical_name, entity_type='ORGANIZATION',
                                 description='', source_chunk_id=[])
        canonical_second = Entity(entity_name=canonical_name, entity_type='ORGANIZATION',
                                  description='', source_chunk_id=[])
        assert variant.id != canonical_first.id
        assert canonical_first.id == canonical_second.id
        relation = Relation(subject_id=variant.id, object_id=canonical_first.id, relation_type='PART_OF',
                            subject_name=variant.entity_name, object_name=canonical_name,
                            description='', source_chunk_id=[])
        canonical_entities, canonical_relations = canonicalize_artifacts(
            [variant, canonical_first, canonical_second], [relation]
        )
        assert all(e.entity_name == canonical_name for e in canonical_entities)
        assert len(canonical_relations) == 1
        assert canonical_relations[0].subject_id == canonical_first.id
        assert canonical_relations[0].object_id == canonical_first.id
        assert canonical_relations[0].subject_name == canonical_name
        assert canonical_relations[0].object_name == canonical_name

    def test_already_canonical_artifacts_are_untouched(self):
        entity = Entity(entity_name='Сибирь', entity_type='LOCATION', description='крупный регион',
                        source_chunk_id=['chunk-a'])
        relation = Relation(subject_id=entity.id, object_id=entity.id, relation_type='PART_OF',
                            subject_name='Сибирь', object_name='Сибирь', description='',
                            source_chunk_id=['chunk-a'])
        canonical_entities, canonical_relations = canonicalize_artifacts([entity], [relation])
        assert canonical_entities == [entity]
        assert canonical_relations == [relation]


class _FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def chat_completion(self, conversation, output_schema=str, **kwargs):
        self.calls.append(conversation)
        return self.response


def _make_extractor(llm, abbreviations):
    return AbbreviationExpandingExtractor(
        llm=llm, embedder=None, icl_config=None, ontology=None,
        abbreviations=abbreviations
    )


class TestAbbreviationExpandingExtractor:
    def test_rewrite_preserves_the_original_chunk_id(self):
        llm = _FakeLLM('Механико-математический факультет Новосибирского государственного университета '
                       'объединяет многих студентов.')
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content='ММФ НГУ объединяет многих студентов.',
                               chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert len(rewritten) == 1
        assert rewritten[0] is not original_chunk
        assert rewritten[0].content.startswith('Механико-математический')
        assert rewritten[0].id == original_chunk.id
        assert rewritten[0].doc_id == original_chunk.doc_id
        assert extractor.num_rewritten_chunks == 1
        assert extractor.num_rejected_rewrites == 0

    def test_chunk_without_candidates_is_not_sent_to_llm(self):
        llm = _FakeLLM('любой ответ')
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content='Простой текст без аббревиатур.', chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert rewritten == [original_chunk]
        assert len(llm.calls) == 0
        assert extractor.num_rewritten_chunks == 0

    def test_suspicious_rewrite_is_rejected_and_original_kept(self):
        llm = _FakeLLM('Совсем другой текст, в котором нет никаких расшифровок аббревиатур.')
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content='ММФ НГУ объединяет многих студентов.',
                               chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert rewritten == [original_chunk]
        assert extractor.num_rejected_rewrites == 1
        assert extractor.num_rewritten_chunks == 0

    def test_paraphrased_rewrite_is_rejected_by_edit_distance(self):
        original_text = ('ММФ НГУ готовит математиков, механиков и программистов. '
                         'Студенты изучают анализ, алгебру, геометрию и программирование. ' * 3).strip()
        paraphrased_text = ('Механико-математический факультет Новосибирского государственного университета '
                            'выпускает бухгалтеров, юристов и поваров. '
                            'Выпускники осваивают кулинарию, дизайн и вождение автомобилей. ' * 3).strip()
        llm = _FakeLLM(paraphrased_text)
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content=original_text, chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert rewritten == [original_chunk]
        assert extractor.num_rejected_rewrites == 1
        assert extractor.num_rewritten_chunks == 0

    def test_declined_expansion_is_accepted(self):
        llm = _FakeLLM('Студенты учатся в Новосибирском государственном университете и живут в общежитии.')
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content='Студенты учатся в НГУ и живут в общежитии.',
                               chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert len(rewritten) == 1
        assert rewritten[0] is not original_chunk
        assert 'Новосибирском государственном университете' in rewritten[0].content
        assert rewritten[0].id == original_chunk.id
        assert extractor.num_rewritten_chunks == 1

    REPEATED_ORIGINAL = (
        'НГУ расположен в Новосибирске. НГУ готовит студентов. '
        'НГУ ведёт научную работу. НГУ сотрудничает со школами. '
        'НГУ известен далеко за пределами региона.'
    )

    def test_repeated_abbreviation_expansion_is_accepted(self):
        original_text = self.REPEATED_ORIGINAL
        rewritten_text = original_text.replace('НГУ', 'Новосибирский государственный университет')
        llm = _FakeLLM(rewritten_text)
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content=original_text, chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert len(rewritten) == 1
        assert rewritten[0] is not original_chunk
        assert rewritten[0].content == rewritten_text
        assert rewritten[0].id == original_chunk.id
        assert extractor.num_rewritten_chunks == 1
        assert extractor.num_rejected_rewrites == 0

    def test_partial_expansion_of_repeated_abbreviation_is_accepted(self):
        original_text = self.REPEATED_ORIGINAL
        rewritten_text = original_text.replace('НГУ', 'Новосибирский государственный университет', 2)
        llm = _FakeLLM(rewritten_text)
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content=original_text, chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert len(rewritten) == 1
        assert rewritten[0] is not original_chunk
        assert rewritten[0].content == rewritten_text
        assert extractor.num_rewritten_chunks == 1
        assert extractor.num_rejected_rewrites == 0

    def test_expansion_of_abbreviation_heavy_chunk_is_accepted(self):
        original_text = 'В НГУ, НГУ и снова НГУ — везде одно и то же: НГУ.'
        rewritten_text = original_text.replace('НГУ', 'Новосибирский государственный университет')
        llm = _FakeLLM(rewritten_text)
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content=original_text, chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert len(rewritten) == 1
        assert rewritten[0] is not original_chunk
        assert rewritten[0].content == rewritten_text
        assert extractor.num_rewritten_chunks == 1
        assert extractor.num_rejected_rewrites == 0

    def test_declined_ege_expansion_is_accepted(self):
        llm = _FakeLLM('Подготовка школьников к единому государственному экзамену требует времени.')
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content='Подготовка школьников к ЕГЭ требует времени.',
                               chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert len(rewritten) == 1
        assert rewritten[0] is not original_chunk
        assert 'единому государственному экзамену' in rewritten[0].content
        assert extractor.num_rewritten_chunks == 1
        assert extractor.num_rejected_rewrites == 0

    def test_similar_name_substitution_is_rejected_by_stem(self):
        llm = _FakeLLM('ММФ Новгородский государственный университет объединяет многих студентов.')
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content='ММФ НГУ объединяет многих студентов.',
                               chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert rewritten == [original_chunk]
        assert extractor.num_rejected_rewrites == 1
        assert extractor.num_rewritten_chunks == 0

    def test_same_length_substitution_without_expansion_is_rejected(self):
        llm = _FakeLLM('ММФ МГУ объединяет многих студентов.')
        extractor = _make_extractor(llm, _make_abbr_dict())
        original_chunk = Chunk(content='ММФ НГУ объединяет многих студентов.',
                               chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert rewritten == [original_chunk]
        assert extractor.num_rejected_rewrites == 1
        assert extractor.num_rewritten_chunks == 0

    def test_empty_dictionary_skips_the_stage(self):
        llm = _FakeLLM('любой ответ')
        extractor = _make_extractor(llm, {})
        original_chunk = Chunk(content='ММФ НГУ объединяет многих студентов.',
                               chunk_order_idx=0, doc_id='doc_1')
        rewritten = asyncio.run(extractor._expand_abbreviations([original_chunk]))
        assert rewritten == [original_chunk]
        assert len(llm.calls) == 0


def test_collection_manifest_keeps_names_after_graph_directory_moves(tmp_path):
    from graph_utils.graph_utils import PortableQdrantStorage
    storage = PortableQdrantStorage(
        filename=str(tmp_path / 'moved' / 'vdb_chunk.json'),
        embedding_dim=3, storage_folder=str(tmp_path / 'moved'),
        collection_names={'vdb_chunk': 'original_chunks'}, location=':memory:',
    )
    assert storage.collection_name == 'original_chunks'
