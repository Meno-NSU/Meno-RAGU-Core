import asyncio
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graph_utils.checkpoints import (
    RELATION_REVIEW_PREVIOUS_CODE,
    CheckpointedExtractor,
    CheckpointedGraphBuilder,
    CheckpointError,
    GraphCheckpoints,
    checkpoint_configuration,
    previous_completion_configuration,
    previous_relation_review_configuration,
    previous_summary_configuration,
)
from ragu import KnowledgeGraph, Settings, StorageArguments
from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import EntitiesExtractionModel, RelationsExtractionModel
from ragu.graph.graph_builder_pipeline import BuilderArguments, InMemoryGraphBuilder
from ragu.graph.types import Community, CommunitySummary, Entity, Relation
from ragu.models.llm import LLM
from ragu.storage.types import ClusterInfo
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage

PASSES = [
    'Extracting entities from chunks', 'Validating extracted entities',
    'Extracting relations from chunks', 'Validating extracted relations',
]


class Model(LLM):
    def __init__(self, *, stop=None):
        self.stop = stop
        self.calls = []

    async def chat_completion(self, conversation, **kwargs):
        raise AssertionError('Unexpected unbatched LLM call')

    async def batch_chat_completion(self, conversations, output_schema, desc=None, **kwargs):
        self.calls.append((desc, len(conversations)))
        if desc == self.stop:
            raise asyncio.CancelledError()
        if output_schema is EntitiesExtractionModel:
            return [EntitiesExtractionModel(entities=[
                {'entity_name': 'Иванов', 'entity_type': 'PERSON', 'description': 'Преподаватель НГУ'},
                {'entity_name': 'НГУ', 'entity_type': 'ORGANIZATION', 'description': 'Университет'},
            ]) for _ in conversations]
        assert output_schema is RelationsExtractionModel
        return [RelationsExtractionModel(relations=[{
            'source_entity': 'Иванов', 'target_entity': 'НГУ', 'relation_type': 'WORKPLACE',
            'description': 'Иванов преподаёт в НГУ', 'relationship_strength': 5,
        }]) for _ in conversations]


def chunks():
    return [Chunk(f'Иванов преподаёт в НГУ. Корпус {i}.', i, f'doc_{i}') for i in range(3)]


def extractor(model, store=None):
    result = CheckpointedExtractor(
        llm=model, ontology=None, do_entity_validation=True, do_relation_validation=True,
    )
    result.checkpoints = store
    return result


@pytest.mark.parametrize('stop', PASSES)
def test_interrupted_extraction_resumes_without_repeating_completed_llm_passes(tmp_path, stop):
    async def run():
        first = Model(stop=stop)
        with GraphCheckpoints(tmp_path, {'corpus': 'unchanged'}) as store:
            with pytest.raises(asyncio.CancelledError):
                await extractor(first, store).extract(chunks())
        resumed = Model()
        with GraphCheckpoints(tmp_path, {'corpus': 'unchanged'}) as store:
            entities, relations = await extractor(resumed, store).extract(chunks())
        assert [name for name, _ in resumed.calls] == PASSES[PASSES.index(stop):]
        assert len(entities) == 6 and len(relations) == 3
        assert {cid for entity in entities for cid in entity.source_chunk_id} == {c.id for c in chunks()}
        with GraphCheckpoints(tmp_path, {'corpus': 'unchanged'}) as store:
            blocked = Model(stop=PASSES[0])
            restored = await extractor(blocked, store).extract(chunks())
        assert blocked.calls == []
        assert [[dataclasses.asdict(item) for item in group] for group in restored] == [
            [dataclasses.asdict(item) for item in group] for group in (entities, relations)
        ]
    asyncio.run(run())


def test_rewritten_chunk_preserves_original_id_on_restore(tmp_path):
    original = chunks()[0]
    rewritten = Chunk(original.content.replace('НГУ', 'Новосибирском государственном университете'),
                      original.chunk_order_idx, original.doc_id, original.num_tokens)
    rewritten.id = original.id
    with GraphCheckpoints(tmp_path, {}) as store:
        store.save('01-abbreviations', ([rewritten],), (Chunk,))
    with GraphCheckpoints(tmp_path, {}) as store:
        result = store.load('01-abbreviations', (Chunk,))[0][0]
    assert dataclasses.asdict(result) == dataclasses.asdict(rewritten)
    assert result.id == original.id


def test_batch_retries_only_failed_items_and_invalidates_dependent_results(tmp_path):
    async def run():
        operation = AsyncMock(return_value=[EntitiesExtractionModel(), None, EntitiesExtractionModel()])
        with GraphCheckpoints(tmp_path, {}) as store:
            await store.batch('02-entities', EntitiesExtractionModel, operation, (['a', 'b', 'c'],))
            store.save('03-entity-validation', ([EntitiesExtractionModel()] * 3,), (EntitiesExtractionModel,))
        retry = AsyncMock(return_value=[EntitiesExtractionModel()])
        with GraphCheckpoints(tmp_path, {}) as store:
            assert store.has_missing()
            result = await store.batch('02-entities', EntitiesExtractionModel, retry, (['a', 'b', 'c'],))
            assert all(item is not None for item in result)
            assert not store.has_missing()
            assert store.load('03-entity-validation', (EntitiesExtractionModel,)) is None
        retry.assert_awaited_once_with(['b'])
    asyncio.run(run())


def pipeline(model, store):
    parent = InMemoryGraphBuilder(
        embedder=object(), llm=model, artifact_extractor=extractor(model, store),
        build_parameters=BuilderArguments(use_llm_summarization=False, use_clustering=False,
                                         make_community_summary=True),
    )
    return CheckpointedGraphBuilder.from_pipeline(parent, store)


def test_community_resume_preserves_cluster_memberships_and_ready_graph_skips_all_processing(tmp_path):
    async def cluster(entities, relations):
        for entity in entities:
            entity.clusters = [ClusterInfo(level=0, cluster_id=7)]
        return [Community(0, 7, entities, relations)]

    async def summarize(communities):
        return [CommunitySummary(community.id, 'Преподаватель работает в университете') for community in communities]

    async def run():
        with GraphCheckpoints(tmp_path, {}) as store:
            first = pipeline(Model(), store)
            first.cluster_graph = AsyncMock(side_effect=cluster)
            first.community_summarizer.summarize = AsyncMock(side_effect=asyncio.CancelledError())
            with pytest.raises(asyncio.CancelledError):
                await first.extract_graph(chunks())
        with GraphCheckpoints(tmp_path, {}) as store:
            model = Model(stop=PASSES[0])
            resumed = pipeline(model, store)
            resumed.cluster_graph = AsyncMock(side_effect=AssertionError('Clustering must be restored'))
            resumed.entity_summarizer.run = AsyncMock(side_effect=AssertionError('Entities must be restored'))
            resumed.relation_summarizer.run = AsyncMock(side_effect=AssertionError('Relations must be restored'))
            resumed.community_summarizer.summarize = AsyncMock(side_effect=summarize)
            result = await resumed.extract_graph(chunks())
            assert model.calls == []
            assert result[0][0].clusters == [ClusterInfo(level=0, cluster_id=7)]
            assert result[3][0].entities[0].clusters == result[0][0].clusters
        with GraphCheckpoints(tmp_path, {}) as store:
            restored = pipeline(Model(), store)
            restored.artifact_extractor = AsyncMock(side_effect=AssertionError('Ready graph must be restored'))
            ready = await restored.extract_graph(chunks())
        assert [[dataclasses.asdict(item) for item in group] for group in ready] == [
            [dataclasses.asdict(item) for item in group] for group in result
        ]
    asyncio.run(run())


def test_interrupted_vectorization_restores_ready_graph_and_writes_real_in_memory_index(tmp_path, monkeypatch):
    class Embedder:
        dim = 4

        async def embed_text(self, text, **kwargs):
            return [1.0, 0.0, 0.0, 0.0]

        async def batch_embed_text(self, texts, **kwargs):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    graph_dir = tmp_path / 'graph'
    graph_dir.mkdir()
    monkeypatch.setattr(Settings, 'storage_folder', str(graph_dir))

    def graph(model, store):
        result = KnowledgeGraph(
            llm=model, embedder=Embedder(), chunker=None, artifact_extractor=extractor(model, store),
            builder_settings=BuilderArguments(use_llm_summarization=False, use_clustering=False,
                                             make_community_summary=False),
            storage_settings=StorageArguments(vdb_storage_type=QdrantVectorDBStorage,
                                              vdb_storage_kwargs={'location': ':memory:'}),
        )
        result.pipeline = CheckpointedGraphBuilder.from_pipeline(result.pipeline, store)
        return result

    async def run():
        with GraphCheckpoints(tmp_path / 'checkpoints', {}) as store:
            first = graph(Model(), store)
            first.upsert_entities = AsyncMock(side_effect=asyncio.CancelledError())
            try:
                with pytest.raises(asyncio.CancelledError):
                    await first.build_from_docs([chunk.content for chunk in chunks()])
                assert '11-graph-ready' in store.manifest['stages']
            finally:
                await first.index.close()
        with GraphCheckpoints(tmp_path / 'checkpoints', {}) as store:
            model = Model(stop=PASSES[0])
            resumed = graph(model, store)
            try:
                await resumed.build_from_docs([chunk.content for chunk in chunks()])
                assert model.calls == []
                entities = store.load('11-graph-ready', (Entity, Relation, CommunitySummary, Community))[0]
                actual = await resumed.get_entities([entity.id for entity in entities])
                assert all(entity is not None for entity in actual)
                assert (graph_dir / 'knowledge_graph.gml').is_file()
            finally:
                await resumed.index.close()
    asyncio.run(run())


def test_corrupted_snapshot_fails_without_regenerating_or_overwriting(tmp_path):
    with GraphCheckpoints(tmp_path, {}) as store:
        store.save('02-entities', ([EntitiesExtractionModel()],), (EntitiesExtractionModel,))
        path = tmp_path / store.manifest['stages']['02-entities']['file']
    path.write_text('damaged')
    with GraphCheckpoints(tmp_path, {}) as store:
        with pytest.raises(CheckpointError, match='Повреждён чекпоинт'):
            store.load('02-entities', (EntitiesExtractionModel,))
    assert path.read_text() == 'damaged'


def test_checkpoint_error_escapes_ragu_validation_fallback_without_new_llm_calls(tmp_path):
    async def run():
        with GraphCheckpoints(tmp_path, {}) as store:
            with pytest.raises(asyncio.CancelledError):
                await extractor(Model(stop=PASSES[2]), store).extract(chunks())
            path = tmp_path / store.manifest['stages']['03-entity-validation']['file']
        path.write_text('damaged')
        with GraphCheckpoints(tmp_path, {}) as store:
            model = Model()
            with pytest.raises(CheckpointError, match='Повреждён чекпоинт'):
                await extractor(model, store).extract(chunks())
            assert model.calls == []
            assert '06-artifacts' not in store.manifest['stages']
    asyncio.run(run())


def test_checkpoint_disk_failure_is_not_swallowed_as_llm_failure(tmp_path):
    async def run():
        def fail():
            raise OSError('disk full')
        with GraphCheckpoints(tmp_path, {}) as store:
            store.save('01-abbreviations', (chunks(),), (Chunk,))
            store.before_save = fail
            model = Model()
            with pytest.raises(OSError, match='disk full'):
                await extractor(model, store).extract(chunks())
            assert model.calls == [(PASSES[0], 3)]
            assert '02-entities' not in store.manifest['stages']
    asyncio.run(run())


@pytest.mark.parametrize('phase', ['data', 'manifest'])
def test_interrupted_checkpoint_commit_keeps_previous_snapshot(tmp_path, monkeypatch, phase):
    old = EntitiesExtractionModel()
    new = EntitiesExtractionModel(entities=[{'entity_name': 'НГУ', 'entity_type': 'ORGANIZATION', 'description': 'НГУ'}])
    with GraphCheckpoints(tmp_path, {}) as store:
        store.save('02-entities', ([old],), (EntitiesExtractionModel,))
    original = Path.replace

    def fail_replace(path, target):
        if (Path(target).name == 'manifest.json') == (phase == 'manifest'):
            raise OSError('simulated disk failure')
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', fail_replace)
        with GraphCheckpoints(tmp_path, {}) as store:
            with pytest.raises(OSError, match='simulated disk failure'):
                store.save('02-entities', ([new],), (EntitiesExtractionModel,))
    with GraphCheckpoints(tmp_path, {}) as store:
        assert store.load('02-entities', (EntitiesExtractionModel,)) == ([old],)
    assert not list(tmp_path.glob('*.part'))


def test_configuration_mismatch_refuses_existing_checkpoints(tmp_path):
    with GraphCheckpoints(tmp_path, {'source': 'first'}) as store:
        store.save('02-entities', ([EntitiesExtractionModel()],), (EntitiesExtractionModel,))
    manifest = (tmp_path / 'manifest.json').read_bytes()
    with pytest.raises(CheckpointError, match='Изменились входы'):
        with GraphCheckpoints(tmp_path, {'source': 'changed'}):
            pytest.fail('Must not reuse changed inputs')
    assert (tmp_path / 'manifest.json').read_bytes() == manifest


@pytest.mark.parametrize('changed', [None, 'settings', 'files', 'code'])
@pytest.mark.parametrize('upgrade', ['budget', 'completion'])
def test_summary_budget_upgrade_preserves_01_to_06_and_invalidates_only_later_snapshots(tmp_path, changed, upgrade):
    configuration = {'settings': {'model': 'gemma'}, 'files': {'source': 'unchanged'},
                     'code': {'graph_utils/bounded_summaries.py': 'new-helper', 'other-helper': 'unchanged'}}
    previous = (previous_summary_configuration(configuration) if upgrade == 'budget'
                else previous_completion_configuration(configuration))
    with GraphCheckpoints(tmp_path, previous) as store:
        store.save('06-artifacts', ([], []), (Entity, Relation))
        store.save('07-entity-summaries', ([],), (Entity,))
        core_path = tmp_path / store.manifest['stages']['06-artifacts']['file']
        summary_path = tmp_path / store.manifest['stages']['07-entity-summaries']['file']
    before = (tmp_path / 'manifest.json').read_bytes()
    if changed:
        configuration[changed]['different'] = 'must not migrate'
        with pytest.raises(CheckpointError, match='Изменились входы'):
            with GraphCheckpoints(tmp_path, configuration):
                pytest.fail('Changed content must not migrate')
        assert (tmp_path / 'manifest.json').read_bytes() == before
    else:
        with GraphCheckpoints(tmp_path, configuration) as store:
            assert set(store.manifest['stages']) == {'06-artifacts'}
            assert store.load('06-artifacts', (Entity, Relation)) == ([], [])
        backup = 'manifest.before-summary-budget.json' if upgrade == 'budget' else 'manifest.before-summary-retries.json'
        assert (tmp_path / backup).read_bytes() == before
        assert core_path.exists() and summary_path.exists()


def test_previous_relation_review_configuration_is_the_code_before_the_review():
    configuration = {'settings': {'model': 'gemma', 'relation_review': True}, 'files': {'source': 'unchanged'},
                     'code': {'graph_utils/relation_review.py': 'new-stage', 'graph_utils/checkpoints.py': 'new',
                              'other-helper': 'unchanged'}}
    previous = previous_relation_review_configuration(configuration)
    assert previous['settings'] == {'model': 'gemma'}
    assert previous['files'] == {'source': 'unchanged'}
    assert previous['code'] == {'other-helper': 'unchanged', **RELATION_REVIEW_PREVIOUS_CODE}
    assert configuration['settings']['relation_review'] is True


@pytest.mark.parametrize('changed', [None, 'settings', 'files'])
def test_relation_review_upgrade_preserves_01_to_04_and_invalidates_later_snapshots(tmp_path, changed):
    configuration = {'settings': {'model': 'gemma', 'relation_review': True}, 'files': {'source': 'unchanged'},
                     'code': {'graph_utils/relation_review.py': 'new-stage', 'other-helper': 'unchanged'}}
    with GraphCheckpoints(tmp_path, previous_relation_review_configuration(configuration)) as store:
        store.save('04-relations', ([RelationsExtractionModel()],), (RelationsExtractionModel,))
        store.save('05-relation-validation', ([RelationsExtractionModel()],), (RelationsExtractionModel,))
        store.save('06-artifacts', ([], []), (Entity, Relation))
        kept_path = tmp_path / store.manifest['stages']['04-relations']['file']
    before = (tmp_path / 'manifest.json').read_bytes()
    if changed:
        configuration[changed]['different'] = 'must not migrate'
        with pytest.raises(CheckpointError, match='Изменились входы'):
            with GraphCheckpoints(tmp_path, configuration):
                pytest.fail('Changed content must not migrate')
        assert (tmp_path / 'manifest.json').read_bytes() == before
        return
    with GraphCheckpoints(tmp_path, configuration) as store:
        assert set(store.manifest['stages']) == {'04-relations'}
        assert store.load('04-relations', (RelationsExtractionModel,)) == ([RelationsExtractionModel()],)
    assert (tmp_path / 'manifest.before-relation-review.json').read_bytes() == before
    assert kept_path.exists()


def test_corrupted_manifest_fails_before_running_any_pass(tmp_path):
    with GraphCheckpoints(tmp_path, {}) as store:
        store.save('02-entities', ([EntitiesExtractionModel()],), (EntitiesExtractionModel,))
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['stages']['02-entities']['dependencies']
    path.write_text(json.dumps(manifest))
    with pytest.raises(CheckpointError, match='Повреждён манифест'):
        with GraphCheckpoints(tmp_path, {}):
            pytest.fail('Malformed checkpoint metadata must not trigger regeneration')


def test_checkpoint_lock_prevents_two_writers_and_releases_after_failure(tmp_path):
    with GraphCheckpoints(tmp_path, {}):
        with pytest.raises(CheckpointError, match='Другой процесс'):
            with GraphCheckpoints(tmp_path, {}):
                pytest.fail('Concurrent writer acquired the lock')
    with GraphCheckpoints(tmp_path, {}):
        pass


def test_context_budget_failure_prevents_checkpoint_publication(tmp_path):
    def reject():
        raise ValueError('context overflow')
    with GraphCheckpoints(tmp_path, {}, before_save=reject) as store:
        with pytest.raises(ValueError, match='context overflow'):
            store.save('02-entities', ([EntitiesExtractionModel()],), (EntitiesExtractionModel,))
    assert not (tmp_path / 'manifest.json').exists()


def test_graph_checkpoint_configuration_detects_content_changes_but_allows_execution_tuning(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text('original')
    args = SimpleNamespace(input_name=str(corpus), reverse_index_name=str(corpus), abbr_fname=None,
                           llm_tokenizer=None, embedder_tokenizer=None, rate_max_simultaneous=32,
                           log_level='INFO', openai_api_key='secret', llm_name='model')
    original = checkpoint_configuration(args)
    args.rate_max_simultaneous = 64
    args.log_level = 'DEBUG'
    args.openai_api_key = 'rotated'
    assert checkpoint_configuration(args) == original
    args.llm_name = 'other-model'
    assert checkpoint_configuration(args) != original
    args.llm_name = 'model'
    corpus.write_text('modified')
    assert checkpoint_configuration(args) != original
    assert 'secret' not in json.dumps(original)
