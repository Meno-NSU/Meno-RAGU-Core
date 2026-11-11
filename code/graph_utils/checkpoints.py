import dataclasses
import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.common.prompts.default_models import EntitiesExtractionModel, RelationsExtractionModel
from ragu.graph.graph_builder_pipeline import InMemoryGraphBuilder
from ragu.graph.types import Community, CommunitySummary, Entity, Relation

from graph_utils.bounded_summaries import install_budgeted_summarizers
from graph_utils.graph_utils import AbbreviationExpandingExtractor
from graph_utils.relation_review import REVIEW_STAGE, RelationReviewModel

SUMMARY_PREVIOUS_CODE = {
    'prepare_graph_index.py': 'd7dc69817e82634f3484eafeae227fdca25fb042592a38799d000fefe6aebd3a',
    'graph_utils/checkpoints.py': '3a7154bfe94215a2a00038a32fcc8bde56e96d0923536a6173ff172b7b0d86a7',
}
COMPLETION_PREVIOUS_CODE = {
    'graph_utils/checkpoints.py': 'ee05701c77cb59a463acc8bec85386fdfbb4e62e4c5c4e72333f244fffa7932f',
    'graph_utils/bounded_summaries.py': 'f287246bfefc15b88c24e3725773a77594a7a6c6800123772cb9449d9eb26b71',
}


# The code of the builds made before the relation review (463b937). The review changes only stage 05 and later, so
# such a build keeps its stages 01-04 and continues from 05 with the review.
RELATION_REVIEW_PREVIOUS_CODE = {
    'prepare_graph_index.py': 'dd8a29027bd6ef02380f5008757bd1b657ff91bcb20838ece95567f89787cf3a',
    'graph_utils/checkpoints.py': 'b6d302d995981d1cfa33cf5bfc3df355539f40de5632da6978c72508b835ee9c',
    'graph_utils/graph_utils.py': '771e82195a41c49c184e5be0d34e829153d4de86d935af35f82fcd7b6ea29aa8',
}
RELATION_REVIEW_NEW_CODE = ('graph_utils/relation_review.py',)
RELATION_REVIEW_SETTINGS = ('relation_review', 'relation_review_sources')
# The code of the review of every chunk (fe27823), before --relation_review_sources limited it to chosen sources.
REVIEW_SOURCES_PREVIOUS_CODE = {
    'prepare_graph_index.py': 'bd6e08549a6372192fd3060417c97797d2217c73edff7b979aecf6bb56fb7250',
    'graph_utils/checkpoints.py': 'fff858d5b87e590d7c2faaee0858993d12478fb011a1274037d8ef03d6afdce3',
    'graph_utils/relation_review.py': '52cc0caf25f1f06ac4b428fe80a4643897244934ad8c96d0d3718fee7fab5b00',
}


def _without_settings(configuration, names):
    previous = {**configuration}
    if 'settings' in configuration:
        previous['settings'] = {key: value for key, value in configuration['settings'].items() if key not in names}
    return previous


def previous_relation_review_configuration(configuration):
    previous = _without_settings(configuration, RELATION_REVIEW_SETTINGS)
    code = {key: value for key, value in configuration.get('code', {}).items() if key not in RELATION_REVIEW_NEW_CODE}
    previous['code'] = {**code, **RELATION_REVIEW_PREVIOUS_CODE}
    return previous


def previous_review_sources_configuration(configuration):
    previous = _without_settings(configuration, ('relation_review_sources',))
    previous['code'] = {**configuration.get('code', {}), **REVIEW_SOURCES_PREVIOUS_CODE}
    return previous


# The code of the two graph v2 builds that stopped on a summary prompt planned for 60000 tokens while gemma serves
# 32000 (--llm_context_tokens is now capped by the served limit, see llm_limits): the baseline build (relation review,
# 72163fe; no --graph_cleanup option yet) and the cleanup variant (11162d7). Both keep their stages 01-06.
CONTEXT_BASELINE_PREVIOUS_CODE = {
    'prepare_graph_index.py': 'db3c1d42e09125d118fb5565c7e108870027e265f0190e9f232fd03ce5552b1a',
    'graph_utils/checkpoints.py': '016eebb7afcbf7b002687460212bdf192d7450add2a078a57ac80960ffeacc8d',
    'graph_utils/graph_utils.py': '3fc8f88661904cb33fac653de84346e8844eadb05e8cfba0fe51b26001e0cd9a',
}
CONTEXT_CLEANUP_PREVIOUS_CODE = {
    'prepare_graph_index.py': 'c8f8da384d169702d7526b49b4effd854d84bd3e81f8aee697b281c5039da271',
    'graph_utils/checkpoints.py': '016eebb7afcbf7b002687460212bdf192d7450add2a078a57ac80960ffeacc8d',
    'graph_utils/graph_utils.py': 'b4d2a84213f1da0045348eb44fb7fcbb4f1295aa0dc083459515bcb3acbdcbc5',
}


# The code of both graph v2 builds (ef684e8) before stage 07 prepared descriptions without a row-by-row DataFrame
# update and sized prompts by their bytes first. The results are the same; stages 01-06 stay.
SUMMARY_SPEED_PREVIOUS_CODE = {
    'graph_utils/checkpoints.py': '1e664ce91c13cbbb175442eef3f69db63dbeaed32658c5190f7e127214477a10',
    'graph_utils/bounded_summaries.py': 'fc94811eacad24e2fd6f909223a324b7615dd891d74628ecc286243deaa95b47',
}


def _previous_code(configuration, code, new_files, new_settings):
    previous = _without_settings(configuration, new_settings)
    kept = {key: value for key, value in configuration.get('code', {}).items() if key not in new_files}
    previous['code'] = {**kept, **code}
    return previous


def previous_context_baseline_configuration(configuration):
    return _previous_code(configuration, CONTEXT_BASELINE_PREVIOUS_CODE,
                          ('graph_utils/llm_limits.py', 'graph_utils/graph_cleanup.py'), ('graph_cleanup',))


def previous_context_cleanup_configuration(configuration):
    return _previous_code(configuration, CONTEXT_CLEANUP_PREVIOUS_CODE, ('graph_utils/llm_limits.py',), ())


def previous_summary_speed_configuration(configuration):
    return _previous_code(configuration, SUMMARY_SPEED_PREVIOUS_CODE, (), ())


def previous_completion_configuration(configuration):
    return {**configuration, 'code': {**configuration.get('code', {}), **COMPLETION_PREVIOUS_CODE}}


def previous_summary_configuration(configuration):
    code = {key: value for key, value in configuration.get('code', {}).items()
            if key != 'graph_utils/bounded_summaries.py'}
    return {**configuration, 'code': {**code, **SUMMARY_PREVIOUS_CODE}}


# Known code changes that keep the earlier snapshots: (configuration before the change, first recomputed stage,
# backup of the old manifest, log message).
UPGRADES = (
    (previous_summary_configuration, '07-', 'manifest.before-summary-budget.json',
     'Сохраняю снимки 01–06; обобщение 07–11 учитывает контекст LLM и повторяет оборванные ответы.'),
    (previous_completion_configuration, '07-', 'manifest.before-summary-retries.json',
     'Сохраняю снимки 01–06; обобщение 07–11 учитывает контекст LLM и повторяет оборванные ответы.'),
    (previous_relation_review_configuration, '05-', 'manifest.before-relation-review.json',
     'Сохраняю снимки 01–04; проверка связей и следующие этапы считаются заново.'),
    (previous_review_sources_configuration, '05-', 'manifest.before-review-sources.json',
     'Сохраняю снимки 01–04; проверка связей выбранных источников и следующие этапы считаются заново.'),
    (previous_context_baseline_configuration, '07-', 'manifest.before-served-context.json',
     'Сохраняю снимки 01–06; обобщение 07–11 планирует промпты по контексту, который обслуживает модель.'),
    (previous_context_cleanup_configuration, '07-', 'manifest.before-served-context.json',
     'Сохраняю снимки 01–06; обобщение 07–11 планирует промпты по контексту, который обслуживает модель.'),
    (previous_summary_speed_configuration, '07-', 'manifest.before-summary-speed.json',
     'Сохраняю снимки 01–06; обобщение 07–11 готовит описания без построчного обновления таблицы.'),
)


class CheckpointError(RuntimeError):
    pass


class _CheckpointAbort(BaseException):
    pass


class SavedChunk(BaseModel):
    id: str
    content: str
    chunk_order_idx: int
    doc_id: str
    num_tokens: int | None


Checksum = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


class SavedPass(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    file: str
    sha256: Checksum
    counts: list[Annotated[int, Field(ge=0)]]
    missing: Annotated[int, Field(ge=0)]
    dependencies: dict[str, Checksum]


class SavedManifest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    version: Literal[1]
    fingerprint: Checksum
    stages: dict[str, SavedPass]


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_configuration(args):
    root = Path(__file__).resolve().parents[1]
    ignored = {
        'log_level', 'rate_max_simultaneous', 'rate_max_per_minute', 'rate_min_delay',
        'cache_dir', 'checkpoint_dir', 'openai_api_key', 'embedder_api_key', 'qdrant_key',
    }
    settings = {key: value for key, value in vars(args).items() if key not in ignored}
    files = {}
    for name in ('input_name', 'reverse_index_name', 'abbr_fname'):
        value = getattr(args, name, None)
        if value is not None:
            files[name] = file_hash(value)
    for name in ('llm_tokenizer', 'embedder_tokenizer'):
        value = getattr(args, name, None)
        if value and Path(value).is_dir():
            files[name] = {
                path.name: file_hash(path) for path in sorted(Path(value).iterdir())
                if path.is_file() and path.suffix in {'.json', '.jinja', '.txt', '.model'}
            }
    code = {'prepare_graph_index.py': file_hash(root / 'prepare_graph_index.py')}
    for folder in ('graph_utils', 'rewriting_utils', 'tokenization_utils'):
        for path in sorted((root / folder).glob('*.py')):
            code[str(path.relative_to(root))] = file_hash(path)
    code['uv.lock'] = file_hash(root.parent / 'uv.lock')
    import ragu
    package = Path(ragu.__file__).parent
    code['ragu'] = {str(path.relative_to(package)): file_hash(path) for path in sorted(package.rglob('*.py'))}
    return {'version': 1, 'settings': settings, 'files': files, 'code': code}


class GraphCheckpoints:
    def __init__(self, directory, configuration, *, before_save=None):
        self.directory = Path(directory)
        self.configuration = configuration
        self.fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        self.before_save = before_save
        self.manifest = {'version': 1, 'fingerprint': self.fingerprint, 'stages': {}}
        self.lock = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / '.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            self.lock = None
            raise CheckpointError('Другой процесс уже использует чекпоинты графа.') from exc
        try:
            path = self.directory / 'manifest.json'
            if path.exists():
                try:
                    self.manifest = SavedManifest.model_validate_json(path.read_text(encoding='utf-8')).model_dump()
                except ValidationError as exc:
                    raise CheckpointError('Повреждён манифест чекпоинтов графа.') from exc
                if self.manifest['fingerprint'] != self.fingerprint:
                    upgrade = next((item for item in UPGRADES if self.manifest['fingerprint'] == hashlib.sha256(
                        json.dumps(item[0](self.configuration), sort_keys=True).encode()).hexdigest()), None)
                    if upgrade is None:
                        raise CheckpointError('Изменились входы, код или параметры графа; чекпоинты оставлены без изменений.')
                    _, first_recomputed, backup_name, message = upgrade
                    backup = self.directory / backup_name
                    if not backup.exists():
                        self.publish_metadata(backup, self.manifest)
                    upgraded = {
                        **self.manifest, 'fingerprint': self.fingerprint,
                        'stages': {name: entry for name, entry in self.manifest['stages'].items()
                                   if name < first_recomputed},
                    }
                    self.publish_metadata(path, upgraded)
                    self.manifest = upgraded
                    logger.info('[ГРАФ: ОБНОВЛЕНИЕ] {}', message)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def dependencies(self, stage):
        return {name: entry['sha256'] for name, entry in self.manifest['stages'].items() if name < stage}

    def has_missing(self):
        return any(entry['missing'] for entry in self.manifest['stages'].values())

    @staticmethod
    def adapters(types):
        return [TypeAdapter(SavedChunk if item_type is Chunk else item_type) for item_type in types]

    def sync_directory(self):
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def publish_metadata(self, path, value):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.directory,
                                             suffix='.part', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            self.sync_directory()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, stage, types):
        entry = self.manifest['stages'].get(stage)
        if entry is None or entry['dependencies'] != self.dependencies(stage):
            return None
        logger.info('[ГРАФ: ВОССТАНОВЛЕНИЕ] {}: читаю готовый результат.', stage)
        started = time.monotonic()
        path = self.directory / entry['file']
        if path.parent != self.directory or not path.is_file() or file_hash(path) != entry['sha256']:
            raise CheckpointError(f'Повреждён чекпоинт {stage}; файл и LLM-кэш оставлены без изменений.')
        adapters = self.adapters(types)
        groups = [[] for _ in types]
        completed = 0
        previous_report = started
        try:
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    group = row['group']
                    if type(group) is not int or not 0 <= group < len(groups):
                        raise ValueError('Invalid checkpoint group')
                    item = adapters[group].validate_python(row['value'])
                    if types[group] is Chunk:
                        saved = item
                        item = Chunk(saved.content, saved.chunk_order_idx, saved.doc_id, saved.num_tokens)
                        item.id = saved.id
                    groups[group].append(item)
                    completed += 1
                    if completed % 1024 == 0 and time.monotonic() - previous_report >= 30:
                        logger.info('[ГРАФ: ЧТЕНИЕ] {}: {}/{} записей.', stage, completed, sum(entry['counts']))
                        previous_report = time.monotonic()
            if [len(group) for group in groups] != entry['counts']:
                raise ValueError('Invalid checkpoint record count')
        except (ValueError, KeyError, TypeError) as exc:
            raise CheckpointError(f'Некорректные данные чекпоинта {stage}.') from exc
        logger.info('[ГРАФ: ВОССТАНОВЛЕН] {}: записей {}; {:.1f} с.',
                    stage, sum(entry['counts']), time.monotonic() - started)
        return tuple(groups)

    def save(self, stage, groups, types):
        if self.before_save is not None:
            self.before_save()
        logger.info('[ГРАФ: СОХРАНЕНИЕ] {}: записываю результат.', stage)
        started = time.monotonic()
        adapters = self.adapters(types)
        digest = hashlib.sha256()
        temporary = None
        completed = 0
        previous_report = started
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.part', delete=False) as stream:
                temporary = Path(stream.name)
                for group, (items, adapter, item_type) in enumerate(zip(groups, adapters, types, strict=True)):
                    for item in items:
                        value = dataclasses.asdict(item) if item_type is Chunk else adapter.dump_python(item, mode='json')
                        line = (json.dumps({'group': group, 'value': value}, ensure_ascii=False) + '\n').encode()
                        stream.write(line)
                        digest.update(line)
                        completed += 1
                        if completed % 1024 == 0 and time.monotonic() - previous_report >= 30:
                            logger.info('[ГРАФ: ЗАПИСЬ] {}: {}/{} записей.',
                                        stage, completed, sum(len(items) for items in groups))
                            previous_report = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
            checksum = digest.hexdigest()
            name = f'{stage}-{checksum}.jsonl'
            temporary.replace(self.directory / name)
            self.sync_directory()
            self.manifest['stages'][stage] = {
                'file': name, 'sha256': checksum, 'counts': [len(group) for group in groups],
                'missing': sum(item is None for group in groups for item in group),
                'dependencies': self.dependencies(stage),
            }
            self.publish_metadata(self.directory / 'manifest.json', self.manifest)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        logger.info('[ГРАФ: СОХРАНЁН] {}: записей {}; {:.1f} с.',
                    stage, sum(len(group) for group in groups), time.monotonic() - started)

    async def run(self, stage, types, operation, *, reusable=True):
        cached = self.load(stage, types) if reusable else None
        if cached is not None:
            return cached
        result = await operation()
        self.save(stage, result, types)
        return result

    async def batch(self, stage, item_type, operation, inputs):
        types = (item_type | None,)
        cached = self.load(stage, types)
        result = cached[0] if cached is not None else [None] * len(inputs[0])
        if len(result) != len(inputs[0]):
            raise CheckpointError(f'Число фрагментов изменилось для {stage}.')
        missing = [index for index, item in enumerate(result) if item is None]
        if not missing:
            return result
        logger.info('[ГРАФ: ЗАПУСК] {}: требуется {} из {} фрагментов.', stage, len(missing), len(result))
        values = await operation(*[[items[index] for index in missing] for items in inputs])
        if len(values) != len(missing):
            raise CheckpointError(f'Неполный ответ прохода {stage}.')
        for index, value in zip(missing, values, strict=True):
            result[index] = value
        self.save(stage, (result,), types)
        return result


class CheckpointedExtractor(AbbreviationExpandingExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoints = None

    async def extract(self, chunks, *args, **kwargs):
        if self.checkpoints is None:
            return await super().extract(chunks, *args, **kwargs)
        method = super().extract

        async def extract():
            result = await method(chunks, *args, **kwargs)
            required = ['02-entities', '04-relations']
            if self.do_entity_validation:
                required.append('03-entity-validation')
            if self.do_relation_validation:
                required.append(REVIEW_STAGE if self.relation_review else '05-relation-validation')
            if any(name not in self.checkpoints.manifest['stages'] for name in required):
                raise CheckpointError('Проход графа завершился с ошибкой; неполный граф не сохранён.')
            return result

        try:
            return await self.checkpoints.run(
                '06-artifacts', (Entity, Relation), extract, reusable=not self.checkpoints.has_missing(),
            )
        except _CheckpointAbort as exc:
            raise exc.__cause__ from None

    async def _checkpointed(self, operation):
        try:
            return await operation
        except (CheckpointError, OSError) as exc:
            raise _CheckpointAbort() from exc

    async def _expand_abbreviations(self, chunks):
        if self.checkpoints is None:
            return await super()._expand_abbreviations(chunks)
        method = super()._expand_abbreviations

        async def expand():
            return (await method(chunks),)

        result = await self.checkpoints.run('01-abbreviations', (Chunk,), expand)
        return result[0]

    async def _extract_entities(self, context):
        if self.checkpoints is None:
            return await super()._extract_entities(context)
        return await self._checkpointed(self.checkpoints.batch(
            '02-entities', EntitiesExtractionModel, super()._extract_entities, (context,),
        ))

    async def _validate_entities(self, context, entities):
        if self.checkpoints is None:
            return await super()._validate_entities(context, entities)
        return await self._checkpointed(self.checkpoints.batch(
            '03-entity-validation', EntitiesExtractionModel, super()._validate_entities, (context, entities),
        ))

    async def _extract_relations(self, context, entities_payload):
        if self.checkpoints is None:
            return await super()._extract_relations(context, entities_payload)
        return await self._checkpointed(self.checkpoints.batch(
            '04-relations', RelationsExtractionModel, super()._extract_relations, (context, entities_payload),
        ))

    async def _validate_relations(self, context, entities_payload, relations):
        if self.checkpoints is None or self.relation_review:
            return await super()._validate_relations(context, entities_payload, relations)
        return await self._checkpointed(self.checkpoints.batch(
            '05-relation-validation', RelationsExtractionModel, super()._validate_relations,
            (context, entities_payload, relations),
        ))

    async def _review_relations(self, context, entities_payload, relations):
        if self.checkpoints is None:
            return await super()._review_relations(context, entities_payload, relations)
        return await self._checkpointed(self.checkpoints.batch(
            REVIEW_STAGE, RelationReviewModel, super()._review_relations,
            (context, entities_payload, relations),
        ))


class CheckpointedGraphBuilder(InMemoryGraphBuilder):
    @classmethod
    def from_pipeline(cls, pipeline, checkpoints):
        result = cls(
            embedder=pipeline.embedder, llm=pipeline.llm, chunker=pipeline.chunker,
            artifact_extractor=pipeline.artifact_extractor, build_parameters=pipeline.build_parameters,
            additional_pipeline=pipeline.additional_pipeline, language=pipeline.language,
        )
        result.checkpoints = checkpoints
        install_budgeted_summarizers(result)
        return result

    async def extract_graph(self, chunks):
        if self.build_parameters.build_only_vector_context:
            return await super().extract_graph(chunks)
        result = await self.checkpoints.run(
            '11-graph-ready', (Entity, Relation, CommunitySummary, Community),
            lambda: self._extract_checkpointed(chunks), reusable=not self.checkpoints.has_missing(),
        )
        return (*result, chunks)

    async def _extract_checkpointed(self, chunks):
        if self.artifact_extractor is None:
            raise ValueError('The graph artifact extractor is missing.')
        entities, relations = await self.artifact_extractor(chunks)

        async def summarize_entities():
            return (await self.entity_summarizer.run(entities),)

        entities, = await self.checkpoints.run('07-entity-summaries', (Entity,), summarize_entities)

        async def summarize_relations():
            return (await self.relation_summarizer.run(relations),)

        relations, = await self.checkpoints.run('08-relation-summaries', (Relation,), summarize_relations)
        for module in self.additional_pipeline or []:
            output = await module.run(entities, relations)
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError('Graph builder module must return an (entities, relations) tuple.')
            entities, relations = output
        communities = []
        summaries = []
        if self.build_parameters.make_community_summary:
            if self.community_summarizer is None:
                raise ValueError('Community summarization requires an LLM client.')

            async def cluster():
                return entities, await self.cluster_graph(entities, relations)

            entities, communities = await self.checkpoints.run('09-communities', (Entity, Community), cluster)
            if communities:
                async def summarize_communities():
                    return (await self.community_summarizer.summarize(communities),)

                summaries, = await self.checkpoints.run(
                    '10-community-summaries', (CommunitySummary,), summarize_communities,
                )
        return entities, relations, summaries, communities
