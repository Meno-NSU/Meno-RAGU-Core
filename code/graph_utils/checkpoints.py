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

from graph_utils.graph_utils import AbbreviationExpandingExtractor


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
                    raise CheckpointError('Изменились входы, код или параметры графа; чекпоинты оставлены без изменений.')
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
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.directory,
                                             suffix='.part', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.manifest, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.directory / 'manifest.json')
            self.sync_directory()
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
                required.append('05-relation-validation')
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
        if self.checkpoints is None:
            return await super()._validate_relations(context, entities_payload, relations)
        return await self._checkpointed(self.checkpoints.batch(
            '05-relation-validation', RelationsExtractionModel, super()._validate_relations,
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
