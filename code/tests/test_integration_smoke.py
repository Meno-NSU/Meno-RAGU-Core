import asyncio
import hashlib
import math
import os

import pytest

from graph_utils.graph_utils import (
    GraphSearchEngines,
    RANDOM_SEED,
    build_chunk_mapping,
    find_relevant_items_by_graph,
    save_chunk_mapping,
)
from ragu import BuilderArguments, KnowledgeGraph, LocalSearchEngine, NaiveSearchEngine, Settings, StorageArguments
from ragu.graph.types import Entity, Relation
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage
from ragu.utils.ragu_utils import compute_mdhash_id

import sys
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


CHUNK_ID_PREFIX = 'chunk-'
EMBEDDING_DIM = 64

CHUNK_TEXTS = [
    'Иванов Иван Иванович читает лекции по математическому анализу в НГУ.',
    'Петров Пётр Петрович ведёт семинары по программированию на факультете информационных технологий.',
    'НГУ расположен в Академгородке Новосибирска.',
]

QUERY = 'Где Иванов читает лекции?'


def _fake_vector(text: str) -> list:
    vector = [0.0] * EMBEDDING_DIM
    for token in str(text).lower().replace('ё', 'е').split():
        digest = hashlib.md5(token.encode('utf-8')).hexdigest()
        vector[int(digest, 16) % EMBEDDING_DIM] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0.0:
        vector = [value / norm for value in vector]
    return vector


class FakeEmbedder:
    """
    Deterministic embedder for the hermetic integration tests: hashed bag-of-words
    vectors, no models and no network.
    """

    def __init__(self):
        self._dim = EMBEDDING_DIM

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text, **kwargs):
        return _fake_vector(text)

    async def batch_embed_text(self, texts, desc=None, **kwargs):
        return [_fake_vector(text) for text in texts]


def _make_dummy_llm():
    dummy_client = CachedAsyncOpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
    return LLMOpenAI(client=dummy_client, model_name='unused')


def _make_knowledge_graph(embedder, sparse_embedder=None, qdrant_location=':memory:'):
    Settings.language = 'russian'
    return KnowledgeGraph(
        llm=None,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        chunker=None,
        builder_settings=BuilderArguments(
            use_llm_summarization=False,
            use_clustering=False,
            make_community_summary=False,
            remove_isolated_nodes=False,
            build_only_vector_context=True,
            random_seed=RANDOM_SEED
        ),
        storage_settings=StorageArguments(
            vdb_storage_type=QdrantVectorDBStorage,
            vdb_storage_kwargs={'location': qdrant_location}
        )
    )


def _build_mini_graph(graph_dir):
    Settings.storage_folder = graph_dir
    embedder = FakeEmbedder()
    knowledge_graph = _make_knowledge_graph(embedder)
    chunk_ids = [compute_mdhash_id(text, prefix=CHUNK_ID_PREFIX) for text in CHUNK_TEXTS]

    async def build_all():
        await knowledge_graph.build_from_docs(list(CHUNK_TEXTS))
        entities = [
            Entity(entity_name='Иванов Иван Иванович', entity_type='PER',
                   description='доцент, читает лекции по математическому анализу',
                   source_chunk_id=[chunk_ids[0]]),
            Entity(entity_name='НГУ', entity_type='ORG',
                   description='Новосибирский государственный университет',
                   source_chunk_id=[chunk_ids[0], chunk_ids[2]]),
        ]
        await knowledge_graph.upsert_entities(entities)
        await knowledge_graph.upsert_relations([
            Relation(subject_id=entities[0].id, object_id=entities[1].id,
                     subject_name=entities[0].entity_name, object_name=entities[1].entity_name,
                     relation_type='работает в',
                     description='Иванов Иван Иванович работает в НГУ',
                     source_chunk_id=[chunk_ids[0]])
        ])

    asyncio.run(build_all())
    chunk_to_global = build_chunk_mapping(CHUNK_TEXTS, list(range(len(CHUNK_TEXTS))))
    save_chunk_mapping(chunk_to_global, graph_dir)
    return knowledge_graph, chunk_to_global


def _make_engines(knowledge_graph, chunk_to_global):
    embedder = FakeEmbedder()
    dummy_llm = _make_dummy_llm()
    local_engine = LocalSearchEngine(llm=dummy_llm, knowledge_graph=knowledge_graph, embedder=embedder)
    naive_engine = NaiveSearchEngine(llm=dummy_llm, knowledge_graph=knowledge_graph, embedder=embedder)
    return GraphSearchEngines(local_engine=local_engine, naive_engine=naive_engine,
                              chunk_to_global=chunk_to_global)


@pytest.mark.integration
class TestMiniGraphSearch:
    def test_naive_and_local_search_on_mini_graph(self, tmp_path):
        graph_dir = str(tmp_path / 'graph')
        os.makedirs(graph_dir, exist_ok=True)
        knowledge_graph, chunk_to_global = _build_mini_graph(graph_dir)
        assert len(chunk_to_global) == len(CHUNK_TEXTS)

        engines = _make_engines(knowledge_graph, chunk_to_global)
        pool = find_relevant_items_by_graph(
            graph_engines=engines,
            search_query=QUERY,
            local_top_k=10,
            naive_top_k=5
        )

        assert len(pool['chunks']) > 0
        naive_chunk_indices = [chunk_index for chunk_index, _ in pool['chunks']]
        assert 0 in naive_chunk_indices

        assert len(pool['entities']) > 0
        entity_names = {cur_item['name'] for cur_item in pool['entities']}
        assert 'Иванов Иван Иванович' in entity_names
        ivanov_items = [cur_item for cur_item in pool['entities']
                        if cur_item['name'] == 'Иванов Иван Иванович']
        assert ivanov_items[0]['sources'] == [0]
        assert ivanov_items[0]['type'] == 'PER'
        assert 0.0 < ivanov_items[0]['score'] <= 1.0

        assert len(pool['relations']) > 0
        relation_types = {cur_item['relation_type'] for cur_item in pool['relations']}
        assert 'работает в' in relation_types
        assert abs(sum(cur_item['score'] for cur_item in pool['relations']) - 1.0) < 1e-6
