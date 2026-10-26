from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from nltk.stem.snowball import SnowballStemmer

from meno_rag.config import Settings
from meno_rag.stand.rewriting import load_abbreviations


class GraphRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragu")
        self.engines: Any = None
        try:
            self.executor.submit(self._load).result()
            assert self.engines is not None
            self.executor.submit(self.engines.event_loop.run_forever)
        except BaseException:
            self.executor.shutdown(wait=False, cancel_futures=True)
            raise

    def _load(self):
        from graph_utils.graph_utils import load_graph_engines

        self.engines = load_graph_engines(
            graph_dir=str(self.settings.graph_dir),
            embedder_name=self.settings.embedder_model,
            embedder_url=self.settings.embedder_url,
            embedder_key=self.settings.embedder_api_key,
            qdrant_location=self.settings.qdrant_url,
            engine_reranker_name=self.settings.engine_reranker_model,
            engine_reranker_url=self.settings.engine_reranker_url,
            engine_reranker_key=self.settings.engine_reranker_api_key,
        )
        try:
            self.engines.event_loop.run_until_complete(self._validate_collections())
        except BaseException:
            self.engines.event_loop.run_until_complete(self._close_clients())
            self.engines.event_loop.close()
            raise

    async def _validate_collections(self):
        index = self.engines.local_engine.knowledge_graph.index
        for storage in (index.nodes_vector_db, index.edges_vector_db, index.chunks_vector_db):
            client = storage._get_client()
            if not await client.collection_exists(storage.collection_name):
                raise ValueError(
                    f"Missing Qdrant collection {storage.collection_name!r}; restore its snapshot and qdrant_collections.json."
                )
            info = await client.get_collection(storage.collection_name)
            if info.points_count == 0:
                raise ValueError(f"Empty Qdrant collection: {storage.collection_name}")

    async def _close_clients(self):
        await self.engines.local_engine.knowledge_graph.index.close()
        clients = set()
        for engine in (self.engines.local_engine, self.engines.naive_engine):
            for obj in (engine.llm, engine.retriever.embedder, engine.reranker):
                client = getattr(getattr(obj, "client", None), "client", None)
                if client is not None and id(client) not in clients:
                    clients.add(id(client))
                    await client.close()

    async def search(self, query: str) -> dict[str, Any]:
        fn = partial(
            self.engines.asearch,
            search_query=query,
            local_top_k=self.settings.graph_top_k,
            naive_top_k=self.settings.top_k,
            local_rerank_top_k=self.settings.engine_rerank_top_k if self.settings.engine_reranker_model else None,
            naive_rerank_top_k=self.settings.engine_rerank_top_k if self.settings.engine_reranker_model else None,
        )
        return await asyncio.wait_for(
            asyncio.wrap_future(asyncio.run_coroutine_threadsafe(fn(), self.engines.event_loop)),
            timeout=self.settings.graph_timeout_seconds,
        )

    async def aclose(self):
        if self.engines is not None:
            loop = self.engines.event_loop
            future = asyncio.run_coroutine_threadsafe(self._close_clients(), loop)
            await asyncio.wrap_future(future)
            loop.call_soon_threadsafe(loop.stop)
            await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
            loop.close()
        else:
            self.executor.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class StandResources:
    documents: list[dict[str, Any]]
    chunk_mapping: dict[str, dict[str, int]]
    graph: Any
    stemmer: SnowballStemmer
    abbreviations: dict
    missing_quality_count: int = 0
    fewshots_enabled: bool = False
    fewshot_retriever: Any = None


def _load_documents(path: Path) -> tuple[list[dict[str, Any]], int]:
    documents = []
    missing = 0
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            doc = json.loads(line)
            if "doc_full_text" not in doc:
                doc["doc_full_text"] = doc["text"]
            doc.setdefault("doc_title", "")
            doc.setdefault("doc_annotation", "")
            if "quality_score" not in doc:
                doc["quality_score"] = 1.0
                missing += 1
            documents.append(doc)
    return documents, missing


def _load_chunk_mapping(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_mapping(mapping: dict, documents: list[dict] | None = None) -> None:
    if not mapping or set(mapping) != {str(i) for i in range(len(mapping))}:
        raise ValueError("Chunk mapping must be nonempty and contiguous from 0.")
    if documents is None:
        return
    for item in mapping.values():
        if not 0 <= item["doc_index"] < len(documents):
            raise ValueError("Invalid document index in chunk mapping.")
        doc = documents[item["doc_index"]]
        if not 0 <= item["local_chunk_index"] < len(doc["chunks"]):
            raise ValueError("Invalid local chunk index in chunk mapping.")
        chunk = doc["chunks"][item["local_chunk_index"]]
        if not 0 <= chunk["start_char"] < chunk["end_char"] <= len(doc["doc_full_text"]):
            raise ValueError("Chunk mapping does not match corpus boundaries.")


def load_stand_resources(settings: Settings) -> StandResources:
    documents, missing = _load_documents(settings.corpus_path)
    mapping = _load_chunk_mapping(settings.chunk_mapping_path)
    _validate_mapping(mapping, documents)
    for name in ("knowledge_graph.gml", "kv_chunks.json"):
        if not (settings.graph_dir / name).is_file():
            raise FileNotFoundError(settings.graph_dir / name)
    sidecar = settings.graph_dir / "menon_chunk_mapping.json"
    graph_mapping = json.loads(sidecar.read_text(encoding="utf-8"))
    if not graph_mapping or any(str(i) not in mapping for i in graph_mapping.values()):
        raise ValueError("RAGU graph mapping does not match the corpus chunk mapping.")
    abbreviations = load_abbreviations(settings.abbreviations_path)
    return StandResources(
        documents=documents,
        chunk_mapping=mapping,
        graph=GraphRuntime(settings),
        stemmer=SnowballStemmer("russian"),
        abbreviations=abbreviations,
        missing_quality_count=missing,
    )
