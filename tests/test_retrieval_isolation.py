"""Retrieval thread-pool isolation: BM25 concurrency is bounded (so a burst of
concurrent requests can't flood the shared thread pool), and the backend runs
retrieval on a dedicated executor."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import StandRagPipeline


class _Res:
    faiss_retriever = "FAISS"
    bm25_retriever = "BM25"
    stemmer = "stem"
    embedder = ("tok", "mdl", "cpu")
    documents: list = []
    chunk_mapping: dict = {}


class _NoLLM:
    async def chat_completion(self, **kwargs):  # pragma: no cover
        return {}

    async def chat_completion_text(self, **kwargs):  # pragma: no cover
        return ""

    async def stream_chat_completion(self, **kwargs):  # pragma: no cover
        if False:
            yield ""


def _pipeline(bm25_semaphore):
    return StandRagPipeline(
        settings=get_settings(),
        resources=_Res(),
        llm_router=_NoLLM(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(8),
        bm25_semaphore=bm25_semaphore,
    )


def test_lifespan_installs_retrieval_executor():
    from meno_rag.api.main import app

    with TestClient(app):
        assert isinstance(app.state.retrieval_executor, ThreadPoolExecutor)
