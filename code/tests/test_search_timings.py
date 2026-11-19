"""Search timings (graph_utils.search_timings): each search gets its own numbers, the wrapped calls are unchanged."""
import asyncio
from types import SimpleNamespace

import pytest
from graph_utils import search_timings
from graph_utils.graph_utils import GraphSearchEngines
from graph_utils.local_search import rerank_with_scores


class Embedder:
    def __init__(self):
        self.calls = []

    async def batch_embed_text(self, texts):
        self.calls.append(list(texts))
        await asyncio.sleep(0)
        return [[float(len(text)), 1.0] for text in texts]

    async def embed_text(self, text):
        return [float(len(text)), 1.0]


class Scorer:
    async def score(self, text_1, text_2):
        await asyncio.sleep(0)
        return [(index, 1.0 / (index + 1)) for index in range(len(text_2))]


class Storage:
    async def query(self, points, top_k=10):
        await asyncio.sleep(0)
        return [[] for _ in points]


def instrumented():
    embedder, scorer, storage = Embedder(), Scorer(), Storage()
    search_timings.instrument_embedder(embedder)
    search_timings.instrument_scorer(scorer)
    search_timings.instrument_storage(storage)
    return embedder, scorer, storage


def engines(embedder, scorer, storage, entity_pairs):
    """A local engine that embeds, queries Qdrant and scores ``entity_pairs`` entities; a naive one scoring 3 chunks."""
    async def local(query, params=None):
        await embedder.batch_embed_text([query])
        await storage.query([query])
        await rerank_with_scores(query, list(range(entity_pairs)), str, scorer, kind='entity')
        return SimpleNamespace(result=SimpleNamespace(chunks=[], entities=[], relations=[], summaries=[]),
                               metrics={'entities': []})

    async def naive(query, params=None):
        await embedder.batch_embed_text([query])
        await storage.query([query])
        await scorer.score(query, ['a', 'b', 'c'])
        return SimpleNamespace(result=SimpleNamespace(chunks=[], scores=[]), metrics={})

    return GraphSearchEngines(local_engine=SimpleNamespace(search=local), naive_engine=SimpleNamespace(search=naive),
                              chunk_to_global={})


def test_a_search_reports_its_embedder_qdrant_and_cross_encoder_calls():
    embedder, scorer, storage = instrumented()
    pool = asyncio.run(engines(embedder, scorer, storage, 4).asearch('вопрос'))
    timings = pool['timings']
    assert (timings['embed_calls'], timings['embed_texts']) == (2, 2)
    assert timings['qdrant_calls'] == 2
    assert (timings['ce_calls'], timings['ce_pairs']) == (2, 7)
    assert timings['ce_pairs_by_kind'] == {'entity': 4, 'chunk_naive': 3}
    assert all(timings[key] >= 0 for key in ('search_ms', 'embed_ms', 'qdrant_ms', 'ce_ms', 'graph_cpu_ms'))
    # the pool itself is what the engines returned
    assert {key: value for key, value in pool.items() if key != 'timings'} == {
        'chunks': [], 'entities': [], 'relations': [], 'num_relations': 0, 'num_summaries': 0}


def test_concurrent_searches_keep_their_own_numbers():
    embedder, scorer, storage = instrumented()

    async def run():
        return await asyncio.gather(engines(embedder, scorer, storage, 2).asearch('один'),
                                    engines(embedder, scorer, storage, 10).asearch('два'))

    first, second = asyncio.run(run())
    assert first['timings']['ce_pairs_by_kind'] == {'entity': 2, 'chunk_naive': 3}
    assert second['timings']['ce_pairs_by_kind'] == {'entity': 10, 'chunk_naive': 3}
    assert first['timings']['embed_calls'] == second['timings']['embed_calls'] == 2


def test_the_chunk_channel_alone_is_timed_too():
    embedder, scorer, storage = instrumented()
    pool = asyncio.run(engines(embedder, scorer, storage, 1).asearch('вопрос', use_graph=False))
    assert pool['timings']['embed_calls'] == 1 and pool['timings']['ce_pairs_by_kind'] == {'chunk_naive': 3}


def test_wrapped_methods_return_the_same_results_and_raise_the_same_errors():
    embedder, scorer, _ = instrumented()
    plain = Embedder()
    assert asyncio.run(embedder.batch_embed_text(['ab', 'c'])) == asyncio.run(plain.batch_embed_text(['ab', 'c']))
    assert asyncio.run(scorer.score('q', ['x', 'y'])) == asyncio.run(Scorer().score('q', ['x', 'y']))

    class Failing:
        async def score(self, text_1, text_2):
            raise RuntimeError('down')

    failing = Failing()
    search_timings.instrument_scorer(failing)
    timings = search_timings.new_timings()

    async def run():
        with search_timings.collecting(timings):
            await failing.score('q', ['x'])

    with pytest.raises(RuntimeError, match='down'):
        asyncio.run(run())
    assert timings['ce_calls'] == 1 and timings['ce_pairs'] == 1


def test_instrumenting_twice_wraps_once_and_nothing_is_recorded_outside_a_search():
    embedder = Embedder()
    search_timings.instrument_embedder(embedder)
    wrapped = embedder.batch_embed_text
    search_timings.instrument_embedder(embedder)
    assert embedder.batch_embed_text is wrapped
    asyncio.run(embedder.batch_embed_text(['outside']))  # no running search: no error, nothing to record into
    assert search_timings._CURRENT.get() is None
