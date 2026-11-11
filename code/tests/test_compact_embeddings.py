import asyncio

import numpy as np
from graph_utils.compact_embeddings import CompactEmbeddingClient
from ragu.models.embedder import EmbedderOpenAI


class Client:
    base_url = 'http://embedder.test/v1'

    def __init__(self):
        self.calls = []

    async def embed_texts(self, model_name, texts, **kwargs):
        self.calls.append(list(texts))
        return [[float(np.float32(0.1 * len(text))), float(np.float32(-1.5)), 0.0] for text in texts]

    async def embed_text(self, model_name, text, **kwargs):
        return (await self.embed_texts(model_name, [text]))[0]


def test_vectors_arrive_as_float32_arrays_with_the_server_values():
    client = Client()
    embedder = EmbedderOpenAI(client=CompactEmbeddingClient(client), model_name='bge-m3', batch_size=2,
                              embedder_token_limit=10**6, tokenizer_backend='tiktoken')
    embedder._truncator = lambda text: text
    vectors = asyncio.run(embedder.batch_embed_text(['ab', 'abcd', 'a']))
    assert sorted(client.calls) == [['a'], ['ab', 'abcd']]  # sub-batches run concurrently
    assert all(isinstance(v, np.ndarray) and v.dtype == np.float32 for v in vectors)
    assert [v.tolist() for v in vectors] == [
        [float(np.float32(0.2)), -1.5, 0.0], [float(np.float32(0.4)), -1.5, 0.0], [float(np.float32(0.1)), -1.5, 0.0]]
    # a vector of 1024 numbers takes 4 KiB instead of the ~32 KiB of a list of Python floats
    assert vectors[0].nbytes == 3 * 4


def test_other_client_calls_reach_the_wrapped_client():
    client = Client()
    compact = CompactEmbeddingClient(client)
    assert compact.base_url == client.base_url
    assert asyncio.run(compact.embed_text('bge-m3', 'abc'))[1] == -1.5
