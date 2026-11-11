"""
Embedding client of the graph build that keeps every vector as a float32 array.

The graph index embeds all relations (or entities, or chunks) of a build in one call and keeps every vector until it
has stored them all in Qdrant. The OpenAI client returns a vector as a list of Python floats, about 32 KiB for the
1024 numbers of bge-m3 against the 4 KiB of float32: the 1.94M relations of graph v2 took about 100 GB of memory. The
server sends float32 numbers, so the arrays hold the same values; each sub-batch is converted as soon as it arrives.
"""
import numpy as np


class CompactEmbeddingClient:
    def __init__(self, client):
        self._client = client

    async def embed_texts(self, *args, **kwargs):
        return [np.asarray(vector, dtype=np.float32) for vector in await self._client.embed_texts(*args, **kwargs)]

    def __getattr__(self, name):
        return getattr(self._client, name)
