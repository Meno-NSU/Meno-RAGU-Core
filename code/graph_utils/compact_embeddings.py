"""
Embedding client of the graph build that keeps every vector as a float32 array.

The graph index embeds all relations (or entities, or chunks) of a build in one call and keeps every vector until it
has stored them all in Qdrant. The OpenAI client returns a vector as a list of Python floats, about 32 KiB for the
1024 numbers of bge-m3 against the 4 KiB of float32: the 1.94M relations of graph v2 took about 100 GB of memory. The
server sends float32 numbers, so the arrays hold the same values; each sub-batch is converted as soon as it arrives.

Several servers of the same model (--embedder_extra_urls) share the sub-batches: each goes to the server with the
fewest requests in flight, so a faster server takes more. One server caps at about 320 texts a second on meno.
"""
import numpy as np


class CompactEmbeddingClient:
    def __init__(self, client, *others):
        self._clients = [client, *others]
        self._busy = [0] * len(self._clients)

    async def embed_texts(self, *args, **kwargs):
        index = min(range(len(self._clients)), key=self._busy.__getitem__)
        self._busy[index] += 1
        try:
            vectors = await self._clients[index].embed_texts(*args, **kwargs)
        finally:
            self._busy[index] -= 1
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    def __getattr__(self, name):
        return getattr(self._clients[0], name)
