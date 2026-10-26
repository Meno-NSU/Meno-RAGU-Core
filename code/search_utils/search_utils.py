import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


def load_dense_embedder(embedder_name: str, device: str = 'cpu') -> SentenceTransformer:
    try:
        return SentenceTransformer(embedder_name, trust_remote_code=True, device=device)
    except:
        return SentenceTransformer(os.path.normpath(embedder_name), trust_remote_code=True, device=device)


def vectorize_texts(texts: List[str], embedder, batch_size: int = 64) -> np.ndarray:
    return embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )


class DenseEmbedderService:
    """
    Adapter over an OpenAI-compatible embeddings service with the encode()
    interface of SentenceTransformer, so that the scoring pipeline can use a
    served model (e.g. bge-m3 in vLLM) instead of a locally loaded one.
    """

    def __init__(self, model_name: str, base_url: str = 'http://localhost:8001/v1',
                 api_key: str = 'EMPTY', timeout: float = 120.0, num_retries: int = 3):
        self.model_name = model_name
        self.num_retries = num_retries
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def encode(self, texts: List[str], batch_size: int = 64,
               normalize_embeddings: bool = True, show_progress_bar: bool = False,
               convert_to_numpy: bool = True) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for start_idx in range(0, len(texts), batch_size):
            batch = [str(cur_text) for cur_text in texts[start_idx:start_idx + batch_size]]
            response = None
            for attempt in range(self.num_retries + 1):
                try:
                    response = self.client.embeddings.create(model=self.model_name, input=batch)
                    break
                except Exception as err:
                    if attempt >= self.num_retries:
                        raise err
                    time.sleep(min(10.0, 2.0 ** attempt))
            all_embeddings.extend(
                [np.asarray(cur_item.embedding, dtype=np.float32) for cur_item in response.data]
            )
        del batch
        result = np.vstack(all_embeddings)
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            result = result / norms
        return result


def combine_relevant_chunks(chunk_list_1: Optional[List[Tuple[int, float]]] = None,
                            chunk_list_2: Optional[List[Tuple[int, float]]] = None) -> List[Tuple[int, float]]:
    united_chunks = dict()
    if chunk_list_1 is not None:
        for idx, score in chunk_list_1:
            if idx in united_chunks:
                if score > united_chunks[idx]:
                    united_chunks[idx] = score
            else:
                united_chunks[idx] = score
    if chunk_list_2 is not None:
        for idx, score in chunk_list_2:
            if idx in united_chunks:
                if score > united_chunks[idx]:
                    united_chunks[idx] = score
            else:
                united_chunks[idx] = score
    united_chunks_ = sorted(
        list(map(lambda it1: (it1, united_chunks[it1]), united_chunks.keys())),
        key=lambda it2: (-it2[1], it2[0])
    )
    del united_chunks
    return united_chunks_
