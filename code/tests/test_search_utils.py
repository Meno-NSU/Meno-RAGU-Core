from types import SimpleNamespace

import numpy as np
import pytest

from search_utils.search_utils import DenseEmbedderService, combine_relevant_chunks, vectorize_texts


class _FakeEmbeddings:
    def __init__(self, vectors_by_text):
        self._vectors_by_text = vectors_by_text
        self.requests = []

    def create(self, model, input):
        self.requests.append(list(input))
        return SimpleNamespace(data=[
            SimpleNamespace(embedding=self._vectors_by_text[text]) for text in input
        ])


class _FakeClient:
    def __init__(self, vectors_by_text):
        self.embeddings = _FakeEmbeddings(vectors_by_text)


def _make_service(vectors_by_text):
    service = DenseEmbedderService('BAAI/bge-m3', base_url='http://fake:8001/v1', api_key='EMPTY')
    service.client = _FakeClient(vectors_by_text)
    return service


class TestCombineRelevantChunks:
    def test_none_inputs_give_empty_list(self):
        assert combine_relevant_chunks(None, None) == []

    def test_single_list_is_sorted_by_score_descending(self):
        result = combine_relevant_chunks([(3, 0.5), (1, 0.9), (2, 0.7)])
        assert result == [(1, 0.9), (2, 0.7), (3, 0.5)]

    def test_duplicates_take_the_maximum_score(self):
        result = combine_relevant_chunks([(1, 0.3), (2, 0.8)], [(1, 0.6)])
        assert result == [(2, 0.8), (1, 0.6)]

    def test_equal_scores_are_ordered_by_index(self):
        result = combine_relevant_chunks([(5, 0.5), (2, 0.5)])
        assert result == [(2, 0.5), (5, 0.5)]


class TestDenseEmbedderService:
    def test_encode_normalizes_rows(self):
        texts = ['a', 'b']
        vectors = {'a': [3.0, 0.0, 0.0, 0.0], 'b': [0.0, 0.0, 5.0, 0.0]}
        service = _make_service(vectors)
        result = service.encode(texts, batch_size=8, normalize_embeddings=True)
        assert result.shape == (2, 4)
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
        assert np.allclose(result[0], [1.0, 0.0, 0.0, 0.0])

    def test_encode_without_normalization_keeps_vectors(self):
        texts = ['a']
        vectors = {'a': [3.0, 4.0]}
        service = _make_service(vectors)
        result = service.encode(texts, batch_size=8, normalize_embeddings=False)
        assert np.allclose(result[0], [3.0, 4.0])

    def test_encode_splits_into_batches_preserving_order(self):
        texts = ['a', 'b', 'c', 'd', 'e']
        vectors = {text: [float(idx), 1.0] for idx, text in enumerate(texts)}
        service = _make_service(vectors)
        result = service.encode(texts, batch_size=2, normalize_embeddings=False)
        assert result.shape == (5, 2)
        assert [service.client.embeddings.requests[0], service.client.embeddings.requests[1],
                service.client.embeddings.requests[2]] == [['a', 'b'], ['c', 'd'], ['e']]
        assert np.allclose(result[:, 0], np.arange(5, dtype=float))

    def test_zero_vector_is_kept_without_division_error(self):
        texts = ['a', 'b']
        vectors = {'a': [0.0, 0.0], 'b': [1.0, 0.0]}
        service = _make_service(vectors)
        result = service.encode(texts, batch_size=8, normalize_embeddings=True)
        assert np.isfinite(result).all()

    def test_vectorize_texts_works_with_service(self):
        texts = ['a', 'b']
        vectors = {'a': [2.0, 0.0], 'b': [0.0, 2.0]}
        service = _make_service(vectors)
        result = vectorize_texts(texts, service, batch_size=8)
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
