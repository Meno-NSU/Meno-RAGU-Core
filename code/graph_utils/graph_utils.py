import asyncio
import codecs
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from nltk.stem.snowball import SnowballStemmer
from rapidfuzz.distance import Levenshtein

try:
    from ragu import BuilderArguments, KnowledgeGraph, LocalSearchEngine, NaiveSearchEngine, Settings, StorageArguments, TwoStageArtifactsExtractorLLM
    from ragu.chunker.types import Chunk
    from ragu.graph.types import Entity, Relation
    from ragu.models.embedder import Embedder, EmbedderOpenAI
    from ragu.models.llm import LLMOpenAI
    from ragu.models.openai import CachedAsyncOpenAI
    from ragu.models.scorer import Scorer, ScorerOpenAI
    from ragu.models.sparse_embedder import BM25 as RagUBM25
    from ragu.search_engine.local_search import LocalParams
    from ragu.search_engine.naive_search import NaiveSearchParams
    from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage
    from ragu.utils.ragu_utils import compute_mdhash_id
except ImportError as err:
    raise ImportError('The "ragu" package (https://github.com/RaguTeam/RAGU) is required for '
                      'the graph-based search. Please install it first.') from err

from search_utils.search_utils import combine_relevant_chunks

try:
    from context_utils.context_utils import global_chunk_index_to_local
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from context_utils.context_utils import global_chunk_index_to_local


RANDOM_SEED: int = 42
CHUNK_ID_PREFIX: str = 'chunk-'
GRAPH_MAPPING_FILE_NAME: str = 'menon_chunk_mapping.json'
SPARSE_MODEL_LANGUAGE: str = 'russian'
RUSSIAN_STEMMER = SnowballStemmer('russian')


def _normalize_metadata_text(value: Any) -> str:
    return ' '.join(str(value).strip().split()).strip()


def enrich_chunk_text(chunk_text: str, document: Dict[str, Union[str, float, List[Dict[str, int]]]],
                      mode: str) -> str:
    """
    Prepends the frozen document metadata header to the chunk text (the contextual
    enrichment of the graph chunks). The header format is a part of the chunk text,
    so changing it changes the md5-based chunk identifiers and invalidates the whole
    graph. The modes are:
      'none'             - the chunk text is returned as is;
      'title'            - the "**Документ:** <название>." header is prepended;
      'title_annotation' - the "**Аннотация:** <аннотация>." line is also added.
    The missing metadata silently degrade the mode: without a title the header is
    not added at all, and without an annotation the "title_annotation" mode falls
    back to "title".
    """
    if mode not in ('title', 'title_annotation'):
        return chunk_text
    doc_title = _normalize_metadata_text(document.get('doc_title', ''))
    if len(doc_title) == 0:
        return chunk_text
    header = f'**Документ:** {doc_title}'
    if header[-1] not in {'.', ',', '?', '!', ':', ';'}:
        header += '.'
    if mode == 'title_annotation':
        doc_annotation = _normalize_metadata_text(document.get('doc_annotation', ''))
        if len(doc_annotation) > 0:
            annotation_line = f'**Аннотация:** {doc_annotation}'
            if annotation_line[-1] not in {'.', ',', '?', '!', ':', ';'}:
                annotation_line += '.'
            header += f'\n\n{annotation_line}'
    return f'{header}\n\n{chunk_text}'


def collect_knowledge_base_chunks(documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                                  chunk_mapping: Dict[str, Dict[str, int]],
                                  content_sources: Optional[set] = None,
                                  max_num_chunks: Optional[int] = None,
                                  enrich_mode: str = 'none',
                                  enrich_sources: Optional[set] = None,
                                  stats: Optional[Dict[str, int]] = None
                                  ) -> Tuple[List[str], List[int]]:
    """
    Collects the exact chunk texts of the Meno knowledge base in the order of the
    global chunk indices. Only the documents with the requested content sources are used
    (all sources are used if the content_sources set is None). If enrich_mode is not
    'none', the chunks of the documents whose content_source is in enrich_sources
    (all sources if it is None) are prepended with the document metadata header (see
    enrich_chunk_text). If the stats dictionary is passed, it is filled with the
    'total' (all collected chunks) and 'enriched' (chunks with the added header) counts.
    """
    chunk_texts = []
    global_indices = []
    num_enriched_chunks = 0
    for str_global_idx in sorted(chunk_mapping.keys(), key=lambda idx: int(idx)):
        global_idx = int(str_global_idx)
        document_index, local_chunk_index = global_chunk_index_to_local(global_idx, documents, chunk_mapping)
        cur_document = documents[document_index]
        cur_source = cur_document.get('content_source')
        if content_sources is not None:
            if cur_source not in content_sources:
                continue
        try:
            doc_text = cur_document['doc_full_text']
        except:
            doc_text = cur_document['text']
        chunk_boundaries = cur_document['chunks'][local_chunk_index]
        chunk_text = doc_text[chunk_boundaries['start_char']:chunk_boundaries['end_char']].strip()
        if (enrich_mode != 'none') and ((enrich_sources is None) or (cur_source in enrich_sources)):
            enriched_text = enrich_chunk_text(chunk_text, cur_document, enrich_mode)
            if enriched_text != chunk_text:
                num_enriched_chunks += 1
            chunk_text = enriched_text
        chunk_texts.append(chunk_text)
        global_indices.append(global_idx)
        if (max_num_chunks is not None) and (len(chunk_texts) >= max_num_chunks):
            break
    if stats is not None:
        stats['total'] = len(chunk_texts)
        stats['enriched'] = num_enriched_chunks
    return chunk_texts, global_indices


def build_chunk_mapping(chunk_texts: List[str], global_indices: List[int]) -> Dict[str, int]:
    """
    Builds the mapping of the RAGU chunk identifiers (md5-based hashes of the chunk texts)
    to the global chunk indices of the Meno knowledge base. If the same text appears in
    several chunks, the smallest global index wins.
    """
    chunk_to_global = dict()
    for cur_text, cur_global_idx in zip(chunk_texts, global_indices):
        cur_chunk_id = compute_mdhash_id(cur_text, prefix=CHUNK_ID_PREFIX)
        if cur_chunk_id not in chunk_to_global:
            chunk_to_global[cur_chunk_id] = cur_global_idx
    return chunk_to_global


def chunk_mapping_fname(graph_dir: str) -> str:
    return os.path.join(graph_dir, GRAPH_MAPPING_FILE_NAME)


def save_chunk_mapping(chunk_to_global: Dict[str, int], graph_dir: str) -> str:
    dst_fname = chunk_mapping_fname(graph_dir)
    with codecs.open(dst_fname, mode='w', encoding='utf-8') as dst_fp:
        json.dump(obj=chunk_to_global, fp=dst_fp, ensure_ascii=False, indent=1, sort_keys=True)
    return dst_fname


def load_chunk_mapping(graph_dir: str) -> Dict[str, int]:
    src_fname = chunk_mapping_fname(graph_dir)
    if not os.path.isfile(src_fname):
        raise IOError(f'The file "{src_fname}" does not exist! Please build the graph index '
                      f'by the prepare_graph_index.py script first.')
    with codecs.open(src_fname, mode='r', encoding='utf-8') as src_fp:
        chunk_to_global: Dict[str, int] = json.load(src_fp)
    return chunk_to_global


def _rank_scores(num_items: int) -> List[float]:
    return [1.0 / (rank + 1) for rank in range(num_items)]


def _normalize_scores(raw_scores: List[Optional[float]]) -> List[float]:
    known_scores = [score for score in raw_scores if score is not None]
    sum_scores = sum(known_scores) if len(known_scores) == len(raw_scores) else 0.0
    if sum_scores > 0.0:
        return [float(score) / sum_scores for score in raw_scores]
    return [score / sum(_rank_scores(len(raw_scores)))
            for score in _rank_scores(len(raw_scores))]


class GraphSearchEngines:
    def __init__(self, local_engine: LocalSearchEngine, naive_engine: NaiveSearchEngine,
                 chunk_to_global: Dict[str, int], event_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.local_engine = local_engine
        self.naive_engine = naive_engine
        self.chunk_to_global = chunk_to_global
        self.event_loop = event_loop if event_loop is not None else asyncio.new_event_loop()

    def search(self, search_query: str, local_top_k: int = 20, naive_top_k: int = 60,
               local_rerank_top_k: Optional[int] = None, naive_rerank_top_k: Optional[int] = None
               ) -> Dict[str, Any]:
        local_params = LocalParams(top_k=local_top_k, use_summary=False, use_chunks=True,
                                   rerank_top_k=local_rerank_top_k)
        naive_params = NaiveSearchParams(top_k=naive_top_k, rerank_top_k=naive_rerank_top_k)

        async def search_both() -> Tuple[Any, Any]:
            local_retrieval, naive_retrieval = await asyncio.gather(
                self.local_engine.search(search_query, params=local_params),
                self.naive_engine.search(search_query, params=naive_params)
            )
            return local_retrieval, naive_retrieval

        local_retrieval, naive_retrieval = self.event_loop.run_until_complete(search_both())

        local_chunks = self._chunks_from_retrieval(local_retrieval, rank_based=True)
        naive_chunks = self._chunks_from_retrieval(naive_retrieval, rank_based=False)
        chunks = combine_relevant_chunks(local_chunks, naive_chunks)
        entities = self._entities_from_local(local_retrieval)
        relations = self._relations_from_local(local_retrieval)
        return {
            'chunks': chunks,
            'entities': entities,
            'relations': relations,
            'num_relations': len(local_retrieval.result.relations),
            'num_summaries': len(local_retrieval.result.summaries)
        }

    def _chunks_from_retrieval(self, retrieval: Any, rank_based: bool) -> List[Tuple[int, float]]:
        result = retrieval.result
        if rank_based:
            raw_scores = _rank_scores(len(result.chunks))
        else:
            raw_scores = list(result.scores)
            if len(raw_scores) != len(result.chunks):
                raw_scores = _rank_scores(len(result.chunks))
        normalized_scores = _normalize_scores(raw_scores)
        united_chunks = dict()
        num_unknown_chunks = 0
        for chunk, normalized_score in zip(result.chunks, normalized_scores):
            global_idx = self.chunk_to_global.get(chunk.id)
            if global_idx is None:
                global_idx = self.chunk_to_global.get(
                    compute_mdhash_id(chunk.content, prefix=CHUNK_ID_PREFIX)
                )
            if global_idx is None:
                num_unknown_chunks += 1
                continue
            if global_idx not in united_chunks:
                united_chunks[global_idx] = normalized_score
            else:
                united_chunks[global_idx] = max(united_chunks[global_idx], normalized_score)
        if num_unknown_chunks > 0:
            print(f'Warning: {num_unknown_chunks} found chunks are unknown to the graph mapping!')
        ordered_chunks = sorted(united_chunks.items(), key=lambda it: (-it[1], it[0]))
        del united_chunks
        return ordered_chunks

    def _map_source_chunks(self, source_chunk_ids: List[str]) -> List[int]:
        source_global_indices = []
        for cur_chunk_id in (source_chunk_ids or []):
            global_idx = self.chunk_to_global.get(cur_chunk_id)
            if global_idx is not None:
                source_global_indices.append(global_idx)
        return sorted(set(source_global_indices))

    def _entities_from_local(self, local_retrieval: Any) -> List[Dict[str, Any]]:
        entities = list(getattr(local_retrieval.result, 'entities', []))
        metrics = local_retrieval.metrics if isinstance(local_retrieval.metrics, dict) else {}
        metric_entities = metrics.get('entities', [])
        items = []
        for pos, cur_entity in enumerate(entities):
            raw_score = None
            if pos < len(metric_entities):
                cur_metric = metric_entities[pos]
                if isinstance(cur_metric, dict):
                    raw_score = cur_metric.get('relevance_score')
                    if raw_score is not None:
                        raw_score = float(raw_score)
            items.append({
                'key': str(cur_entity.id),
                'kind': 'entity',
                'name': str(cur_entity.entity_name),
                'type': str(cur_entity.entity_type),
                'description': str(cur_entity.description),
                'score': raw_score,
                'sources': self._map_source_chunks(list(cur_entity.source_chunk_id or []))
            })
        normalized_scores = _normalize_scores([it['score'] for it in items])
        for it, normalized_score in zip(items, normalized_scores):
            it['score'] = normalized_score
        return items

    def _relations_from_local(self, local_retrieval: Any) -> List[Dict[str, Any]]:
        relations = list(getattr(local_retrieval.result, 'relations', []))
        normalized_scores = _normalize_scores(_rank_scores(len(relations)))
        items = []
        for cur_relation, normalized_score in zip(relations, normalized_scores):
            items.append({
                'key': str(cur_relation.id),
                'kind': 'relation',
                'subject': str(cur_relation.subject_name),
                'object': str(cur_relation.object_name),
                'relation_type': str(cur_relation.relation_type),
                'description': str(cur_relation.description),
                'score': normalized_score,
                'sources': self._map_source_chunks(list(cur_relation.source_chunk_id or []))
            })
        return items


def find_relevant_items_by_graph(graph_engines: GraphSearchEngines, search_query: str,
                                 local_top_k: int = 20, naive_top_k: int = 60,
                                 local_rerank_top_k: Optional[int] = None,
                                 naive_rerank_top_k: Optional[int] = None
                                 ) -> Dict[str, Any]:
    """
    Runs the local (graph neighborhood) and naive (hybrid chunk vector + sparse) search
    modes of RAGU for one search query and merges their results into a united pool of
    chunks, entities, and relations. The chunk identifiers are mapped back to the global
    chunk indices of the Meno knowledge base. The scores of each source are normalized
    so that they sum up to one (the local chunks and the relations use rank-based scores,
    because RAGU does not provide per-item relevance scores for them).
    """
    return graph_engines.search(
        search_query=search_query,
        local_top_k=local_top_k,
        naive_top_k=naive_top_k,
        local_rerank_top_k=local_rerank_top_k,
        naive_rerank_top_k=naive_rerank_top_k
    )


def build_sparse_embedder() -> RagUBM25:
    return RagUBM25(language=SPARSE_MODEL_LANGUAGE)


def load_graph_engines(graph_dir: str, embedder_name: str,
                       embedder_url: str = 'http://localhost:8000/v1', embedder_key: str = 'EMPTY',
                       qdrant_location: Optional[str] = None,
                       engine_reranker_name: Optional[str] = None,
                       engine_reranker_url: Optional[str] = None, engine_reranker_key: Optional[str] = None,
                       embedder: Optional[Embedder] = None, scorer: Optional[Scorer] = None
                       ) -> GraphSearchEngines:
    if not os.path.isdir(graph_dir):
        raise IOError(f'The directory "{graph_dir}" does not exist!')
    Settings.storage_folder = graph_dir
    Settings.language = SPARSE_MODEL_LANGUAGE
    if embedder is None:
        embedder_client = CachedAsyncOpenAI(base_url=embedder_url, api_key=embedder_key)
        embedder = EmbedderOpenAI(client=embedder_client, model_name=embedder_name)
    sparse_embedder = build_sparse_embedder()
    event_loop = asyncio.new_event_loop()
    try:
        event_loop.run_until_complete(embedder.initialize())
    except BaseException:
        event_loop.close()
        raise
    dummy_client = CachedAsyncOpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
    dummy_llm = LLMOpenAI(client=dummy_client, model_name='unused')
    if scorer is None:
        if (engine_reranker_name is not None) and (len(engine_reranker_name.strip()) > 0):
            reranker_url = engine_reranker_url if engine_reranker_url is not None else embedder_url
            reranker_key = engine_reranker_key if engine_reranker_key is not None else embedder_key
            scorer_client = CachedAsyncOpenAI(base_url=reranker_url, api_key=reranker_key)
            scorer = ScorerOpenAI(client=scorer_client, model_name=engine_reranker_name)
    vdb_storage_kwargs = {'sparse_type': 'bm25'}
    if qdrant_location is not None:
        vdb_storage_kwargs['location'] = qdrant_location
    knowledge_graph = KnowledgeGraph(
        llm=None,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        chunker=None,
        builder_settings=BuilderArguments(
            use_llm_summarization=False,
            use_clustering=False,
            make_community_summary=False,
            remove_isolated_nodes=False,
            random_seed=RANDOM_SEED
        ),
        storage_settings=StorageArguments(
            vdb_storage_type=QdrantVectorDBStorage,
            vdb_storage_kwargs=vdb_storage_kwargs
        )
    )
    local_engine = LocalSearchEngine(
        llm=dummy_llm, knowledge_graph=knowledge_graph, embedder=embedder,
        sparse_embedder=sparse_embedder, reranker=scorer
    )
    naive_engine = NaiveSearchEngine(
        llm=dummy_llm, knowledge_graph=knowledge_graph, embedder=embedder,
        sparse_embedder=sparse_embedder, reranker=scorer
    )
    if knowledge_graph.sparse_embedder is not sparse_embedder:
        raise RuntimeError('The sparse embedder is not wired into the knowledge graph index! '
                           'The hybrid retrieval would silently degrade to the dense-only mode.')
    if getattr(local_engine, 'retriever', None) is not None:
        if local_engine.retriever.sparse_embedder is not sparse_embedder:
            raise RuntimeError('The sparse embedder is not wired into the local search engine!')
    if getattr(naive_engine, 'retriever', None) is not None:
        if naive_engine.retriever.sparse_embedder is not sparse_embedder:
            raise RuntimeError('The sparse embedder is not wired into the naive search engine!')
    return GraphSearchEngines(
        local_engine=local_engine,
        naive_engine=naive_engine,
        chunk_to_global=load_chunk_mapping(graph_dir),
        event_loop=event_loop
    )


ABBREVIATION_REWRITE_SYSTEM_PROMPT = (
    'Вы — модуль расшифровки аббревиатур. Вам даны фрагмент текста и перечень встретившихся в нём аббревиатур '
    'с их возможными расшифровками. Замените каждую аббревиатуру на подходящую по контексту расшифровку; '
    'если ни одна расшифровка не подходит по смыслу, оставьте аббревиатуру без изменений. Всё остальное '
    'перепишите дословно: не исправляйте опечатки, не меняйте порядок слов, пунктуацию и форматирование. '
    'Верните только преобразованный текст фрагмента, без каких-либо пояснений.'
)


def find_abbreviation_candidates(text: str,
                                 abbreviations: Dict[str, Dict[str, Any]]) -> List[str]:
    norm_abbreviations = []
    for cur_norm_abbr, cur_record in abbreviations.items():
        cur_original = cur_record.get('abbreviation', cur_norm_abbr) if isinstance(cur_record, dict) else cur_norm_abbr
        if re.search(rf'(?<![\w]){re.escape(cur_original)}(?![\w])', text):
            norm_abbreviations.append(cur_norm_abbr)
    return norm_abbreviations


def count_abbreviation_occurrences(text: str,
                                   abbreviations: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    occurrences = {}
    for cur_norm_abbr, cur_record in abbreviations.items():
        cur_original = cur_record.get('abbreviation', cur_norm_abbr) if isinstance(cur_record, dict) else cur_norm_abbr
        cur_count = len(re.findall(rf'(?<![\w]){re.escape(cur_original)}(?![\w])', text))
        if cur_count > 0:
            occurrences[cur_norm_abbr] = cur_count
    return occurrences


def canonicalize_artifacts(entities: List[Entity],
                           relations: List[Relation]) -> Tuple[List[Entity], List[Relation]]:
    def normalize_name(name: str) -> str:
        return ' '.join(re.sub(r'[^\w\s]', ' ', str(name).casefold().replace('ё', 'е')).split())

    def trim_edge_punctuation(name: str) -> str:
        return re.sub(r'^[^\w]+|[^\w]+$', '', name)

    name_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for cur_entity in entities:
        cur_key = (normalize_name(cur_entity.entity_name), cur_entity.entity_type)
        if cur_key not in name_counts:
            name_counts[cur_key] = {}
        name_counts[cur_key][cur_entity.entity_name] = name_counts[cur_key].get(cur_entity.entity_name, 0) + 1
    canonical_name_by_key = {
        cur_key: max(cur_counts.items(), key=lambda it: (it[1], it[0]))[0]
        for cur_key, cur_counts in name_counts.items()
    }
    id_replacement: Dict[str, str] = {}
    name_by_id: Dict[str, str] = {}
    canonical_entities: List[Entity] = []
    for cur_entity in entities:
        cur_key = (normalize_name(cur_entity.entity_name), cur_entity.entity_type)
        new_name = canonical_name_by_key[cur_key]
        trimmed_name = trim_edge_punctuation(new_name)
        if len(trimmed_name) > 0:
            new_name = trimmed_name
        if new_name != cur_entity.entity_name:
            new_entity = Entity(
                entity_name=new_name,
                entity_type=cur_entity.entity_type,
                description=cur_entity.description,
                source_chunk_id=cur_entity.source_chunk_id,
                documents_id=cur_entity.documents_id,
                clusters=cur_entity.clusters
            )
            id_replacement[cur_entity.id] = new_entity.id
            cur_entity = new_entity
        name_by_id[cur_entity.id] = cur_entity.entity_name
        canonical_entities.append(cur_entity)
    canonical_relations: List[Relation] = []
    for cur_relation in relations:
        new_subject_id = id_replacement.get(cur_relation.subject_id, cur_relation.subject_id)
        new_object_id = id_replacement.get(cur_relation.object_id, cur_relation.object_id)
        if (new_subject_id == cur_relation.subject_id) and (new_object_id == cur_relation.object_id):
            canonical_relations.append(cur_relation)
            continue
        canonical_relations.append(Relation(
            subject_id=new_subject_id,
            object_id=new_object_id,
            relation_type=cur_relation.relation_type,
            subject_name=name_by_id.get(new_subject_id, cur_relation.subject_name),
            object_name=name_by_id.get(new_object_id, cur_relation.object_name),
            description=cur_relation.description,
            relation_strength=cur_relation.relation_strength,
            source_chunk_id=cur_relation.source_chunk_id
        ))
    return canonical_entities, canonical_relations


class AbbreviationExpandingExtractor(TwoStageArtifactsExtractorLLM):
    """
    Two-stage extractor with an abbreviation-expansion pre-stage and artifact
    canonicalization. Chunks are stored in the knowledge base verbatim: the
    rewritten text is passed to the extraction stages only, and every rewritten
    copy keeps the original chunk id so artifact source references stay valid.
    The canonicalization renames case-only variants of an entity name to the
    most frequent spelling and remaps relation endpoints, so the downstream
    grouping merges such duplicates.
    """

    def __init__(self, *args, abbreviations: Optional[Dict[str, Dict[str, Any]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.abbreviations = abbreviations if abbreviations is not None else {}
        self.num_rewritten_chunks = 0
        self.num_rejected_rewrites = 0

    async def extract(self, chunks: List[Chunk], *args: Any, **kwargs: Any) -> Tuple[List[Entity], List[Relation]]:
        rewritten_chunks = await self._expand_abbreviations(chunks)
        extracted_entities, extracted_relations = await super().extract(rewritten_chunks, *args, **kwargs)
        return canonicalize_artifacts(extracted_entities, extracted_relations)

    async def _expand_abbreviations(self, chunks: List[Chunk]) -> List[Chunk]:
        if len(self.abbreviations) == 0:
            return list(chunks)
        result_chunks: List[Chunk] = []
        for cur_chunk in chunks:
            candidates = find_abbreviation_candidates(cur_chunk.content, self.abbreviations)
            if len(candidates) == 0:
                result_chunks.append(cur_chunk)
                continue
            occurrences = count_abbreviation_occurrences(cur_chunk.content, self.abbreviations)
            dictionary_lines = []
            expected_edit_cost = 0
            expansion_stems = []
            for cur_abbr in candidates:
                cur_record = self.abbreviations[cur_abbr]
                dictionary_lines.append(
                    f"{cur_record['abbreviation']} — " + '; '.join(cur_record['explanation'])
                )
                expected_edit_cost += occurrences.get(cur_abbr, 1) * max(
                    0,
                    max(len(cur_e) for cur_e in cur_record['explanation']) - len(cur_record['abbreviation'])
                )
                for cur_explanation in cur_record['explanation']:
                    first_words = cur_explanation.split()
                    if len(first_words) > 0:
                        expansion_stems.append(RUSSIAN_STEMMER.stem(first_words[0]))
            user_prompt = 'СЛОВАРЬ АББРЕВИАТУР:\n' + '\n'.join(dictionary_lines)
            user_prompt += '\n\nТЕКСТ ФРАГМЕНТА:\n' + cur_chunk.content
            rewritten_text = ''
            try:
                response = await self.llm.chat_completion(
                    conversation=[
                        {'role': 'system', 'content': ABBREVIATION_REWRITE_SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    output_schema=str
                )
                rewritten_text = str(response).strip()
            except Exception:
                rewritten_text = ''
            edit_slack = max(48, round(0.10 * len(cur_chunk.content)))
            max_rewritten_len = len(cur_chunk.content) + expected_edit_cost + edit_slack
            is_sane = (
                (len(rewritten_text) > 0)
                and (0.6 * len(cur_chunk.content) <= len(rewritten_text) <= max_rewritten_len)
                and (Levenshtein.distance(cur_chunk.content, rewritten_text)
                     <= expected_edit_cost + edit_slack)
            )
            rewritten_stems = {
                RUSSIAN_STEMMER.stem(cur_token)
                for cur_token in re.findall(r'\w+', rewritten_text)
            }
            any_expansion = any(
                (cur_stem in rewritten_stems) for cur_stem in expansion_stems
            )
            if is_sane and any_expansion:
                rewritten_chunk = Chunk(
                    content=rewritten_text,
                    chunk_order_idx=cur_chunk.chunk_order_idx,
                    doc_id=cur_chunk.doc_id,
                    num_tokens=cur_chunk.num_tokens
                )
                rewritten_chunk.id = cur_chunk.id
                result_chunks.append(rewritten_chunk)
                self.num_rewritten_chunks += 1
            else:
                if len(rewritten_text) > 0:
                    self.num_rejected_rewrites += 1
                result_chunks.append(cur_chunk)
        print(f'Abbreviation expansion: {self.num_rewritten_chunks} chunks are rewritten, '
              f'{self.num_rejected_rewrites} rewrites are rejected.')
        return result_chunks
