from argparse import ArgumentParser
import codecs
import gc
import json
import os
from typing import Any, Dict, List, Union
import warnings

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['SOCKS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['socks_proxy'] = ''
os.environ['no_proxy'] = '*'

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from openai import OpenAI
from vllm import LLM

from context_utils.context_utils import (
    format_document_reference,
    graph_element_to_text,
    prepare_context,
    prepare_graph_elements_context,
)
from rerank_utils.rerank_utils import MAX_RERANKER_TOKENS, rerank_documents_with_vllm, rerank_documents_with_api
from rewriting_utils.rewriting_utils import load_abbreviations
from vllm_utils.vllm_utils import finalize_vllm


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database.')
    parser.add_argument('-r', '--reverse', dest='reverse_index_name', type=str, required=True,
                        help='The input JSON file with prepared reverse chunk index.')
    parser.add_argument('-q', '--query', dest='search_query', type=str, required=True,
                        help='The search query.')
    parser.add_argument('-a', '--abbr', dest='abbreviation_name', type=str, required=False,
                        default=None, help='The JSON file with abbreviations and their explanations '
                                           '(it is required for the LLM-based reranking).')
    parser.add_argument('-g', '--graph', dest='graph_dir', type=str, required=True,
                        help='The directory with the RAGU knowledge graph (prepared by prepare_graph_index.py).')
    parser.add_argument('--graph_top_k', dest='graph_top_k', type=int, required=False, default=20,
                        help='The maximum number of graph entities for the local search mode.')
    parser.add_argument('--top_k', dest='top_k', type=int, required=False, default=60,
                        help='The maximum number of chunks for the naive search mode.')
    parser.add_argument('--engine_reranker', dest='engine_reranker_name', type=str, required=False,
                        default='BAAI/bge-reranker-v2-m3',
                        help='The name of the cross-encoder reranker model served by an OpenAI-compatible '
                             'scoring API (the first pass of the default two-pass reranking scheme inside '
                             'the RAGU search engines). An empty string disables the first pass.')
    parser.add_argument('--engine_rerank_top_k', dest='engine_rerank_top_k', type=int, required=False, default=30,
                        help='The number of chunks kept by the cross-encoder reranker inside the naive '
                             'search engine.')
    parser.add_argument('-e', '--embedder', dest='embedder_name', type=str, required=True,
                        help='The name of the embedding model served by an OpenAI-compatible embeddings API.')
    parser.add_argument('--embedder_url', dest='embedder_base_url', type=str, required=False, default=None,
                        help='Your URL prefix for the OpenAI-compatible embeddings API '
                             '(the "--url" value is used if it is not specified).')
    parser.add_argument('--embedder_key', dest='embedder_api_key', type=str, required=False, default=None,
                        help='Your secret OpenAI API key for the embeddings API '
                             '(the "--key" value is used if it is not specified).')
    parser.add_argument('--qdrant_location', dest='qdrant_location', type=str, required=False, default='',
                        help='The Qdrant location: an URL of the Qdrant server (it is required for the full '
                             'knowledge graph with entities and relations), ":memory:" for the in-memory '
                             'mode, or an empty string (default) for the local on-disk storage inside the '
                             'graph directory (only the vector-only builds work in this mode, because the '
                             'local on-disk Qdrant supports a single client per folder). The same value '
                             'that was used for the graph building is expected here.')
    parser.add_argument('--reranker', dest='reranker_name', type=str, required=False, default=None,
                        help='The name of LLM-as-reranker (it can be Meno, Qwen, Gemma, or any other '
                             'generative LLM) for the second reranking pass.')
    parser.add_argument('--rerank_top_k', dest='rerank_top_k', type=int, required=False, default=12,
                        help='The number of the most relevant pool items kept after the LLM-based reranking.')
    parser.add_argument('--rerank_weight', dest='rerank_weight', type=float, required=False, default=0.8,
                        help='The reranking score weight.')
    parser.add_argument('--min_quality', dest='min_document_quality', type=float, required=False,
                        default=0.0, help='The minimum document quality.')
    parser.add_argument('--gpu', dest='gpu_mem_part', type=float, required=False,
                        default=0.6, help='The GPU memory part for the vLLM-based inference of reranker.')
    parser.add_argument('--num_batched_tokens', dest='num_batched_tokens', type=int, required=False,
                        default=MAX_RERANKER_TOKENS // 2,
                        help='Maximum number of tokens that can be processed in a single iteration '
                             'of the vLLM-based inference.')
    parser.add_argument('--url', dest='openai_base_url', type=str, required=False,
                        default='http://localhost:8000/v1', help='Your URL prefix for OpenAI API calls.')
    parser.add_argument('--key', dest='openai_api_key', type=str, required=False,
                        default='EMPTY', help='Your secret OpenAI API key.')
    parser.add_argument('--vllm', dest='use_vllm', action='store_true',
                        help='Will offline VLLM-based inference be used? '
                             'If not specified, the OpenAI API will be used.')
    args = parser.parse_args()

    if (args.engine_reranker_name is not None) and (len(args.engine_reranker_name.strip()) == 0):
        args.engine_reranker_name = None

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    mapping_fname = os.path.normpath(args.reverse_index_name)
    if not os.path.isfile(mapping_fname):
        raise IOError(f'The file "{mapping_fname}" does not exist!')

    if args.abbreviation_name is None:
        abbreviations = None
    else:
        abbreviations_fname = os.path.normpath(args.abbreviation_name)
        if not os.path.isfile(abbreviations_fname):
            raise IOError(f'The file "{abbreviations_fname}" does not exist!')
        abbreviations = load_abbreviations(abbreviations_fname)
        print(f'There are {len(abbreviations)} abbreviations.')

    with codecs.open(mapping_fname, mode='r', encoding='utf-8') as fp:
        chunk_reverse_index: Dict[str, Dict[str, int]] = json.load(fp)
    total_num_chunks = len(chunk_reverse_index)
    print(f'{total_num_chunks} chunks are loaded from the mapping "{mapping_fname}".')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]] = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), fp.readlines())
            )
        ))
    print(f'{len(documents)} documents are loaded from the corpus "{input_fname}".')

    search_query_text = ' '.join(args.search_query.strip().split()).strip()
    if len(search_query_text) == 0:
        raise IOError('The search query is empty!')

    graph_dir = os.path.normpath(args.graph_dir)
    if not os.path.isdir(graph_dir):
        raise IOError(f'The directory "{graph_dir}" does not exist!')
    from graph_utils.graph_utils import find_relevant_items_by_graph, load_graph_engines
    embedder_base_url = args.embedder_base_url if args.embedder_base_url is not None else args.openai_base_url
    embedder_api_key = args.embedder_api_key if args.embedder_api_key is not None else args.openai_api_key
    qdrant_location = args.qdrant_location.strip() if len(args.qdrant_location.strip()) > 0 else None
    graph_engines = load_graph_engines(
        graph_dir=graph_dir,
        embedder_name=args.embedder_name,
        embedder_url=embedder_base_url,
        embedder_key=embedder_api_key,
        qdrant_location=qdrant_location,
        engine_reranker_name=args.engine_reranker_name
    )
    print(f'The RAGU graph search engines are successfully loaded from the "{graph_dir}". '
          f'There are {len(graph_engines.chunk_to_global)} chunks in the graph mapping.')

    search_pool = find_relevant_items_by_graph(
        graph_engines=graph_engines,
        search_query=search_query_text,
        local_top_k=args.graph_top_k,
        naive_top_k=args.top_k,
        naive_rerank_top_k=(args.engine_rerank_top_k if args.engine_reranker_name is not None else None)
    )
    pool_chunks = search_pool['chunks']
    pool_elements = search_pool['entities'] + search_pool['relations']

    print('')
    print(f'The {len(search_pool["entities"])} most relevant entities are found by the local search:')
    for cur_item in search_pool['entities']:
        print(f'  [{round(cur_item["score"], 4)}] {cur_item["name"]} ({cur_item["type"]}) '
              f'-> chunks {cur_item["sources"]}')
    print('')
    print(f'The local search returns {search_pool["num_relations"]} relations '
          f'({len(search_pool["relations"])} of them are kept in the pool):')
    for cur_item in search_pool['relations']:
        print(f'  [{round(cur_item["score"], 4)}] {cur_item["subject"]} '
              f'—[{cur_item["relation_type"]}]→ {cur_item["object"]} -> chunks {cur_item["sources"]}')
    print('')
    print(f'The {len(pool_chunks)} most relevant chunks are found by the hybrid naive search '
          f'(dense vectors + sparse BM25, reciprocal rank fusion):')
    print([(idx, round(score, 4)) for idx, score in pool_chunks])
    print(f'\nSize of the united search pool is {len(pool_chunks) + len(pool_elements)} '
          f'({len(pool_chunks)} chunks, {len(search_pool["entities"])} entities, '
          f'and {len(search_pool["relations"])} relations).')

    if len(pool_chunks) + len(pool_elements) == 0:
        print('\nThe search pool is empty.')
        return

    if args.reranker_name is None:
        reranker = None
        client = None
    else:
        if args.rerank_weight <= 0.0:
            err_msg = (f'The reranking score weight is too small! '
                       f'Expected a floating-point number greater than 0.0 and less than 1.0, '
                       f'got {args.rerank_weight}.')
            raise ValueError(err_msg)
        if args.rerank_weight >= 1.0:
            err_msg = (f'The reranking score weight is too large! '
                       f'Expected a floating-point number greater than 0.0 and less than 1.0, '
                       f'got {args.rerank_weight}.')
            raise ValueError(err_msg)
        if args.use_vllm:
            client = None
            try:
                reranker = LLM(
                    model=args.reranker_name,
                    gpu_memory_utilization=args.gpu_mem_part,
                    max_model_len=MAX_RERANKER_TOKENS,
                    max_num_batched_tokens=args.num_batched_tokens,
                    enable_chunked_prefill=True,
                    tensor_parallel_size=1
                )
            except Exception as err:
                warnings.warn(f'{args.reranker_name}: ' + str(err))
                reranker = LLM(
                    model=os.path.normpath(args.reranker_name),
                    gpu_memory_utilization=args.gpu_mem_part,
                    max_model_len=MAX_RERANKER_TOKENS,
                    max_num_batched_tokens=args.num_batched_tokens,
                    enable_chunked_prefill=True,
                    tensor_parallel_size=1
                )
            print(f'The LLM-as-reranker is successfully loaded from the "{args.reranker_name}".\n')
        else:
            client = OpenAI(
                api_key=args.openai_api_key,
                base_url=args.openai_base_url
            )
            reranker = args.reranker_name

    if reranker is not None:
        if abbreviations is None:
            raise ValueError('The abbreviations are not provided (the "-a" argument is required '
                             'for the LLM-based reranking).')
        try:
            print(f'There are {len(pool_chunks) + len(pool_elements)} pool items before reranking.')
            element_document_texts = [graph_element_to_text(cur_item, documents, chunk_reverse_index)
                                      for cur_item in pool_elements]
            if client is None:
                context_scores = rerank_documents_with_vllm(
                    user_question=search_query_text,
                    dialogue_history='',
                    stemmer=None,
                    abbr_dict=abbreviations,
                    global_indices_of_relevant_chunks=[it[0] for it in pool_chunks],
                    documents=documents, chunk_mapping=chunk_reverse_index, llm=reranker,
                    extra_document_texts=element_document_texts
                )
            else:
                context_scores = rerank_documents_with_api(
                    user_question=search_query_text,
                    dialogue_history='',
                    stemmer=None,
                    abbr_dict=abbreviations,
                    global_indices_of_relevant_chunks=[it[0] for it in pool_chunks],
                    documents=documents, chunk_mapping=chunk_reverse_index, client=client,
                    llm_name=reranker,
                    extra_document_texts=element_document_texts
                )
            expected_num_items = len(pool_chunks) + len(pool_elements)
            if len(context_scores) != expected_num_items:
                err_msg = (f'The number of items in the search pool does not equal to the number '
                           f'of relevance scores! {expected_num_items} ! {len(context_scores)}.')
                raise RuntimeError(err_msg)
            alpha = args.rerank_weight
            scored_items = []
            for idx, val in enumerate(pool_chunks):
                item_score = 0.0
                if context_scores[idx] > 0.0:
                    item_score = (1.0 - alpha) * val[1] + alpha * context_scores[idx]
                scored_items.append(({'kind': 'chunk', 'chunk_index': val[0]}, item_score))
            for offset, cur_item in enumerate(pool_elements):
                cur_score = context_scores[len(pool_chunks) + offset]
                item_score = 0.0
                if cur_score > 0.0:
                    item_score = (1.0 - alpha) * cur_item['score'] + alpha * cur_score
                scored_items.append(({'kind': 'element', 'item': cur_item}, item_score))
            ordered_items = list(filter(
                lambda it3: it3[1] > 0.0,
                sorted(scored_items, key=lambda it2: (-it2[1], it2[0]['kind'], it2[0].get('chunk_index', 0)))
            ))
            del context_scores, scored_items
        finally:
            if client is None:
                del reranker
                finalize_vllm()
        gc.collect()
        if len(ordered_items) > args.rerank_top_k:
            ordered_items = ordered_items[:args.rerank_top_k]
        ordered_chunk_indices = [(it[0]['chunk_index'], it[1])
                                 for it in ordered_items if it[0]['kind'] == 'chunk']
        ordered_elements = [it[0]['item'] for it in ordered_items if it[0]['kind'] == 'element']
        print(f'\nThere are {len(ordered_items)} pool items after reranking '
              f'({len(ordered_chunk_indices)} chunks and {len(ordered_elements)} graph elements):')
        for cur_item, cur_score in ordered_items:
            if cur_item['kind'] == 'chunk':
                print(f'  [{round(cur_score, 4)}] chunk {cur_item["chunk_index"]}')
            else:
                if cur_item['item']['kind'] == 'entity':
                    print(f'  [{round(cur_score, 4)}] entity "{cur_item["item"]["name"]}"')
                else:
                    print(f'  [{round(cur_score, 4)}] relation "{cur_item["item"]["subject"]} '
                          f'—[{cur_item["item"]["relation_type"]}]→ {cur_item["item"]["object"]}"')
        print('')
        del ordered_items
        pool_chunks = ordered_chunk_indices
        pool_elements = ordered_elements
        del ordered_chunk_indices, ordered_elements

    document_texts, document_references = prepare_context(
        indices_of_relevant_chunks=[it[0] for it in pool_chunks],
        scores_of_relevant_chunks=[it[1] for it in pool_chunks],
        documents=documents,
        chunk_mapping=chunk_reverse_index,
        min_document_quality=args.min_document_quality
    )
    element_texts, _ = prepare_graph_elements_context(
        items=pool_elements,
        documents=documents,
        chunk_mapping=chunk_reverse_index
    )
    print('')
    for idx, val in enumerate(document_texts):
        print('==========')
        print(f'DOCUMENT {idx + 1}')
        print('==========')
        print(val + '\n')
    if len(element_texts) > 0:
        print('==========')
        print('GRAPH ELEMENTS')
        print('==========')
        for val in element_texts:
            print(val + '\n')
    if len(document_references) > 0:
        print('**Дополнительные источники**\n')
        print('\n\n'.join(document_references))


if __name__ == '__main__':
    main()
