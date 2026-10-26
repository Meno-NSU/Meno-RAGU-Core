from argparse import ArgumentParser
import codecs
import json
import os
from typing import Dict, List, Tuple, Union

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

from context_utils.context_utils import global_chunk_index_to_local
from graph_utils.graph_utils import find_relevant_items_by_graph, load_graph_engines


def main():
    parser = ArgumentParser()
    parser.add_argument('-g', '--graph', dest='graph_dir', type=str, required=True,
                        help='The directory with the RAGU knowledge graph (prepared by prepare_graph_index.py).')
    parser.add_argument('-q', '--query', dest='search_query', type=str, required=True,
                        help='The search query.')
    parser.add_argument('-e', '--embedder', dest='embedder_name', type=str, required=True,
                        help='The name of the embedding model served by an OpenAI-compatible embeddings API '
                             '(the same model that was used for the graph building).')
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
    parser.add_argument('--engine_reranker', dest='engine_reranker_name', type=str, required=False, default=None,
                        help='The name of the cross-encoder reranker model served by an OpenAI-compatible '
                             'scoring API.')
    parser.add_argument('--engine_rerank_top_k', dest='engine_rerank_top_k', type=int, required=False, default=30,
                        help='The number of chunks kept by the cross-encoder reranker inside the naive '
                             'search engine.')
    parser.add_argument('--url', dest='openai_base_url', type=str, required=False,
                        default='http://localhost:8000/v1', help='Your URL prefix for OpenAI API calls.')
    parser.add_argument('--key', dest='openai_api_key', type=str, required=False,
                        default='EMPTY', help='Your secret OpenAI API key.')
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=False, default=None,
                        help='The input JSONL knowledge database; if it is specified, the document titles '
                             'are shown for each found chunk.')
    parser.add_argument('-r', '--reverse', dest='reverse_index_name', type=str, required=False, default=None,
                        help='The input JSON file with prepared reverse chunk index (it is required '
                             'if the "-i" argument is specified).')
    parser.add_argument('--local_top_k', dest='local_top_k', type=int, required=False, default=20,
                        help='The maximum number of graph entities for the local search.')
    parser.add_argument('--naive_top_k', dest='naive_top_k', type=int, required=False, default=10,
                        help='The maximum number of chunks for the naive hybrid search.')
    args = parser.parse_args()

    graph_dir = os.path.normpath(args.graph_dir)
    if not os.path.isdir(graph_dir):
        raise IOError(f'The directory "{graph_dir}" does not exist!')

    if args.input_name is None:
        documents = None
        chunk_reverse_index = None
    else:
        if args.reverse_index_name is None:
            raise ValueError('The reverse chunk index is not specified, but it is required '
                             'if the "-i" argument is specified!')
        input_fname = os.path.normpath(args.input_name)
        if not os.path.isfile(input_fname):
            raise IOError(f'The file "{input_fname}" does not exist!')
        mapping_fname = os.path.normpath(args.reverse_index_name)
        if not os.path.isfile(mapping_fname):
            raise IOError(f'The file "{mapping_fname}" does not exist!')
        with codecs.open(mapping_fname, mode='r', encoding='utf-8') as fp:
            chunk_reverse_index: Dict[str, Dict[str, int]] = json.load(fp)
        with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
            documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]] = list(map(
                lambda it3: json.loads(it3),
                filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), fp.readlines())
                )
            ))
        print(f'{len(documents)} documents are loaded from the corpus "{input_fname}".')

    def chunk_to_title(global_chunk_index: int) -> str:
        if (documents is None) or (chunk_reverse_index is None):
            return ''
        document_index, _ = global_chunk_index_to_local(global_chunk_index, documents, chunk_reverse_index)
        return ' '.join(documents[document_index]['doc_title'].strip().split()).strip()

    search_query_text = ' '.join(args.search_query.strip().split()).strip()
    if len(search_query_text) == 0:
        raise IOError('The search query is empty!')

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
    print(f'The RAGU graph search engines are successfully loaded from the "{graph_dir}".')

    search_result = find_relevant_items_by_graph(
        graph_engines=graph_engines,
        search_query=search_query_text,
        local_top_k=args.local_top_k,
        naive_top_k=args.naive_top_k,
        naive_rerank_top_k=(args.engine_rerank_top_k if args.engine_reranker_name is not None else None)
    )

    print('')
    print(f'The {len(search_result["entities"])} most relevant entities are found by the local search:')
    for cur_item in search_result['entities']:
        print(f'  [{round(cur_item["score"], 3)}] {cur_item["name"]} ({cur_item["type"]}) '
              f'-> chunks {cur_item["sources"]}')

    print('')
    print(f'The local search returns {search_result["num_relations"]} relations '
          f'({len(search_result["relations"])} of them are kept in the pool), '
          f'{search_result["num_summaries"]} community summaries, '
          f'and the following chunks:')
    for global_idx, score in [(it[0], it[1]) for it in search_result['chunks']]:
        title = chunk_to_title(global_idx)
        if len(title) > 0:
            print(f'  ({global_idx}, {round(score, 4)}) "{title}"')
        else:
            print(f'  ({global_idx}, {round(score, 4)})')


if __name__ == '__main__':
    main()
