from argparse import ArgumentParser
import asyncio
import codecs
import gc
import json
import networkx as nx
import os
import random
import sys
from typing import Dict, List, Optional, Set, Union

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

try:
    from ragu import BuilderArguments, KnowledgeGraph, Settings, StorageArguments
    from ragu.common.prompts import ICLConfig
    from ragu.common.logger import logger
    from ragu.models.embedder import EmbedderOpenAI
    from ragu.models.llm import LLMOpenAI
    from ragu.models.openai import CachedAsyncOpenAI
    from ragu.models.sparse_embedder import BM25 as RagUBM25
    from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage
except ImportError as err:
    raise ImportError('The "ragu" package (https://github.com/RaguTeam/RAGU) is required for '
                      'the graph index building. Please install it first.') from err

from graph_utils.graph_utils import (
    RANDOM_SEED,
    SPARSE_MODEL_LANGUAGE,
    AbbreviationExpandingExtractor,
    build_chunk_mapping,
    collect_knowledge_base_chunks,
    save_chunk_mapping,
)
from rewriting_utils.rewriting_utils import load_abbreviations
from tokenization_utils.tokenization_utils import load_hf_tokenizer


def configure_logging(level):
    logger.remove()
    logger.add(sys.stdout, colorize=sys.stdout.isatty(), enqueue=True, level=level, diagnose=False,
               format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}')


class ContextBudgetLLM(LLMOpenAI):
    def __init__(self, *, tokenizer, context_tokens, completion_tokens, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens
        self.completion_tokens = completion_tokens
        self.overflow_count = 0

    async def chat_completion(self, conversation, output_schema=str, **kwargs):
        prompt_tokens = len(self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        ))
        if prompt_tokens + self.completion_tokens > self.context_tokens:
            self.overflow_count += 1
            raise ValueError(
                f'Graph LLM prompt needs {prompt_tokens} tokens plus {self.completion_tokens} completion tokens, '
                f'but the configured context is {self.context_tokens}. Increase --llm_context_tokens '
                'to the served model limit or reduce --max_completion_tokens.'
            )
        return await super().chat_completion(conversation, output_schema=output_schema, **kwargs)


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database (the final corpus with metadata and scores).')
    parser.add_argument('-r', '--reverse', dest='reverse_index_name', type=str, required=True,
                        help='The input JSON file with prepared reverse chunk index '
                             '(i.e. mapping of the global chunk index to the document index '
                             'and the chunk-in-document index).')
    parser.add_argument('-o', '--output', dest='output_dir', type=str, required=False,
                        default='../data/knowledge/ragu_graph',
                        help='The output directory for the RAGU knowledge graph.')
    parser.add_argument('--sources', dest='content_sources', type=str, required=False,
                        default='summary,teachers,news,base',
                        help='The comma-separated list of content sources whose documents are '
                             'included into the knowledge graph.')
    parser.add_argument('--chunk_header', dest='chunk_header', type=str, required=False,
                        default='none', choices=['none', 'title', 'title_annotation'],
                        help='The contextual enrichment of the graph chunks: a document metadata header '
                             'is prepended to each chunk text. The "title" mode prepends the frozen header '
                             '"**Документ:** <название>.", the "title_annotation" mode also adds the '
                             'document annotation line. The "title" mode is recommended, because a long '
                             'generated annotation dilutes the BM25 statistics. WARNING: the header format '
                             'is a part of the chunk text, so changing it changes the chunk identifiers '
                             'and requires the full graph rebuilding.')
    parser.add_argument('--chunk_header_sources', dest='chunk_header_sources', type=str, required=False,
                        default='',
                        help='The comma-separated list of content sources whose chunks are enriched with '
                             'the document header (all sources are enriched if it is not specified). '
                             'The self-sufficient summary handbook does not need the enrichment, '
                             'so "base,news" is the recommended value.')
    parser.add_argument('-a', '--abbr', dest='abbr_fname', type=str, required=False, default=None,
                        help='The path to the JSON dictionary of abbreviations. Chunks containing dictionary '
                             'abbreviations are rewritten by the LLM (contextual abbreviation expansion, the '
                             'rest of the text verbatim) before the artifact extraction; the knowledge base '
                             'keeps the original chunk texts, and the rewritten copies keep the original '
                             'chunk ids. Without this flag the rewriting stage is skipped.')
    parser.add_argument('--llm_tokenizer', default=None,
                        help='Local path or Hugging Face tokenizer id, independent of the served LLM name.')
    parser.add_argument('--embedder_tokenizer', default=None,
                        help='Local path or Hugging Face tokenizer id for embeddings.')
    parser.add_argument('--llm_context_tokens', type=int, default=16384,
                        help='Total LLM context window; checked before graph LLM requests.')
    parser.add_argument('--embedder_tokens', type=int, default=8192,
                        help='Maximum embedding input length, including special tokens.')
    parser.add_argument('--qdrant_key', default=os.getenv('QDRANT_API_KEY'),
                        help='Qdrant API key; defaults to QDRANT_API_KEY.')
    parser.add_argument('--vector_only', dest='vector_only', action='store_true',
                        help='Will only the chunk vector context be built (without entity and relation '
                             'extraction)? Such a graph supports only the naive search mode.')
    parser.add_argument('--llm', dest='llm_name', type=str, required=False, default=None,
                        help='The name of the generative LLM (it must be served by an OpenAI-compatible API).')
    parser.add_argument('--url', dest='openai_base_url', type=str, required=False,
                        default='http://localhost:8000/v1',
                        help='Your URL prefix for the OpenAI-compatible API of the generative LLM.')
    parser.add_argument('--key', dest='openai_api_key', type=str, required=False,
                        default=os.getenv('MENO_INDEX_LLM_KEY', 'EMPTY'), help='Your secret OpenAI API key for the generative LLM.')
    parser.add_argument('-e', '--embedder', dest='embedder_name', type=str, required=True,
                        help='The name of the embedding model (it must be served by an OpenAI-compatible '
                             'embeddings API, for example the bge-m3 model served by vLLM).')
    parser.add_argument('--embedder_url', dest='embedder_base_url', type=str, required=False, default=None,
                        help='Your URL prefix for the OpenAI-compatible embeddings API '
                             '(the "--url" value is used if it is not specified).')
    parser.add_argument('--embedder_key', dest='embedder_api_key', type=str, required=False, default=os.getenv('MENO_INDEX_EMBEDDER_KEY'),
                        help='Your secret OpenAI API key for the embeddings API '
                             '(the "--key" value is used if it is not specified).')
    parser.add_argument('--qdrant_location', dest='qdrant_location', type=str, required=False, default='',
                        help='The Qdrant location: an URL of the Qdrant server (it is required for the full '
                             'knowledge graph with entities and relations), ":memory:" for the in-memory '
                             'mode, or an empty string (default) for the local on-disk storage inside the '
                             'graph directory (only the vector-only builds work in this mode, because the '
                             'local on-disk Qdrant supports a single client per folder).')
    parser.add_argument('--cache', dest='cache_dir', type=str, required=False, default=None,
                        help='The directory for the LLM response cache. It makes the graph building '
                             'resumable across restarts.')
    parser.add_argument('--rate_max_simultaneous', dest='rate_max_simultaneous', type=int, required=False,
                        default=8, help='The maximum number of simultaneous LLM requests (0 means no limit).')
    parser.add_argument('--rate_max_per_minute', dest='rate_max_per_minute', type=int, required=False,
                        default=0, help='The maximum number of LLM requests per minute (0 means no limit).')
    parser.add_argument('--rate_min_delay', dest='rate_min_delay', type=float, required=False,
                        default=0.0, help='The minimum delay in seconds between LLM requests (0 means no delay).')
    parser.add_argument('--temperature', dest='temperature', type=float, required=False, default=0.0,
                        help='The sampling temperature for the extraction and validation requests '
                             '(0.0 means the deterministic greedy decoding).')
    parser.add_argument('--max_completion_tokens', dest='max_completion_tokens', type=int, required=False,
                        default=4096, help='The maximum number of generated tokens per one LLM request; '
                             'it prevents runaway generations on JSON schemas with unbounded lists.')
    parser.add_argument('--icl_examples', dest='icl_examples', type=int, required=False, default=2,
                        help='The number of in-context examples for the entity and relation extraction (0 disables '
                             'the in-context learning).')
    parser.add_argument('--no_validation', dest='no_validation', action='store_true',
                        help='Disable the entity and relation validation during the extraction.')
    parser.add_argument('--no_clustering', dest='no_clustering', action='store_true',
                        help='Disable the entity clustering before their summarization.')
    parser.add_argument('--no_community_summary', dest='no_community_summary', action='store_true',
                        help='Disable the community summary generation.')
    parser.add_argument('--max_chunks', dest='max_chunks', type=int, required=False, default=None,
                        help='The maximum number of chunks to be included into the knowledge graph '
                             '(for smoke runs).')
    parser.add_argument('--seed', dest='random_seed', type=int, required=False, default=RANDOM_SEED,
                        help='The random seed for the reproducibility.')
    parser.add_argument('--log_level', choices=['INFO', 'DEBUG'], default='INFO',
                        help='RAGU log level, including model request and cache diagnostics.')
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.llm_context_tokens <= args.max_completion_tokens:
        raise ValueError('LLM context must exceed the completion token budget.')
    if args.embedder_tokens <= 0:
        raise ValueError('Embedding token limit must be positive.')

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    mapping_fname = os.path.normpath(args.reverse_index_name)
    if not os.path.isfile(mapping_fname):
        raise IOError(f'The file "{mapping_fname}" does not exist!')

    graph_dir = os.path.normpath(args.output_dir)
    if os.path.isfile(graph_dir):
        raise IOError(f'The path "{graph_dir}" is a file, but a directory is expected!')
    os.makedirs(graph_dir, exist_ok=True)

    content_sources: Optional[Set[str]] = None
    if len(args.content_sources.strip()) > 0:
        content_sources = set(filter(
            lambda it: len(it) > 0,
            map(lambda it: it.strip().lower(), args.content_sources.split(','))
        ))
        if len(content_sources) == 0:
            content_sources = None

    chunk_header_sources: Optional[Set[str]] = None
    if len(args.chunk_header_sources.strip()) > 0:
        chunk_header_sources = set(filter(
            lambda it: len(it) > 0,
            map(lambda it: it.strip().lower(), args.chunk_header_sources.split(','))
        ))
        if len(chunk_header_sources) == 0:
            chunk_header_sources = None

    if (not args.vector_only) and (args.llm_name is None):
        raise ValueError('The generative LLM name is not specified, and the graph extraction cannot be done! '
                         'Please specify the "--llm" argument or use the "--vector_only" mode.')

    embedder_base_url = args.embedder_base_url if args.embedder_base_url is not None else args.openai_base_url
    embedder_api_key = args.embedder_api_key if args.embedder_api_key is not None else args.openai_api_key
    qdrant_location = args.qdrant_location.strip() if len(args.qdrant_location.strip()) > 0 else None
    if (qdrant_location is None) and (not args.vector_only):
        print('Warning: the local on-disk Qdrant storage supports a single client per folder, so the full '
              'knowledge graph with entities and relations cannot be built in this mode. Please use the '
              '"--qdrant_location" with an URL of a Qdrant server, or ":memory:" for one-process smoke runs '
              '(the built collections are lost when the process exits).')

    random.seed(args.random_seed)

    with codecs.open(mapping_fname, mode='r', encoding='utf-8') as fp:
        chunk_reverse_index: Dict[str, Dict[str, int]] = json.load(fp)
    print(f'{len(chunk_reverse_index)} chunks are loaded from the mapping "{mapping_fname}".')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]] = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), fp.readlines())
            )
        ))
    print(f'{len(documents)} documents are loaded from the corpus "{input_fname}".')

    if content_sources is not None:
        num_documents_by_source = dict()
        for cur_document in documents:
            cur_source = cur_document.get('content_source')
            num_documents_by_source[cur_source] = num_documents_by_source.get(cur_source, 0) + 1
        print('The documents by source: ' + json.dumps(num_documents_by_source, ensure_ascii=False) + '.')

    enrichment_stats: Dict[str, int] = dict()
    chunk_texts, global_indices = collect_knowledge_base_chunks(
        documents=documents,
        chunk_mapping=chunk_reverse_index,
        content_sources=content_sources,
        max_num_chunks=args.max_chunks,
        enrich_mode=args.chunk_header,
        enrich_sources=chunk_header_sources,
        stats=enrichment_stats
    )
    del documents, chunk_reverse_index
    gc.collect()
    print(f'{len(chunk_texts)} chunks are selected for the knowledge graph.')
    if args.chunk_header != 'none':
        sources_msg = (args.chunk_header_sources.strip()
                       if len(args.chunk_header_sources.strip()) > 0 else 'all')
        print(f'The chunk header mode is "{args.chunk_header}": {enrichment_stats["enriched"]} of '
              f'{enrichment_stats["total"]} chunks are enriched with the document header '
              f'(sources: {sources_msg}).')

    chunk_to_global = build_chunk_mapping(chunk_texts, global_indices)
    mapping_output_fname = save_chunk_mapping(chunk_to_global, graph_dir)
    print(f'{len(chunk_to_global)} unique chunk identifiers are saved into "{mapping_output_fname}".')
    del chunk_to_global, global_indices

    Settings.storage_folder = graph_dir
    Settings.language = SPARSE_MODEL_LANGUAGE
    Settings.tokenizer_embedder_backend = 'local'
    Settings.tokenizer_embedder_name = args.embedder_tokenizer or args.embedder_name
    Settings.embedder_token_limit = args.embedder_tokens

    embedder_client = CachedAsyncOpenAI(base_url=embedder_base_url, api_key=embedder_api_key)
    dense_embedder = EmbedderOpenAI(client=embedder_client, model_name=args.embedder_name)
    sparse_embedder = RagUBM25(language=SPARSE_MODEL_LANGUAGE)
    print(f'The sparse embedder is the fastembed BM25 model for the "{SPARSE_MODEL_LANGUAGE}" language.')

    vdb_storage_kwargs = {'sparse_type': 'bm25'}
    if args.qdrant_key:
        vdb_storage_kwargs['api_key'] = args.qdrant_key
    if qdrant_location is not None:
        vdb_storage_kwargs['location'] = qdrant_location
        location_msg = f'"{qdrant_location}"'
    else:
        location_msg = f'the local on-disk storage inside "{graph_dir}"'
    print(f'The Qdrant vector storage is used with the location {location_msg} and the "bm25" sparse vectors.')

    if args.vector_only:
        llm = None
        artifact_extractor = None
        print('The vector-only mode is used: no entity and relation extraction will be performed.')
    else:
        llm_tokenizer = load_hf_tokenizer(args.llm_tokenizer or args.llm_name)
        llm_client = CachedAsyncOpenAI(
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            rate_max_simultaneous=args.rate_max_simultaneous if args.rate_max_simultaneous > 0 else None,
            rate_max_per_minute=args.rate_max_per_minute if args.rate_max_per_minute > 0 else None,
            rate_min_delay=args.rate_min_delay if args.rate_min_delay > 0 else None,
            max_completion_tokens=args.max_completion_tokens,
            cache=args.cache_dir if args.cache_dir is not None else os.path.join(graph_dir, 'llm_cache'),
            debug_errors_storage=os.path.join(graph_dir, 'llm_debug')
        )
        llm = ContextBudgetLLM(
            client=llm_client, model_name=args.llm_name, temperature=args.temperature,
            tokenizer=llm_tokenizer, context_tokens=args.llm_context_tokens,
            completion_tokens=args.max_completion_tokens,
        )
        Settings.tokenizer_llm_backend = 'local'
        Settings.tokenizer_llm_name = args.llm_tokenizer or args.llm_name
        Settings.llm_context_token_limit = args.llm_context_tokens
        icl_config = None
        icl_embedder = None
        if args.icl_examples > 0:
            icl_config = ICLConfig(
                enabled=True,
                num_examples=args.icl_examples,
                selection_strategy='semantic'
            )
            icl_embedder = dense_embedder
        abbreviations = None
        if args.abbr_fname is not None:
            abbr_fname = os.path.normpath(args.abbr_fname)
            if not os.path.isfile(abbr_fname):
                raise IOError(f'The file "{abbr_fname}" does not exist!')
            abbreviations = load_abbreviations(abbr_fname)
            print(f'There are {len(abbreviations)} abbreviations for the chunk rewriting.')
        artifact_extractor = AbbreviationExpandingExtractor(
            llm=llm,
            embedder=icl_embedder,
            icl_config=icl_config,
            do_entity_validation=not args.no_validation,
            do_relation_validation=not args.no_validation,
            abbreviations=abbreviations,
            rewrite_concurrency=args.rate_max_simultaneous if args.rate_max_simultaneous > 0 else 8,
        )
        info_msg = (f'The two-stage artifact extractor is prepared (ICL examples: {args.icl_examples}, '
                    f'validation: {not args.no_validation}, '
                    f'abbreviations: {len(abbreviations) if abbreviations is not None else 0}).')
        print(info_msg)

    builder_settings = BuilderArguments(
        use_llm_summarization=not args.vector_only,
        use_clustering=not args.no_clustering,
        build_only_vector_context=args.vector_only,
        make_community_summary=(not args.no_community_summary) and (not args.vector_only),
        remove_isolated_nodes=True,
        random_seed=args.random_seed
    )

    async def build_graph():
        await dense_embedder.initialize()
        knowledge_graph = KnowledgeGraph(
            llm=llm,
            embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            chunker=None,
            artifact_extractor=artifact_extractor,
            builder_settings=builder_settings,
            storage_settings=StorageArguments(
                vdb_storage_type=QdrantVectorDBStorage,
                vdb_storage_kwargs=vdb_storage_kwargs
            )
        )
        try:
            await knowledge_graph.build_from_docs(chunk_texts)
            if llm is not None and llm.overflow_count:
                raise RuntimeError(f'{llm.overflow_count} graph LLM requests exceeded the context budget. '
                                   'The graph is incomplete; review context/completion limits.')
            collections = {
                'vdb_entity': knowledge_graph.index.nodes_vector_db.collection_name,
                'vdb_relation': knowledge_graph.index.edges_vector_db.collection_name,
                'vdb_chunk': knowledge_graph.index.chunks_vector_db.collection_name,
            }
            with open(os.path.join(graph_dir, 'qdrant_collections.json'), 'w', encoding='utf-8') as fp:
                json.dump(collections, fp, ensure_ascii=False, indent=2)
        finally:
            await knowledge_graph.index.close()

    asyncio.run(build_graph())

    if not args.vector_only:
        gml_fname = os.path.join(graph_dir, 'knowledge_graph.gml')
        num_graph_nodes = 0
        num_graph_edges = 0
        if os.path.isfile(gml_fname):
            knowledge_gml = nx.read_gml(gml_fname)
            num_graph_nodes = knowledge_gml.number_of_nodes()
            num_graph_edges = knowledge_gml.number_of_edges()
        if (num_graph_nodes == 0) and (num_graph_edges == 0):
            err_msg = ('The built knowledge graph is empty: no entities and no relations are extracted. '
                       'Check the health of the LLM service and the debug logs in the '
                       f'"{os.path.join(graph_dir, "llm_debug")}" directory.')
            raise RuntimeError(err_msg)

    print(f'The knowledge graph is successfully built and saved into the "{graph_dir}" directory.')


if __name__ == '__main__':
    main()
