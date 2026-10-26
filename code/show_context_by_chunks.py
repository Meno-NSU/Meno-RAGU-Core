from argparse import ArgumentParser
import codecs
import json
import os
import random
from typing import Dict, List, Union

from context_utils.context_utils import prepare_context


def find_global_chunk_idx(chunk_mapping: Dict[str, Dict[str, int]], document_index: int, local_chunk_index: int) -> int:
    found_idx = -1
    for k in chunk_mapping:
        if chunk_mapping[k]['doc_index'] != document_index:
            continue
        if chunk_mapping[k]['local_chunk_index'] != local_chunk_index:
            continue
        found_idx = int(k)
        break
    return found_idx


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database.')
    parser.add_argument('-r', '--reverse', dest='reverse_index_name', type=str, required=True,
                        help='The input JSON file with prepared reverse chunk index '
                             '(i.e. mapping of the global chunk index to the document index '
                             'and the chunk-in-document index).')
    parser.add_argument('--num_chunks', dest='num_chunks', type=int, required=False, default=None,
                        help='Number of randomly selected chunks from knowledge database.')
    parser.add_argument('--chunk_idx', dest='chunk_index', type=int, required=False, action='append',
                        default=None, help='The global index of selected chunk.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    mapping_fname = os.path.normpath(args.reverse_index_name)
    if not os.path.isfile(mapping_fname):
        raise IOError(f'The file "{mapping_fname}" does not exist!')

    with codecs.open(mapping_fname, mode='r', encoding='utf-8') as fp:
        chunk_reverse_index: Dict[str, Dict[str, int]] = json.load(fp)
    total_num_chunks = len(chunk_reverse_index)
    print(f'{total_num_chunks} chunks are loaded from the mapping "{mapping_fname}".')
    all_global_indices = sorted(list(map(lambda idx: int(idx), chunk_reverse_index.keys())))
    min_global_index = all_global_indices[0]
    max_global_index = all_global_indices[-1]
    if (min_global_index != 0) or (max_global_index != (total_num_chunks - 1)):
        raise ValueError(f'The chunk mapping "{mapping_fname}" is wrong!')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]] = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), fp.readlines())
            )
        ))
    print(f'{len(documents)} documents are loaded from the corpus "{input_fname}".')

    if args.chunk_index is None:
        global_indices_of_selected_chunks = []
    elif len(args.chunk_index) == 0:
        global_indices_of_selected_chunks = []
    else:
        global_indices_of_selected_chunks = []
        for global_index in args.chunk_index:
            if global_index is not None:
                global_indices_of_selected_chunks.append(global_index)
    if len(global_indices_of_selected_chunks) == 0:
        if args.num_chunks is None:
            global_indices_of_selected_chunks = []
            for doc_idx in random.sample(population=list(range(len(documents))), k=random.randint(1, 4)):
                num_chunks = len(documents[doc_idx]['chunks'])
                if num_chunks > 1:
                    num_selected_chunks = random.randint(1, min(num_chunks, 7))
                else:
                    num_selected_chunks = num_chunks
                if num_selected_chunks < num_chunks:
                    local_chunk_indices = random.sample(population=list(range(num_chunks)), k=num_selected_chunks)
                else:
                    local_chunk_indices = list(range(num_chunks))
                for chunk_idx in local_chunk_indices:
                    global_chunk_idx = find_global_chunk_idx(chunk_reverse_index, doc_idx, chunk_idx)
                    if global_chunk_idx < 0:
                        err_msg = (f'The global chunk index cannot be found by its local coordinates '
                                   f'(doc_idx = {doc_idx}, local_chunk_idx = {chunk_idx}).')
                        raise RuntimeError(err_msg)
                    global_indices_of_selected_chunks.append(global_chunk_idx)
            print('All global chunk indices for demo are prepared.')
        else:
            global_indices_of_selected_chunks = list(set(
                [random.randint(0, total_num_chunks - 1) for _ in range(args.num_chunks)]
            ))
        random.shuffle(global_indices_of_selected_chunks)
    info_msg = (f'{len(global_indices_of_selected_chunks)} chunks are specified for demo. '
                f'Their global indices are: {global_indices_of_selected_chunks}.')
    print(info_msg)
    print('')

    relevances_of_selected_chunks = [
        (0.1 + random.random()) / 1.1
        for _ in range(len(global_indices_of_selected_chunks))
    ]

    prepared_context, _ = prepare_context(
        indices_of_relevant_chunks=global_indices_of_selected_chunks,
        scores_of_relevant_chunks=relevances_of_selected_chunks,
        documents=documents,
        chunk_mapping=chunk_reverse_index,
        min_document_quality=0.0
    )
    if len(prepared_context) == 0:
        print('The prepared context is empty.')
    else:
        if len(prepared_context) == 1:
            print(prepared_context[0])
        for idx, val in enumerate(prepared_context):
            print('==========')
            print(f'DOCUMENT {idx + 1}')
            print('==========')
            print(val + '\n')


if __name__ == '__main__':
    main()
