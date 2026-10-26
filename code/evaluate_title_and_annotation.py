from argparse import ArgumentParser
import codecs
import gc
import json
import logging
import math
import os
import random
import sys

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch
from tqdm import trange

from search_utils.search_utils import DenseEmbedderService, load_dense_embedder, vectorize_texts


summary_eval_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The bge-m3-like embedder for title and annotation evaluation: a model name '
                             'in the embeddings service (together with --url) or a local path / hub id '
                             'of a SentenceTransformer (without --url).')
    parser.add_argument('--url', dest='openai_base_url', type=str, required=False, default=None,
                        help='The URL prefix of the OpenAI-compatible embeddings service. If it is not '
                             'specified, the model is loaded locally as a SentenceTransformer.')
    parser.add_argument('--key', dest='openai_api_key', type=str, required=False, default='EMPTY',
                        help='The secret API key for the OpenAI-compatible embeddings service.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output JSONL file knowledge database with eval scores.')
    parser.add_argument('--minibatch', dest='minibatch', type=int, required=False, default=64,
                        help='The mini-batch for embeddings calculation.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        err_msg = f'The file "{input_fname}" does not exist!'
        summary_eval_logger.error(err_msg)
        raise IOError(err_msg)

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        output_dirname = os.path.dirname(output_fname)
        if not os.path.isdir(output_dirname):
            err_msg = f'The directory "{output_dirname}" does not exist!'
            summary_eval_logger.error(err_msg)
            raise IOError(err_msg)

    with codecs.open(input_fname, mode='r', encoding='utf-8') as src_fp:
        documents = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), src_fp.readlines())
            )
        ))
    summary_eval_logger.info(f'{len(documents)} documents are loaded from the corpus "{input_fname}".')

    if args.openai_base_url is not None:
        dense_embedder = DenseEmbedderService(
            args.model_name,
            base_url=args.openai_base_url,
            api_key=args.openai_api_key
        )
        summary_eval_logger.info(f'The embeddings service at "{args.openai_base_url}" is used '
                                 f'for evaluation with the model "{args.model_name}".')
    elif torch.cuda.is_available():
        dense_embedder = load_dense_embedder(args.model_name, device='cuda:0')
        summary_eval_logger.info(f'The bge-m3-like embedder is successfully loaded from the "{args.model_name}". '
                                 f'CUDA is used.')
    else:
        dense_embedder = load_dense_embedder(args.model_name, device='cpu')
        summary_eval_logger.info(f'The bge-m3-like embedder is successfully loaded from the "{args.model_name}".')

    all_scores = []
    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as dst_fp:
        for doc_idx in trange(len(documents), desc='Evaluation'):
            if (len(documents[doc_idx]['doc_title'].strip()) == 0) or (len(documents[doc_idx]['doc_annotation']) == 0):
                documents[doc_idx]['quality_score'] = 0.0
            else:
                texts = [documents[doc_idx]['doc_title']]
                texts += [documents[doc_idx]['doc_annotation']]
                full_text = documents[doc_idx]['doc_full_text']
                num_chunks = min(3, len(documents[doc_idx]['chunks']))
                for cur_chunk in documents[doc_idx]['chunks'][0:num_chunks]:
                    texts.append(full_text[cur_chunk['start_char']:cur_chunk['end_char']].strip())
                vectors_of_texts = vectorize_texts(texts, dense_embedder, batch_size=args.minibatch)
                gc.collect()
                title_distance = np.mean(
                    euclidean_distances(vectors_of_texts[0:1], vectors_of_texts[2:]),
                    keepdims=False
                )
                annotation_distance = np.mean(
                    euclidean_distances(vectors_of_texts[1:2], vectors_of_texts[2:]),
                    keepdims=False
                )
                title_proba = (1.0 + math.exp(-1.0)) / (1.0 + math.exp(float(title_distance) - 1.0))
                del title_distance
                annotation_proba = (1.0 + math.exp(-1.0)) / (1.0 + math.exp(float(annotation_distance) - 1.0))
                p = 2 * title_proba * annotation_proba / (title_proba + annotation_proba)
                documents[doc_idx]['quality_score'] = p
                del vectors_of_texts, texts
                del title_proba, annotation_proba, p
            all_scores.append(documents[doc_idx]['quality_score'])
            dst_fp.write(json.dumps(obj=documents[doc_idx], ensure_ascii=False) + '\n')
    all_scores.sort()
    info_msg = (f'The minimal eval score is {round(all_scores[0], 4)}, '
                f'the maximal one is {round(all_scores[-1], 4)}, '
                f'the median one is {round(all_scores[(len(all_scores) - 1) // 2], 4)}, '
                f'and the mean one is {round(sum(all_scores) / len(all_scores), 4)}')
    summary_eval_logger.info(info_msg)
    summary_eval_logger.info(f'The scored documents are saved into the "{output_fname}".')


if __name__ == '__main__':
    summary_eval_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    summary_eval_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('document_summary_eval.log')
    file_handler.setFormatter(formatter)
    summary_eval_logger.addHandler(file_handler)
    main()
