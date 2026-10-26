from argparse import ArgumentParser
import codecs
import json
import os
import random

from transformers import AutoTokenizer
from tqdm import trange


def main():
    random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output JSONL knowledge database enriched with the lengths of texts in tokens.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The name of LLM whose tokenizer will be used to estimate the number of tokens.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        base_dirname = os.path.dirname(output_fname)
        if len(base_dirname) > 0:
            if not os.path.isdir(base_dirname):
                raise IOError(f'The directory "{base_dirname}" does not exist!')

    try:
        tok = AutoTokenizer.from_pretrained(args.model_name)
    except:
        tok = AutoTokenizer.from_pretrained(os.path.normpath(args.model_name))
    print(f'The tokenizer is successfully loaded from "{args.model_name}".')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as src_fp:
        samples = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), src_fp.readlines())
            )
        ))
        print(f'There are {len(samples)} samples in the "{input_fname}".')

    chunk_lengths = []
    full_text_lengths = []
    for sample_idx in trange(len(samples), desc='Tokenization'):
        chunk_boundaries = []
        for chunk_start, chunk_end in samples[sample_idx]['chunk_boundaries']:
            chunk_text = samples[sample_idx]['text'][chunk_start:chunk_end]
            num_tokens_in_chunk = len(tok.tokenize(chunk_text))
            chunk_boundaries.append({
                'start_char': chunk_start,
                'end_char': chunk_end,
                'num_tokens': num_tokens_in_chunk
            })
            chunk_lengths.append(num_tokens_in_chunk)
        num_tokens_in_text = len(tok.tokenize(samples[sample_idx]['text']))
        full_text_lengths.append(num_tokens_in_text)
        samples[sample_idx]['chunks'] = chunk_boundaries
        samples[sample_idx]['num_tokens'] = num_tokens_in_text
        del samples[sample_idx]['chunk_boundaries']
        del chunk_boundaries
    print('All texts are tokenized.')

    sorted_chunk_lengths = sorted(chunk_lengths)
    print(f'\nMinimal chunk length is {sorted_chunk_lengths[0]}.')
    print(f'Maximal chunk length is {sorted_chunk_lengths[-1]}.')
    print(f'Median chunk length is {sorted_chunk_lengths[(len(sorted_chunk_lengths) - 1) // 2]}.')
    print(f'Mean chunk length is {round(sum(sorted_chunk_lengths) / len(sorted_chunk_lengths), 2)}.')
    del sorted_chunk_lengths

    sorted_full_text_lengths = sorted(full_text_lengths)
    print(f'\nMinimal text length is {sorted_full_text_lengths[0]}.')
    print(f'Maximal text length is {sorted_full_text_lengths[-1]}.')
    print(f'Median text length is {sorted_full_text_lengths[(len(sorted_full_text_lengths) - 1) // 2]}.')
    print(f'Mean text length is {round(sum(sorted_full_text_lengths) / len(sorted_full_text_lengths), 2)}.')
    del sorted_full_text_lengths

    samples.sort(key=lambda it: it['num_tokens'])
    with codecs.open(output_fname, mode='w', encoding='utf-8') as dst_fp:
        for cur_sample in samples:
            dst_fp.write(json.dumps(cur_sample, ensure_ascii=False) + '\n')
    print(f'\nEnriched samples are successfully saved into the "{output_fname}".')


if __name__ == '__main__':
    main()
