from argparse import ArgumentParser
import codecs
import json
import os
import sys

from context_utils.context_utils import build_reverse_chunk_index


def main():
    parser = ArgumentParser(
        description='Build the reverse chunk index (the mapping of global chunk indices to their '
                    'local coordinates: the document index and the chunk index inside the document) '
                    'for the final knowledge base corpus.')
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database with chunk boundaries '
                             '(the output of the calculate_num_tokens.py step).')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output JSON file with the reverse chunk index.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')
    output_fname = os.path.normpath(args.output_name)
    output_dirname = os.path.dirname(output_fname)
    if (len(output_dirname) > 0) and (not os.path.isdir(output_dirname)):
        raise IOError(f'The directory "{output_dirname}" does not exist!')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as src_fp:
        samples = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), src_fp.readlines())
            )
        ))
        print(f'There are {len(samples)} samples in the "{input_fname}".')

    reverse_chunk_index = build_reverse_chunk_index(samples)

    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as dst_fp:
        json.dump(obj=reverse_chunk_index, fp=dst_fp, ensure_ascii=False, indent=4)
    print(f'The reverse chunk index of {len(reverse_chunk_index)} chunks is successfully '
          f'saved into the "{output_fname}".', file=sys.stderr)


if __name__ == '__main__':
    main()
