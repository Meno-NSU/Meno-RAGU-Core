from argparse import ArgumentParser
import codecs
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output image file with histogram of document scores distribution.')
    parser.add_argument('--zeros', dest='with_zeros', action='store_true',
                        help='Zero quality scores should be taken into account when constructing a histogram.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')
    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        output_dirname = os.path.dirname(output_fname)
        if not os.path.isdir(output_dirname):
            raise IOError(f'The directory "{output_dirname}" does not exist!')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as src_fp:
        quality_scores = list(map(
            lambda it4: it4['quality_score'],
            map(
                lambda it3: json.loads(it3),
                filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), src_fp.readlines())
                )
            )
        ))
    print(f'There are {len(quality_scores)} scored documents.')
    if not args.with_zeros:
        quality_scores = list(filter(lambda it: abs(it) > 1e-6, quality_scores))
        print(f'There are {len(quality_scores)} scored documents after zero scores removing.')
    quality_mean = float(np.mean(quality_scores))
    quality_std = float(np.std(quality_scores))
    print(f'The mean of quality score is {round(quality_mean, 4)}.')
    print(f'The standard deviation of quality score is {round(quality_std, 4)}.')

    min_score = min(quality_scores)
    max_score = max(quality_scores)
    dscore = 0.005
    num_bins = round((max_score - min_score) / dscore)
    print(f'Number of bins for histogram is {num_bins}.')

    plt.hist(quality_scores, color='blue', edgecolor='black', bins=num_bins)
    plt.title('Histogram of Document Quality Scores')
    plt.xlabel('Quality Score (from 0 to 1, higher value is better)')
    plt.ylabel('Documents')
    plt.savefig(output_fname, dpi=300, format='png')
    print(f'The histogram image is successfully saved into the "{output_fname}".')


if __name__ == '__main__':
    main()
