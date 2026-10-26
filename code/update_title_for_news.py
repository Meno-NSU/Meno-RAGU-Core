from argparse import ArgumentParser
import codecs
from datetime import datetime
import json
import locale
import os
import random
import re

from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from filter_utils.filter_utils import normalize_text



def main():
    random.seed(42)
    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')

    parser = ArgumentParser()
    parser.add_argument('-b', '--base', dest='base_data', type=str, required=True,
                        help='The input JSONL knowledge database.')
    parser.add_argument('-n', '--news', dest='news_data', type=str, required=True, action='append',
                        help='The additional JSONL news knowledge (crawled from the VK pages corresponding to NSU).')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output JSONL knowledge database with enriched document titles.')
    args = parser.parse_args()

    knowledge_input_fname = os.path.normpath(args.base_data)
    if not os.path.isfile(knowledge_input_fname):
        raise IOError(f'The file "{knowledge_input_fname}" does not exist!')

    news_fnames = set()
    re_for_news_pattern = re.compile(r'\w+_\d\d\d\d\d\d\d\d_\d\d\d\d\d\d\d\d.jsonl')
    for it in args.news_data:
        normalized_news_fname = os.path.normpath(it)
        if not os.path.isfile(normalized_news_fname):
            raise IOError(f'The file "{normalized_news_fname}" does not exist!')
        if re_for_news_pattern.match(os.path.basename(normalized_news_fname)) is None:
            raise IOError(f'The file "{normalized_news_fname}" has incorrect name!')
        news_fnames.add(normalized_news_fname)
    if len(news_fnames) == 0:
        raise IOError(f'There are no news knowledges about NSU!')
    news_fnames = sorted(list(news_fnames), key=lambda it: -int(it[-14:-6]))

    knowledge_output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(knowledge_output_fname):
        knowledge_output_dirname = os.path.dirname(knowledge_output_fname)
        if len(knowledge_output_dirname) > 0:
            if not os.path.isdir(knowledge_output_dirname):
                raise IOError(f'The directory "{knowledge_output_dirname}" does not exist!')

    ru_stemmer = SnowballStemmer('russian')

    uniques_news = dict()
    for cur_fname in news_fnames:
        with codecs.open(cur_fname, mode='r', encoding='utf-8') as src_fp:
            samples = sorted(
                list(map(
                    lambda it3: json.loads(it3),
                    filter(
                        lambda it2: len(it2) > 0,
                        map(lambda it1: it1.strip(), src_fp.readlines())
                    )
                )),
                key=lambda it4: -it4['date']
            )
        print(f'\nThe file "{cur_fname}" contains {len(samples)} samples.')
        for cur_sample in tqdm(samples, desc=os.path.basename(cur_fname)):
            news_text = normalize_text(cur_sample['content'], ru_stemmer)
            if len(news_text) > 0:
                news_timestamp = cur_sample['date']
                news_date = '(новость опубликована ' + datetime.fromtimestamp(news_timestamp).strftime('%d %B %Y') + ')'
                if news_text in uniques_news:
                    if news_timestamp > uniques_news[news_text]['timestamp']:
                        uniques_news[news_text] = {
                            'timestamp': news_timestamp,
                            'textualized_date': news_date
                        }
                else:
                    uniques_news[news_text] = {
                        'timestamp': news_timestamp,
                        'textualized_date': news_date
                    }
        del samples
    print(f'There are {len(uniques_news)} unique news about NSU!')

    with codecs.open(knowledge_input_fname, mode='r', encoding='utf-8') as knowledge_fp:
        main_samples = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), knowledge_fp.readlines())
            )
        ))
    print(f'There are {len(main_samples)} main knowledge samples about NSU!')

    total_number_of_news = 0
    number_of_updated_news = 0
    for idx in range(len(main_samples)):
        if main_samples[idx]['content_source'] == 'news':
            total_number_of_news += 1
            news_text = normalize_text(main_samples[idx]['doc_full_text'], ru_stemmer)
            if (news_text in uniques_news) and ('doc_title' in main_samples[idx]):
                number_of_updated_news += 1
                main_samples[idx]['doc_title'] += (' ' + uniques_news[news_text]['textualized_date'])
    print(f'{number_of_updated_news} news titles of {total_number_of_news} are updated.')

    with codecs.open(knowledge_output_fname, mode='w', encoding='utf-8', errors='ignore') as dst_fp:
        for it in main_samples:
            dst_fp.write(json.dumps(obj=it, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
