from argparse import ArgumentParser
import codecs
import gc
import json
import os
import random
import re
from typing import List, Tuple, Union

from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from filter_utils.filter_utils import MIN_NUM_WORDS, text_is_good, normalize_text
from news_utils import read_news_samples


MAX_F1: float = 0.9
RATIO_EPSILON: float = 1e-9


def can_reach_f1(a_size: int, b_size: int, max_f1: float = MAX_F1) -> bool:
    """
    Checks whether two word-set sizes can reach the F1 score max_f1 or higher.
    The maximal possible F1 for the sizes is 2 * min(A, B) / (A + B), so the
    size ratio must lie inside [max_f1 / (2 - max_f1), (2 - max_f1) / max_f1].
    The band is widened by a tiny epsilon so that the floating point rounding
    of the actual F1 calculation cannot hide a borderline candidate.
    """
    if a_size <= 0 or b_size <= 0:
        return False
    ratio = b_size / a_size
    lower_bound = max_f1 / (2.0 - max_f1) * (1.0 - RATIO_EPSILON)
    upper_bound = (2.0 - max_f1) / max_f1 * (1.0 + RATIO_EPSILON)
    return lower_bound <= ratio <= upper_bound


def iter_json_samples(src_fp):
    for cur_line in src_fp:
        cur_line = cur_line.strip()
        if len(cur_line) > 0:
            yield json.loads(cur_line)


def normalize_urls(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        result: List[str] = []
        for v in value:
            s = v.strip() if isinstance(v, str) else str(v).strip()
            if s and s not in result:
                result.append(s)
        return result
    s = str(value).strip()
    return [s] if s else []


def urls_field(urls: List[str]) -> Union[str, List[str]]:
    if len(urls) == 0:
        return ''
    if len(urls) == 1:
        return urls[0]
    return urls


def url_dedup_key(value) -> Tuple[str, ...]:
    norm = [' '.join(u.strip().lower().split()).strip() for u in normalize_urls(value)]
    return tuple(sorted([u for u in norm if u]))


def main():
    random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('-s', '--summary', dest='summary_data', type=str, required=True,
                        help='The input JSONL summary knowledge (prepared from the curated handbook).')
    parser.add_argument('-b', '--base', dest='base_data', type=str, required=True,
                        help='The input JSONL base knowledge (crawled from the main NSU website).')
    parser.add_argument('-n', '--news', dest='news_data', type=str, required=True, action='append',
                        help='The input JSONL news knowledge (crawled from the VK pages corresponding to NSU).')
    parser.add_argument('-t', '--teachers', dest='teachers_data', type=str, required=True,
                        help='The input JSONL knowledge about NSU teachers and employees '
                             '(prepared by the prepare_teachers_corpus.py script).')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output JSONL knowledge database.')
    parser.add_argument('--timezone', default='Asia/Novosibirsk',
                        help='Timezone for doc_date values without an explicit offset.')
    args = parser.parse_args()

    summary_input_fname = os.path.normpath(args.summary_data)
    if not os.path.isfile(summary_input_fname):
        raise IOError(f'The file "{summary_input_fname}" does not exist!')

    teachers_input_fname = os.path.normpath(args.teachers_data)
    if not os.path.isfile(teachers_input_fname):
        raise IOError(f'The file "{teachers_input_fname}" does not exist!')

    base_input_fname = os.path.normpath(args.base_data)
    if not os.path.isfile(base_input_fname):
        raise IOError(f'The file "{base_input_fname}" does not exist!')

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
        raise IOError('There are no news knowledges about NSU!')
    news_fnames = sorted(list(news_fnames), key=lambda it: -int(it[-14:-6]))

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        output_dirname = os.path.dirname(output_fname)
        if len(output_dirname) > 0:
            if not os.path.isdir(output_dirname):
                raise IOError(f'The directory "{output_dirname}" does not exist!')

    ru_stemmer = SnowballStemmer('russian')

    unique_texts = set()
    unique_urls = set()
    num_saved_texts = 0
    num_summary_texts = 0
    num_empty_texts = 0
    num_duplicated_texts = 0
    num_bad_texts = 0
    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as dst_fp:
        with codecs.open(summary_input_fname, mode='r', encoding='utf-8') as src_fp:
            summary_samples = list(map(
                lambda it3: json.loads(it3),
                filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), src_fp.readlines())
                )
            ))
        print(f'\nThe file "{summary_input_fname}" contains {len(summary_samples)} samples.')
        for cur_sample in tqdm(summary_samples, desc=os.path.basename(summary_input_fname)):
            cur_text = cur_sample['text']
            cur_url = cur_sample.get('url', '')
            if len(cur_text.strip()) == 0:
                num_empty_texts += 1
                continue
            new_sample = {
                'text': cur_text,
                'content_source': 'summary',
                'url': urls_field(normalize_urls(cur_url))
            }
            dst_fp.write(json.dumps(obj=new_sample, ensure_ascii=False) + '\n')
            del new_sample
            num_saved_texts += 1
            num_summary_texts += 1
        del summary_samples

        num_teachers_texts = 0
        with codecs.open(teachers_input_fname, mode='r', encoding='utf-8') as src_fp:
            teachers_samples = list(map(
                lambda it3: json.loads(it3),
                filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), src_fp.readlines())
                )
            ))
        print(f'\nThe file "{teachers_input_fname}" contains {len(teachers_samples)} samples.')
        for cur_sample in tqdm(teachers_samples, desc=os.path.basename(teachers_input_fname)):
            cur_text = cur_sample['text']
            cur_url = cur_sample.get('url', '')
            if len(cur_text.strip()) == 0:
                num_empty_texts += 1
                continue
            new_sample = {
                'text': cur_text,
                'content_source': 'teachers',
                'url': urls_field(normalize_urls(cur_url))
            }
            dst_fp.write(json.dumps(obj=new_sample, ensure_ascii=False) + '\n')
            del new_sample
            num_saved_texts += 1
            num_teachers_texts += 1
        del teachers_samples

        for cur_fname in news_fnames:
            samples = read_news_samples(cur_fname, timezone=args.timezone)
            print(f'\nThe file "{cur_fname}" contains {len(samples)} samples.')
            for cur_sample in tqdm(samples, desc=os.path.basename(cur_fname)):
                new_url_key = url_dedup_key(cur_sample['url'])
                new_text = normalize_text(cur_sample['content'], ru_stemmer)
                if len(new_url_key) == 0:
                    raise IOError(f'The file "{cur_fname}" contains a wrong url "{cur_sample["url"]}"!')
                if len(new_text) == 0:
                    raise IOError(f'The file "{cur_fname}" contains a wrong text "{cur_sample["content"]}"!')
                if (new_url_key not in unique_urls) and (new_text not in unique_texts):
                    unique_urls.add(new_url_key)
                    unique_texts.add(new_text)
                    if (len(new_text.split()) >= MIN_NUM_WORDS) and text_is_good(cur_sample['content']):
                        new_sample = {
                            'text': cur_sample['content'],
                            'content_source': 'news',
                            'url': urls_field(normalize_urls(cur_sample['url']))
                        }
                        dst_fp.write(json.dumps(obj=new_sample, ensure_ascii=False) + '\n')
                        del new_sample
                        num_saved_texts += 1
                elif new_url_key not in unique_urls:
                    unique_urls.add(new_url_key)
                elif new_text not in unique_texts:
                    unique_texts.add(new_text)
            del samples

        del unique_urls
        search_index = dict()
        ordered_news_texts = list(unique_texts)
        ordered_all_texts = []
        text_sizes = []
        gc.collect()
        info_msg = (f'\nThere are {num_saved_texts} news texts are prepared, '
                    f'and there are {len(ordered_news_texts)} unqiue normalized texts.')
        print(info_msg)

        for cur_text in tqdm(ordered_news_texts, desc='Search index'):
            words_of_cur_text = set(cur_text.split())
            if len(words_of_cur_text) >= MIN_NUM_WORDS:
                text_idx = len(ordered_all_texts)
                ordered_all_texts.append(words_of_cur_text)
                text_sizes.append(len(words_of_cur_text))
                for cur_word in words_of_cur_text:
                    if cur_word in search_index:
                        search_index[cur_word].add(text_idx)
                    else:
                        search_index[cur_word] = {text_idx}
                del text_idx
            else:
                num_empty_texts += 1
            del words_of_cur_text
        del ordered_news_texts
        gc.collect()
        print(f'\nThere are {len(search_index)} unique words in the search index.')

        with codecs.open(base_input_fname, mode='r', encoding='utf-8') as src_fp:
            num_base_samples = sum(1 for cur_line in src_fp if len(cur_line.strip()) > 0)
        print(f'\nThe file "{base_input_fname}" contains {num_base_samples} samples.')
        with codecs.open(base_input_fname, mode='r', encoding='utf-8') as src_fp:
            for cur_sample in tqdm(iter_json_samples(src_fp), total=num_base_samples,
                                   desc=os.path.basename(base_input_fname)):
                cur_text = cur_sample['text']
                cur_url = cur_sample['url']
                if text_is_good(cur_text):
                    norm_text = normalize_text(cur_text, ru_stemmer)
                    if len(norm_text) > 0:
                        if norm_text in unique_texts:
                            num_duplicated_texts += 1
                        else:
                            unique_texts.add(norm_text)
                            words_of_base_text = set(norm_text.split())
                            if len(words_of_base_text) >= MIN_NUM_WORDS:
                                indices_of_other_texts = set()
                                for cur_word in words_of_base_text:
                                    if cur_word in search_index:
                                        indices_of_other_texts |= search_index[cur_word]
                                is_found = False
                                if len(indices_of_other_texts) > 0:
                                    base_size = len(words_of_base_text)
                                    for other_text_idx in indices_of_other_texts:
                                        if not can_reach_f1(base_size, text_sizes[other_text_idx]):
                                            continue
                                        words_of_other_text = ordered_all_texts[other_text_idx]
                                        num_common = len(words_of_other_text & words_of_base_text)
                                        precision = num_common / len(words_of_other_text)
                                        recall = num_common / len(words_of_base_text)
                                        if (precision + recall) > 0:
                                            f1 = 2.0 * precision * recall / (precision + recall)
                                        else:
                                            f1 = 0.0
                                        if f1 >= MAX_F1:
                                            is_found = True
                                            break
                                del indices_of_other_texts
                                if is_found:
                                    num_duplicated_texts += 1
                                else:
                                    new_sample = {
                                        'text': cur_text,
                                        'content_source': 'base',
                                        'url': urls_field(normalize_urls(cur_url))
                                    }
                                    dst_fp.write(json.dumps(obj=new_sample, ensure_ascii=False) + '\n')
                                    del new_sample
                                    num_saved_texts += 1
                                text_idx = len(ordered_all_texts)
                                ordered_all_texts.append(words_of_base_text)
                                text_sizes.append(len(words_of_base_text))
                                for cur_word in words_of_base_text:
                                    if cur_word in search_index:
                                        search_index[cur_word].add(text_idx)
                                    else:
                                        search_index[cur_word] = {text_idx}
                                del text_idx
                            else:
                                num_empty_texts += 1
                            del words_of_base_text
                    else:
                        num_empty_texts += 1
                    del cur_text, cur_url, norm_text
                else:
                    num_bad_texts += 1
    print(f'\n{num_saved_texts} texts are successfully saved into "{output_fname}" '
          f'({num_summary_texts} of them are from the summary source, '
          f'{num_teachers_texts} of them are from the teachers source).')
    print(f'Filtered out: {num_empty_texts} empty texts, {num_duplicated_texts} duplicated texts, '
          f'{num_bad_texts} bad texts.')


if __name__ == '__main__':
    main()
