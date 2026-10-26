from argparse import ArgumentParser
import codecs
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set

import certifi
import urllib3
from tqdm import tqdm

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['SOCKS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

from filter_utils.filter_utils import MIN_NUM_WORDS, text_is_good, count_words

_http_pool: urllib3.PoolManager = None


def extract_urls(record: dict) -> List[str]:
    value = record.get('url', '')
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


def check_url(url: str) -> bool:
    try:
        resp = _http_pool.request(
            'GET', url, redirect=True,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; URLValidator/1.0)'})
        code = resp.status
        resp.release_conn()
        return code < 400 or code in (403, 500, 502, 503)
    except Exception:
        return False


def main():
    parser = ArgumentParser(
        description='Validate base JSONL records by text quality and URL accessibility.')
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL base knowledge file.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output filtered JSONL file.')
    parser.add_argument('--concurrency', dest='concurrency', type=int, required=False,
                        default=500,
                        help='Max concurrent HTTP requests (default: %(default)s).')
    parser.add_argument('--timeout', dest='timeout', type=int, required=False,
                        default=15,
                        help='HTTP request timeout in seconds (default: %(default)s).')
    parser.add_argument('--retries', dest='retries', type=int, required=False,
                        default=1,
                        help='Retries on timeout/connection error (default: %(default)s).')
    parser.add_argument('--max-per-host', dest='max_per_host', type=int, required=False,
                        default=10,
                        help='Max concurrent connections per host (default: %(default)s).')
    args = parser.parse_args()

    global _http_pool
    _http_pool = urllib3.PoolManager(
        num_pools=2000,
        maxsize=args.max_per_host,
        retries=args.retries,
        timeout=urllib3.Timeout(connect=args.timeout, read=args.timeout),
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
    )

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    output_fname = os.path.normpath(args.output_name)
    output_dirname = os.path.dirname(output_fname)
    if len(output_dirname) > 0:
        if not os.path.isdir(output_dirname):
            raise IOError(f'The directory "{output_dirname}" does not exist!')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as src_fp:
        all_records = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), src_fp.readlines())
            )
        ))
    print(f'Loaded {len(all_records)} records from "{input_fname}".', file=sys.stderr)

    num_empty_text = 0
    num_bad_phrases = 0
    num_too_short = 0
    candidates: List[dict] = []
    for record in all_records:
        text = record.get('text', '')
        if len(text.strip()) == 0:
            num_empty_text += 1
            continue
        if not text_is_good(text):
            num_bad_phrases += 1
            continue
        if count_words(text) < MIN_NUM_WORDS:
            num_too_short += 1
            continue
        candidates.append(record)

    print(f'\n=== Phase 1: text filtering ===', file=sys.stderr)
    print(f'Total records: {len(all_records)}', file=sys.stderr)
    print(f'Filtered out (empty text): {num_empty_text}', file=sys.stderr)
    print(f'Filtered out (bad phrases): {num_bad_phrases}', file=sys.stderr)
    print(f'Filtered out (too short): {num_too_short}', file=sys.stderr)
    print(f'Remaining after text filter: {len(candidates)}', file=sys.stderr)

    unique_urls: Set[str] = set()
    record_urls: Dict[int, List[str]] = {}
    for idx, record in enumerate(candidates):
        urls = extract_urls(record)
        record_urls[idx] = urls
        unique_urls.update(urls)
    print(f'\n=== Phase 2: URL checking ===', file=sys.stderr)
    print(f'Unique URLs to check: {len(unique_urls)}', file=sys.stderr)

    url_alive: Dict[str, bool] = {}
    if len(unique_urls) > 0:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_url = {
                executor.submit(check_url, url): url
                for url in unique_urls
            }
            with tqdm(total=len(unique_urls), desc='Checking URLs',
                      file=sys.stderr) as pbar:
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        url_alive[url] = future.result()
                    except Exception:
                        url_alive[url] = False
                    pbar.update(1)

    alive_count = sum(1 for v in url_alive.values() if v)
    dead_count = sum(1 for v in url_alive.values() if not v)
    print(f'Alive: {alive_count} ({100 * alive_count / max(1, len(url_alive)):.1f}%)', file=sys.stderr)
    print(f'Dead: {dead_count} ({100 * dead_count / max(1, len(url_alive)):.1f}%)', file=sys.stderr)

    num_all_dead = 0
    output_records: List[dict] = []
    for idx, record in enumerate(candidates):
        urls = record_urls[idx]
        if len(urls) == 0:
            output_records.append(record)
            continue
        all_dead = all(not url_alive.get(url, False) for url in urls)
        if all_dead:
            num_all_dead += 1
        else:
            output_records.append(record)

    print(f'\n=== Phase 3: final filtering ===', file=sys.stderr)
    print(f'Records with all URLs dead: {num_all_dead}', file=sys.stderr)
    print(f'Final output: {len(output_records)} records', file=sys.stderr)

    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as dst_fp:
        for record in output_records:
            dst_fp.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'\nFiltered records saved into "{output_fname}".', file=sys.stderr)


if __name__ == '__main__':
    main()
