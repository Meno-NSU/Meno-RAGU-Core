"""CPU-only exact prompt comparison; keep original requests on server and export aggregates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import statistics
import time
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand import rerank
from meno_rag.stand.abbreviation_cache import AbbreviationTokenCache
from meno_rag.stand.rewriting import load_abbreviations

HEADERS = re.compile(r'==========\n(ABBREVIATION DICTIONARY|DOCUMENT 1|DIALOGUE HISTORY|CURRENT QUESTION|INSTRUCTION)\n==========\n\n')


def parts(messages):
    user = next(message['content'] for message in reversed(messages) if message['role'] == 'user')
    found = list(HEADERS.finditer(user))
    sections = {}
    for index, match in enumerate(found):
        body = user[match.end():found[index + 1].start() if index + 1 < len(found) else len(user)]
        if index + 1 < len(found):
            if not body.endswith('\n\n'):
                raise ValueError('Unexpected section boundary')
            body = body[:-2]
        sections[match.group(1)] = body
    return sections['DOCUMENT 1'], sections.get('DIALOGUE HISTORY', ''), sections['CURRENT QUESTION']


def identity(messages):
    return hashlib.sha256('\0'.join(parts(messages)).encode()).hexdigest()


def requests(path):
    for line in path.open():
        row = json.loads(line)
        request = row.get('request') or {}
        if row.get('status') == 200 and request.get('max_tokens') == 1 and request.get('logprobs'):
            yield request['messages']


def summary(values):
    ordered = sorted(values)
    return {'n': len(values), 'sum_ms': sum(values), 'mean_ms': statistics.mean(values) if values else None,
            'median_ms': statistics.median(values) if values else None,
            'p95_ms': ordered[round((len(ordered) - 1) * .95)] if ordered else None}


def profile(stream, abbreviations, *, compare_saved, limit=None, deduplicate=True):
    stemmer = SnowballStemmer('russian')
    cache = AbbreviationTokenCache(stemmer)
    original = rerank.find_candidates_to_abbreviations
    old_times, new_times = [], []
    first_50 = []
    document_keys, common_keys = set(), set()
    wall_started = time.perf_counter()
    old_saved_mismatches = cached_mismatches = 0
    seen = set()
    for messages in stream:
        key = identity(messages)
        if key in seen and deduplicate:
            continue
        seen.add(key)
        document, history, question = parts(messages)
        document_keys.add(hashlib.sha256(document.encode()).hexdigest())
        common_keys.add(hashlib.sha256((history + "\n\n" + question).encode()).hexdigest())
        if len(first_50) < 50:
            first_50.append((document, history, question))
        try:
            rerank.find_candidates_to_abbreviations = original
            started = time.perf_counter()
            baseline = rerank.build_prompt(question, history, document, abbreviations, stemmer, order='legacy')
            old_times.append((time.perf_counter() - started) * 1000)
            if compare_saved:
                old_saved_mismatches += baseline != messages
            started = time.perf_counter()
            found = cache.find(document, history, question, abbreviations)
            rerank.find_candidates_to_abbreviations = lambda _found=found, **kwargs: _found
            candidate = rerank.build_prompt(question, history, document, abbreviations, stemmer, order='legacy')
            new_times.append((time.perf_counter() - started) * 1000)
            cached_mismatches += candidate != baseline
            # Also verify the alternative existing order without changing its sections or matching rules.
            candidate_q = rerank.build_prompt(question, history, document, abbreviations, stemmer, order='question_first')
            rerank.find_candidates_to_abbreviations = original
            baseline_q = rerank.build_prompt(question, history, document, abbreviations, stemmer, order='question_first')
            cached_mismatches += candidate_q != baseline_q
        finally:
            rerank.find_candidates_to_abbreviations = original
        if limit and len(old_times) >= limit:
            break
    warm_times = []
    warm_cache = AbbreviationTokenCache(stemmer)
    try:
        for repeat in range(2):
            for document, history, question in first_50:
                started = time.perf_counter()
                found = warm_cache.find(document, history, question, abbreviations)
                rerank.find_candidates_to_abbreviations = lambda _found=found, **kwargs: _found
                rerank.build_prompt(question, history, document, abbreviations, stemmer, order='legacy')
                if repeat:
                    warm_times.append((time.perf_counter() - started) * 1000)
    finally:
        rerank.find_candidates_to_abbreviations = original
    return {'processed_requests': len(old_times), 'unique_inputs': len(seen), 'saved_legacy_prompt_mismatches': old_saved_mismatches,
            'unique_document_hashes': len(document_keys), 'unique_question_history_hashes': len(common_keys),
            'controlled_repeated_first_50': summary(warm_times),
            'controlled_repeat_caution': 'An intentionally warmed replay, not an estimate of unseen production questions',
            'wall_elapsed_ms_including_two_order_parity': (time.perf_counter() - wall_started) * 1000,
            'cached_complete_message_mismatches_both_orders': cached_mismatches,
            'baseline': summary(old_times), 'cache_first_pass': summary(new_times), 'cache_stats': cache.stats(),
            'parity_ok': not old_saved_mismatches and not cached_mismatches}


def main(args):
    abbreviations = load_abbreviations(args.abbreviations)
    result = {'cpu_only': True, 'scope': 'Tokenization/build_prompt only; no inference/network clients', 'runs': {},
              'limit': args.limit, 'abbr_sha256': hashlib.sha256(args.abbreviations.read_bytes()).hexdigest(),
              'packages': {name: importlib.metadata.version(name) for name in ('nltk', 'razdel')},
              'source_sha256': {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                for module in (rerank, __import__('meno_rag.stand.abbreviation_cache', fromlist=['x']),
                                               __import__('meno_rag.stand.rewriting', fromlist=['x']),
                                               __import__('meno_rag.stand.tokenization', fromlist=['x']))}}
    if args.focus:
        payload = json.loads(args.focus.read_text())
        stream = (candidate['request']['L']['messages'] for case in payload['cases'].values() for candidate in case['pool'])
        result['runs']['fixed_pool'] = profile(stream, abbreviations, compare_saved=True, limit=args.limit, deduplicate=False)
    if args.capture_a and args.capture_b:
        a = {identity(messages) for messages in requests(args.capture_a)}
        b = {identity(messages) for messages in requests(args.capture_b)}
        common = a & b
        result['pair_counts'] = {'common': len(common), 'only_A': len(a - b), 'only_B': len(b - a)}
        stream = (messages for messages in requests(args.capture_a) if identity(messages) in common)
        result['runs']['all_common_pairs'] = profile(stream, abbreviations, compare_saved=True, limit=args.limit)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--abbreviations', type=Path, required=True)
    parser.add_argument('--focus', type=Path)
    parser.add_argument('--capture-a', type=Path)
    parser.add_argument('--capture-b', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
