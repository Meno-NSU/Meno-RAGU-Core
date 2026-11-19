"""
Runs a prepared model pass of graph v6.1 (graph_utils/graph_patch_jobs.py) against an OpenAI-compatible endpoint.
Cached and resumable: an answer already in --cache is not asked again; stop it at any time and start it again.

    python graph_v61_llm_runner.py --jobs llm/pairs.jobs.jsonl --concurrency 16
    (results: llm/pairs.results.jsonl, cache: llm/cache.jsonl)
"""
import argparse
import json
import time
from pathlib import Path

from graph_utils import graph_patch_jobs as jobs_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--base_url', default='http://everest.nsu.ru:9022/v1')
    parser.add_argument('--model', default='gemma-4-12B-it')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--max_tokens', type=int, default=900)
    parser.add_argument('--limit', type=int, default=0, help='only the first N jobs (a trial run)')
    args = parser.parse_args()
    out = args.out or args.jobs.with_name(args.jobs.name.replace('.jobs.jsonl', '.results.jsonl'))
    cache = args.cache or args.jobs.with_name('cache.jsonl')
    jobs = [json.loads(row) for row in args.jobs.read_text(encoding='utf-8').splitlines() if row.strip()]
    if args.limit:
        jobs = jobs[:args.limit]
    started = time.time()

    def progress(done, total):
        rate = done / max(time.time() - started, 1)
        print(time.strftime('%H:%M:%S'), f'{done}/{total} asked, {rate:.2f} calls/s, '
              f'{(total - done) / max(rate, 1e-9) / 60:.0f} min left', flush=True)

    counts = jobs_.run_jobs(jobs, cache, out, args.model,
                            lambda prompt: jobs_.ask(args.base_url, args.model, prompt, args.max_tokens),
                            args.concurrency, progress=progress)
    print(json.dumps({**counts, 'seconds': round(time.time() - started), 'results': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
