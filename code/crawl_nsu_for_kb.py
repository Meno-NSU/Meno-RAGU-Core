import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit, quote
from typing import Dict, List, Set, Tuple

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['SOCKS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['socks_proxy'] = ''
os.environ['no_proxy'] = '*'

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pymupdf
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import DomainFilter, FilterChain, URLPatternFilter


SEEDS_BY_TOPIC: Dict[str, List[str]] = {
    'general': [
        'https://www.nsu.ru/n/university/',
        'https://www.nsu.ru/n/university/top20/',
        'https://www.nsu.ru/n/university/rankings/',
        'https://www.nsu.ru/n/university/governance/',
        'https://www.nsu.ru/n/university/details/',
        'https://www.nsu.ru/n/university/history/',
        'https://www.nsu.ru/n/university/how-to-reach/',
        'https://www.nsu.ru/n/university/location/',
        'https://www.nsu.ru/n/university/maps/',
        'https://www.nsu.ru/n/contacts/',
        'https://www.nsu.ru/n/sveden/common/',
    ],
    'admission': [
        'https://www.nsu.ru/n/education/apply/',
        'https://education.nsu.ru/bachelor/',
        'https://www.nsu.ru/n/education/olympiads/',
        'https://www.nsu.ru/n/education/programs/preparatory-courses/',
        'https://www.nsu.ru/n/education/programs/preparatory-courses-special-groups/',
        'https://events.nsu.ru/admission_time/',
        'https://education.nsu.ru/useful-links/',
        'https://www.nsu.ru/n/international/education/',
        'https://www.nsu.ru/n/education/programs/master-s-degree/',
        'https://www.nsu.ru/n/education/apply-info/scholarship/',
        'https://education.nsu.ru/nsu-for-pupils/',
        'https://education.nsu.ru/ege-podgotovka/',
    ],
    'faculties': [
        'https://www.nsu.ru/n/education/departments-sections/',
        'https://www.nsu.ru/n/mathematics-mechanics-department/',
        'https://www.nsu.ru/n/physics-department/',
        'https://www.nsu.ru/n/natural-sciences-department/',
        'https://www.nsu.ru/n/information-technologies-department/',
        'https://www.nsu.ru/n/geology-geophysics-department/',
        'https://www.nsu.ru/n/economics-department/',
        'https://www.nsu.ru/n/humanities-institute/',
        'https://www.nsu.ru/n/med-psy-institute/',
        'https://www.nsu.ru/n/ppl-institute/',
        'https://education.nsu.ru/mechatronics/',
        'https://www.nsu.ru/n/saes/',
        'https://sesc.nsu.ru/',
        'https://ci.nsu.ru/',
    ],
    'campus': [
        'https://www.nsu.ru/n/life/campus/',
        'https://www.nsu.ru/n/life/campus/dormitories/',
        'https://www.nsu.ru/n/life/campus/check-in/',
        'https://www.nsu.ru/n/life/campus/cost-of-living/',
        'https://www.nsu.ru/n/life/campus/rules/',
        'https://www.nsu.ru/n/life/campus/documents/',
        'https://www.nsu.ru/n/life/campus/contacts/',
    ],
    'sport': [
        'https://www.nsu.ru/n/life/sport/',
        'https://www.nsu.ru/n/life/sport/gym/',
        'https://www.nsu.ru/n/life/sport/swimming/',
        'https://www.nsu.ru/n/life/sport/ski/',
        'https://www.nsu.ru/n/life/sport/sauna/',
        'https://www.nsu.ru/n/life/sport/sportarea/',
        'https://www.nsu.ru/n/life/sport/fisculture/',
    ],
    'clubs': [
        'https://www.nsu.ru/n/life/student-clubs/',
        'https://www.nsu.ru/n/life/student-clubs/music-clubs/',
        'https://www.nsu.ru/n/life/student-clubs/dance-clubs/',
        'https://www.nsu.ru/n/life/student-clubs/art-clubs/',
        'https://www.nsu.ru/n/life/student-clubs/comedy-clubs/',
        'https://www.nsu.ru/n/life/student-clubs/intellectual-clubs/',
        'https://www.nsu.ru/n/life/student-clubs/community/',
        'https://www.nsu.ru/n/life/student-clubs/student-media/',
        'https://www.nsu.ru/n/life/student-clubs/international-clubs/',
        'https://www.nsu.ru/n/life/main-annual-events/',
        'https://education.nsu.ru/youth-policy/',
        'https://baza.nsu.ru/',
    ],
    'study': [
        'https://www.nsu.ru/n/students/',
        'https://www.nsu.ru/n/education/stipendii/',
        'https://www.nsu.ru/n/education/social/',
        'https://www.nsu.ru/n/education/academic-calendar/',
        'https://education.nsu.ru/additional-courses/',
        'https://www.nsu.ru/n/students/military-registration/',
        'https://education.nsu.ru/freshmen/',
        'https://www.nsu.ru/n/education/programs/foreign-languages/',
        'https://www.nsu.ru/n/education/inclusive-education/',
        'https://www.nsu.ru/n/education/',
    ],
    'research': [
        'https://www.nsu.ru/n/research/',
        'https://www.nsu.ru/n/research/divisions/',
        'https://www.nsu.ru/n/issc/',
        'https://www.nsu.ru/n/research/conferences/',
        'https://www.nsu.ru/n/research/dissertation-committee/',
        'https://new-research.nsu.ru/portal',
    ],
    'career': [
        'https://www.nsu.ru/n/career/',
        'https://www.nsu.ru/n/alumni/',
        'https://www.nsu.ru/n/alumni/history/',
        'http://alumninsu.ru/news/',
    ],
    'exchange': [
        'https://www.nsu.ru/n/international/',
    ],
    'phd': [
        'https://education.nsu.ru/phd/',
        'https://www.nsu.ru/n/phd/',
        'https://www.nsu.ru/n/education/programs/residency-training/',
    ],
    'psychology': [
        'https://www.nsu.ru/n/life/psikhologicheskaya-podderzhka.php',
        'https://msec.nsu.ru/',
    ],
}


DEEP_ROOTS_BY_TOPIC: Dict[str, List[str]] = {
    'faculties': [
        'https://www.nsu.ru/n/mathematics-mechanics-department/',
        'https://www.nsu.ru/n/physics-department/',
        'https://www.nsu.ru/n/natural-sciences-department/',
        'https://www.nsu.ru/n/information-technologies-department/',
        'https://www.nsu.ru/n/geology-geophysics-department/',
        'https://www.nsu.ru/n/economics-department/',
        'https://www.nsu.ru/n/humanities-institute/',
        'https://www.nsu.ru/n/med-psy-institute/',
        'https://www.nsu.ru/n/ppl-institute/',
        'https://www.nsu.ru/n/saes/',
    ],
    'admission': [
        'https://www.nsu.ru/n/education/apply/',
        'https://education.nsu.ru/bachelor/',
    ],
    'campus': [
        'https://www.nsu.ru/n/life/campus/',
    ],
    'research': [
        'https://www.nsu.ru/n/research/divisions/',
    ],
}


EXCLUDE_PATTERNS = [
    '*/media/news/*',
    '*/media/events/*',
    '*/media/advert/*',
    '*/media/nsu-life/*',
    '*/upload/resize_cache/*',
    '*/upload/iblock/*',
    '*/upload/medialibrary/*',
    '*?PAGEN*',
    '*?menu=*',
    '*/switch-lang/*',
    '*mailto:*',
    '*tel:*',
    '*.jpg', '*.jpeg', '*.png', '*.gif', '*.svg',
    '*.css', '*.js', '*.ico', '*.woff', '*.woff2',
    '*.mp4', '*.mp3', '*.webm',
    '*/wp-admin/*', '*/wp-login*',
    '*#*',
    '*/en/*', '*/cn/*',
    '*/rss/*',
]

ALLOWED_DOMAINS = [
    'nsu.ru', 'www.nsu.ru', 'education.nsu.ru', 'sesc.nsu.ru', 'ci.nsu.ru',
    'saes.nsu.ru', 'msec.nsu.ru', 'new-research.nsu.ru', 'events.nsu.ru',
    'calc-ege.nsu.ru', 'table.nsu.ru', 'baza.nsu.ru', 'alumninsu.ru',
]


def build_filter_chain() -> FilterChain:
    return FilterChain(filters=[
        DomainFilter(allowed_domains=ALLOWED_DOMAINS),
        URLPatternFilter(patterns=EXCLUDE_PATTERNS, reverse=True),
    ])


def base_run_config(deep: bool = False, max_pages: int = 40) -> CrawlerRunConfig:
    md_gen = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.40, threshold_type='fixed', min_word_threshold=8)
    )
    kwargs = dict(
        markdown_generator=md_gen,
        remove_overlay_elements=True,
        excluded_tags=['nav', 'header', 'footer', 'aside', 'script', 'style', 'noscript'],
        word_count_threshold=15,
        only_text=True,
        exclude_all_images=True,
        page_timeout=35000,
        verbose=False,
        mean_delay=0.3,
        semaphore_count=8,
    )
    if deep:
        kwargs['deep_crawl_strategy'] = BFSDeepCrawlStrategy(
            max_depth=2, max_pages=max_pages, filter_chain=build_filter_chain()
        )
    return CrawlerRunConfig(**kwargs)


def extract_text(result) -> str:
    md = getattr(result, 'markdown', None)
    if md is None:
        return ''
    full = getattr(md, 'full_markdown', None)
    if full:
        return full
    fit = getattr(md, 'fit_markdown', None)
    if fit:
        return fit
    return str(md)


def extract_links(result) -> List[str]:
    out: List[str] = []
    links = getattr(result, 'links', None)
    if not isinstance(links, dict):
        return out
    for key in ('internal', 'external'):
        bucket = links.get(key) or []
        for item in bucket:
            if isinstance(item, dict):
                href = item.get('href') or item.get('url')
            else:
                href = item
            if isinstance(href, str):
                out.append(href)
    return out


def result_to_records(result, topic: str, seen: Set[str]) -> Tuple[List[dict], List[str]]:
    records: List[dict] = []
    pdf_links: List[str] = []
    url = getattr(result, 'url', None)
    if not url:
        meta = getattr(result, 'metadata', None) or {}
        url = meta.get('og:url') or meta.get('url')
    if not url:
        return records, pdf_links
    url = url.split('#')[0]
    if url in seen:
        return records, pdf_links
    text = extract_text(result)
    if len(text.strip()) < 120:
        return records, pdf_links
    seen.add(url)
    meta = getattr(result, 'metadata', None) or {}
    title = meta.get('title') or ''
    records.append({
        'url': url,
        'title': title.strip(),
        'topic': topic,
        'source_type': 'html',
        'text': text,
    })
    for href in extract_links(result):
        if href.lower().split('?')[0].endswith('.pdf'):
            pdf_links.append(href)
    return records, pdf_links


def ingest_result(result, topic: str, seen: Set[str]) -> Tuple[List[dict], List[str]]:
    all_records: List[dict] = []
    all_pdfs: List[str] = []
    items = result if isinstance(result, list) else [result]
    for r in items:
        recs, pdfs = result_to_records(r, topic, seen)
        all_records.extend(recs)
        all_pdfs.extend(pdfs)
    return all_records, all_pdfs


async def crawl_phase(crawler: AsyncWebCrawler, urls: List[str], topic: str,
                      deep: bool, seen: Set[str], max_pages: int = 40):
    records: List[dict] = []
    pdf_links: List[str] = []
    if deep:
        for url in urls:
            try:
                res = await crawler.arun(url=url, config=base_run_config(deep=True, max_pages=max_pages))
                recs, pdfs = ingest_result(res, topic, seen)
                records.extend(recs)
                pdf_links.extend(pdfs)
            except Exception as e:
                print(f'[WARN] deep crawl failed for {url}: {type(e).__name__}: {e}', file=sys.stderr)
    else:
        try:
            res = await crawler.arun_many(urls=urls, config=base_run_config(deep=False))
            recs, pdfs = ingest_result(res, topic, seen)
            records.extend(recs)
            pdf_links.extend(pdfs)
        except Exception as e:
            print(f'[WARN] arun_many failed for topic {topic}: {type(e).__name__}: {e}', file=sys.stderr)
    return records, pdf_links


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


def safe_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        path = quote(parts.path, safe='/%+')
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    except Exception:
        return url


def download_pdf(url: str, timeout: int = 40) -> bytes:
    url = safe_url(url)
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def pdf_to_text(data: bytes) -> str:
    doc = pymupdf.open(stream=data, filetype='pdf')
    parts: List[str] = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return '\n'.join(parts)


def topic_from_pdf_url(url: str) -> str:
    u = url.lower()
    if any(k in u for k in ['med-psy', 'medicine', 'medic', 'residency', 'ordinatur', 'klinic']):
        return 'faculties'
    if any(k in u for k in ['economics', 'economy']):
        return 'faculties'
    if any(k in u for k in ['ppl-institute', 'law', 'philosoph']):
        return 'faculties'
    if any(k in u for k in ['dormitor', 'campus', 'obshchezhit', 'check-in', 'rules']):
        return 'campus'
    if any(k in u for k in ['stipend', 'scholarship', 'social']):
        return 'study'
    if any(k in u for k in ['career', 'alumni', 'trudoustroystvo']):
        return 'career'
    return 'faculties'


def load_existing(out_dir: str) -> Tuple[List[dict], Set[str]]:
    records: List[dict] = []
    seen: Set[str] = set()
    path = os.path.join(out_dir, 'pages.jsonl')
    if not os.path.exists(path):
        return records, seen
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
            for u in normalize_urls(rec.get('url', '')):
                seen.add(u)
    print(f'[resume] loaded {len(records)} existing records', file=sys.stderr)
    return records, seen


def dedupe_pdfs(pdf_candidates: List[str]) -> List[str]:
    unique: List[str] = []
    seen_pdf: Set[str] = set()
    for u in pdf_candidates:
        u = u.split('#')[0]
        if u in seen_pdf:
            continue
        seen_pdf.add(u)
        if not u.lower().startswith('http'):
            continue
        if not u.lower().split('?')[0].endswith('.pdf'):
            continue
        unique.append(u)
    return unique


def collect_pdfs_from_records(records: List[dict]) -> List[str]:
    return dedupe_pdfs([u for rec in records for u in (rec.get('pdf_links') or [])])


def write_output(out_dir: str, records: List[dict], pdf_link_count: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pages.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    summary = {
        'total': len(records),
        'html_pages': sum(1 for r in records if r.get('source_type') == 'html'),
        'pdf_documents': sum(1 for r in records if r.get('source_type') == 'pdf'),
        'unique_pdf_links_found': pdf_link_count,
        'topics': {},
    }
    for rec in records:
        summary['topics'][rec['topic']] = summary['topics'].get(rec['topic'], 0) + 1
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)


def process_pdfs(pdf_links: List[str], pdf_budget: int, seen_pdf_urls: Set[str]) -> List[dict]:
    pdf_records: List[dict] = []
    for url in pdf_links:
        if len(pdf_records) >= pdf_budget:
            break
        if url in seen_pdf_urls:
            continue
        seen_pdf_urls.add(url)
        try:
            data = download_pdf(url)
            text = pdf_to_text(data)
        except Exception as e:
            print(f'[WARN] pdf failed {url}: {type(e).__name__}: {e}', file=sys.stderr)
            continue
        if len(text.strip()) < 200:
            continue
        pdf_records.append({
            'url': url,
            'title': text.strip()[:160].replace('\n', ' '),
            'topic': topic_from_pdf_url(url),
            'source_type': 'pdf',
            'text': text,
        })
        print(f'[pdf] +{url[:80]} ({len(text)} chars)', file=sys.stderr)
    return pdf_records


async def run(out_dir: str, mode: str, deep_max_pages: int, deep_time_budget: float, pdf_budget: int):
    records, seen = load_existing(out_dir)
    if mode == 'all':
        records, seen = [], set()
    pdf_candidates: List[str] = collect_pdfs_from_records(records)
    seen_pdf_urls: Set[str] = set()
    for r in records:
        if r.get('source_type') == 'pdf':
            for u in normalize_urls(r.get('url', '')):
                seen_pdf_urls.add(u)

    browser_cfg = BrowserConfig(headless=True, text_mode=True, verbose=False, light_mode=True)
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        if mode in ('curated', 'all'):
            for topic, urls in SEEDS_BY_TOPIC.items():
                print(f'[curated] {topic}: {len(urls)} urls', file=sys.stderr)
                recs, pdfs = await crawl_phase(crawler, urls, topic, deep=False, seen=seen)
                for r in recs:
                    r['pdf_links'] = []
                records.extend(recs)
                pdf_candidates = dedupe_pdfs(pdf_candidates + pdfs)
            write_output(out_dir, records, len(pdf_candidates))

        if mode in ('deep', 'all'):
            if deep_time_budget <= 0:
                print('[budget] deep phase disabled (deep_time_budget<=0)', file=sys.stderr)
            else:
                deep_roots = DEEP_ROOTS_BY_TOPIC.get('faculties', [])
                t_deep = time.time()
                for url in deep_roots:
                    if (time.time() - t_deep) > deep_time_budget:
                        print('[budget] deep time budget reached', file=sys.stderr)
                        break
                    print(f'[deep] {url} (elapsed {time.time() - t_deep:.0f}s)', file=sys.stderr)
                    try:
                        res = await crawler.arun(url=url, config=base_run_config(deep=True, max_pages=deep_max_pages))
                    except Exception as e:
                        print(f'[WARN] deep failed {url}: {type(e).__name__}: {e}', file=sys.stderr)
                        continue
                    recs, pdfs = ingest_result(res, 'faculties', seen)
                    records.extend(recs)
                    pdf_candidates = dedupe_pdfs(pdf_candidates + pdfs)
                    write_output(out_dir, records, len(pdf_candidates))

    if mode in ('curated', 'all'):
        try:
            with open(os.path.join(out_dir, 'pdf_candidates.json'), 'w', encoding='utf-8') as f:
                json.dump(pdf_candidates, f, ensure_ascii=False)
        except Exception:
            pass
        pdf_records = process_pdfs(pdf_candidates, pdf_budget, seen_pdf_urls)
        records.extend(pdf_records)

    write_output(out_dir, records, len(pdf_candidates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', dest='output_dir', type=str, default='/tmp/opencode/nsu_crawl',
                        help='Output directory for pages.jsonl and summary.json.')
    parser.add_argument('--mode', dest='mode', type=str, default='all',
                        choices=['curated', 'deep', 'all'],
                        help='curated: only curated seeds + pdfs; deep: resume and deep-crawl faculty roots; all: both.')
    parser.add_argument('--deep_max_pages', dest='deep_max_pages', type=int, default=12,
                        help='Max pages per deep-crawl root.')
    parser.add_argument('--deep_time_budget', dest='deep_time_budget', type=float, default=600.0,
                        help='Wall-clock budget in seconds for the whole deep phase.')
    parser.add_argument('--pdf_budget', dest='pdf_budget', type=int, default=100,
                        help='Maximum number of PDF documents to download and parse.')
    args = parser.parse_args()
    t0 = time.time()
    asyncio.run(run(args.output_dir, args.mode, args.deep_max_pages, args.deep_time_budget, args.pdf_budget))
    print(f'[done] elapsed {time.time() - t0:.1f}s -> {args.output_dir}', file=sys.stderr)


if __name__ == '__main__':
    main()
