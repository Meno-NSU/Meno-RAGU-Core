from argparse import ArgumentParser
import codecs
import json
import os
import re
import sys
from typing import List, Tuple, Dict, Optional, Union


RANDOM_SEED: int = 42

DEFAULT_MIN_CHARS: int = 250
DEFAULT_MAX_CHARS: int = 12000

UNIVERSITY_CONTEXT: str = 'Новосибирский государственный университет (НГУ)'
HOMEPAGE_URL: str = 'https://www.nsu.ru/n/'

SUBSECTION_LEVEL_SECTIONS = {'4'}

UNIT_EXPANSIONS: List[Tuple[str, str]] = [
    ('руб./мес.', 'рублей в месяц'),
    ('руб./год.', 'рублей в год'),
    ('руб./год', 'рублей в год'),
    ('руб.', 'рублей'),
    ('мес.', 'месяцев'),
    ('чел.', 'человек'),
]


def normalize_urls(value: Union[str, List[str], None]) -> List[str]:
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
    return list(urls)


def clean_title(title: str) -> str:
    title = title.strip()
    title = re.sub(r'^\d+(?:\.\d+)*\.?\s+', '', title)
    return title.strip()


def section_number(title: str) -> Optional[str]:
    m = re.match(r'^(\d+)\.', title.strip())
    return m.group(1) if m else None


def expand_units(text: str) -> str:
    for abbr, full in UNIT_EXPANSIONS:
        text = text.replace(abbr, full)
    return text


def strip_bold(text: str) -> str:
    return text.replace('**', '')


def parse_toc(lines: List[str]) -> Dict[str, str]:
    toc: Dict[str, str] = {}
    pattern = re.compile(r'^(\d+)\.\s+\[[^\]]+\]\((https?://[^)]+)\)')
    for line in lines:
        m = pattern.match(line.strip())
        if m:
            toc[m.group(1)] = m.group(2)
    return toc


def parse_h2_blocks(lines: List[str]) -> List[Tuple[Optional[str], str, List[str]]]:
    blocks: List[Tuple[Optional[str], str, List[str]]] = []
    cur_number: Optional[str] = None
    cur_title: str = ''
    cur_body: List[str] = []
    started: bool = False
    for line in lines:
        m = re.match(r'^##(?!#)\s+(.+?)\s*$', line)
        if m:
            if started:
                blocks.append((cur_number, cur_title, cur_body))
            cur_title = m.group(1).strip()
            cur_number = section_number(cur_title)
            cur_body = []
            started = True
        elif started:
            cur_body.append(line)
    if started:
        blocks.append((cur_number, cur_title, cur_body))
    return blocks


def split_by_h3(body_lines: List[str]) -> List[Tuple[str, List[str]]]:
    subs: List[Tuple[str, List[str]]] = []
    parent_intro: List[str] = []
    cur_title: Optional[str] = None
    cur_body: List[str] = []
    for line in body_lines:
        m = re.match(r'^###(?!#)\s+(.+?)\s*$', line)
        if m:
            if cur_title is not None:
                subs.append((cur_title, cur_body))
            cur_title = m.group(1).strip()
            cur_body = []
        elif cur_title is None:
            parent_intro.append(line)
        else:
            cur_body.append(line)
    if cur_title is not None:
        subs.append((cur_title, cur_body))
    if subs and any(ln.strip() for ln in parent_intro):
        first_title, first_body = subs[0]
        subs[0] = (first_title, parent_intro + [''] + first_body)
    return subs


def unwrap_details_and_html(lines: List[str]) -> List[str]:
    tag_re = re.compile(r'</?(?:details|summary)>')
    result: List[str] = []
    for line in lines:
        cleaned = tag_re.sub('', line)
        if cleaned.strip():
            result.append(cleaned)
    return result


def strip_blockquote(lines: List[str]) -> List[str]:
    result: List[str] = []
    for line in lines:
        m = re.match(r'^\s*>\s?(.*)$', line)
        if m:
            result.append(m.group(1))
        else:
            result.append(line)
    return result


def convert_internal_headers(lines: List[str]) -> List[str]:
    result: List[str] = []
    for line in lines:
        m = re.match(r'^#{2,6}\s+(.+?)\s*$', line)
        if m:
            title = clean_title(m.group(1).strip())
            if title:
                result.append(f'**{title}**')
            continue
        result.append(line)
    return result


def is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith('|') and s.endswith('|') and len(s) > 1


def split_table_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [cell.strip() for cell in s.split('|')]


def is_separator_row(line: str) -> bool:
    cells = split_table_row(line)
    if len(cells) == 0:
        return False
    for cell in cells:
        if not re.fullmatch(r':?-{2,}:?', cell):
            return False
    return True


def make_caption(raw_caption: str, fallback_titles: List[str]) -> str:
    text = strip_bold(raw_caption).strip()
    text = expand_units(text)
    text = re.sub(r'[:]\s*$', '.', text)
    if text and not (text.endswith('.') or text.endswith('!') or text.endswith('?')):
        text = text + '.'
    if len(text) == 0:
        caption_base = '. '.join([clean_title(t) for t in fallback_titles if clean_title(t)])
        text = caption_base + '.' if caption_base else ''
    return text


def render_table(table_lines: List[str], raw_caption: str, fallback_titles: List[str]) -> List[str]:
    rows: List[List[str]] = []
    for line in table_lines:
        if is_separator_row(line):
            continue
        rows.append(split_table_row(line))
    if len(rows) < 2:
        return list(table_lines)
    header = rows[0]
    data_rows = rows[1:]
    caption = make_caption(raw_caption, fallback_titles)
    rendered: List[str] = []
    for row_idx, row in enumerate(data_rows):
        if row_idx > 0:
            rendered.append('')
        if caption:
            rendered.append(caption)
        for col_idx, col_name in enumerate(header):
            value = row[col_idx].strip() if col_idx < len(row) else ''
            col_name_clean = strip_bold(col_name).strip()
            rendered.append(f'{col_name_clean}: {value}')
    return rendered


def render_body(body_lines: List[str], fallback_titles: List[str]) -> List[str]:
    body_lines = unwrap_details_and_html(body_lines)
    body_lines = strip_blockquote(body_lines)
    body_lines = convert_internal_headers(body_lines)
    output: List[str] = []
    i = 0
    n = len(body_lines)
    while i < n:
        if is_table_row(body_lines[i]):
            table_start = i
            while i < n and is_table_row(body_lines[i]):
                i += 1
            table_block = body_lines[table_start:i]
            cap_idx = len(output) - 1
            while cap_idx >= 0 and output[cap_idx].strip() == '':
                cap_idx -= 1
            raw_caption = ''
            consumed = False
            if cap_idx >= 0:
                candidate = output[cap_idx]
                if '\n' not in candidate and not candidate.strip().startswith('-'):
                    raw_caption = candidate
                    del output[cap_idx]
                    while output and output[-1].strip() == '':
                        output.pop()
                    consumed = True
            if not consumed:
                raw_caption = '. '.join([clean_title(t) for t in fallback_titles if clean_title(t)])
            rendered = render_table(table_block, raw_caption, fallback_titles)
            if output and output[-1].strip() != '':
                output.append('')
            output.extend(rendered)
            output.append('')
        else:
            output.append(body_lines[i])
            i += 1
    return output


def collapse_blank_lines(lines: List[str]) -> List[str]:
    result: List[str] = []
    prev_blank = False
    for line in lines:
        is_blank = (line.strip() == '')
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank
    while result and result[0].strip() == '':
        result.pop(0)
    while result and result[-1].strip() == '':
        result.pop()
    return result


def extract_all_urls(text: str) -> List[str]:
    urls: List[str] = []
    for m in re.finditer(r'\[[^\]]+\]\((https?://[^)\s]+)\)', text):
        urls.append(m.group(1))
    for m in re.finditer(r'(?<![\(\[])(https?://[^\s)\]\}]+)', text):
        urls.append(m.group(0).rstrip('.,;:'))
    return urls


def url_priority_score(url: str) -> int:
    u = url.lower()
    is_binary = u.endswith('.pdf') or u.endswith('.docx') or '/upload/' in u
    is_nsu = 'nsu.ru' in u
    is_edu = 'education.nsu.ru' in u
    is_n_page = ('nsu.ru/n/' in u) and (not is_binary)
    if is_n_page:
        return 100
    if is_edu and not is_binary:
        return 90
    if is_nsu and not is_binary:
        return 70
    if not is_binary:
        return 50
    return 30


def collect_urls(text: str) -> List[str]:
    seen: List[str] = []
    seen_set = set()
    for u in extract_all_urls(text):
        if u not in seen_set:
            seen.append(u)
            seen_set.add(u)
    seen.sort(key=lambda u: -url_priority_score(u))
    return seen


def build_context_header(path: List[str]) -> str:
    titles = [clean_title(t) for t in path]
    titles = [t for t in titles if t]
    if not titles:
        return UNIVERSITY_CONTEXT
    return UNIVERSITY_CONTEXT + '. ' + '. '.join(titles)


def main():
    parser = ArgumentParser(
        description='Convert data/nsu_main_info.md into a self-contained JSONL corpus '
                    'for the Meno RAG system (one document per top-level section, '
                    'except the faculties section which is split per subsection).')
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='Path to the source Markdown file (nsu_main_info.md).')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='Path to the output JSONL file.')
    parser.add_argument('--min_chars', dest='min_chars', type=int, required=False,
                        default=DEFAULT_MIN_CHARS,
                        help='Minimal size of a standalone document in characters; '
                             'shorter parent intros are merged into the first child '
                             '(default: %(default)s).')
    parser.add_argument('--max_chars', dest='max_chars', type=int, required=False,
                        default=DEFAULT_MAX_CHARS,
                        help='Soft threshold for emitting a length warning '
                             '(default: %(default)s).')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')
    output_fname = os.path.normpath(args.output_name)
    output_dir = os.path.dirname(output_fname)
    if (len(output_dir) > 0) and (not os.path.isdir(output_dir)):
        raise IOError(f'The directory "{output_dir}" does not exist!')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        raw_lines = fp.read().splitlines()

    toc = parse_toc(raw_lines)
    blocks = parse_h2_blocks(raw_lines)

    chunks: List[Tuple[str, Union[str, List[str]]]] = []
    seen_norm: set = set()

    def emit(path: List[str], body: List[str]):
        body_text = '\n'.join(body).strip()
        if len(body_text.replace(' ', '')) == 0:
            return
        rendered_lines = render_body(body, path)
        rendered_lines = collapse_blank_lines(rendered_lines)
        rendered = '\n'.join(rendered_lines).strip()
        if len(rendered.replace(' ', '')) == 0:
            return
        context_header = build_context_header(path)
        chunk_text = f'**{context_header}.**\n\n{rendered}'
        norm = re.sub(r'\s+', ' ',
                      re.sub(r'[^0-9a-zа-яё]', '', chunk_text.lower().replace('ё', 'е'))).strip()
        if len(norm) == 0 or norm in seen_norm:
            return
        seen_norm.add(norm)
        h2_number = None
        for t in path:
            n = section_number(t)
            if n is not None:
                h2_number = n
                break
        urls = collect_urls(chunk_text)
        if len(urls) == 0 and h2_number is not None and h2_number in toc:
            urls = [toc[h2_number]]
        if len(urls) == 0:
            urls = [HOMEPAGE_URL]
        chunks.append((chunk_text, urls_field(urls)))

    for h2_number, h2_title, body in blocks:
        if clean_title(h2_title).lower() == 'содержание':
            continue
        if h2_number in SUBSECTION_LEVEL_SECTIONS:
            subs = split_by_h3(body)
            if len(subs) == 0:
                emit([h2_title], body)
            else:
                for sub_title, sub_body in subs:
                    emit([h2_title, sub_title], sub_body)
        else:
            emit([h2_title], body)

    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as dst:
        for text, url in chunks:
            record = {'text': text, 'url': url}
            dst.write(json.dumps(record, ensure_ascii=False) + '\n')

    long_count = sum(1 for text, _ in chunks if len(text) > args.max_chars)
    print(f'{len(chunks)} documents are successfully saved into "{output_fname}".', file=sys.stderr)
    if long_count > 0:
        print(f'Warning: {long_count} document(s) exceed --max_chars={args.max_chars}.',
              file=sys.stderr)


if __name__ == '__main__':
    main()
