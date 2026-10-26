from argparse import ArgumentParser
import codecs
from functools import partial
import json
import math
from multiprocessing import Pool
import os
import random
import re
import time
from typing import List, Tuple, Union

from razdel import sentenize, tokenize
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import trange

from tokenization_utils.tokenization_utils import load_hf_tokenizer

RANDOM_SEED: int = 42

SEPARATOR_RE = re.compile(r'^[-*|:\s]+$')


def tokenize_text(s: str, word_as_unit: bool) -> List[Tuple[int, int]]:
    bounds_of_text_items = []
    prev_item_end = 0
    for cur_item in (tokenize(s) if word_as_unit else sentenize(s)):
        found_idx = s[prev_item_end:].find(cur_item.text)
        if found_idx < 0:
            raise RuntimeError(f'The text cannot be tokenized!\n{s}')
        item_start = found_idx + prev_item_end
        item_end = item_start + len(cur_item.text)
        bounds_of_text_items.append((item_start, item_end))
        prev_item_end = item_end
    return bounds_of_text_items


def split_by_lang_items(full_text: str, max_tokens_per_chunk: int,
                        tok: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                        word_as_unit: bool, initial_pos: int = 0) -> List[Tuple[int, int]]:
    full_text_ = full_text.strip()
    if len(full_text_) == 0:
        return []
    num_tokens = len(tok.tokenize(full_text))
    if num_tokens <= max_tokens_per_chunk:
        start_idx = full_text.find(full_text_)
        end_idx = start_idx + len(full_text_)
        return [(start_idx + initial_pos, end_idx + initial_pos)]
    del full_text_
    lang_item_boundaries = tokenize_text(full_text, word_as_unit=word_as_unit)
    num_lang_items = len(lang_item_boundaries)
    if num_lang_items == 0:
        raise RuntimeError(f'There are no language items in the text!\n{full_text}')
    if num_lang_items < 3:
        if word_as_unit:
            raise RuntimeError(f'The text cannot be splitted into the chunks!\n{full_text}')
        return split_by_lang_items(full_text, max_tokens_per_chunk, tok, True, initial_pos)
    lang_item_lengths = []
    for item_start, item_end in lang_item_boundaries:
        lang_item_lengths.append(len(tok.tokenize(full_text[item_start:item_end])))
    best_split_idx = -1
    best_diff = math.inf
    for split_idx in range(1, num_lang_items - 1):
        num_left_tokens = sum(lang_item_lengths[0:(split_idx + 1)])
        num_right_tokens = sum(lang_item_lengths[split_idx:])
        cur_diff = abs(num_left_tokens - num_right_tokens)
        if best_split_idx < 0:
            best_split_idx = split_idx
            best_diff = cur_diff
        elif cur_diff < best_diff:
            best_split_idx = split_idx
            best_diff = cur_diff
        if best_diff < 1:
            break
    left_chunk_start = lang_item_boundaries[0][0]
    left_chunk_end = lang_item_boundaries[best_split_idx][1]
    right_chunk_start = lang_item_boundaries[best_split_idx][0]
    right_chunk_end = lang_item_boundaries[-1][1]
    chunks = split_by_lang_items(full_text[left_chunk_start:left_chunk_end], max_tokens_per_chunk, tok,
                                 word_as_unit, initial_pos + left_chunk_start)
    chunks += split_by_lang_items(full_text[right_chunk_start:right_chunk_end], max_tokens_per_chunk, tok,
                                  word_as_unit, initial_pos + right_chunk_start)
    return chunks


def split_by_paragraphs(full_text: str, max_tokens_per_chunk: int,
                        tok: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                        initial_pos: int = 0) -> List[Tuple[int, int]]:
    full_text_ = full_text.strip()
    if len(full_text_) == 0:
        return []
    num_tokens = len(tok.tokenize(full_text))
    if num_tokens <= max_tokens_per_chunk:
        start_idx = full_text.find(full_text_)
        end_idx = start_idx + len(full_text_)
        return [(start_idx + initial_pos, end_idx + initial_pos)]
    del full_text_
    paragraphs = list(filter(
        lambda it2: len(it2) > 0,
        map(lambda it1: it1.strip(), full_text.strip().split('\n'))
    ))
    num_paragraphs = len(paragraphs)
    if num_paragraphs == 0:
        raise RuntimeError(f'There are no paragraphs in the text!\n{full_text}')
    if num_paragraphs < 3:
        return split_by_lang_items(full_text, max_tokens_per_chunk, tok, False, initial_pos)
    paragraph_boundaries = []
    paragraph_lengths = []
    prev_paragraph_end = 0
    for cur_paragraph in paragraphs:
        found_idx = full_text[prev_paragraph_end:].find(cur_paragraph)
        if found_idx < 0:
            raise RuntimeError(f'The text cannot be tokenized!\n{full_text}')
        paragraph_start = found_idx + prev_paragraph_end
        paragraph_end = paragraph_start + len(cur_paragraph)
        paragraph_boundaries.append((paragraph_start, paragraph_end))
        paragraph_lengths.append(len(tok.tokenize(full_text[paragraph_start:paragraph_end])))
        prev_paragraph_end = paragraph_end
    best_split_idx = -1
    best_diff = math.inf
    for split_idx in range(1, num_paragraphs - 1):
        num_left_tokens = sum(paragraph_lengths[0:(split_idx + 1)])
        num_right_tokens = sum(paragraph_lengths[split_idx:])
        cur_diff = abs(num_left_tokens - num_right_tokens)
        if best_split_idx < 0:
            best_split_idx = split_idx
            best_diff = cur_diff
        elif cur_diff < best_diff:
            best_split_idx = split_idx
            best_diff = cur_diff
        if best_diff < 1:
            break
    left_chunk_start = paragraph_boundaries[0][0]
    left_chunk_end = paragraph_boundaries[best_split_idx][1]
    right_chunk_start = paragraph_boundaries[best_split_idx][0]
    right_chunk_end = paragraph_boundaries[-1][1]
    chunks = split_by_paragraphs(full_text[left_chunk_start:left_chunk_end], max_tokens_per_chunk, tok,
                                 initial_pos + left_chunk_start)
    chunks += split_by_paragraphs(full_text[right_chunk_start:right_chunk_end], max_tokens_per_chunk, tok,
                                  initial_pos + right_chunk_start)
    return chunks


def _is_list_line(s: str) -> bool:
    return bool(re.match(r'^[-*+]\s', s)) or bool(re.match(r'^\d+[.)]\s', s))


def _is_bold_header(s: str) -> bool:
    s = s.strip()
    return s.startswith('**') and not _is_list_line(s)


def _classify_line(s: str) -> str:
    if not s.strip():
        return 'blank'
    if s.strip().startswith('#'):
        return 'atx_header'
    if _is_bold_header(s):
        return 'bold_header'
    if _is_list_line(s):
        return 'list'
    return 'text'


def _split_oversized(lines: List[str], start: int, end: int, max_tokens: int,
                     line_starts: List[int], line_end_fn,
                     tok: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> List[Tuple[int, int]]:
    def _tokens(s, e):
        return len(tok.tokenize('\n'.join(lines[s:e])))

    result = []
    stack = [(start, end)]
    while stack:
        s, e = stack.pop()
        total = _tokens(s, e)
        if total <= max_tokens:
            if '\n'.join(lines[s:e]).strip():
                result.append((line_starts[s], line_end_fn(e - 1)))
            continue
        if e - s <= 1:
            base = line_starts[s]
            sub = split_by_lang_items(lines[s], max_tokens, tok, False, 0)
            result.extend((base + ss, base + ee) for ss, ee in sub)
            continue
        best_j = -1
        best_score = math.inf
        acc = 0
        for j in range(s, e):
            acc += len(tok.tokenize(lines[j]))
            diff = abs(acc - (total - acc))
            cls = _classify_line(lines[j])
            prev_cls = _classify_line(lines[j - 1]) if j > s else 'blank'
            penalty = 0
            if cls == 'blank' or prev_cls == 'blank':
                penalty = -50
            elif (cls == 'list') != (prev_cls == 'list'):
                penalty = -30
            score = diff + penalty
            if score < best_score:
                best_score = score
                best_j = j + 1
        if best_j <= s or best_j >= e:
            best_j = (s + e) // 2
        stack.append((best_j, e))
        stack.append((s, best_j))
    return result


def _consolidate_chunks(boundaries: List[Tuple[int, int]],
                        full_text: str,
                        tok: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                        max_tokens: int, min_tokens: int) -> List[Tuple[int, int]]:
    if len(boundaries) == 0:
        return []
    kept = []
    for s, e in boundaries:
        txt = full_text[s:e].strip()
        if len(txt) == 0 or SEPARATOR_RE.match(txt):
            continue
        n = len(tok.tokenize(full_text[s:e]))
        if n == 0:
            continue
        kept.append([s, e, n])
    if len(kept) == 0:
        return []
    merged = [kept[0]]
    for s, e, n in kept[1:]:
        ps, pe, pn = merged[-1]
        new_e = max(pe, e)
        new_n = len(tok.tokenize(full_text[ps:new_e]))
        if n < min_tokens and new_n <= max_tokens:
            merged[-1][1] = new_e
            merged[-1][2] = new_n
        else:
            merged.append([s, e, n])
    i = len(merged) - 2
    while i >= 0:
        s, e, n = merged[i]
        if n < min_tokens:
            ns, ne, nn = merged[i + 1]
            new_s = min(s, ns)
            new_n = len(tok.tokenize(full_text[new_s:ne]))
            if new_n <= max_tokens:
                merged[i + 1][0] = new_s
                merged[i + 1][2] = new_n
                merged.pop(i)
        i -= 1
    return [(s, e) for s, e, n in merged if n >= min_tokens]


def split_by_semantic_blocks(full_text: str, max_tokens_per_chunk: int,
                             tok: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                             min_tokens_per_chunk: int = 30,
                             initial_pos: int = 0) -> List[Tuple[int, int]]:
    full_text_ = full_text.strip()
    if len(full_text_) == 0:
        return []
    num_tokens = len(tok.tokenize(full_text))
    if num_tokens <= max_tokens_per_chunk:
        start_idx = full_text.find(full_text_)
        end_idx = start_idx + len(full_text_)
        return _consolidate_chunks(
            [(start_idx + initial_pos, end_idx + initial_pos)],
            full_text, tok, max_tokens_per_chunk, min_tokens_per_chunk)

    lines = full_text.split('\n')
    n = len(lines)

    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    def line_end(i):
        return line_starts[i] + len(lines[i])

    def slice_tokens(s, e):
        return len(tok.tokenize('\n'.join(lines[s:e])))

    group_leaders = [0]
    for i in range(1, n):
        cls = _classify_line(lines[i])
        prev_cls = _classify_line(lines[i - 1])
        if cls == 'blank':
            group_leaders.append(i)
        elif cls in ('atx_header', 'bold_header'):
            group_leaders.append(i)
        elif cls == 'list' and prev_cls != 'list':
            group_leaders.append(i)

    groups = []
    for gi in range(len(group_leaders)):
        start = group_leaders[gi]
        end = group_leaders[gi + 1] if gi + 1 < len(group_leaders) else n
        groups.append((start, end))

    merged = []
    i = 0
    while i < len(groups):
        g = groups[i]
        first_cls = _classify_line(lines[g[0]])
        if first_cls in ('atx_header', 'bold_header') and i + 1 < len(groups):
            ng = groups[i + 1]
            if slice_tokens(g[0], ng[1]) <= max_tokens_per_chunk:
                merged.append((g[0], ng[1]))
                i += 2
                continue
        merged.append(g)
        i += 1
    groups = merged

    merged = []
    i = 0
    while i < len(groups):
        g = groups[i]
        glc = _classify_line(lines[g[1] - 1]) if g[1] > g[0] else 'blank'
        while i + 1 < len(groups):
            ng = groups[i + 1]
            nfc = _classify_line(lines[ng[0]])
            ct = slice_tokens(g[0], ng[1])
            if ct > max_tokens_per_chunk:
                break
            if glc == 'list' and nfc == 'list':
                g = (g[0], ng[1])
                glc = _classify_line(lines[g[1] - 1]) if g[1] > g[0] else 'blank'
                i += 1
            else:
                break
        merged.append(g)
        i += 1
    groups = merged

    merged = []
    i = 0
    while i < len(groups):
        g = groups[i]
        gt = slice_tokens(g[0], g[1])
        while i + 1 < len(groups):
            ng = groups[i + 1]
            ct = slice_tokens(g[0], ng[1])
            if ct > max_tokens_per_chunk:
                break
            nt = slice_tokens(ng[0], ng[1])
            if gt < max_tokens_per_chunk * 0.35 or nt < max_tokens_per_chunk * 0.35 or ct <= max_tokens_per_chunk * 0.65:
                g = (g[0], ng[1])
                gt = ct
                i += 1
            else:
                break
        merged.append(g)
        i += 1
    groups = merged

    result = []
    for g in groups:
        gt = slice_tokens(g[0], g[1])
        group_text = '\n'.join(lines[g[0]:g[1]]).strip()
        if not group_text or SEPARATOR_RE.match(group_text):
            continue
        if gt <= max_tokens_per_chunk:
            result.append((line_starts[g[0]] + initial_pos, line_end(g[1] - 1) + initial_pos))
        else:
            sub = _split_oversized(lines, g[0], g[1], max_tokens_per_chunk, line_starts, line_end, tok)
            result.extend([(s + initial_pos, e + initial_pos) for s, e in sub])

    return _consolidate_chunks(result, full_text, tok, max_tokens_per_chunk, min_tokens_per_chunk)


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output JSONL knowledge database enriched with the lengths of texts in tokens.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The name of LLM whose tokenizer will be used to estimate the number of tokens.')
    parser.add_argument('--maxtokens', dest='max_num_tokens', type=int, required=False, default=256,
                        help='The maximum number of tokens in the chunk.')
    parser.add_argument('--mintokens', dest='min_num_tokens', type=int, required=False, default=30,
                        help='The minimum number of tokens in the chunk. Ultra-short chunks are merged '
                             'into neighbors or dropped.')
    args = parser.parse_args()

    if args.min_num_tokens > 100:
        raise ValueError(f'The "--mintokens" value must not be greater than 100, '
                         f'but {args.min_num_tokens} is given!')
    if args.max_num_tokens < round(1.5 * args.min_num_tokens):
        raise ValueError(f'The "--maxtokens" value ({args.max_num_tokens}) must be at least 1.5 times greater '
                         f'than the "--mintokens" value ({args.min_num_tokens}).')

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        base_dirname = os.path.dirname(output_fname)
        if len(base_dirname) > 0:
            if not os.path.isdir(base_dirname):
                raise IOError(f'The directory "{base_dirname}" does not exist!')

    llm_tokenizer = load_hf_tokenizer(args.model_name)
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

    num_processes = max(1, os.cpu_count() - 1)
    if num_processes > 1:
        print(f'There are {num_processes} parallel processes for chunking.')
    chunking_func = partial(split_by_semantic_blocks, max_tokens_per_chunk=args.max_num_tokens,
                            min_tokens_per_chunk=args.min_num_tokens, tok=llm_tokenizer)
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        chunking_res = pool.map(chunking_func, [cur['text'] for cur in samples])
    end_time = time.time()
    print(f'The preliminary results of chunking have been prepared. It took {round(end_time - start_time)} seconds.')
    del chunking_func, llm_tokenizer
    if len(samples) != len(chunking_res):
        err_msg = (f'The number of source samples does not equal to the number of the preliminary chunking results! '
                   f'{len(samples)} != {len(chunking_res)}')
        raise RuntimeError(err_msg)

    num_skipped = 0
    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as dst_fp:
        for idx in trange(len(samples)):
            chunk_boundaries = chunking_res[idx]
            if len(chunk_boundaries) < 1:
                num_skipped += 1
                continue
            set_of_chunk_texts = set()
            deduplicated_chunk_boundaries = [chunk_boundaries[0]]
            normalized_chunk_text = ' '.join(
                samples[idx]['text'][chunk_boundaries[0][0]:chunk_boundaries[0][1]].strip().lower().split()
            ).strip()
            set_of_chunk_texts.add(normalized_chunk_text)
            del normalized_chunk_text
            for chunk_start, chunk_end in chunk_boundaries[1:]:
                normalized_chunk_text = ' '.join(
                    samples[idx]['text'][chunk_start:chunk_end].strip().lower().split()
                ).strip()
                if normalized_chunk_text in set_of_chunk_texts:
                    continue
                set_of_chunk_texts.add(normalized_chunk_text)
                deduplicated_chunk_boundaries.append((chunk_start, chunk_end))
                del normalized_chunk_text
            samples[idx]['chunk_boundaries'] = deduplicated_chunk_boundaries
            dst_fp.write(json.dumps(samples[idx], ensure_ascii=False) + '\n')
            del deduplicated_chunk_boundaries, chunk_boundaries, set_of_chunk_texts
    print(f'Chunked samples are successfully saved into the "{output_fname}".')
    if num_skipped > 0:
        print(f'{num_skipped} samples were skipped because they are shorter than '
              f'the minimum chunk size ({args.min_num_tokens} tokens).')


if __name__ == '__main__':
    main()
