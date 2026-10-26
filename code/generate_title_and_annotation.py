from argparse import ArgumentParser
import codecs
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['SOCKS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['socks_proxy'] = ''
os.environ['all_proxy'] = ''
os.environ['no_proxy'] = '*'

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from pydantic import BaseModel
from tqdm import tqdm
import openai
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
except:
    from vllm.sampling_params import GuidedDecodingParams as StructuredOutputsParams
from transformers import AutoTokenizer, AutoConfig

from vllm_utils.vllm_utils import finalize_vllm


class DocumentDescription(BaseModel):
    title: str
    annotation: str


summarization_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42
MAX_NEW_TOKENS: int = 1024
SYSTEM_PROMPT: str = 'Вы — эксперт по саммаризации документов для RAG-систем.'
PROMPT_TEMPLATE_FOR_LONG_DOCUMENT: str = '''Ваша задача — на основе **неполных фрагментов** большого документа создать его **название** и **аннотацию** в формате JSON. Эти метаданные будут использоваться для того, чтобы в будущем эти метаданные — название и аннотация — могли подаваться во входной контекст LLM вместе с релевантными чанками этого документа.

### Важные ограничения
- Оригинальный документ может быть очень большим (до 1 млн токенов). Вы видите только **5 фрагментов**: первый, второй и три случайных из середины/конца.
- Фрагменты могут не содержать явного названия или введения. Нужно **экстраполировать** общую тему документа на основе имеющихся кусков.
- **Не выдумывайте** факты, которых нет ни в одном фрагменте. Тем не менее, разумные обобщения допустимы.
- Название должно быть кратким (до 10 слов), аннотация — 2–4 предложения (до 150 слов).
- Аннотация должна отвечать на вопрос: «О чём этот документ в целом? Какие основные разделы/темы он охватывает?»

### Входные данные
<Фрагмент 1 (начало документа)>
{first_chunk}
</Фрагмент 1>

<Фрагмент 2 (сразу после первого)>
{second_chunk}
</Фрагмент 2>

<Случайный фрагмент A>
{random_chunk_A}
</Случайный фрагмент A>

<Случайный фрагмент B>
{random_chunk_B}
</Случайный фрагмент B>

<Случайный фрагмент C>
{random_chunk_C}
</Случайный фрагмент C>

### Инструкция
1. Проанализируйте все пять фрагментов.
2. Сформулируйте название документа.
3. Напишите аннотацию (утвердительно, без фраз «на основе фрагментов»).
4. Выведите результат строго в JSON по образцу.

### Формат вывода (только JSON, без лишнего текста)
{
    "title": "Название документа",
    "annotation": "Аннотация документа."
}

Теперь выполните задание.
'''

PROMPT_TEMPLATE_FOR_SHORT_DOCUMENT: str = '''Ваша задача — на основе **полного текста** документа создать его **название** и **аннотацию** в формате JSON. Эти метаданные будут использоваться для того, чтобы в будущем название и аннотация могли подаваться во входной контекст LLM вместе с релевантными чанками этого документа.

### Важные ограничения
- Документ полностью находится в вашем контексте. Вы видите его целиком.
- **Не выдумывайте** факты, которых нет в документе. Обобщайте только то, что реально присутствует.
- Название должно быть кратким (до 10 слов), аннотация — 2–4 предложения (до 150 слов).
- Аннотация должна отвечать на вопрос: «О чём этот документ в целом? Какие основные разделы/темы он охватывает?»

### Входные данные
<Полный текст документа>
{document_text}
</Полный текст документа>

### Инструкция
1. Внимательно прочитайте весь документ.
2. Сформулируйте название, отражающее главную тему и, если уместно, жанр (статья, инструкция, отчёт, руководство и т.п.).
3. Напишите аннотацию в утвердительном ключе, без фраз «в данном документе» или «документ описывает» (лучше сразу: «Рассматриваются...», «Приводится...»).
4. Выведите результат строго в JSON по образцу.

### Формат вывода (только JSON, без лишнего текста)
{
    "title": "Название документа",
    "annotation": "Аннотация документа."
}

Теперь выполните задание.
'''


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL knowledge database.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The LLM name for title and annotation generation.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output JSONL file knowledge database, enriched with document titles and annotations.')
    parser.add_argument('--url', dest='openai_base_url', type=str, required=False,
                        default='http://localhost:8000/v1', help='Your URL prefix for OpenAI API calls.')
    parser.add_argument('--key', dest='openai_api_key', type=str, required=False,
                        default='EMPTY', help='Your secret OpenAI API key.')
    parser.add_argument('--vllm', dest='use_vllm', action='store_true',
                        help='Will offline VLLM-based inference be used?')
    parser.add_argument('--max_input_len', dest='max_input_len', type=int, required=False,
                        default=4096, help='The maximum number of input tokens.')
    parser.add_argument('--gpu', dest='gpu_mem_part', type=float, required=False,
                        default=0.9, help='The GPU memory part for the vLLM-based inference.')
    parser.add_argument('--concurrency', dest='concurrency', type=int, required=False,
                        default=8, help='Number of parallel requests to the OpenAI API '
                                        '(used only without --vllm).')
    parser.add_argument('--timeout', dest='timeout', type=float, required=False,
                        default=60.0, help='Timeout (in seconds) for a single OpenAI API request '
                                           '(used only without --vllm).')
    parser.add_argument('--max_retries', dest='max_retries', type=int, required=False,
                        default=5, help='Max number of retries for transient OpenAI API errors '
                                        '(used only without --vllm).')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='Skip documents already present in the output file and continue from '
                             'there (used only without --vllm).')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        err_msg = f'The file "{input_fname}" does not exist!'
        summarization_logger.error(err_msg)
        raise IOError(err_msg)

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        output_dirname = os.path.dirname(output_fname)
        if not os.path.isdir(output_dirname):
            err_msg = f'The directory "{output_dirname}" does not exist!'
            summarization_logger.error(err_msg)
            raise IOError(err_msg)

    with codecs.open(input_fname, mode='r', encoding='utf-8') as src_fp:
        documents = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), src_fp.readlines())
            )
        ))
    summarization_logger.info(f'{len(documents)} documents are loaded from the corpus "{input_fname}".')

    instructions = []
    num_short_docs = 0
    num_long_docs = 0
    for cur_doc in documents:
        num_chunks = len(cur_doc['chunks'])
        if num_chunks > 7:
            chunk_idx = 0
            chunk_start = cur_doc['chunks'][chunk_idx]['start_char']
            chunk_end = cur_doc['chunks'][chunk_idx]['end_char']
            first_chunk = ' '.join(cur_doc['text'][chunk_start:chunk_end].strip().split()).strip()
            chunk_idx = 1
            chunk_start = cur_doc['chunks'][chunk_idx]['start_char']
            chunk_end = cur_doc['chunks'][chunk_idx]['end_char']
            second_chunk = ' '.join(cur_doc['text'][chunk_start:chunk_end].strip().split()).strip()
            indices_of_additional_chunks = [1 + (num_chunks - 2) // 3, 1 + (2 * (num_chunks - 2)) // 3, num_chunks - 1]
            three_additional_chunks = []
            for chunk_idx in indices_of_additional_chunks:
                chunk_start = cur_doc['chunks'][chunk_idx]['start_char']
                chunk_end = cur_doc['chunks'][chunk_idx]['end_char']
                three_additional_chunks.append(' '.join(cur_doc['text'][chunk_start:chunk_end].strip().split()).strip())
            user_prompt = PROMPT_TEMPLATE_FOR_LONG_DOCUMENT.replace('{first_chunk}', first_chunk)
            user_prompt = user_prompt.replace('{second_chunk}', second_chunk)
            user_prompt = user_prompt.replace('{random_chunk_A}', three_additional_chunks[0])
            user_prompt = user_prompt.replace('{random_chunk_B}', three_additional_chunks[1])
            user_prompt = user_prompt.replace('{random_chunk_C}', three_additional_chunks[2])
            del chunk_idx, chunk_start, chunk_end
            del indices_of_additional_chunks, three_additional_chunks
            num_long_docs += 1
        else:
            user_prompt = PROMPT_TEMPLATE_FOR_SHORT_DOCUMENT.replace('{document_text}', cur_doc['text'].strip())
            num_short_docs += 1
        instructions.append([
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ])
        del user_prompt, num_chunks
    info_msg = (f'{len(instructions)} input instructions are prepared. '
                f'There are {num_short_docs} instructions for short documents and '
                f'{num_long_docs} ones for long documents.')
    summarization_logger.info(info_msg)

    num_parsing_errors = 0
    if args.use_vllm:
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            llm_config = AutoConfig.from_pretrained(args.model_name)
        except Exception as err:
            summarization_logger.error(str(err))
            raise
        if llm_tokenizer.padding_side != 'left':
            llm_tokenizer.padding_side = 'left'

        json_schema = DocumentDescription.model_json_schema()
        structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
        try:
            sampling_params_json = SamplingParams(
                structured_outputs=structured_outputs_params_json,
                temperature=0.0,
                max_tokens=MAX_NEW_TOKENS,
                seed=RANDOM_SEED,
                skip_special_tokens=True
            )
        except Exception as err:
            summarization_logger.warning(str(err))
            sampling_params_json = SamplingParams(
                guided_decoding=structured_outputs_params_json,
                temperature=0.0,
                max_tokens=MAX_NEW_TOKENS,
                seed=RANDOM_SEED,
                skip_special_tokens=True
            )
        max_model_len = min(args.max_input_len + MAX_NEW_TOKENS, llm_config.max_position_embeddings - 15)
        info_msg = (f'The LLM configuration is loaded from the "{args.model_name}". '
                    f'The LLM context length is {max_model_len}.')
        summarization_logger.info(info_msg)

        try:
            large_language_model = LLM(
                model=args.model_name,
                gpu_memory_utilization=args.gpu_mem_part,
                max_model_len=max_model_len,
                max_num_batched_tokens=max(16_384, max_model_len),
                enable_chunked_prefill=True,
                seed=RANDOM_SEED,
            )
        except Exception as err:
            summarization_logger.error(str(err))
            raise
        summarization_logger.info(f'The LLM weights are loaded from {args.model_name}')

        try:
            input_prompts = []
            lengths_of_input_prompts = []
            for cur_msg in tqdm(instructions, desc='Input prompts preparation'):
                new_input_prompt = llm_tokenizer.apply_chat_template(
                    cur_msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
                input_prompts.append(new_input_prompt)
                lengths_of_input_prompts.append(len(llm_tokenizer.tokenize(new_input_prompt)))
                del new_input_prompt
            lengths_of_input_prompts.sort()
            info_msg = (f'The minimum length of input prompt is {lengths_of_input_prompts[0]}, '
                        f'the maximum one is {lengths_of_input_prompts[-1]}, the median one is '
                        f'{lengths_of_input_prompts[(len(lengths_of_input_prompts) - 1) // 2]}, '
                        f'and the mean one is '
                        f'{round(sum(lengths_of_input_prompts) / len(lengths_of_input_prompts))}.')
            summarization_logger.info(info_msg)
            del lengths_of_input_prompts
            responses = large_language_model.generate(
                prompts=input_prompts,
                sampling_params=sampling_params_json,
                use_tqdm=True
            )
            del input_prompts
        finally:
            del large_language_model
            finalize_vllm()

        for doc_idx in range(len(instructions)):
            predicted_metadata = None
            try:
                predicted_metadata = DocumentDescription.model_validate_json(responses[doc_idx].outputs[0].text)
                doc_title = ' '.join(predicted_metadata.title.strip().split()).strip()
                doc_annotation = ' '.join(predicted_metadata.annotation.strip().split()).strip()
            except Exception as err:
                summarization_logger.warning(str(err))
                doc_title = ''
                doc_annotation = ''
                num_parsing_errors += 1
            documents[doc_idx]['doc_title'] = doc_title
            documents[doc_idx]['doc_annotation'] = doc_annotation
            documents[doc_idx]['doc_full_text'] = documents[doc_idx]['text']
            del documents[doc_idx]['text']
            if predicted_metadata is not None:
                del predicted_metadata
            del doc_title, doc_annotation

        with codecs.open(output_fname, mode='w', encoding='utf-8') as dst_fp:
            for cur_doc in documents:
                dst_fp.write(json.dumps(obj=cur_doc, ensure_ascii=False) + '\n')
    else:
        try:
            client = openai.OpenAI(
                api_key=args.openai_api_key,
                base_url=args.openai_base_url
            )
        except Exception as err:
            summarization_logger.error(str(err))
            raise
        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'document-description',
                'schema': DocumentDescription.model_json_schema()
            },
        }
        summarization_logger.info(f'OpenAI client with url={args.openai_base_url} is created.')

        num_already_done = 0
        if args.resume and os.path.isfile(output_fname):
            kept_lines = []
            with codecs.open(output_fname, mode='r', encoding='utf-8') as fp:
                existing_lines = [ln for ln in fp.read().split('\n')]
            for ln in existing_lines:
                s = ln.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    break
                if ('doc_title' not in obj) or ('doc_annotation' not in obj):
                    break
                kept_lines.append(s)
            num_already_done = len(kept_lines)
            non_empty_existing = sum(1 for ln in existing_lines if ln.strip())
            if num_already_done != non_empty_existing:
                with codecs.open(output_fname, mode='w', encoding='utf-8') as fp:
                    for s in kept_lines:
                        fp.write(s + '\n')
            if num_already_done > 0:
                summarization_logger.info(
                    f'Resume is enabled. {num_already_done} already-processed documents '
                    f'are found in the "{output_fname}" and will be skipped.'
                )

        write_mode = 'a' if num_already_done > 0 else 'w'

        def process_one(doc_idx, cur_msg):
            response = None
            for attempt in range(args.max_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=cur_msg,
                        response_format=response_format,
                        temperature=0.0,
                        max_tokens=MAX_NEW_TOKENS,
                        timeout=args.timeout,
                    )
                    break
                except (openai.RateLimitError, openai.APITimeoutError,
                        openai.APIConnectionError, openai.InternalServerError) as err:
                    if attempt >= args.max_retries:
                        summarization_logger.error(
                            f'doc_idx={doc_idx} failed after {attempt + 1} attempts: {err}'
                        )
                        response = None
                        break
                    sleep_time = min(60.0, (2.0 ** attempt) + random.random())
                    time.sleep(sleep_time)
            out_rec = dict(documents[doc_idx])
            is_error = False
            if response is None:
                out_rec['doc_title'] = ''
                out_rec['doc_annotation'] = ''
                is_error = True
            else:
                try:
                    predicted_metadata = DocumentDescription.model_validate_json(
                        response.choices[0].message.content
                    )
                    out_rec['doc_title'] = ' '.join(predicted_metadata.title.strip().split()).strip()
                    out_rec['doc_annotation'] = ' '.join(predicted_metadata.annotation.strip().split()).strip()
                except Exception as err:
                    summarization_logger.warning(f'doc_idx={doc_idx} JSON parsing error: {err}')
                    out_rec['doc_title'] = ''
                    out_rec['doc_annotation'] = ''
                    is_error = True
            out_rec['doc_full_text'] = out_rec.pop('text')
            return doc_idx, out_rec, is_error

        next_to_write = num_already_done
        results_buffer = {}
        write_lock = threading.Lock()

        def try_flush(dst_fp):
            nonlocal next_to_write
            with write_lock:
                while next_to_write in results_buffer:
                    rec = results_buffer.pop(next_to_write)
                    dst_fp.write(json.dumps(obj=rec, ensure_ascii=False) + '\n')
                    dst_fp.flush()
                    next_to_write += 1

        work_items = list(enumerate(instructions))[num_already_done:]
        total_to_do = len(work_items)
        summarization_logger.info(
            f'{total_to_do} documents to process with concurrency={args.concurrency}, '
            f'timeout={args.timeout}s, max_retries={args.max_retries}.'
        )

        with codecs.open(output_fname, mode=write_mode, encoding='utf-8') as dst_fp:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(process_one, doc_idx, cur_msg): doc_idx
                    for doc_idx, cur_msg in work_items
                }
                with tqdm(total=total_to_do, desc='LLM summarization') as pbar:
                    for fut in as_completed(futures):
                        doc_idx, out_rec, is_error = fut.result()
                        with write_lock:
                            results_buffer[doc_idx] = out_rec
                        if is_error:
                            num_parsing_errors += 1
                        try_flush(dst_fp)
                        pbar.update(1)

    if num_parsing_errors > 0:
        info_msg = (f'Titles and annotations are generated for {len(instructions) - num_parsing_errors} '
                    f'documents, and predictions for {num_parsing_errors} documents are processed '
                    f'with JSON parsing errors.')
    else:
        info_msg = f'Titles and annotations are successfully generated for {len(instructions)} documents.'
    summarization_logger.info(info_msg)
    summarization_logger.info(f'The enriched documents are saved into the "{output_fname}".')


if __name__ == '__main__':
    summarization_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    summarization_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('document_summarization.log')
    file_handler.setFormatter(formatter)
    summarization_logger.addHandler(file_handler)
    main()
