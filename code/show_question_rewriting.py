from argparse import ArgumentParser
import json
import os
import random

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['SOCKS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['socks_proxy'] = ''
os.environ['no_proxy'] = '*'

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from nltk.stem.snowball import SnowballStemmer
import openai
from vllm import LLM, SamplingParams
from transformers import AutoConfig

from rewriting_utils.rewriting_utils import load_abbreviations, prepare_prompt_for_rewriting
from dialogue_history_utils.dialogue_history_utils import prepare_dialogue_history
from inference_utils.inference_utils import generate_responses_with_vllm, generate_responses_with_api
from vllm_utils.vllm_utils import finalize_vllm


RANDOM_SEED: int = 42
MAX_NEW_TOKENS: int = 512


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-q', '--question', dest='source_question', type=str, required=True,
                        help='The source question for rewriting.')
    parser.add_argument('-a', '--abbr', dest='abbreviation_name', type=str, required=True,
                        help='The JSON file with abbreviations and their explanations.')
    parser.add_argument('-s', '--hist', dest='dialogue_history', type=str, required=False, default=None,
                        help='The jsonified string with dialogue history in chat template.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The LLM name for title and annotation generation.')
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
    parser.add_argument('--max_words', dest='max_words', type=int, required=False,
                        default=None, help='The maximum number of words in the assistant\'s response.')
    args = parser.parse_args()

    source_question = ' '.join(args.source_question.strip().split()).strip()
    if len(source_question) == 0:
        raise ValueError('The source question for rewriting is empty!')

    if args.dialogue_history is None:
        dialogue_history = []
    else:
        dialogue_history = json.loads(args.dialogue_history)
        if not isinstance(dialogue_history, list):
            raise ValueError('The dialogue history is wrong!')
        if (len(dialogue_history) % 2) != 0:
            raise ValueError('The dialogue history is wrong!')
        for idx, val in enumerate(dialogue_history):
            if not isinstance(val, dict):
                raise ValueError(f'The dialogue history item {idx} is wrong!')
            if ('role' not in val) or ('content' not in val):
                raise ValueError(f'The dialogue history item {idx} is wrong!')
            if val['role'] not in {'user', 'assistant'}:
                raise ValueError(f'The dialogue history item {idx} is wrong!')
            if not isinstance(val['content'], str):
                raise ValueError(f'The dialogue history item {idx} is wrong!')

    abbreviation_fname = os.path.normpath(args.abbreviation_name)
    if not os.path.isfile(abbreviation_fname):
        raise IOError(f'The file "{abbreviation_fname}" does not exist!')
    abbreviations = load_abbreviations(abbreviation_fname)
    selected_for_demo = random.sample(population=list(abbreviations.keys()), k=3)
    print(f'Some examples of the abbreviation file:')
    print(json.dumps(obj=dict([(k, abbreviations[k]) for k in selected_for_demo]), ensure_ascii=False, indent=4) + '\n')

    ru_stemmer = SnowballStemmer('russian')

    prepared_history = prepare_dialogue_history(source_dialogue=dialogue_history, max_words=args.max_words)
    if len(dialogue_history) > 0:
        print('Prepared dialogue history:\n')
        print(prepared_history + '\n')

    input_messages = prepare_prompt_for_rewriting(
        source_question,
        prepared_history,
        abbreviations,
        ru_stemmer
    )
    print('Prepared prompt:\n')
    print(json.dumps(obj=input_messages, ensure_ascii=False, indent=4) + '\n')

    if args.use_vllm:
        llm_config = AutoConfig.from_pretrained(args.model_name)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS,
            seed=RANDOM_SEED,
            skip_special_tokens=True
        )
        max_model_len = min(args.max_input_len + MAX_NEW_TOKENS, llm_config.max_position_embeddings - 15)
        info_msg = (f'The LLM configuration is loaded from the "{args.model_name}". '
                    f'The LLM context length is {max_model_len}.')
        print(info_msg)

        large_language_model = LLM(
            model=args.model_name,
            gpu_memory_utilization=args.gpu_mem_part,
            max_model_len=max_model_len,
            max_num_batched_tokens=max(16_384, max_model_len),
            enable_chunked_prefill=True,
            seed=RANDOM_SEED,
        )

        try:
            rewritten_question = generate_responses_with_vllm(
                [input_messages],
                large_language_model, sampling_params
            )[0]
        finally:
            del large_language_model
            finalize_vllm()
    else:
        client = openai.OpenAI(
            api_key=args.openai_api_key,
            base_url=args.openai_base_url
        )
        rewritten_question = generate_responses_with_api(
            [input_messages],
            client,
            args.model_name,
            MAX_NEW_TOKENS
        )[0]

    print('\n')
    print(f'The source question:\n{source_question}\n')
    print(f'The rewritten question:\n{rewritten_question}')


if __name__ == '__main__':
    main()
