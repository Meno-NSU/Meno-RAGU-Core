from typing import Dict, List

import openai
from tqdm import tqdm
from vllm import LLM, SamplingParams


def generate_responses_with_api(inputs: List[List[Dict[str, str]]], client: openai.OpenAI, model_name: str,
                                max_new_tokens: int, verbose: bool = False) -> List[str]:
    outputs = []
    for cur_input in (tqdm(inputs) if verbose else inputs):
        response = client.chat.completions.create(
            model=model_name,
            messages=cur_input,
            max_tokens=max_new_tokens
        )
        outputs.append(response.choices[0].message.content.strip())
        del response
    return outputs


def generate_responses_with_vllm(inputs: List[List[Dict[str, str]]], llm: LLM, params: SamplingParams,
                                 verbose: bool = False) -> List[str]:
    tokenizer = llm.get_tokenizer()
    textualized_inputs = []
    for cur_input in inputs:
        new_input_prompt = tokenizer.apply_chat_template(
            cur_input,
            tokenize=False,
            add_generation_prompt=True
        )
        textualized_inputs.append(new_input_prompt)
    responses = llm.generate(
        prompts=textualized_inputs,
        sampling_params=params,
        use_tqdm=verbose
    )
    del textualized_inputs, tokenizer
    return [it.outputs[0].text.strip() for it in responses]
