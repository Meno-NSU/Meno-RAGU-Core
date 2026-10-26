import gc
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import warnings

from nltk.stem.snowball import SnowballStemmer
from openai import OpenAI
from vllm import LLM, SamplingParams

try:
    from context_utils.context_utils import prepare_context
    from rewriting_utils.rewriting_utils import find_candidates_to_abbreviations
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from context_utils.context_utils import prepare_context
    from rewriting_utils.rewriting_utils import find_candidates_to_abbreviations


SYSTEM_PROMPT_FOR_RELEVANCE = '''Вы — компонент RAG-системы (Retrieve Augmented Generation), который оценивает, насколько документ пригоден как фактическое основание для ответа на вопрос пользователя.

Вам подаётся:
- актуализированный вопрос пользователя (уже с разрешёнными местоимениями и восстановленным контекстом диалога);
- один документ-кандидат (название, аннотация, фрагменты);
- словарь возможных расшифровок аббревиатур (ненадёжный, вспомогательный).

Ваша задача — оценить НЕ тематическую близость, а ПРИГОДНОСТЬ документа как источника фактов для ответа. Представьте, что отвечающая модель будет использовать ТОЛЬКО этот документ и не имеет права использовать свои собственные знания. Вопрос: сможет ли она извлечь отсюда фактическое содержание для ответа?

## Классы пригодности

- **2 — документ содержит прямое фактическое основание для ответа.** В тексте (названии, аннотации или фрагментах) присутствуют конкретные факты, утверждения, данные или формулировки, которые прямо отвечают на вопрос или на его существенную часть. Отвечающая модель может построить корректный, обоснованный ответ, опираясь только на этот документ.

- **1 — документ содержит частичное или косвенное основание.** Текст затрагивает предмет вопроса и даёт вспомогательные сведения (контекст, смежные факты, упоминания), но прямого и достаточного ответа в нём нет. Из документа можно извлечь полезный, но неполный материал; для полного ответа потребуются другие источники.

- **0 — документ не пригоден как источник ответа.** В тексте нет фактов, которые помогают ответить на вопрос. Документ может быть про ту же общую тему, упоминать те же служебные слова или ту же организацию, но конкретного отвечающего содержания в нём нет. Сюда же относятся документы, состоящие из навигационных элементов, оглавлений, расписаний, списков ссылок и т. п., если они не содержат искомого факта.

## Принципиально важное различие

Тематическая близость — это НЕ то же самое, что пригодность для ответа.
- Документ может быть целиком посвящён нужной персоне/проекту/теме, но НЕ содержать ответа на конкретный вопрос → это класс 0 или 1, а не 2.
- Короткий фрагмент, не «про тему вообще», но содержащий ровно нужный факт → это класс 2.
Оценивайте наличие ОТВЕЧАЮЩЕГО ФАКТА, а не совпадение темы.

## Работа с именованными сущностями

При сопоставлении людей, организаций, проектов, мест действуйте по принципу "специфичность запроса против специфичности документа":

1. **Фамилия — главный идентификатор человека.** Совпадение фамилии важнее совпадения имени или отчества. Совпадение только имени и/или отчества без совпадения фамилии НЕ свидетельствует о том, что речь об одном человеке.

2. **Недоспецификация — это НЕ противоречие.** Если одна из сторон указывает сущность менее подробно, чем другая, но не вступает с ней в конфликт, они могут описывать один и тот же объект.
   - Запрос «Александр Люлько» (без отчества) совместим и с «Александром Николаевичем Люлько», и с «Александром Михайловичем Люлько»: запрос просто не различает их, поэтому оба документа потенциально подходят.
   - Запрос «Александр Николаевич Люлько» совместим с документом, где упомянута только фамилия «Люлько» без имени и отчества: документ не противоречит запросу, речь может идти о том же человеке.

3. **Конфликт атрибутов — это запрет.** Если запрос специфицирует атрибут (отчество, имя, инициал, название), а документ называет ДРУГОЕ, НЕСОВМЕСТИМОЕ значение того же атрибута, то речь о разных объектах. Такой документ нерелевантен ни по какому критерию — это класс 0, даже при совпадении остальных частей имени.
   - Запрос «Александр Николаевич Люлько» и документ «Александр Михайлович Люлько» — это РАЗНЫЕ люди (конфликт по отчеству): класс 0, несмотря на совпадение имени и фамилии.

4. **Те же принципы для организаций, проектов, продуктов.** Главный идентификатор — собственное название. Совпадение общих слов («факультет», «нейросеть», «модель») без совпадения собственного названия не делает документ пригодным. Если запрос называет «факультет информационных технологий», а документ — «механико-математический факультет», это разные факультеты (конфликт): класс 0. Если же запрос не конкретизирует («какие факультеты есть в университете?»), то документ про любой факультет может подходить.

5. **Совпадение сущности — необходимое, но не достаточное условие.** Даже когда сущности совпадают (или совместимы), документ относится к классу 2 ТОЛЬКО если он содержит факт, отвечающий на сам вопрос, а не просто упоминает сущность.

## Работа со словарём аббревиатур

- Используйте словарь только чтобы понять, что аббревиатура из вопроса и аббревиатура из документа обозначают одно и то же.
- Словарь ненадёжен: не считайте факт пригодным только потому, что в словаре есть запись.
- Если запись словаря не согласуется с контекстом документа — игнорируйте её.

## Жёсткие ограничения

- Опирайтесь ИСКЛЮЧИТЕЛЬНО на текст документа. НЕ используйте свои фоновые знания о мире, об НГУ, о людях или проектах. Если факт не написан в документе — его для вас не существует.
- Не додумывайте и не достраивайте недостающие факты.
- Оценивайте по совокупности названия, аннотации и всех фрагментов.

## Формат ответа

Ответьте строго одним символом: 2, 1 или 0. Без пояснений и любого другого текста.
'''
MAX_RERANKER_TOKENS: int = 8192


def build_prompt(user_question: str, dialogue_history: str, estimated_document: str,
                 abbr_dict: Dict[str, Dict[str, Union[str, List[str]]]],
                 stemmer: Optional[SnowballStemmer] = None, is_json: bool=False) -> List[Dict[str, str]]:
    if len(estimated_document) == 0:
        raise ValueError('The estimated document is empty.')
    found_abbreviations = find_candidates_to_abbreviations(
        source_text=estimated_document + '\n\n' + dialogue_history + '\n\n' + user_question,
        all_abbreviations=abbr_dict,
        stemmer=stemmer
    )
    user_prompt = ''
    if len(found_abbreviations) > 0:
        user_prompt += '==========\nABBREVIATION DICTIONARY\n==========\n\n'
        user_prompt += ('**Природа данных:** предположительные расшифровки, подобранные автоматически'
                        ' по совпадению буквенной формы. Записи могут быть лишними, ошибочными или '
                        'неоднозначными (омонимия). Используйте словарь как подсказку, а не как '
                        'достоверный источник.\n\n')
        user_prompt += ('**Записи:**\n' + found_abbreviations)
        user_prompt += '\n\n'
    user_prompt += (estimated_document + '\n\n')
    if len(dialogue_history) > 0:
        user_prompt += '==========\nDIALOGUE HISTORY\n==========\n\n'
        user_prompt += dialogue_history
        user_prompt += '\n\n'
    user_prompt += '==========\nCURRENT QUESTION\n==========\n\n'
    user_prompt += (user_question + '\n\n')
    user_prompt += '==========\nINSTRUCTION\n==========\n\n'
    user_prompt += 'Оцените, пожалуйста, возможность ответа на вопрос из раздела CURRENT QUESTION'
    user_prompt += ', опираясь только на факты из разделов DOCUMENT 1.'
    if len(dialogue_history) > 0:
        user_prompt += ('\n\nРаздел DIALOGUE HISTORY используйте исключительно для разрешения местоимений, '
                        'уточнений и восстановления темы — не извлекайте из него фактические сведения.')
    if len(found_abbreviations) > 0:
        user_prompt += '\n\nРаздел ABBREVIATION DICTIONARY используйте по следующим правилам:\n'
        user_prompt += ('1. Применяйте расшифровку из словаря, только если сокращение действительно встречается '
                        'в CURRENT QUESTION')
        if len(dialogue_history) > 0:
            user_prompt += ', DIALOGUE HISTORY или DOCUMENT 1'
        else:
            user_prompt += ' или DOCUMENT 1'
        user_prompt += ' и по контексту является аббревиатурой.'
        user_prompt += ('2. Если одной аббревиатуре соответствует несколько расшифровок (омонимия), '
                        'выберите ту, которая согласуется с темой диалога и содержанием документов.\n')
        user_prompt += ('3. Если в тексте встречается буквосочетание, совпадающее с ключом словаря, '
                        'но по контексту аббревиатурой не является (например, это команда, междометие, '
                        'имя собственное, часть слова), — игнорируйте запись словаря.\n')
        user_prompt += ('4. Если подходящая расшифровка в словаре отсутствует, а в документах её тоже нет, — '
                        'не придумывайте расшифровку; используйте сокращение как есть.\n')
        user_prompt += ('5. Словарь не является источником фактов: сам факт наличия записи «X — Y» не доказывает, '
                        'что Y упомянут в документах или релевантен вопросу.')
    if is_json:
        modified_chapter_name = '## Формат ответа'
        found_idx = SYSTEM_PROMPT_FOR_RELEVANCE.find(modified_chapter_name)
        if found_idx < 0:
            raise RuntimeError(f'The system prompt does not contain the chapter `{modified_chapter_name}`.')
        modified_system_prompt = SYSTEM_PROMPT_FOR_RELEVANCE[:found_idx] + modified_chapter_name + '\n\n'
        modified_system_prompt += ('Ответьте в формате JSON вида `{"label": <your_label>}`, где '
                                   '`<your_label>` — это цифра 2, 1 или 0. Не добавляйте никаких пояснений, '
                                   'комментариев или дополнительного текста.\n')
        messages = [
            {'role': 'system', 'content': modified_system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    else:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT_FOR_RELEVANCE},
            {'role': 'user', 'content': user_prompt}
        ]
    del user_prompt
    return messages


def aggregate_logprobs_to_score(label_logprobs: Dict[int, float]) -> float:
    if set(label_logprobs.keys()) != {0, 1, 2}:
        raise ValueError(f'Expected labels {{0, 1, 2}}, got {set(label_logprobs.keys())}.')
    max_lp = max(label_logprobs.values())
    exp_values = {label: math.exp(lp - max_lp) for label, lp in label_logprobs.items()}
    total = sum(exp_values.values())
    probs = {label: exp_val / total for label, exp_val in exp_values.items()}
    expected_label = sum(label * probs[label] for label in probs)
    return expected_label / 2.0


def messages_to_log(inputs: List[Dict[str, str]], reference: str) -> Dict[str, Union[str, List[List[str]]]]:
    sample = dict()
    if len(inputs) == 0:
        raise ValueError('The inputs are empty!')
    if inputs[0]['role'] == 'system':
        sample['system'] = inputs[0]['content']
        start_idx = 1
    else:
        start_idx = 0
    if start_idx >= len(inputs):
        raise ValueError('The inputs do not contain a user message!')
    msg_history = []
    while start_idx < (len(inputs) - 2):
        if inputs[start_idx]['role'] != 'user':
            raise ValueError(f'The inputs are incorrect!\n{inputs}')
        if inputs[start_idx + 1]['role'] != 'assistant':
            raise ValueError(f'The inputs are incorrect!\n{inputs}')
        msg_history.append([inputs[start_idx]['content'], inputs[start_idx + 1]['content']])
        start_idx += 2
    if inputs[start_idx]['role'] != 'user':
        raise ValueError(f'The inputs are incorrect!\n{inputs}')
    sample['query'] = inputs[start_idx]['content']
    sample['response'] = reference
    sample['history'] = msg_history
    return sample


def rerank_documents_with_api(user_question: str, global_indices_of_relevant_chunks: List[int],
                              documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                              chunk_mapping: Dict[str, Dict[str, int]], client: OpenAI, llm_name: str,
                              dialogue_history: str, abbr_dict: Dict[str, Dict[str, Union[str, List[str]]]],
                              stemmer: Optional[SnowballStemmer] = None,
                              extra_document_texts: Optional[List[str]] = None,
                              log_for_distillation: bool = False) -> Union[List[float], Tuple[List[float], List[Dict[str, Union[str, List[List[str]]]]]]]:
    scores = []
    possible_labels = ['0', '1', '2']
    samples_for_log = []
    document_texts = []
    for cur_chunk_idx in global_indices_of_relevant_chunks:
        cur_doc = prepare_context(
            indices_of_relevant_chunks=[cur_chunk_idx],
            scores_of_relevant_chunks=[1.0],
            documents=documents,
            chunk_mapping=chunk_mapping,
            min_document_quality=0.0
        )[0][0]
        document_texts.append(cur_doc)
    if extra_document_texts is not None:
        document_texts.extend(extra_document_texts)
    for cur_doc in document_texts:
        new_prompt = build_prompt(user_question, dialogue_history, cur_doc, abbr_dict, stemmer)
        try:
            response = client.chat.completions.create(
                model=llm_name,
                messages=new_prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
                extra_body={'guided_choice': possible_labels}
            )
        except Exception as err:
            response = None
            warnings.warn(str(err))

        if response is None:
            ok = False
        else:
            if len(response.choices) < 1:
                ok = False
            else:
                ok = (response.choices[0].logprobs is not None)
        if ok:
            if log_for_distillation:
                samples_for_log.append(messages_to_log(new_prompt, response.choices[0].message.content.strip()))
            choice = response.choices[0]
            top_logprobs_list = choice.logprobs.content[0].top_logprobs

            token_to_logprob = {
                item.token: item.logprob
                for item in top_logprobs_list
            }

            raw = {}
            for label in possible_labels:
                raw[int(label)] = token_to_logprob.get(label, -100.0)

            scores.append(aggregate_logprobs_to_score(raw))
            del raw, token_to_logprob
        else:
            del new_prompt
            new_prompt = build_prompt(user_question, dialogue_history, cur_doc, abbr_dict, stemmer, is_json=True)
            try:
                response = client.chat.completions.create(
                    model=llm_name,
                    messages=new_prompt,
                    max_tokens=512,
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "relevance",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "enum": ["0", "1", "2"]
                                    }
                                },
                                "required": ["label"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }
                    }
                )
            except Exception as err:
                response = None
                warnings.warn(str(err))
            if response is None:
                scores.append(0.0)
            else:
                output_text = response.choices[0].message.content.strip()
                if log_for_distillation:
                    samples_for_log.append(messages_to_log(new_prompt, output_text))
                try:
                    predicted_label = int(json.loads(output_text)['label'])
                    if predicted_label not in {0, 1, 2}:
                        raise ValueError(f'Unexpected label: {predicted_label}')
                except:
                    warnings.warn(f'The output text is bad!\n{output_text}')
                    raise
                one_hot_logprobs = {label: (0.0 if label == predicted_label else -100.0)
                                    for label in (0, 1, 2)}
                scores.append(aggregate_logprobs_to_score(one_hot_logprobs))
                del predicted_label, one_hot_logprobs
        del new_prompt
    if log_for_distillation:
        return scores, samples_for_log
    return scores


def rerank_documents_with_vllm(user_question: str, global_indices_of_relevant_chunks: List[int],
                               documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                               chunk_mapping: Dict[str, Dict[str, int]], llm: LLM,
                               dialogue_history: str, abbr_dict: Dict[str, Dict[str, Union[str, List[str]]]],
                               stemmer: Optional[SnowballStemmer] = None,
                               extra_document_texts: Optional[List[str]] = None) -> List[float]:
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=20,
    )
    tokenizer = llm.get_tokenizer()
    possible_tokens = [(tokenizer.encode(str(lbl), add_special_tokens=False)[0], lbl) for lbl in [0, 1, 2]]

    document_texts = []
    for cur_chunk_idx in global_indices_of_relevant_chunks:
        cur_doc = prepare_context(
            indices_of_relevant_chunks=[cur_chunk_idx],
            scores_of_relevant_chunks=[1.0],
            documents=documents,
            chunk_mapping=chunk_mapping,
            min_document_quality=0.0
        )[0][0]
        document_texts.append(cur_doc)
    if extra_document_texts is not None:
        document_texts.extend(extra_document_texts)
    prompts = []
    for cur_doc in document_texts:
        prompts.append(tokenizer.apply_chat_template(
            build_prompt(user_question, dialogue_history, cur_doc, abbr_dict, stemmer),
            tokenize=False,
            add_generation_prompt=True
        ))
        del cur_doc
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    del prompts, tokenizer, document_texts

    scores = []
    for cur_output in outputs:
        token_logprobs = cur_output.outputs[0].logprobs[0]

        raw_logprobs = {}
        for token_id, label in possible_tokens:
            if token_id in token_logprobs:
                raw_logprobs[label] = token_logprobs[token_id].logprob
            else:
                raw_logprobs[label] = -100.0

        scores.append(aggregate_logprobs_to_score(raw_logprobs))

    del sampling_params, possible_tokens
    gc.collect()
    return scores
