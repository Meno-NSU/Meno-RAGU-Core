from typing import Callable, Dict, List, Optional, Tuple

from razdel import tokenize


def tokenize_russian_text(source_text: str) -> List[Tuple[int, int]]:
    if len(source_text.strip()) == 0:
        return []
    tokens = list(tokenize(source_text))
    if len(tokens) == 0:
        return []
    token_boundaries = []
    for cur in tokens:
        token_boundaries.append((cur.start, cur.stop))
    del tokens
    return token_boundaries


def crop_long_text(src: str, tokenize_fn: Callable[[str], List[Tuple[int, int]]], max_words: int) -> str:
    token_boundaries = tokenize_fn(src)
    if len(token_boundaries) <= max_words:
        return src.strip()
    return src[:token_boundaries[max_words - 1][1]].strip() + ' [...]'


def prepare_dialogue_history(source_dialogue: List[Dict[str, str]],
                             tokenize_fn: Optional[Callable[[str], List[Tuple[int, int]]]] = tokenize_russian_text,
                             max_words: Optional[int] = None) -> str:
    if not isinstance(source_dialogue, list):
        raise ValueError(f'The source dialogue is wrong! Expected {type(["1", "2"])}, got {type(source_dialogue)}.')
    if (max_words is not None) and (tokenize_fn is None):
        raise ValueError('The tokenization function is not specified!')
    if max_words is not None:
        if max_words < 5:
            raise ValueError(f'The maximum number of words is wrong! Expected 5 or greater, got {max_words}.')
    if len(source_dialogue) == 0:
        return ''
    possible_roles = {'user', 'assistant'}
    expected_role = 'user'
    for idx, val in enumerate(source_dialogue):
        if not isinstance(val, dict):
            err_msg = f'The source dialogue item {idx} is wrong! Expected {type({"a": "b"})}, got {type(val)}.'
            raise ValueError(err_msg)
        if 'role' not in val:
            err_msg = f'The source dialogue item {idx} is wrong! The field "role" is not found.'
            raise ValueError(err_msg)
        if 'content' not in val:
            err_msg = f'The source dialogue item {idx} is wrong! The field "content" is not found.'
            raise ValueError(err_msg)
        current_role = val['role']
        if not isinstance(current_role, str):
            err_msg = (f'The role of the source dialogue item {idx} is wrong! '
                       f'Expected {type({"a"})}, got {type(current_role)}.')
            raise ValueError(err_msg)
        current_content = val['content']
        if not isinstance(current_content, str):
            err_msg = (f'The content of the source dialogue item {idx} is wrong! '
                       f'Expected {type({"a"})}, got {type(current_content)}.')
            raise ValueError(err_msg)
        if current_role not in possible_roles:
            possible_roles_list = list(possible_roles)
            err_msg = (f'The role of the source dialogue item {idx} is unknown! '
                       f'Expected {", ".join(possible_roles_list[0:-1])} or {possible_roles_list[-1]}, '
                       f'got {current_role}.')
            raise ValueError(err_msg)
        if current_role != expected_role:
            err_msg = (f'The role of the source dialogue item {idx} is wrong! '
                       f'Expected {expected_role}, got {current_role}.')
            raise ValueError(err_msg)
        expected_role = ('assistant' if (expected_role == 'user') else 'user')
    if (len(source_dialogue) % 2) != 0:
        err_msg = 'The last item does not relate to the dialogue history, but describes the user\'s current question.'
        raise ValueError(err_msg)
    user_question = ' '.join(source_dialogue[0]['content'].split()).strip()
    assistant_answer = ' '.join(source_dialogue[1]['content'].split()).strip()
    if max_words is None:
        cropped_answer = assistant_answer
    else:
        cropped_answer = crop_long_text(assistant_answer, tokenize_fn, max_words)
    textualized_dialogue = '----------\nTURN 1\n----------\n**Реплика пользователя:** ' + user_question
    if cropped_answer == assistant_answer:
        textualized_dialogue += '\n**Ответ ассистента:** ' + assistant_answer + '\n'
    else:
        textualized_dialogue += '\n**Ответ ассистента (сокр.):** ' + cropped_answer + '\n'
    num_turns = len(source_dialogue) // 2
    for turn_idx in range(1, num_turns):
        user_question = ' '.join(source_dialogue[turn_idx * 2]['content'].split()).strip()
        assistant_answer = ' '.join(source_dialogue[turn_idx * 2 + 1]['content'].split()).strip()
        if max_words is None:
            cropped_answer = assistant_answer
        else:
            cropped_answer = crop_long_text(assistant_answer, tokenize_fn, max_words)
        textualized_dialogue += f'\n----------\nTURN {turn_idx + 1}\n----------\n**Реплика пользователя:** '
        textualized_dialogue += user_question
        if cropped_answer == assistant_answer:
            textualized_dialogue += '\n**Ответ ассистента:** ' + assistant_answer + '\n'
        else:
            textualized_dialogue += '\n**Ответ ассистента (сокр.):** ' + cropped_answer + '\n'
    return textualized_dialogue
