import re
from typing import Optional

from nltk import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer
from razdel import tokenize


def load_hf_tokenizer(model_name: str):
    from transformers import AutoTokenizer, __version__
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    options = {}
    if int(__version__.split('.')[0]) < 5:
        config = get_tokenizer_config(model_name)
        extra = config.get('extra_special_tokens')
        if isinstance(extra, list):
            # Transformers 4 expects named multimodal tokens, while 5 also writes lists.
            options['extra_special_tokens'] = {}
            options['additional_special_tokens'] = list(dict.fromkeys(
                [*config.get('additional_special_tokens', []), *extra]
            ))
    return AutoTokenizer.from_pretrained(model_name, **options)


def tokenize_and_normalize_text(s: str, stemmer: Optional[SnowballStemmer] = None) -> str:
    re_for_digit = re.compile(r'\d+')
    new_tokens = []
    for cur in tokenize(s):
        if re_for_digit.search(cur.text) is None:
            new_tokens += list(filter(
                lambda x2: (len(x2) > 0) and x2.isalpha(),
                map(lambda x1: x1.strip().lower(), wordpunct_tokenize(cur.text))
            ))
        else:
            token_text = cur.text.lower().strip()
            if len(token_text) > 0:
                new_tokens.append(token_text)
    if stemmer is None:
        stemmed_tokens = new_tokens
    else:
        stemmed_tokens = list(filter(
            lambda it2: len(it2) > 0,
            map(lambda it1: stemmer.stem(it1).strip(), new_tokens)
        ))
    del new_tokens
    if len(stemmed_tokens) == 0:
        return ''
    return ' '.join(stemmed_tokens)
