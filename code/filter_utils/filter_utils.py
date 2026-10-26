from typing import List, Optional

from nltk import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer


MIN_NUM_WORDS: int = 10

BAD_PHRASES: List[str] = [
    '402 please renew your subscription',
    'страница не найдена',
    'page not found',
    'access denied',
    'forbidden',
    'connection timed out',
    'name resolution failed',
]


def text_is_good(s: str) -> bool:
    normalized = ' '.join(s.strip().lower().split()).strip()
    if len(normalized) == 0:
        return False
    is_good = True
    for cur_bad_phrase in BAD_PHRASES:
        if normalized.find(cur_bad_phrase) >= 0:
            is_good = False
            break
    return is_good


def count_words(s: str) -> int:
    words = list(filter(
        lambda it2: (len(it2) > 2) and it2.isalpha(),
        map(
            lambda it1: it1.strip(),
            wordpunct_tokenize(s.strip().lower())
        )
    ))
    return len(words)


def normalize_text(s: str, stemmer: Optional[SnowballStemmer] = None) -> str:
    if stemmer is not None:
        words = list(filter(
            lambda it4: len(it4) > 0,
            map(
                lambda it3: stemmer.stem(it3),
                filter(
                    lambda it2: (len(it2) > 2) and it2.isalpha(),
                    map(
                        lambda it1: it1.strip(),
                        wordpunct_tokenize(s.strip().lower())
                    )
                )
            )
        ))
    else:
        words = list(filter(
            lambda it2: len(it2) > 0,
            map(
                lambda it1: it1.strip(),
                filter(
                    lambda it0: (len(it0) > 2) and it0.isalpha(),
                    wordpunct_tokenize(s.strip().lower())
                )
            )
        ))
    if len(words) == 0:
        return ''
    return ' '.join(words)
