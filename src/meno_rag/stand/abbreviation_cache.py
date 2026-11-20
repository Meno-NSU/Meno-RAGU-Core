"""Bounded token-only memoization for one loaded corpus and its fixed stemmer.

No abbreviation dictionary or formatted result is cached: callers always read its
current keys and explanations. Cached token sets contain no mutable values.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.rewriting import _format_abbreviations
from meno_rag.stand.tokenization import tokenize_and_normalize_text


@dataclass(frozen=True, slots=True)
class TokenSets:
    raw: frozenset[str]
    stemmed: frozenset[str]


class AbbreviationTokenCache:
    def __init__(self, stemmer: SnowballStemmer | None, *, max_entries: int = 2048,
                 max_bytes: int = 64 * 1024 * 1024):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("Cache limits must be nonnegative.")
        self.stemmer = stemmer
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, tuple[TokenSets, int]] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def tokens(self, text: str) -> TokenSets:
        with self._lock:
            entry = self._entries.get(text)
            if entry is not None:
                self._entries.move_to_end(text)
                self.hits += 1
                return entry[0]
            self.misses += 1
        # Compute outside the lock. Duplicate concurrent computations are pure and harmless.
        raw = frozenset(tokenize_and_normalize_text(text).split())
        stemmed = (frozenset(tokenize_and_normalize_text(text, self.stemmer).split())
                   if self.stemmer is not None else frozenset())
        tokens = TokenSets(raw, stemmed)
        cost = (sys.getsizeof(text) + sys.getsizeof(tokens) + sys.getsizeof(raw) + sys.getsizeof(stemmed)
                + sum(sys.getsizeof(token) for token in raw) + sum(sys.getsizeof(token) for token in stemmed)
                + 256)  # conservative allowance for the ordered mapping entry and bookkeeping
        if self.max_entries and cost <= self.max_bytes:
            with self._lock:
                previous = self._entries.pop(text, None)
                if previous is not None:
                    self._bytes -= previous[1]
                while self._entries and (len(self._entries) >= self.max_entries or self._bytes + cost > self.max_bytes):
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= evicted[1]
                    self.evictions += 1
                self._entries[text] = (tokens, cost)
                self._bytes += cost
        return tokens

    def find(self, document: str, history: str, question: str,
             abbreviations: dict[str, dict[str, str | list[str]]]) -> str:
        source = document + "\n\n" + history + "\n\n" + question
        tokens = self.tokens(source)
        selected = tokens.raw | tokens.stemmed
        return _format_abbreviations(selected.intersection(abbreviations), abbreviations)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                    "entries": len(self._entries), "accounted_bytes": self._bytes}
