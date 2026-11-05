"""Relevance pre-filter: watched assets and macro topics, word-boundary matched.

Precompiled, case-insensitive regexes. A keyword only matches as a whole word, so ``sol`` does not hit
"solution", ``eth`` not "method", ``sec`` not "second". ``$BTC`` style cashtags match. Keywords of four or
more letters (plus ``etf``/``ban``) also match simple inflections ("hacked", "exploits", "ETFs").
Known limitation: matching is case-insensitive by specification, so the verb "fed" counts as FED.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_SUFFIX = r"(?:s|es|ed|d|ing|ned|ning)?"
_SUFFIX_SHORT_ALLOWED = frozenset({"etf", "ban"})
MAX_BODY_CHARS = 4000


@dataclass(frozen=True, slots=True)
class RelevanceMatch:
    symbols: tuple[str, ...]
    macro_tags: tuple[str, ...]
    relevant: bool


def _compile(words: Iterable[str]) -> re.Pattern[str] | None:
    unique = sorted({w.strip().lower() for w in words if w and w.strip()}, key=lambda w: (-len(w), w))
    if not unique:
        return None
    parts: list[str] = []
    for word in unique:
        pattern = re.escape(word).replace(r"\ ", r"\s+")
        if word.isalpha() and (len(word) >= 4 or word in _SUFFIX_SHORT_ALLOWED):
            pattern += _SUFFIX
        elif " " in word and word.split()[-1].isalpha() and len(word.split()[-1]) >= 4:
            pattern += _SUFFIX
        parts.append(pattern)
    return re.compile(r"(?<![\w$])\$?(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


class RelevanceFilter:
    def __init__(self, symbol_keywords: Mapping[str, Iterable[str]], macro_keywords: Mapping[str, Iterable[str]]) -> None:
        self._symbols: list[tuple[str, re.Pattern[str]]] = []
        for symbol, words in symbol_keywords.items():
            pattern = _compile([symbol, *words])
            if pattern is not None:
                self._symbols.append((symbol.upper(), pattern))
        self._macros: list[tuple[str, re.Pattern[str]]] = []
        for tag, words in macro_keywords.items():
            pattern = _compile([*words])
            if pattern is not None:
                self._macros.append((tag.upper(), pattern))

    def match(self, title: str | None, body: str | None = None) -> RelevanceMatch:
        text = f"{title or ''}\n{(body or '')[:MAX_BODY_CHARS]}"
        symbols = tuple(sym for sym, rx in self._symbols if rx.search(text))
        tags = tuple(tag for tag, rx in self._macros if rx.search(text))
        return RelevanceMatch(symbols=symbols, macro_tags=tags, relevant=bool(symbols or tags))
