"""Cross-source deduplication (rule R4) with strictly bounded memory (rule R2).

Two layers:

* exact: canonical-URL sha256 and normalised-title fingerprint, kept in an LRU
  ``OrderedDict`` capped at ``NewsSettings.dedup_window`` (URL keys never expire except by LRU;
  title fingerprints only match inside the 6h window so a recurring headline is not suppressed forever);
* near-duplicate: token-set Jaccard / token-level difflib ratio >= ``title_similarity_threshold`` against
  titles seen in the last 6h. Titles live in a ``deque(maxlen=...)``; an inverted token index restricts the
  comparison to a bounded set of candidates (entries sharing the query's up-to-3 rarest tokens when those occur in <=100 titles, else
  the 25 newest entries sharing its rarest token), so cost per check is O(1) in the window size. This is an
  approximation: a duplicate whose shared tokens are all very common and which is older than the 25 newest
  such titles can be missed.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import islice

from app.config.settings import NewsSettings
from app.core.types import utcnow
from app.news.normalize import (
    make_news_id,
    normalize_title,
    title_tokens,
    token_similarity,
    url_hash,
)
from app.news.sources import RawNewsItem

TITLE_WINDOW = timedelta(hours=6)
MAX_TITLE_WINDOW = 4096  # hard cap of the title deque
MIN_NEAR_DUP_TOKENS = 3  # shorter titles only match exactly (too generic otherwise)
_MAX_PER_TOKEN = 100  # a token with a posting list this short is 'rare'
_COMMON_FALLBACK = 25  # newest entries examined when every shared token is common
_MAX_INDEX_TOKENS = 3


@dataclass(frozen=True, slots=True)
class DedupResult:
    is_duplicate: bool
    original_id: str | None
    news_id: str


@dataclass(slots=True)
class _TitleEntry:
    seq: int
    arrived: datetime
    published: datetime
    tokens: tuple[str, ...]
    tokenset: frozenset[str]
    news_id: str


class Deduplicator:
    def __init__(self, settings: NewsSettings, clock: Callable[[], datetime] = utcnow) -> None:
        self._clock = clock
        self._cap = int(settings.dedup_window)
        self._threshold = float(settings.title_similarity_threshold)
        self._seen: OrderedDict[str, tuple[str, datetime]] = OrderedDict()
        self._titles: deque[_TitleEntry] = deque(maxlen=min(self._cap, MAX_TITLE_WINDOW))
        self._by_seq: dict[int, _TitleEntry] = {}
        self._index: dict[str, dict[int, None]] = {}
        self._seq = 0

    # ------------------------------------------------------------------ introspection
    @property
    def seen_size(self) -> int:
        return len(self._seen)

    @property
    def title_window_size(self) -> int:
        return len(self._titles)

    # ------------------------------------------------------------------ public API
    def check(self, raw: RawNewsItem) -> DedupResult:
        """Classify ``raw``; a new story is registered so later copies are recognised."""
        now = self._clock()
        self._expire(now)
        news_id = make_news_id(raw.url, raw.title, raw.source)
        uhash = url_hash(raw.url) if raw.url else ""
        ukey = f"u:{uhash}" if raw.url and uhash else ""
        norm = normalize_title(raw.title, raw.source)
        fkey = f"t:{hashlib.sha256(norm.encode('utf-8')).hexdigest()}" if norm else ""
        published = min(raw.published_at, now) if raw.published_at else now

        if ukey:
            hit = self._seen.get(ukey)
            if hit is not None:
                self._seen.move_to_end(ukey)
                return DedupResult(True, hit[0], news_id)
        if fkey:
            hit = self._seen.get(fkey)
            if hit is not None and now - hit[1] <= TITLE_WINDOW:
                self._seen.move_to_end(fkey)
                self._remember(ukey, hit[0], now)
                return DedupResult(True, hit[0], news_id)

        tokens = title_tokens(norm) if norm else ()
        if len(tokens) >= MIN_NEAR_DUP_TOKENS:
            original = self._find_near_duplicate(tokens, published)
            if original is not None:
                self._remember(ukey, original, now)
                self._remember(fkey, original, now)
                return DedupResult(True, original, news_id)

        self._register(news_id, ukey, fkey, tokens, published, now)
        return DedupResult(False, None, news_id)

    def seed(self, news_id: str, url: str | None, title: str, published_at: datetime, seen_at: datetime | None = None,
             source: str = "") -> None:
        """Register an already-published story (warm start from persistence) without classifying it."""
        now = self._clock()
        arrived = min(seen_at or now, now)
        norm = normalize_title(title, source)
        uhash = url_hash(url) if url else ""
        ukey = f"u:{uhash}" if url and uhash else ""
        fkey = f"t:{hashlib.sha256(norm.encode('utf-8')).hexdigest()}" if norm else ""
        tokens = title_tokens(norm) if norm else ()
        self._register(news_id, ukey, fkey, tokens, min(published_at, now), arrived)

    # ------------------------------------------------------------------ internals
    def _register(self, news_id: str, ukey: str, fkey: str, tokens: tuple[str, ...], published: datetime,
                  arrived: datetime) -> None:
        self._remember(ukey, news_id, arrived)
        self._remember(fkey, news_id, arrived)
        if len(tokens) >= MIN_NEAR_DUP_TOKENS:
            self._add_title(news_id, tokens, published, arrived)

    def _remember(self, key: str, news_id: str, ts: datetime) -> None:
        if not key:
            return
        self._seen[key] = (news_id, ts)
        self._seen.move_to_end(key)
        while len(self._seen) > self._cap:
            self._seen.popitem(last=False)

    def _add_title(self, news_id: str, tokens: tuple[str, ...], published: datetime, arrived: datetime) -> None:
        if len(self._titles) == self._titles.maxlen:
            self._drop(self._titles.popleft())
        self._seq += 1
        entry = _TitleEntry(self._seq, arrived, published, tokens, frozenset(tokens), news_id)
        self._titles.append(entry)
        self._by_seq[entry.seq] = entry
        for tok in entry.tokenset:
            self._index.setdefault(tok, {})[entry.seq] = None

    def _drop(self, entry: _TitleEntry) -> None:
        self._by_seq.pop(entry.seq, None)
        for tok in entry.tokenset:
            posting = self._index.get(tok)
            if posting is not None:
                posting.pop(entry.seq, None)
                if not posting:
                    del self._index[tok]

    def _expire(self, now: datetime) -> None:
        cutoff = now - TITLE_WINDOW
        while self._titles and self._titles[0].arrived < cutoff:
            self._drop(self._titles.popleft())

    def _find_near_duplicate(self, tokens: tuple[str, ...], published: datetime) -> str | None:
        qset = frozenset(tokens)
        postings = sorted((len(self._index[t]), t) for t in qset if t in self._index)
        if not postings:
            return None
        candidates: dict[int, None] = {}
        rare = [t for size, t in postings[:_MAX_INDEX_TOKENS] if size <= _MAX_PER_TOKEN]
        if rare:
            for tok in rare:
                candidates.update(dict.fromkeys(self._index[tok]))
        else:
            candidates.update(dict.fromkeys(islice(reversed(self._index[postings[0][1]]), _COMMON_FALLBACK)))
        best_id: str | None = None
        best = 0.0
        for seq in candidates:
            entry = self._by_seq.get(seq)
            if entry is None or abs(entry.published - published) > TITLE_WINDOW:
                continue
            shared = len(qset & entry.tokenset)
            # cheap upper bounds first: skip the (slow) difflib pass when the threshold is unreachable
            if shared / (len(qset) + len(entry.tokenset) - shared) < self._threshold and (
                2 * shared / (len(tokens) + len(entry.tokens)) < self._threshold
            ):
                continue
            sim = token_similarity(tokens, entry.tokens)
            if sim >= self._threshold and sim > best:
                best, best_id = sim, entry.news_id
        return best_id
