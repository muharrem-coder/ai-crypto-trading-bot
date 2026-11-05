"""Bounded in-memory cache of published news (rule R2).

Capped at ``max_items`` (oldest-inserted evicted first) and TTL-evicted by ``ingested_at``.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta

from app.core.types import utcnow
from app.news.contracts import NewsEvent


class NewsCache:
    def __init__(self, max_items: int, ttl_hours: float, clock: Callable[[], datetime] = utcnow) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self._max = int(max_items)
        self._ttl = timedelta(hours=ttl_hours)
        self._clock = clock
        self._items: OrderedDict[str, NewsEvent] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, news_id: object) -> bool:
        return news_id in self._items

    def add(self, event: NewsEvent) -> None:
        self._items[event.news_id] = event
        self._items.move_to_end(event.news_id)
        self._pop_expired_head()
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def get(self, news_id: str) -> NewsEvent | None:
        return self._items.get(news_id)

    def mark_duplicate(self, news_id: str, source: str) -> NewsEvent | None:
        """Append ``source`` to the cached event's ``duplicate_sources`` (no-op for its own source / repeats)."""
        event = self._items.get(news_id)
        if event is None:
            return None
        if source == event.source or source in event.duplicate_sources:
            return event
        updated = dataclasses.replace(event, duplicate_sources=(*event.duplicate_sources, source))
        self._items[news_id] = updated
        return updated

    def recent(self, symbols: Iterable[str] | None, since: datetime, limit: int, include_macro: bool = True) -> list[NewsEvent]:
        """Events published at/after ``since``, newest first.

        ``symbols=None`` returns everything; otherwise an event matches when it mentions one of the symbols
        or (``include_macro``) carries at least one macro tag.
        """
        wanted = None if symbols is None else {s.upper() for s in symbols}
        out = [
            e
            for e in self._items.values()
            if e.published_at >= since
            and (
                wanted is None
                or (wanted.intersection(e.symbols))
                or (include_macro and e.macro_tags)
            )
        ]
        out.sort(key=lambda e: e.published_at, reverse=True)
        return out[: max(0, limit)]

    def evict_expired(self) -> int:
        """Full scan removing every expired entry; returns how many were removed."""
        cutoff = self._clock() - self._ttl
        stale = [k for k, e in self._items.items() if e.ingested_at < cutoff]
        for key in stale:
            del self._items[key]
        return len(stale)

    def _pop_expired_head(self) -> None:
        cutoff = self._clock() - self._ttl
        while self._items:
            first_key = next(iter(self._items))
            if self._items[first_key].ingested_at >= cutoff:
                break
            del self._items[first_key]
