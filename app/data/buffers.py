"""Bounded in-memory market-data buffers (rule R2: nothing here can grow without limit).

* ``RingBuffer``    - generic ``collections.deque(maxlen=...)`` wrapper.
* ``CandleBuffer``  - OHLCV bars keyed by open time with in-progress / closed semantics.
* ``OrderBookState``- top-N L2 book that rejects crossed / empty updates.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Generic, TypeVar

from app.data.models import Candle, OrderBookSnapshot

T = TypeVar("T")


def to_decimal(x: Any) -> Decimal | None:
    """``Decimal(str(x))`` for ccxt numbers. ``None`` for missing / unparsable / non-finite input."""
    if x is None or x == "":
        return None
    if isinstance(x, Decimal):
        value = x
    else:
        try:
            value = Decimal(str(x))
        except (InvalidOperation, ValueError):
            return None
    return value if value.is_finite() else None


class RingBuffer(Generic[T]):
    """Fixed-capacity FIFO. Appending beyond ``maxlen`` silently evicts the oldest item."""

    __slots__ = ("_dq",)

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("RingBuffer maxlen must be positive (unbounded buffers are forbidden)")
        self._dq: deque[T] = deque(maxlen=maxlen)

    @property
    def maxlen(self) -> int:
        return self._dq.maxlen or 0

    def append(self, item: T) -> None:
        self._dq.append(item)

    def extend(self, items: Iterable[T]) -> None:
        self._dq.extend(items)

    def __len__(self) -> int:
        return len(self._dq)

    def __iter__(self) -> Iterator[T]:
        return iter(self._dq)

    def last(self, n: int) -> tuple[T, ...]:
        """The newest ``n`` items in chronological order (oldest first)."""
        if n <= 0:
            return ()
        newest_first = tuple(itertools.islice(reversed(self._dq), n))
        return newest_first[::-1]

    @property
    def latest(self) -> T | None:
        return self._dq[-1] if self._dq else None

    def clear(self) -> None:
        self._dq.clear()


class CandleBuffer(RingBuffer[Candle]):
    """OHLCV bars in chronological order. At most the newest bar is in progress.

    ``upsert`` treats the exchange stream as "latest state of bar X": a bar closes when a newer
    bar arrives (or when an update arrives with ``closed=True``).
    """

    def upsert(self, candle: Candle) -> Candle | None:
        """Insert / replace a bar and return the candle that has just become closed (or ``None``).

        * same ``open_time`` as the newest bar: replace it (ignored if that bar is already closed);
        * newer ``open_time``: the previous in-progress bar is re-created with ``closed=True`` and
          returned, then ``candle`` is appended;
        * older ``open_time``: ignored (duplicate / out-of-order).
        """
        dq = self._dq
        if not dq:
            dq.append(candle)
            return candle if candle.closed else None
        last = dq[-1]
        if candle.open_time < last.open_time:
            return None
        if candle.open_time == last.open_time:
            if last.closed:
                return None
            dq[-1] = candle
            return candle if candle.closed else None
        newly_closed: Candle | None = None
        if not last.closed:
            newly_closed = replace(last, closed=True)
            dq[-1] = newly_closed
        dq.append(candle)
        if newly_closed is None and candle.closed:
            return candle
        return newly_closed

    def seed(self, candles: Iterable[Candle]) -> int:
        """Load historical bars silently (no close events). Returns how many were stored."""
        stored = 0
        for candle in sorted(candles, key=lambda c: c.open_time):
            last = self._dq[-1] if self._dq else None
            if last is not None and candle.open_time <= last.open_time:
                continue
            self._dq.append(candle if candle.closed else replace(candle, closed=True))
            stored += 1
        return stored

    def closed(self) -> tuple[Candle, ...]:
        return tuple(c for c in self._dq if c.closed)

    @property
    def current(self) -> Candle | None:
        """The in-progress bar, if any."""
        last = self._dq[-1] if self._dq else None
        return last if last is not None and not last.closed else None

    @property
    def last_closed(self) -> Candle | None:
        for candle in reversed(self._dq):
            if candle.closed:
                return candle
        return None


class OrderBookState:
    """Top-N order book for one symbol. Never stores more than ``depth`` levels per side."""

    def __init__(self, symbol: str, depth: int) -> None:
        if depth <= 0:
            raise ValueError("order book depth must be positive")
        self.symbol = symbol
        self.depth = depth
        self.crossed_rejects = 0
        self.invalid_rejects = 0  # empty side / unparsable update
        self.last_accepted = False  # outcome of the most recent apply_snapshot call
        self._snapshot: OrderBookSnapshot | None = None

    @property
    def snapshot(self) -> OrderBookSnapshot | None:
        return self._snapshot

    @staticmethod
    def _levels(raw: Iterable[Sequence[Any]], *, descending: bool) -> list[tuple[Decimal, Decimal]]:
        out: list[tuple[Decimal, Decimal]] = []
        for level in raw:
            if len(level) < 2:
                continue
            price = to_decimal(level[0])
            qty = to_decimal(level[1])
            if price is None or qty is None or price <= 0 or qty <= 0:
                continue
            out.append((price, qty))
        out.sort(key=lambda lv: lv[0], reverse=descending)
        return out

    def apply_snapshot(
        self,
        bids: Iterable[Sequence[Any]],
        asks: Iterable[Sequence[Any]],
        exchange_ts: datetime | None,
        received_at: datetime,
    ) -> OrderBookSnapshot | None:
        """Replace the book with the given levels (top ``depth`` kept, best first).

        A crossed (best_bid >= best_ask) or one-sided book is rejected: the previous snapshot is
        kept and returned (``None`` if there is none yet), the matching counter is incremented and
        ``last_accepted`` is False. Callers must NOT treat a rejected update as a fresh message.
        """
        bid_levels = self._levels(bids, descending=True)[: self.depth]
        ask_levels = self._levels(asks, descending=False)[: self.depth]
        if not bid_levels or not ask_levels:
            self.invalid_rejects += 1
            self.last_accepted = False
            return self._snapshot
        if bid_levels[0][0] >= ask_levels[0][0]:
            self.crossed_rejects += 1
            self.last_accepted = False
            return self._snapshot
        self._snapshot = OrderBookSnapshot(
            symbol=self.symbol,
            bids=tuple(bid_levels),
            asks=tuple(ask_levels),
            exchange_ts=exchange_ts,
            received_at=received_at,
        )
        self.last_accepted = True
        return self._snapshot
