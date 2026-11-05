"""Exponential reconnect backoff with jitter."""

from __future__ import annotations

import random


class ExponentialBackoff:
    """``delay = min(maximum, base * 2**attempt)`` randomised by +/- ``jitter`` (a fraction).

    The jittered delay never exceeds ``maximum`` and is always positive. ``attempt`` counts the
    delays handed out since the last ``reset()``.
    """

    def __init__(
        self,
        base: float,
        maximum: float,
        jitter: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        if base <= 0 or maximum <= 0:
            raise ValueError("base and maximum must be positive")
        if maximum < base:
            raise ValueError("maximum must be >= base")
        if not 0.0 <= jitter <= 1.0:
            raise ValueError("jitter must be within [0, 1]")
        self.base = base
        self.maximum = maximum
        self.jitter = jitter
        self._rng = rng or random.Random()
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt

    def raw_delay(self, attempt: int | None = None) -> float:
        """Un-jittered delay for ``attempt`` (defaults to the current one)."""
        n = self._attempt if attempt is None else attempt
        exponent = min(max(n, 0), 62)  # avoid huge floats; the cap is hit long before this
        return min(self.maximum, self.base * (2.0**exponent))

    def next_delay(self) -> float:
        raw = self.raw_delay()
        self._attempt += 1
        if self.jitter:
            raw *= 1.0 + self._rng.uniform(-self.jitter, self.jitter)
        return max(min(raw, self.maximum), 1e-9)

    def reset(self) -> None:
        self._attempt = 0
