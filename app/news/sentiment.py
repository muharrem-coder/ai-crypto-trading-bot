"""Deterministic lexicon sentiment + importance scoring.

IMPORTANT: this is only a cheap, reproducible PRE-FILTER signal (no ML, no network, no randomness).
The real interpretation of a headline - context, magnitude, whether it is already priced in - is done by
Claude in Phase 3, which receives the raw title/summary alongside these numbers.

Score = x / (|x| + 4) where x is the weighted lexicon sum (title weight 1.0, body 0.4), quantized to 0.01 with
Decimal arithmetic (fully deterministic). Negators ("not", "no", "denies", ...) within the three preceding
tokens flip a word's polarity and dampen it by 20%.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, Decimal

BULLISH_THRESHOLD = Decimal("0.15")
_QUANT = Decimal("0.01")
_K = Decimal(4)
_BODY_WEIGHT = Decimal("0.4")
_NEGATION_DAMPING = Decimal("0.8")
_NEGATION_WINDOW = 3
_BODY_CHARS = 1500

_PHRASES = (
    ("all time high", "ath"), ("record high", "ath"), ("sell off", "selloff"), ("rate cut", "ratecut"),
    ("rate cuts", "ratecut"), ("rate hike", "ratehike"), ("rate hikes", "ratehike"), ("short squeeze", "squeeze"),
    ("wind down", "winddown"), ("shut down", "shutdown"), ("shuts down", "shutdown"),
)

LEXICON: dict[str, int] = {
    # bullish
    "approval": 3, "approve": 3, "approves": 3, "approved": 3, "inflow": 2, "inflows": 2, "adoption": 2, "adopt": 2,
    "adopts": 2, "adopted": 2, "rally": 2, "rallies": 2, "rallied": 2, "partnership": 2, "partner": 1, "partners": 1,
    "surge": 2, "surges": 2, "surged": 2, "soar": 2, "soars": 2, "soared": 2, "jump": 1, "jumps": 1, "record": 1,
    "ath": 2, "bullish": 2, "breakout": 2, "gain": 1, "gains": 1, "upgrade": 1, "launch": 1, "launches": 1,
    "listing": 1, "lists": 1, "buyback": 2, "accumulate": 1, "accumulation": 1, "integration": 1, "win": 2,
    "wins": 2, "won": 2, "ratecut": 1, "squeeze": 1, "recover": 1, "recovery": 1, "rebound": 1, "support": 1,
    "backing": 1, "invest": 1, "invests": 1, "investment": 1, "milestone": 1, "greenlight": 2,
    # bearish
    "hack": -4, "hacked": -4, "hacks": -4, "hacker": -3, "hackers": -3, "exploit": -4, "exploits": -4,
    "exploited": -4, "drained": -3, "drain": -3, "stolen": -3, "theft": -3, "breach": -3, "breached": -3,
    "vulnerability": -2, "lawsuit": -3, "lawsuits": -3, "sue": -3, "sues": -3, "sued": -3, "ban": -3, "bans": -3,
    "banned": -3, "banning": -3, "outflow": -2, "outflows": -2, "liquidation": -2, "liquidations": -2,
    "liquidated": -2, "delist": -3, "delists": -3, "delisted": -3, "delisting": -3, "insolvency": -4,
    "insolvent": -4, "bankrupt": -4, "bankruptcy": -4, "crash": -3, "crashes": -3, "crashed": -3, "plunge": -3,
    "plunges": -3, "plunged": -3, "dump": -2, "dumps": -2, "dumped": -2, "selloff": -2, "fraud": -3, "scam": -3,
    "rug": -3, "rejected": -2, "rejects": -2, "rejection": -2, "denied": -1, "denies": -1, "charges": -2,
    "charged": -2, "investigation": -2, "probe": -2, "halt": -2, "halted": -2, "suspend": -2, "suspended": -2,
    "suspends": -2, "bearish": -2, "ratehike": -1, "sanction": -2, "sanctions": -2, "sanctioned": -2, "fined": -2,
    "fine": 0, "fear": -1, "fears": -1, "decline": -1, "declines": -1, "drop": -1, "drops": -1, "dropped": -1,
    "falls": -1, "fall": -1, "fell": -1, "loss": -1, "losses": -2, "warning": -1, "warns": -1, "risk": -1,
    "concerns": -1, "shutdown": -2, "winddown": -3, "withdrawals": -1, "depeg": -3, "depegged": -3, "default": -3,
}

NEGATORS = frozenset(
    {"not", "no", "never", "without", "denies", "denied", "deny", "fails", "failed", "fail", "unable", "cannot",
     "cant", "wont", "isnt", "arent", "wasnt", "doesnt", "didnt", "dont", "nor", "neither", "stops", "stopped"}
)

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")

_OFFICIAL_SOURCES = frozenset({"sec", "fed", "x:federalreserve", "x:secgov", "rss:sec", "rss:fed"})
_TAG_WEIGHTS = {"HACK": 4, "FED": 4, "CPI": 3, "ETF": 3, "SEC": 3, "REGULATION": 2, "EXCHANGE": 2}


def _tokens(text: str) -> list[str]:
    lowered = text.lower().replace("-", " ")
    for phrase, repl in _PHRASES:
        lowered = lowered.replace(phrase, repl)
    return [t.replace("'", "") for t in _TOKEN.findall(lowered)]


def _raw_score(text: str) -> Decimal:
    tokens = _tokens(text)
    total = Decimal(0)
    for i, tok in enumerate(tokens):
        weight = LEXICON.get(tok, 0)
        if not weight:
            continue
        value = Decimal(weight)
        if any(t in NEGATORS for t in tokens[max(0, i - _NEGATION_WINDOW): i]):
            value = -value * _NEGATION_DAMPING
        total += value
    return total


def score(title: str | None, body: str | None = None) -> tuple[Decimal, str]:
    """Return (score in [-1, 1] quantized to 0.01, BULLISH | BEARISH | NEUTRAL)."""
    x = _raw_score(title or "")
    if body:
        x += _raw_score(body[:_BODY_CHARS]) * _BODY_WEIGHT
    value = x / (abs(x) + _K) if x else Decimal(0)
    value = max(Decimal(-1), min(Decimal(1), value)).quantize(_QUANT, rounding=ROUND_HALF_EVEN)
    if value >= BULLISH_THRESHOLD:
        label = "BULLISH"
    elif value <= -BULLISH_THRESHOLD:
        label = "BEARISH"
    else:
        label = "NEUTRAL"
    return value, label


def importance(source: str, macro_tags: tuple[str, ...] | list[str], symbols: tuple[str, ...] | list[str],
               score_value: Decimal) -> int:
    """0..10: official sources (SEC/FED) and HACK/FED/CPI/ETF tags weigh most; strong sentiment adds up to 3."""
    points = 1
    if source.lower() in _OFFICIAL_SOURCES:
        points += 3
    weights = sorted((_TAG_WEIGHTS.get(t.upper(), 1) for t in macro_tags), reverse=True)
    if weights:
        points += weights[0] + min(2, len(weights) - 1)
    if symbols:
        points += 1
    points += int((abs(Decimal(score_value)) * 3).to_integral_value(rounding=ROUND_HALF_EVEN))
    return max(0, min(10, points))
