"""Normalisation, dedup core, relevance, sentiment and cache unit tests (no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.config.settings import NewsSettings
from app.news.cache import NewsCache
from app.news.contracts import NewsEvent
from app.news.dedup import Deduplicator
from app.news.normalize import (
    canonical_url,
    normalize_title,
    title_fingerprint,
    title_similarity,
    url_hash,
)
from app.news.relevance import RelevanceFilter
from app.news.sentiment import importance, score
from app.news.sources import RawNewsItem

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def _rel() -> RelevanceFilter:
    s = NewsSettings()
    return RelevanceFilter(s.symbol_keywords, s.macro_keywords)


def _raw(i: int, source: str = "rss:test", title: str | None = None, url: str | None = None) -> RawNewsItem:
    return RawNewsItem(
        source=source,
        external_id=str(i),
        title=title or f"Bitcoin update {i} headline",
        url=url if url is not None else f"https://example.com/story/{i}",
        published_at=NOW - timedelta(minutes=5),
    )


# ------------------------------------------------------------------------------ URLs
def test_canonical_url_strips_tracking_fragment_www_and_trailing_slash():
    a = "HTTPS://www.CoinDesk.com/markets/btc-etf/?utm_source=rss&utm_medium=feed&fbclid=abc&gclid=x&ref=tw#comments"
    b = "https://coindesk.com/markets/btc-etf"
    assert canonical_url(a) == b == "https://coindesk.com/markets/btc-etf"
    assert url_hash(a) == url_hash(b)


def test_canonical_url_amp_variants_and_query_order():
    base = canonical_url("https://site.com/news/story")
    assert canonical_url("https://site.com/news/story/amp/") == base
    assert canonical_url("https://site.com/amp/news/story") == base
    assert canonical_url("https://site.com/news/story?outputType=amp") == base
    assert canonical_url("https://site.com/news/story.amp") == base
    assert canonical_url("https://amp.site.com/news/story") == base
    assert canonical_url("https://site.com/p?b=2&a=1&utm_campaign=z") == "https://site.com/p?a=1&b=2"
    assert canonical_url("https://site.com:443/x") == "https://site.com/x"
    assert canonical_url(None) == "" and canonical_url("  ") == ""
    assert url_hash("https://a.com/x") != url_hash("https://a.com/y")


# ------------------------------------------------------------------------------ titles
def test_normalize_title_strips_source_suffix_punctuation_and_urls():
    assert normalize_title("Bitcoin ETF Sees Record $1B Inflow - CoinDesk") == "bitcoin etf sees record 1b inflow"
    assert normalize_title("Bitcoin ETF sees record $1B inflow | The Block") == "bitcoin etf sees record 1b inflow"
    assert normalize_title("  Bitcoin   ETF sees record $1B inflow!!  https://t.co/abc ") == "bitcoin etf sees record 1b inflow"
    assert normalize_title("RT @whale_alert: 500 BTC moved") == "500 btc moved"
    # a real dash inside the headline is not a publisher suffix
    assert normalize_title("Bitcoin rallies - analysts say rally is just starting") == "bitcoin rallies analysts say rally is just starting"
    assert title_fingerprint("Bitcoin ETF Sees Record $1B Inflow - CoinDesk") == title_fingerprint("bitcoin etf sees record $1b inflow")
    assert title_fingerprint("") == ""


def test_title_similarity_thresholds():
    assert title_similarity("Bitcoin ETF sees record $1B inflow", "Bitcoin ETF Sees Record $1B Inflow - CoinDesk") == 1.0
    assert title_similarity("Bitcoin ETF sees record 1B inflow today", "Bitcoin ETF sees record 1B inflow") >= 0.85
    assert title_similarity("SEC approves spot Bitcoin ETF", "SEC rejects spot Bitcoin ETF") < 0.85
    assert title_similarity("Bitcoin falls below $60,000", "Bitcoin rises above $60,000") < 0.85
    assert title_similarity("", "x") == 0.0


# ------------------------------------------------------------------------------ relevance
@pytest.mark.parametrize(
    "text",
    [
        "Our new solution simplifies things",
        "A better method for the problem",
        "Wait a second for the result",
        "Hackathon winners announced",
        "Ethical questions about AI",
        "Consolation prize for the local bakery",
    ],
)
def test_relevance_false_positives_not_matched(text):
    m = _rel().match(text, None)
    assert not m.relevant, (text, m)


@pytest.mark.parametrize(
    "text,symbol",
    [("Solana network upgrade", "SOL"), ("SOL rallies 10%", "SOL"), ("Why $eth is moving", "ETH"),
     ("btc breaks 100k", "BTC"), ("Ripple wins case", "XRP"), ("Ether staking hits record", "ETH")],
)
def test_relevance_symbol_matches(text, symbol):
    m = _rel().match(text, None)
    assert symbol in m.symbols and m.relevant


def test_relevance_macro_tags_and_inflections():
    f = _rel()
    assert f.match("FOMC minutes show split on rates", "").macro_tags == ("FED",)
    assert "HACK" in f.match("Exchange hacked for $50M", None).macro_tags
    assert "SEC" in f.match("SEC sues exchange", None).macro_tags
    assert "ETF" in f.match("New ETFs launch", None).macro_tags
    assert "CPI" in f.match("US CPI hotter than expected", None).macro_tags
    m = f.match("Quiet day", "Powell speaks about inflation")
    assert m.relevant and not m.symbols and set(m.macro_tags) == {"FED", "CPI"}
    assert not f.match("Local bakery opens", "fresh croissants every morning").relevant


# ------------------------------------------------------------------------------ sentiment
def test_sentiment_bullish_bearish_neutral_and_negation():
    s, label = score("SEC approval of spot Bitcoin ETF brings record inflow")
    assert label == "BULLISH" and s > Decimal("0.15")
    s, label = score("Exchange hacked, $50M drained in exploit")
    assert label == "BEARISH" and s < Decimal("-0.15")
    s, label = score("Bitcoin trades sideways ahead of data")
    assert label == "NEUTRAL" and s == Decimal("0.00")
    approved = score("SEC approved the ETF")[0]
    not_approved = score("SEC not approved the ETF")[0]
    assert approved > 0 > not_approved
    assert score("Binance denies hack rumors")[0] > 0
    for text in ("Massive rally surge soar record gains win approval", "hack exploit crash plunge fraud scam bankrupt ban"):
        value, _ = score(text, text)
        assert Decimal("-1") <= value <= Decimal("1")
        assert value == value.quantize(Decimal("0.01"))
    assert score("Bitcoin rally", "x") == score("Bitcoin rally", "x")  # deterministic


def test_importance_weights():
    official = importance("fed", ("FED",), (), Decimal("0"))
    plain = importance("rss:decrypt", (), ("BTC",), Decimal("0"))
    hack = importance("rss:decrypt", ("HACK",), ("ETH",), Decimal("-0.5"))
    assert 0 <= plain < hack <= 10 and official >= 8 and official > plain
    assert importance("sec", ("SEC", "HACK", "FED", "ETF"), ("BTC",), Decimal("1")) == 10
    assert importance("x:a", (), (), Decimal("0")) >= 0


# ------------------------------------------------------------------------------ memory caps (R2)
def test_deduplicator_bounded_after_50k_inserts():
    settings = NewsSettings(dedup_window=1000)
    dd = Deduplicator(settings, lambda: NOW)
    for i in range(50_000):
        res = dd.check(_raw(i, title=f"Bitcoin update {i} headline"))
        assert not res.is_duplicate
    assert dd.seen_size <= 1000
    assert dd.title_window_size <= 1000
    assert len(dd._by_seq) == dd.title_window_size
    assert sum(len(p) for p in dd._index.values()) <= dd.title_window_size * 6
    # oldest evicted from LRU, newest still remembered
    assert dd.check(_raw(49_999, source="x:a", url="https://example.com/story/49999")).is_duplicate
    assert not dd.check(_raw(0)).is_duplicate


def test_news_cache_capped_and_ttl_evicted():
    clock = {"now": NOW}
    cache = NewsCache(100, 24, lambda: clock["now"])

    def ev(i: int, ingested: datetime, symbols=("BTC",), tags=()) -> NewsEvent:
        return NewsEvent(
            news_id=f"id{i}", source="s", external_id=str(i), title=f"t{i}", url=None,
            published_at=ingested, ingested_at=ingested, symbols=symbols, macro_tags=tags,
            sentiment_score=Decimal("0"), sentiment_label="NEUTRAL", importance=1,
        )

    for i in range(50_000):
        cache.add(ev(i, NOW))
    assert len(cache) == 100
    assert cache.get("id49999") is not None and cache.get("id0") is None

    clock["now"] = NOW + timedelta(hours=25)
    assert cache.evict_expired() == 100 and len(cache) == 0

    clock["now"] = NOW
    old = ev(1, NOW - timedelta(hours=1), ("ETH",))
    macro = ev(2, NOW - timedelta(minutes=30), (), ("FED",))
    btc = ev(3, NOW - timedelta(minutes=10), ("BTC",))
    for e in (old, macro, btc):
        cache.add(e)
    since = NOW - timedelta(hours=2)
    assert [e.news_id for e in cache.recent(["BTC"], since, 10)] == ["id3", "id2"]
    assert [e.news_id for e in cache.recent(["BTC"], since, 10, include_macro=False)] == ["id3"]
    assert [e.news_id for e in cache.recent(None, since, 2)] == ["id3", "id2"]
    assert cache.mark_duplicate("id3", "cryptopanic").duplicate_sources == ("cryptopanic",)
    assert cache.mark_duplicate("id3", "cryptopanic").duplicate_sources == ("cryptopanic",)
    assert cache.mark_duplicate("id3", "s").duplicate_sources == ("cryptopanic",)
    assert cache.mark_duplicate("nope", "x") is None
