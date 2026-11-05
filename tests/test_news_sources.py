"""Source adapters against httpx.MockTransport (no network)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
import pytest

from app.news.sources import (
    CryptoPanicSource,
    NewsSource,
    RSSSource,
    SourceError,
    XSource,
    redact_text,
    strip_html,
)

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)

RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Feed</title><link>https://example.com</link><description>d</description>
<item><title>Bitcoin ETF Sees Record $1B Inflow - CoinDesk</title>
<link>https://www.example.com/markets/etf?utm_source=rss</link><guid>guid-1</guid>
<pubDate>Wed, 05 Nov 2026 10:00:00 GMT</pubDate>
<description>&lt;p&gt;Hello &amp;amp; &lt;b&gt;world&lt;/b&gt; &lt;script&gt;evil()&lt;/script&gt;&lt;/p&gt;</description></item>
<item><title>Long one</title><link>https://example.com/long</link>
<pubDate>Wed, 05 Nov 2026 09:00:00 +0200</pubDate>
<description>{long}</description></item>
</channel></rss>""".replace("{long}", "x" * 5000)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_rss_parses_items_strips_html_and_caps_body():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=RSS_XML.encode(), headers={"ETag": '"v1"'})

    src = RSSSource("rss:coindesk", "https://example.com/feed.xml", clock=lambda: NOW)
    assert isinstance(src, NewsSource)
    async with _client(handler) as client:
        items = await src.fetch(client)
    assert len(items) == 2
    first, second = items
    assert first.source == "rss:coindesk" and first.external_id == "guid-1"
    assert first.url == "https://www.example.com/markets/etf?utm_source=rss"
    assert first.published_at == datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc)
    assert first.body == "Hello & world"
    assert "<" not in first.body and "evil" not in first.body
    assert second.published_at == datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc)
    assert len(second.body) == 2000
    assert second.external_id == "https://example.com/long"


async def test_rss_etag_and_last_modified_304():
    seen: list[dict[str, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"inm": request.headers.get("if-none-match"), "ims": request.headers.get("if-modified-since")})
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(
            200, content=RSS_XML.encode(), headers={"ETag": '"v1"', "Last-Modified": "Wed, 05 Nov 2026 10:00:00 GMT"}
        )

    src = RSSSource("rss:x", "https://example.com/feed.xml", clock=lambda: NOW)
    async with _client(handler) as client:
        assert len(await src.fetch(client)) == 2
        assert await src.fetch(client) == []
    assert seen[0] == {"inm": None, "ims": None}
    assert seen[1] == {"inm": '"v1"', "ims": "Wed, 05 Nov 2026 10:00:00 GMT"}


async def test_rss_errors_and_entity_rejection():
    async def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html><body>not a feed")

    async def entity(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]><rss><channel/></rss>')

    for handler, text in ((boom, "HTTP 503"), (garbage, "not parseable"), (entity, "entity")):
        src = RSSSource("rss:x", "https://example.com/feed.xml")
        async with _client(handler) as client:
            with pytest.raises(SourceError, match=text):
                await src.fetch(client)


V2 = {
    "results": [
        {
            "id": 111, "kind": "news", "title": "Bitcoin ETF sees record $1B inflow",
            "description": "Details &amp; more", "published_at": "2026-11-05T11:00:00Z",
            "original_url": "https://news.example.com/btc-etf?utm_source=cp", "url": "https://cryptopanic.com/news/111/click/",
            "source": {"title": "Example", "domain": "news.example.com"},
            "instruments": [{"code": "BTC"}, {"code": "ETH"}],
            "votes": {"positive": 5, "negative": 1, "important": 2},
        },
        {"id": 112, "title": "", "published_at": "2026-11-05T11:00:00Z"},
    ]
}
V1 = {
    "results": [
        {
            "id": 7, "title": "Solana outage", "created_at": "2026-11-05T09:30:00.000000Z",
            "url": "https://cryptopanic.com/news/7/click/", "domain": "x.com",
            "source": {"title": "X", "domain": "x.com"}, "currencies": [{"code": "SOL"}],
            "votes": {"negative": 3},
        }
    ]
}


async def test_cryptopanic_v2_request_and_parsing():
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=V2)

    src = CryptoPanicSource("TOPSECRET", ["btc", "eth"], clock=lambda: NOW)
    async with _client(handler) as client:
        items = await src.fetch(client)
    req = captured[0]
    assert req.url.host == "cryptopanic.com" and req.url.path == "/api/developer/v2/posts/"
    assert req.url.params["auth_token"] == "TOPSECRET"
    assert req.url.params["currencies"] == "BTC,ETH" and req.url.params["public"] == "true"
    assert len(items) == 1
    item = items[0]
    assert item.source == "cryptopanic" and item.external_id == "111"
    assert item.url == "https://news.example.com/btc-etf?utm_source=cp"
    assert item.published_at == datetime(2026, 11, 5, 11, 0, tzinfo=timezone.utc)
    assert item.extra["votes"] == {"positive": 5, "negative": 1, "important": 2}
    assert item.extra["currencies"] == ["BTC", "ETH"] and item.body == "Details & more"


async def test_cryptopanic_v1_shape_tolerated():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=V1)

    async with _client(handler) as client:
        items = await CryptoPanicSource("t", ["SOL"], clock=lambda: NOW).fetch(client)
    assert items[0].url == "https://cryptopanic.com/news/7/click/"
    assert items[0].published_at == datetime(2026, 11, 5, 9, 30, tzinfo=timezone.utc)
    assert items[0].extra["currencies"] == ["SOL"] and items[0].extra["votes"] == {"negative": 3}


async def test_cryptopanic_errors_never_leak_token(caplog):
    async def http_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def transport_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {request.url}")

    async def bad_shape(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["nope"])

    caplog.set_level(logging.DEBUG)
    for handler in (http_error, transport_error, bad_shape):
        src = CryptoPanicSource("TOPSECRET", ["BTC"])
        async with _client(handler) as client:
            with pytest.raises(SourceError) as info:
                await src.fetch(client)
        assert "TOPSECRET" not in str(info.value)
        assert "TOPSECRET" not in repr(info.value.__cause__) + repr(info.value.__context__)
    assert "TOPSECRET" not in caplog.text


def test_redact_text():
    assert "SECRET" not in redact_text("GET https://x.io/a?auth_token=SECRET&b=1 failed")
    assert "SECRET" not in redact_text("Authorization: Bearer SECRET")
    assert "SECRET" not in redact_text("plain SECRET text", "SECRET")
    assert strip_html("<p>a&nbsp;&amp;<br>b</p>") == "a & b"


async def test_x_source_disabled_without_token(caplog):
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    caplog.set_level(logging.INFO)
    src = XSource("", ["federalreserve"])
    async with _client(handler) as client:
        assert await src.fetch(client) == []
        assert await src.fetch(client) == []
    assert calls == 0 and not src.enabled
    assert sum("X source disabled" in r.message for r in caplog.records) == 1


async def test_x_source_query_since_id_and_parsing():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json={
                "data": [
                    {"id": "1002", "text": "FOMC statement released &amp; rates held", "author_id": "9", "created_at": "2026-11-05T11:30:00.000Z"},
                    {"id": "1001", "text": "older", "author_id": "9", "created_at": "2026-11-05T11:00:00.000Z"},
                ],
                "includes": {"users": [{"id": "9", "username": "federalreserve"}]},
            })
        return httpx.Response(200, json={"meta": {"result_count": 0}})

    src = XSource("BEARERSECRET", ["federalreserve", "@SECGov"], clock=lambda: NOW)
    async with _client(handler) as client:
        items = await src.fetch(client)
        assert await src.fetch(client) == []
    first = requests[0]
    assert first.url.path == "/2/tweets/search/recent"
    assert "from:federalreserve OR from:SECGov" in first.url.params["query"]
    assert first.headers["authorization"] == "Bearer BEARERSECRET"
    assert "since_id" not in first.url.params
    assert requests[1].url.params["since_id"] == "1002"
    assert [i.external_id for i in items] == ["1002", "1001"]
    top = items[0]
    assert top.source == "x:federalreserve" and top.title == "FOMC statement released & rates held"
    assert top.url == "https://x.com/federalreserve/status/1002"
    assert top.published_at == datetime(2026, 11, 5, 11, 30, tzinfo=timezone.utc)


async def test_x_source_http_error_redacted():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    async with _client(handler) as client:
        with pytest.raises(SourceError, match="429") as info:
            await XSource("BEARERSECRET", ["a"]).fetch(client)
    assert "BEARERSECRET" not in str(info.value)
