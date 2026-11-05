"""News sources: RSS/Atom, CryptoPanic and X (Twitter) API v2.

Every source returns ``RawNewsItem`` objects; dedup/relevance/sentiment happen in the engine.
Secrets (CryptoPanic ``auth_token``, X bearer token) are never logged: errors are re-raised as
``SourceError`` with the secret redacted, and the ``httpx`` request logger is filtered.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import feedparser
import httpx

from app.core.types import utcnow

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 2000
MAX_RSS_BYTES = 5_000_000
MAX_RSS_ENTRIES = 100

# ----------------------------------------------------------------------------------------
# Secret redaction
# ----------------------------------------------------------------------------------------

_QUERY_SECRET = re.compile(r"((?:auth_token|access_token|api_?key|bearer_token|token)=)[^&\s'\"]+", re.IGNORECASE)
_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/%=]+", re.IGNORECASE)


def redact_text(text: str, *secrets: str) -> str:
    """Remove known secrets and token-looking query params / bearer values from ``text``."""
    out = str(text)
    for secret in secrets:
        if secret:
            out = out.replace(secret, "***")
    out = _QUERY_SECRET.sub(r"\1***", out)
    return _BEARER.sub(r"\1***", out)


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never break logging
            return True
        redacted = redact_text(message)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).addFilter(_RedactingFilter())


class SourceError(RuntimeError):
    """A source fetch failed; the message is already free of secrets."""


# ----------------------------------------------------------------------------------------
# Contracts
# ----------------------------------------------------------------------------------------


@dataclass(slots=True)
class RawNewsItem:
    source: str  # "cryptopanic", "rss:coindesk", "sec", "fed", "x:<handle>"
    external_id: str | None
    title: str
    url: str | None
    published_at: datetime
    body: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class NewsSource(Protocol):
    name: str

    async def fetch(self, client: httpx.AsyncClient) -> list[RawNewsItem]: ...


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------

_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def strip_html(text: str | None, limit: int = MAX_BODY_CHARS) -> str:
    if not text:
        return ""
    cleaned = _TAG.sub(" ", _SCRIPT_STYLE.sub(" ", text))
    cleaned = _SPACES.sub(" ", html.unescape(cleaned)).strip()
    return cleaned[:limit]


def parse_datetime(value: Any) -> datetime | None:
    """struct_time / ISO-8601 / RFC-2822 string / datetime -> aware UTC datetime, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, time.struct_time):
        return datetime(*value[:6], tzinfo=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime

                dt = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                return None
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    return None


def _safe_url(url: httpx.URL | str) -> str:
    u = httpx.URL(str(url))
    return f"{u.scheme}://{u.host}{u.path}"


async def _get(client: httpx.AsyncClient, url: str, *, secrets: tuple[str, ...] = (), **kwargs: Any) -> httpx.Response:
    """GET that converts transport errors into secret-free ``SourceError`` (raised outside the except block)."""
    failure: str | None = None
    response: httpx.Response | None = None
    try:
        response = await client.get(url, **kwargs)
    except httpx.HTTPError as exc:
        failure = f"{type(exc).__name__}: {redact_text(str(exc), *secrets)}"
    if failure is not None or response is None:
        raise SourceError(f"request to {_safe_url(url)} failed: {failure}")
    return response


def _check_status(response: httpx.Response) -> None:
    if response.status_code >= 400:
        raise SourceError(f"HTTP {response.status_code} from {_safe_url(response.request.url)}")


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        raise SourceError(f"invalid JSON from {_safe_url(response.request.url)}") from None


# ----------------------------------------------------------------------------------------
# RSS
# ----------------------------------------------------------------------------------------


class RSSSource:
    """RSS/Atom feed with conditional GET (ETag / Last-Modified); 304 yields no items."""

    def __init__(self, name: str, url: str, *, clock: Callable[[], datetime] = utcnow) -> None:
        self.name = name
        self.url = url
        self._clock = clock
        self._etag: str | None = None
        self._last_modified: str | None = None

    async def fetch(self, client: httpx.AsyncClient) -> list[RawNewsItem]:
        headers = {"Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.5"}
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._last_modified:
            headers["If-Modified-Since"] = self._last_modified
        response = await _get(client, self.url, headers=headers)
        if response.status_code == 304:
            return []
        _check_status(response)
        content = response.content
        if len(content) > MAX_RSS_BYTES:
            raise SourceError(f"feed {self.name} too large ({len(content)} bytes)")
        if b"<!ENTITY" in content:  # entity-expansion (billion laughs / XXE) guard
            raise SourceError(f"feed {self.name} contains entity declarations; rejected")
        parsed = await asyncio.to_thread(feedparser.parse, content)
        if not parsed.entries:
            if parsed.get("bozo"):
                raise SourceError(f"feed {self.name} is not parseable")
            self._remember_validators(response)
            return []

        now = self._clock()
        items: list[RawNewsItem] = []
        for entry in parsed.entries[:MAX_RSS_ENTRIES]:
            title = strip_html(entry.get("title"), 500)
            if not title:
                continue
            link = entry.get("link") or None
            body = strip_html(entry.get("summary") or entry.get("description"), MAX_BODY_CHARS) or None
            published = (
                parse_datetime(entry.get("published_parsed"))
                or parse_datetime(entry.get("updated_parsed"))
                or now
            )
            items.append(
                RawNewsItem(
                    source=self.name,
                    external_id=str(entry.get("id") or link or title),
                    title=title,
                    url=link,
                    published_at=published,
                    body=body,
                    extra={"feed": self.url},
                )
            )
        self._remember_validators(response)
        return items

    def _remember_validators(self, response: httpx.Response) -> None:
        self._etag = response.headers.get("etag") or self._etag
        self._last_modified = response.headers.get("last-modified") or self._last_modified


# ----------------------------------------------------------------------------------------
# CryptoPanic
# ----------------------------------------------------------------------------------------


class CryptoPanicSource:
    """CryptoPanic developer API (v2 shape; v1 shape tolerated)."""

    name = "cryptopanic"
    min_interval = 30.0
    URL = "https://cryptopanic.com/api/developer/v2/posts/"

    def __init__(self, token: str, currencies: list[str] | None = None, *, clock: Callable[[], datetime] = utcnow) -> None:
        self._token = token
        self.currencies = [c.upper() for c in (currencies or [])]
        self._clock = clock

    async def fetch(self, client: httpx.AsyncClient) -> list[RawNewsItem]:
        params = {"auth_token": self._token, "public": "true"}
        if self.currencies:
            params["currencies"] = ",".join(self.currencies)
        response = await _get(client, self.URL, secrets=(self._token,), params=params)
        _check_status(response)
        data = _json(response)
        posts = data.get("results") if isinstance(data, dict) else None
        if not isinstance(posts, list):
            raise SourceError("unexpected CryptoPanic response shape")

        now = self._clock()
        items: list[RawNewsItem] = []
        for post in posts:
            if not isinstance(post, dict):
                continue
            title = strip_html(post.get("title"), 500)
            if not title:
                continue
            src = post.get("source") if isinstance(post.get("source"), dict) else {}
            url = post.get("original_url") or post.get("url") or None
            codes = [
                str(c.get("code")).upper()
                for c in (post.get("instruments") or post.get("currencies") or [])
                if isinstance(c, dict) and c.get("code")
            ]
            votes = post.get("votes") if isinstance(post.get("votes"), dict) else {}
            published = parse_datetime(post.get("published_at")) or parse_datetime(post.get("created_at")) or now
            post_id = post.get("id")
            items.append(
                RawNewsItem(
                    source=self.name,
                    external_id=str(post_id) if post_id is not None else (url or title),
                    title=title,
                    url=url,
                    published_at=published,
                    body=strip_html(post.get("description") or post.get("body"), MAX_BODY_CHARS) or None,
                    extra={
                        "votes": dict(votes),
                        "currencies": codes,
                        "kind": post.get("kind"),
                        "domain": src.get("domain") or post.get("domain"),
                        "publisher": src.get("title"),
                    },
                )
            )
        return items


# ----------------------------------------------------------------------------------------
# X (Twitter) API v2
# ----------------------------------------------------------------------------------------


class XSource:
    """Recent-search poller for a fixed list of accounts, with ``since_id`` tracking."""

    name = "x"
    min_interval = 60.0  # recent-search rate limits are tight
    URL = "https://api.twitter.com/2/tweets/search/recent"
    MAX_QUERY_LEN = 480

    def __init__(self, bearer_token: str, accounts: list[str], *, clock: Callable[[], datetime] = utcnow) -> None:
        self._token = bearer_token
        self.accounts = [a.lstrip("@") for a in accounts if a.strip()]
        self._clock = clock
        self._since: dict[str, str] = {}
        self._warned = False

    @property
    def enabled(self) -> bool:
        return bool(self._token) and bool(self.accounts)

    def _queries(self) -> list[str]:
        queries: list[str] = []
        current: list[str] = []
        for account in self.accounts:
            candidate = " OR ".join(f"from:{a}" for a in [*current, account])
            if current and len(candidate) > self.MAX_QUERY_LEN:
                queries.append(" OR ".join(f"from:{a}" for a in current))
                current = [account]
            else:
                current.append(account)
        if current:
            queries.append(" OR ".join(f"from:{a}" for a in current))
        return queries

    async def fetch(self, client: httpx.AsyncClient) -> list[RawNewsItem]:
        if not self.enabled:
            if not self._warned:
                self._warned = True
                logger.info("X source disabled: no bearer token or accounts configured")
            return []
        items: list[RawNewsItem] = []
        for query in self._queries():
            items.extend(await self._fetch_query(client, query))
        return items

    async def _fetch_query(self, client: httpx.AsyncClient, query: str) -> list[RawNewsItem]:
        params = {
            "query": f"({query}) -is:retweet",
            "max_results": "100",
            "tweet.fields": "created_at,author_id,public_metrics",
            "expansions": "author_id",
            "user.fields": "username",
        }
        since = self._since.get(query)
        if since:
            params["since_id"] = since
        response = await _get(
            client, self.URL, secrets=(self._token,), params=params,
            headers={"Authorization": f"Bearer {self._token}"},
        )
        _check_status(response)
        data = _json(response)
        if not isinstance(data, dict):
            raise SourceError("unexpected X response shape")
        tweets = data.get("data") or []
        users = {
            str(u.get("id")): str(u.get("username"))
            for u in (data.get("includes") or {}).get("users", [])
            if isinstance(u, dict)
        }
        now = self._clock()
        out: list[RawNewsItem] = []
        newest = int(since) if since and since.isdigit() else 0
        for tweet in tweets:
            if not isinstance(tweet, dict) or not tweet.get("id"):
                continue
            tweet_id = str(tweet["id"])
            if tweet_id.isdigit():
                newest = max(newest, int(tweet_id))
            text = _SPACES.sub(" ", html.unescape(str(tweet.get("text") or ""))).strip()
            if not text:
                continue
            handle = users.get(str(tweet.get("author_id")), "unknown")
            out.append(
                RawNewsItem(
                    source=f"x:{handle}",
                    external_id=tweet_id,
                    title=text[:500],
                    url=f"https://x.com/{handle}/status/{tweet_id}",
                    published_at=parse_datetime(tweet.get("created_at")) or now,
                    body=None,
                    extra={"author": handle, "metrics": tweet.get("public_metrics") or {}},
                )
            )
        if newest:
            self._since[query] = str(newest)
        return out
