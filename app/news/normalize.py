"""URL / title normalisation used by the deduplicator.

Pure functions, no I/O, no global mutable state.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlsplit

# Query parameters that never identify content (tracking / attribution / AMP switches).
_TRACKING_PARAMS = frozenset(
    {
        "ref", "ref_src", "ref_url", "referrer", "fbclid", "gclid", "dclid", "msclkid", "yclid",
        "mc_cid", "mc_eid", "igshid", "cmpid", "campaign", "amp", "amp_js_v", "amp_gsa", "usqp",
        "_ga", "_gl", "spm", "taid", "guccounter",
    }
)

# Publisher suffixes appended to headlines (" - CoinDesk"). Matching is on the normalised suffix.
_KNOWN_SOURCE_SUFFIXES = frozenset(
    {
        "coindesk", "cointelegraph", "the block", "theblock", "decrypt", "cryptopanic", "reuters", "bloomberg",
        "cnbc", "yahoo finance", "yahoo", "forbes", "the defiant", "bitcoin magazine", "beincrypto", "coinmarketcap",
        "benzinga", "investopedia", "fortune", "wsj", "wall street journal", "financial times", "ft", "cryptoslate",
        "u today", "utoday", "newsbtc", "bitcoinist", "cryptonews", "the daily hodl", "daily hodl", "coingape",
        "ambcrypto", "dailycoin", "coinjournal", "crypto briefing", "the street", "thestreet", "marketwatch",
        "business insider", "coinpedia", "cryptopotato", "bitcoin com news", "news bitcoin com", "sec gov",
        "federal reserve", "the verge", "techcrunch", "axios", "barrons", "barron s", "ap news", "associated press",
    }
)

_SEPARATOR_SUFFIX = re.compile(r"\s+([-–—|•·])\s+([^-–—|•·]{1,40})$")
_URL_IN_TEXT = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RETWEET_PREFIX = re.compile(r"^rt\s+@\w+:\s*")
_MENTION = re.compile(r"@\w+")
_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
_SPACES = re.compile(r"\s+")

STOPWORDS = frozenset(
    {"a", "an", "the", "of", "to", "in", "on", "for", "and", "as", "at", "is", "are", "was", "were", "by", "with",
     "from", "its", "it", "s", "that", "this", "be", "has", "have", "will", "after", "amid"}
)


def canonical_url(url: str | None) -> str:
    """Canonical form of ``url`` for identity comparison; ``""`` for empty input.

    Lowercases scheme and host, drops ``www.``/``amp.`` host prefixes, default ports, userinfo, fragments,
    ``utm_*``/``ref``/``fbclid``/``gclid`` (and similar) query params, ``outputType=amp``, AMP path segments
    (``/amp/``, ``/story.amp``) and trailing slashes. Remaining query params are sorted.
    """
    if not url:
        return ""
    raw = url.strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https:" + raw if raw.startswith("//") else "https://" + raw
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return raw.lower()
    scheme = parts.scheme.lower()
    for prefix in ("www.", "amp."):
        if host.startswith(prefix) and host.count(".") >= 2:
            host = host[len(prefix):]
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"

    segments = [s for s in parts.path.split("/") if s]
    segments = [s for s in segments if s.lower() != "amp"]
    if segments:
        last = segments[-1]
        if last.lower().endswith(".amp"):
            segments[-1] = last[:-4]
        elif last.lower() in ("amp.html", "index.amp"):
            segments.pop()
    path = "/" + "/".join(segments) if segments else ""

    query_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lk = key.lower()
        if lk.startswith("utm_") or lk in _TRACKING_PARAMS:
            continue
        if lk == "outputtype" and value.lower() == "amp":
            continue
        query_pairs.append((key, value))
    query_pairs.sort()
    query = urlencode(query_pairs)
    return f"{scheme}://{host}{path}" + (f"?{query}" if query else "")


def url_hash(url: str | None) -> str:
    """sha256 hex digest of the canonical URL (``""`` URL hashes the empty string)."""
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()


def normalize_title(title: str | None, source: str | None = None) -> str:
    """Lowercase, de-HTML, strip URLs/@mentions/publisher suffixes/punctuation and collapse whitespace."""
    if not title:
        return ""
    text = html.unescape(unicodedata.normalize("NFKC", title)).lower().strip()
    text = _RETWEET_PREFIX.sub("", text)
    text = _URL_IN_TEXT.sub(" ", text)
    text = _MENTION.sub(" ", text)
    own = ""
    if source:
        own = _PUNCT.sub(" ", source.lower().split(":")[-1]).strip()
    for _ in range(2):  # at most two stacked suffixes, e.g. "Title - Foo | CoinDesk"
        text = text.strip()
        match = _SEPARATOR_SUFFIX.search(text)
        if not match:
            break
        sep, suffix = match.group(1), _SPACES.sub(" ", _PUNCT.sub(" ", match.group(2))).strip()
        words = suffix.split()
        known = suffix in _KNOWN_SOURCE_SUFFIXES or (own and suffix == own)
        if known or (sep in "|•·" and 0 < len(words) <= 3):
            text = text[: match.start()]
        else:
            break
    text = text.replace("'", "").replace("’", "")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def title_fingerprint(title: str | None, source: str | None = None) -> str:
    """sha256 hex digest of the normalised title (``""`` if the title normalises to nothing)."""
    norm = normalize_title(title, source)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest() if norm else ""


def title_tokens(normalized_title: str) -> tuple[str, ...]:
    """Content tokens of an already normalised title (stopwords removed unless that leaves nothing)."""
    words = normalized_title.split()
    content = tuple(w for w in words if w not in STOPWORDS)
    return content or tuple(words)


def token_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """max(token-set Jaccard, token-level difflib ratio) in [0, 1].

    The difflib ratio runs on token lists (not characters) so a single flipped verb
    ("approves" vs "rejects") in a short headline does not read as 'almost identical'.
    """
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    jaccard = len(sa & sb) / len(sa | sb)
    if jaccard >= 1.0 and len(a) == len(b):
        return 1.0
    ratio = SequenceMatcher(None, list(a), list(b), autojunk=False).ratio()
    return max(jaccard, ratio)


def title_similarity(a: str | None, b: str | None) -> float:
    """Similarity of two raw titles in [0, 1] (see :func:`token_similarity`)."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return token_similarity(title_tokens(na), title_tokens(nb))


def make_news_id(url: str | None, title: str | None, source: str) -> str:
    """Stable id: sha256 of the canonical URL, else of normalised title + source."""
    if url and canonical_url(url):
        return url_hash(url)
    key = f"t:{normalize_title(title, source)}|{source}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
