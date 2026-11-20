"""Fetch a web page and extract its readable text, with SSRF protection.

An agent decides which URLs to fetch, and page content can try to steer it, so the fetcher refuses
anything that is not a public http(s) address: localhost, private networks, link-local and cloud
metadata addresses are blocked, and every redirect hop is checked again.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import asdict, dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from aidesk.config import get_settings
from aidesk.llm.cache import ResponseCache, cache_key

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 4
PAGE_TTL = 24 * 3600
USER_AGENT = "AIDesk/0.1 (+https://github.com/bamboomaster112/ai-engineering-kodebox-projects)"
ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class BlockedURL(ValueError):
    pass


class FetchError(RuntimeError):
    pass


@dataclass
class Page:
    url: str
    final_url: str
    title: str
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


def check_url(url: str) -> str:
    """Raise BlockedURL unless url is http(s) on port 80/443 and every address it resolves to is public."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise BlockedURL(f"only http and https URLs are allowed, not {parts.scheme or 'none'}")
    if not parts.hostname:
        raise BlockedURL("URL has no host")
    if parts.username or parts.password:
        raise BlockedURL("URLs with credentials are not allowed")
    if parts.port not in (None, 80, 443):
        raise BlockedURL("only the standard ports 80 and 443 are allowed")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as err:
        raise BlockedURL(f"cannot resolve {parts.hostname}") from err
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if not addr.is_global or addr.is_multicast:
            raise BlockedURL(f"{parts.hostname} resolves to a non-public address ({addr})")
    return url


def _extract(html: str, url: str) -> tuple[str, str]:
    import trafilatura

    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True,
                               favor_precision=True) or ""
    title = ""
    meta = trafilatura.extract_metadata(html)
    if meta and meta.title:
        title = meta.title
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        title = " ".join(m.group(1).split()) if m else url
    if not text:
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    return title.strip()[:200], " ".join(text.split())


def _fetch_live(url: str) -> Page:
    current = url
    with httpx.Client(follow_redirects=False, timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            check_url(current)
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location", "")
                    if not location:
                        raise FetchError("redirect without a location")
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 400:
                    raise FetchError(f"HTTP {resp.status_code} from {current}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and ctype not in ALLOWED_TYPES:
                    raise FetchError(f"unsupported content type {ctype}")
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        break
                html = body.decode(resp.encoding or "utf-8", errors="replace")
                title, text = _extract(html, current) if ctype != "text/plain" else (current, html)
                return Page(url=url, final_url=current, title=title, text=text)
    raise FetchError("too many redirects")


def _fetch_fixture(url: str) -> Page:
    from aidesk.web.search import load_fixtures

    pages = load_fixtures()["pages"]
    if url not in pages:
        raise FetchError(f"no fixture page for {url}")
    p = pages[url]
    return Page(url=url, final_url=url, title=p["title"], text=p["text"])


def fetch_page(url: str, max_chars: int = 8000) -> Page:
    settings = get_settings()
    url = (url or "").strip()
    if settings.search_backend == "fixture":
        page = _fetch_fixture(url)
    else:
        cache = ResponseCache(settings.db_path)
        key = cache_key("page", {"url": url})
        hit = cache.get(key, max_age=PAGE_TTL)
        if hit is not None:
            page = Page(**hit)
        else:
            page = _fetch_live(url)
            cache.put(key, page.as_dict())
    page.text = page.text[:max_chars]
    return page
