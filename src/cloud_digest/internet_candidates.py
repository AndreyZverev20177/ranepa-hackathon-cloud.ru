"""Opt-in retrieval of web *candidate* articles from public RSS feeds.

Network access happens ONLY when fetch_internet_candidates() is called. Feeds
provide metadata/snippets, not verified full article content. Never treat RSS
results as external verification of facts or as known audience/view counts.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html import unescape
import re
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET

from .config import MSK
from .preprocessing import brand, clean_text

_ALLOWED_FEED_HOSTS = frozenset({'news.google.com', 'habr.com'})
_MAX_RSS_BYTES = 1_500_000
_MAX_ITEMS_PER_FEED = 150


class InternetCandidateError(ValueError):
    pass


def google_news_rss_url(query: str, *, day: date | None = None) -> str:
    """Google News RSS search is unofficial and not a historical archive."""
    if not isinstance(query, str) or not query.strip() or len(query) > 180:
        raise InternetCandidateError('Пустой или слишком длинный запрос к RSS')
    if day is not None:
        from datetime import timedelta
        query += f' after:{day.isoformat()} before:{(day + timedelta(days=1)).isoformat()}'
    return f'https://news.google.com/rss/search?q={quote(query, safe="")}&hl=ru&gl=RU&ceid=RU:ru'


def _validate_feed_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.hostname.lower() not in _ALLOWED_FEED_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
        raise InternetCandidateError('Разрешены только HTTPS RSS с news.google.com и habr.com')


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        _validate_feed_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _download_rss(url: str, timeout: float = 12.0) -> bytes:
    _validate_feed_url(url)
    opener = build_opener(_SafeRedirect())
    req = Request(url, headers={'User-Agent': 'CloudDigestRSS/0.4 (+metadata-only)', 'Accept': 'application/rss+xml,application/atom+xml,application/xml'})
    try:
        with opener.open(req, timeout=timeout) as response:
            _validate_feed_url(response.geturl())
            data = response.read(_MAX_RSS_BYTES + 1)
    except (OSError, URLError, HTTPError) as exc:
        raise InternetCandidateError('Не удалось получить RSS: проверьте сеть и адрес') from exc
    if len(data) > _MAX_RSS_BYTES:
        raise InternetCandidateError('RSS превышает допустимый размер')
    return data


def _node_text(parent: ET.Element, name: str) -> str:
    node = parent.find(name)
    if node is None:
        return ''
    return ''.join(node.itertext()).strip()


def _read_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return None   # Never invent a timezone.
    return dt.astimezone(timezone.utc)


def _rss_items(data: bytes):
    if len(data) > _MAX_RSS_BYTES or re.search(br'<!\s*(DOCTYPE|ENTITY)\b', data, re.I):
        raise InternetCandidateError('Небезопасный или слишком большой RSS')
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise InternetCandidateError('RSS не содержит корректного XML') from exc
    if root.tag.lower().endswith('rss') or root.find('channel') is not None:
        channel = root.find('channel')
        if channel is None:
            return []
        items = []
        for entry in channel.findall('item')[:_MAX_ITEMS_PER_FEED]:
            source = _node_text(entry, 'source') or _node_text(channel, 'title') or 'RSS'
            items.append((
                _node_text(entry, 'title'), _node_text(entry, 'description'),
                _node_text(entry, 'link'), _node_text(entry, 'pubDate'), source,
            ))
        return items
    # Atom: namespace-agnostic for published/updated, title, summary and link.
    if root.tag.endswith('feed'):
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        out = []
        for entry in root.findall('a:entry', ns)[:_MAX_ITEMS_PER_FEED]:
            link_node = next((n for n in entry.findall('a:link', ns) if n.get('rel', 'alternate') == 'alternate'), None)
            link = link_node.get('href', '') if link_node is not None else ''
            out.append((_node_text(entry, 'a:title'.replace('a:', '{http://www.w3.org/2005/Atom}')),
                        _node_text(entry, '{http://www.w3.org/2005/Atom}summary'), link,
                        _node_text(entry, '{http://www.w3.org/2005/Atom}published') or _node_text(entry, '{http://www.w3.org/2005/Atom}updated'),
                        _node_text(entry, '{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name') or 'Atom'))
        return out
    raise InternetCandidateError('Неизвестный формат RSS/Atom')


def fetch_internet_candidates(
    urls: list[str], *, day: date, max_records: int = 100,
    fetcher: Callable[[str], bytes] | None = None,
) -> list[dict]:
    """Retrieve RSS records for a date in Moscow. Metadata-only and opt-in.

    Returns rows with 'internet_candidate' marker, unknown popularity metrics,
    and observed feed publication dates. Ignores undated and irrelevant entries.
    """
    if type(max_records) is not int or not 1 <= max_records <= 500:
        raise InternetCandidateError('max_records должен быть от 1 до 500')
    fetcher = fetcher or _download_rss
    rows: dict[str, dict] = {}
    for url in urls:
        _validate_feed_url(url)
        body = fetcher(url)
        for title, description, link, published, source in _rss_items(body):
            dt = _read_date(published)
            if dt is None or dt.astimezone(MSK).date() != day:
                continue
            if not link.startswith('https://') and not link.startswith('http://'):
                continue
            # The title/snippet can contain formatting and third-party markup.
            title = clean_text(unescape(title))
            snippet = clean_text(unescape(description))
            content = '\n'.join(part for part in (title, snippet) if part).strip()
            if not content or brand(content) is None:
                continue
            ident = 'rss:' + sha256(link.encode('utf-8')).hexdigest()[:24]
            if ident in rows:
                continue
            rows[ident] = dict(
                message_id=ident, published_at=dt.isoformat(),
                source_type='RSS', source_name=source[:200], url=link,
                text=content, title=title or None,
                audience=None, views=None, likes=None, comments=None, reposts=None,
                ba_sentiment=None, ba_topic_hit=None, language='', geo='',
                internet_candidate=True,
                provenance='rss_metadata_only', verified_full_text=False,
            )
            if len(rows) >= max_records:
                return list(rows.values())
    return list(rows.values())
