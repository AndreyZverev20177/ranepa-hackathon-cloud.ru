"""Подготовка данных для дальнейшей кластеризации без изменения baseline-v1.

Важно: исходный текст публикации не перезаписывается. Существующие функции
normalized(), brand() и brand_context() сохранены для совместимости.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
import re
from typing import Any, Iterable
from urllib.parse import urlsplit

from .validation import timestamp

# Сохраняем исходную нормализацию: ее использует старый story_key().
def normalized(text: str) -> str:
    return ' '.join(text.casefold().split())


def brand(text: str) -> tuple[str, str] | None:
    patterns = [
        ('direct', r'(?<![\w.-])(?:cloud[. _-]?ru|клауд[. _-]?ру|клаудру|cloudru)(?![\w-]|\.[\w-])'),
        ('historical', r'(?<!\w)(?:sber[ -]?cloud|сбер[ -]?клауд)(?!\w)'),
        ('alias', r'(?<!\w)(?:clod\.ru|клауд\.руу|cloudру)(?!\w)'),
    ]
    for kind, pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return kind, m.group(0)
    # Product-only names are ambiguous; do not auto-accept them.
    return None


def brand_context(text: str, span: str) -> str:
    """Text local to brand; avoid treating competitor-only negative as our negative."""
    pos = text.casefold().find(span.casefold())
    if pos < 0:
        return text
    clauses = re.split(r'(?<=[.!?;])\s+|\n+', text)
    local = [c for c in clauses if span.casefold() in c.casefold()]
    return ' '.join(local) if local else text[max(0, pos-100):pos+160]


_URL = re.compile(r'(?i)\bhttps?://[^\s<>"\']+')
_HTML_TAG = re.compile(r'<[a-z][\w:-]*(?:\s[^<>]*)?/?>|</[a-z][\w:-]*\s*>|<!--', re.I)
_HIDDEN = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff\u00ad]')
_TECH_LINE = re.compile(r'(?i)^\s*(?:читать далее|продолжение(?: статьи)?|read more)\s*[.·…:»>\-]*\s*$')
_BLOCK_TAGS = frozenset({'p', 'div', 'br', 'article', 'section', 'li', 'ul', 'ol',
                         'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'tr'})
_SKIP_TAGS = frozenset({'script', 'style', 'noscript'})


class _ReadableHTML(HTMLParser):
    """Удаляет разметку, не затрагивая отображаемое содержимое тегов."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self.suppressed.append(tag)
        elif tag in _BLOCK_TAGS and not self.suppressed:
            self.parts.append('\n')

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            if self.suppressed and self.suppressed[-1] == tag:
                self.suppressed.pop()
        elif tag in _BLOCK_TAGS and not self.suppressed:
            self.parts.append('\n')

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS and not self.suppressed:
            self.parts.append('\n')

    def handle_data(self, data: str) -> None:
        if not self.suppressed:
            self.parts.append(data)


def clean_text(text: str) -> str:
    """Очищает видимый текст от HTML и технических символов, сохраняя факты.

    Ссылки намеренно остаются в этом представлении. Их удаление допускается
    только из semantic_text, а исходный текст всегда доступен в raw_text.
    """
    if _HTML_TAG.search(text):
        parser = _ReadableHTML()
        parser.feed(text)
        parser.close()
        text = ''.join(parser.parts)
    text = _HIDDEN.sub('', text).replace('\xa0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n')]
    # Удаляем только самостоятельные UI-подсказки, не тематические сведения.
    lines = [line for line in lines if not _TECH_LINE.fullmatch(line)]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def embedded_urls(text: str) -> tuple[str, ...]:
    """Ссылки из текста в порядке появления, без дубликатов."""
    found: dict[str, None] = {}
    for match in _URL.finditer(text):
        url = match.group().rstrip('.,;:!?)]}»')
        if url:
            found[url] = None
    return tuple(found)


def semantic_text(text: str) -> str:
    """Версия текста без URL для будущих embeddings; факты доступны в raw_text."""
    without_links = _URL.sub(' ', clean_text(text))
    # Не удаляем технически выглядящие слова и числовые факты: SKU, API, 503.
    return clean_text(without_links)


def _infer_title(explicit_title: str | None, readable_body: str) -> str | None:
    if explicit_title and clean_text(explicit_title):
        return clean_text(explicit_title)
    for line in readable_body.splitlines():
        line = line.strip()
        if not line or _URL.fullmatch(line):
            continue
        # Если заголовок явно не выделен, не выдаем произвольные 300 символов за title.
        if 12 <= len(line) <= 160:
            return line
        break
    return None


@dataclass(frozen=True, slots=True)
class PreparedPublication:
    """Нормализованное представление, не заменяющее исходную запись.

    published_at_utc используется для сравнения событий во времени.
    semantic_text предназначен только для следующего этапа (embeddings).
    """

    message_id: str
    published_at_utc: datetime
    source_name: str
    source_type: str
    source_url: str
    title: str | None
    raw_text: str
    clean_text: str
    semantic_text: str
    embedded_urls: tuple[str, ...]
    audience: int | None
    views: int | None
    likes: int | None
    comments: int | None
    reposts: int | None
    semantic_ready: bool


def prepare_publication(row: dict[str, Any]) -> PreparedPublication:
    """Принимает запись после validation.load() или partner.adapt()."""
    raw_text = row['text']
    readable = clean_text(raw_text)
    meaning = semantic_text(raw_text)
    title = _infer_title(row.get('title'), readable)
    # Заголовок может включать ссылки: в векторы передаём только его текст.
    title_for_semantics = semantic_text(title) if title else ''
    if title_for_semantics and normalized(title_for_semantics) not in normalized(meaning):
        meaning = f'{title_for_semantics}\n{meaning}'.strip()
    # URL-only сообщения не создают фиктивное семантическое событие.
    usable = len(meaning.split()) >= 5
    return PreparedPublication(
        message_id=row['message_id'],
        published_at_utc=timestamp(row['published_at']),
        source_name=row['source_name'],
        source_type=row['source_type'],
        source_url=row['url'],
        title=title,
        raw_text=raw_text,
        clean_text=readable,
        semantic_text=meaning,
        embedded_urls=embedded_urls(raw_text),
        audience=row.get('audience'),
        views=row.get('views'),
        likes=row.get('likes'),
        comments=row.get('comments'),
        reposts=row.get('reposts'),
        semantic_ready=usable,
    )


def prepare_publications(rows: Iterable[dict[str, Any]]) -> list[PreparedPublication]:
    return [prepare_publication(row) for row in rows]
