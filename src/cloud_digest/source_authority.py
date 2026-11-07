"""Справочник источников для экспериментального публикационного скоринга v2.

Важное ограничение исходной таблицы: top25_sources.json агрегирован по
домену И ТИПУ площадки (а не по конкретным редакциям и Telegram-каналам).
media_weight отражает активность в партнёрской выборке, НЕ достоверность СМИ.
Поэтому влияние по выборке и экспертные оценки авторитетности разделены.
Ни один из экспертных баллов не является подтверждённым аудитом источника.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any


class SourceAuthorityError(ValueError):
    """Ошибка загрузки или интерпретации справочника площадок."""


# Пилотная, РЕДАКТИРУЕМАЯ шкала (0–100). Ключи соответствуют 25 строкам
# top25_sources.json. Оценка Telegram/Дзена/VK относится к типу платформы,
# а НЕ подтверждает авторитетность конкретного канала или паблика.
TOP25_AUTHORITY = {
    ('telegram.org', 'Мессенджеры каналы'): 40,
    ('max.ru', 'Мессенджеры каналы'): 38,
    ('vk.com', 'Соцсети'): 40,
    ('dzen.ru', 'Блоги'): 44,
    ('instagram.com', 'Соцсети'): 35,
    ('telegram.org', 'Мессенджеры чаты'): 25,
    ('youtube.com', 'Видео'): 40,
    ('setka.ru', 'Соцсети'): 38,
    ('rutube.ru', 'Видео'): 36,
    ('mirtesen.ru', 'Соцсети'): 28,
    ('agroxxi.ru', 'Онлайн-СМИ'): 58,
    ('2ch.hk', 'Форумы'): 20,
    ('applestyle.su', 'Онлайн-СМИ'): 43,
    ('banks.cnews.ru', 'Онлайн-СМИ'): 78,
    ('biz.cnews.ru', 'Онлайн-СМИ'): 78,
    ('agro.cap.ru', 'Онлайн-СМИ'): 68,
    ('corp.cnews.ru', 'Онлайн-СМИ'): 78,
    ('disclosure.1prime.ru', 'Онлайн-СМИ'): 80,
    ('dsmedia.pro', 'Онлайн-СМИ'): 58,
    ('forbes.ru', 'Онлайн-СМИ'): 88,
    ('forpes.ru', 'Онлайн-СМИ'): 45,
    ('habr.com', 'Блоги'): 72,
    ('habr.com', 'Онлайн-СМИ'): 76,
    ('ai.cnews.ru', 'Онлайн-СМИ'): 78,
    ('cheb-news.net', 'Онлайн-СМИ'): 45,
}

# Оценки для НЕ вошедших в топ-25: предварительная эвристика, не фактчекинг.
FALLBACK_BY_SOURCE_TYPE = {
    'Онлайн-СМИ': 55,
    'Блоги': 38,
    'Мессенджеры каналы': 35,
    'Мессенджеры чаты': 22,
    'Соцсети': 32,
    'Видео': 35,
    'Форумы': 22,
    'unknown': 30,
}

# Пилотный справочник известных доменов вне топ-25 (приоритет над типом).
# Применяется к настоящему домену URL, не к произвольному source_name.
FALLBACK_DOMAIN_AUTHORITY = {
    'tass.ru': 90,
    'interfax.ru': 90,
    'rbc.ru': 85,
    'kommersant.ru': 86,
    'vedomosti.ru': 85,
    'ria.ru': 84,
    'cnews.ru': 76,
    'vc.ru': 58,
    'ixbt.com': 63,
    'rg.ru': 82,
}

# Алиасы домена платформы; используются только с совпавшим типом площадки.
HOST_ALIASES = {
    't.me': 'telegram.org',
    'telegram.me': 'telegram.org',
    'telegram.org': 'telegram.org',
    'm.vk.com': 'vk.com',
    'www.vk.com': 'vk.com',
    'vk.com': 'vk.com',
    'www.dzen.ru': 'dzen.ru',
    'dzen.ru': 'dzen.ru',
    'm.youtube.com': 'youtube.com',
    'youtu.be': 'youtube.com',
    'www.youtube.com': 'youtube.com',
}
PLATFORM_DOMAINS = frozenset({
    'telegram.org', 'max.ru', 'vk.com', 'dzen.ru', 'instagram.com',
    'youtube.com', 'setka.ru', 'rutube.ru', 'mirtesen.ru', '2ch.hk',
})


def normalize_host(url_or_host: str) -> str:
    """Получить проверенный host, не позволяя fakeforbes.ru стать forbes.ru."""
    raw = url_or_host.strip()
    parsed = urlsplit(raw if '://' in raw else 'https://' + raw)
    host = (parsed.hostname or '').lower().rstrip('.')
    if not host:
        raise SourceAuthorityError('Некорректный домен источника')
    if host.startswith('www.'):
        host = host[4:]
    if host.startswith('m.') and host[2:] in ('vk.com', 'youtube.com'):
        host = host[2:]
    return HOST_ALIASES.get(host, host)


def _domain_match(host: str, candidates: dict[str, int]) -> str | None:
    """Самый конкретный домен в справочнике, только по границе субдомена."""
    matches = (name for name in candidates if host == name or host.endswith('.' + name))
    return max(matches, key=len, default=None)


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    score: int  # 0...100, предварительная экспертная шкала
    host: str
    source_type: str
    basis: str  # top25_heuristic / domain_heuristic / type_heuristic
    rank: int | None
    media_weight: float | None
    platform_level: bool


@dataclass(frozen=True, slots=True)
class _TopEntry:
    rank: int
    media_weight: float
    score: int


class SourceAuthorityRegistry:
    """Воспроизводимый справочник: 25 строк рейтинга + правила для остальных."""

    def __init__(self, rankings: dict[tuple[str, str], _TopEntry], *,
                 fallback_types: dict[str, int] | None = None,
                 fallback_domains: dict[str, int] | None = None) -> None:
        self._rankings = dict(rankings)
        self._fallback_types = dict(FALLBACK_BY_SOURCE_TYPE if fallback_types is None else fallback_types)
        self._fallback_domains = dict(FALLBACK_DOMAIN_AUTHORITY if fallback_domains is None else fallback_domains)

    @classmethod
    def from_json(cls, path: Path, *, rules: dict[str, Any] | None = None) -> SourceAuthorityRegistry:
        """Первые 25 строк — из файла партнёра, баллы — из rules/пилотной таблицы."""
        rules = {} if rules is None else rules
        if not isinstance(rules, dict):
            raise SourceAuthorityError('Некорректные настройки авторитетности')
        authority = dict(TOP25_AUTHORITY)
        if 'top25_authority' in rules:
            entries = rules['top25_authority']
            if not isinstance(entries, list):
                raise SourceAuthorityError('top25_authority: ожидается список')
            seen: set[tuple[str, str]] = set()
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get('source'), str) or not isinstance(entry.get('source_type'), str):
                    raise SourceAuthorityError('Некорректная запись top25_authority')
                key = (normalize_host(entry['source']), entry['source_type'])
                value = entry.get('authority')
                if key in seen or key not in TOP25_AUTHORITY or type(value) is not int or not 0 <= value <= 100:
                    raise SourceAuthorityError('Некорректный или повторный балл top25_authority')
                seen.add(key)
                authority[key] = value
            if seen != set(TOP25_AUTHORITY):
                raise SourceAuthorityError('top25_authority должен содержать ровно все 25 площадок и типов')

        def scores(field: str, default: dict[str, int]) -> dict[str, int]:
            values = rules.get(field, default)
            if not isinstance(values, dict):
                raise SourceAuthorityError(f'{field}: требуется словарь')
            if any(not isinstance(k, str) or type(v) is not int or not 0 <= v <= 100 for k, v in values.items()):
                raise SourceAuthorityError(f'{field}: баллы должны быть целыми от 0 до 100')
            return dict(values)
        fallback_types = scores('fallback_by_type', FALLBACK_BY_SOURCE_TYPE)
        fallback_domains = scores('fallback_by_domain', FALLBACK_DOMAIN_AUTHORITY)
        try:
            with path.open(encoding='utf-8-sig') as stream:
                payload: Any = json.load(stream)
        except FileNotFoundError:
            raise SourceAuthorityError('Не найден top25_sources.json. Укажите --sources-file') from None
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise SourceAuthorityError('Не удалось прочитать справочник top25_sources.json') from None
        if not isinstance(payload, list) or len(payload) < 25:
            raise SourceAuthorityError('В справочнике top25_sources.json ожидается не менее 25 строк')
        rankings: dict[tuple[str, str], _TopEntry] = {}
        for rank, record in enumerate(payload[:25], 1):
            if not isinstance(record, dict):
                raise SourceAuthorityError(f'Некорректная строка рейтинга: {rank}')
            domain, source_type, media_weight = (record.get(k) for k in ('source', 'source_type', 'media_weight'))
            if not isinstance(domain, str) or not isinstance(source_type, str) or not source_type.strip():
                raise SourceAuthorityError(f'Некорректный домен/тип в строке рейтинга: {rank}')
            key = (normalize_host(domain), source_type.strip())
            if key in rankings:
                raise SourceAuthorityError(f'Повтор домена и типа в строке рейтинга: {rank}')
            if type(media_weight) not in (int, float) or not math.isfinite(media_weight) or not 0 <= media_weight <= 1:
                raise SourceAuthorityError(f'Некорректный media_weight в строке рейтинга: {rank}')
            if key not in TOP25_AUTHORITY:
                raise SourceAuthorityError(f'Нет оценки авторитетности для строки рейтинга: {rank}')
            rankings[key] = _TopEntry(rank=rank, media_weight=float(media_weight), score=authority[key])
        return cls(rankings, fallback_types=fallback_types, fallback_domains=fallback_domains)

    def resolve(self, url: str, source_type: str) -> SourceAuthority:
        host = normalize_host(url)
        key = (host, source_type)
        item = self._rankings.get(key)
        if item is not None:
            return SourceAuthority(score=item.score, host=host, source_type=source_type,
                                   basis='top25_heuristic', rank=item.rank, media_weight=item.media_weight,
                                   platform_level=host in PLATFORM_DOMAINS)
        # В случае субдомена редакции смотрим конкретный домен из top-25.
        for (known_host, known_type), ranked in self._rankings.items():
            if source_type == known_type and host.endswith('.' + known_host):
                return SourceAuthority(score=ranked.score, host=host, source_type=source_type,
                                       basis='top25_heuristic', rank=ranked.rank,
                                       media_weight=ranked.media_weight, platform_level=known_host in PLATFORM_DOMAINS)
        known = _domain_match(host, self._fallback_domains)
        if known:
            return SourceAuthority(score=self._fallback_domains[known], host=host,
                                   source_type=source_type, basis='domain_heuristic',
                                   rank=None, media_weight=None, platform_level=False)
        return SourceAuthority(score=self._fallback_types.get(source_type, 30),
                               host=host, source_type=source_type,
                               basis='type_heuristic', rank=None, media_weight=None,
                               platform_level=host in PLATFORM_DOMAINS)


def load_scoring_rules(path: Path) -> dict[str, Any]:
    """Читаем общий настраиваемый JSON для авторитетности и формулы."""
    try:
        with path.open(encoding='utf-8-sig') as stream:
            rules = json.load(stream)
    except FileNotFoundError:
        raise SourceAuthorityError('Не найден config/publication_scoring_v2.json. Укажите --scoring-rules') from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SourceAuthorityError('Некорректный JSON настроек publication-scoring-v2') from None
    if not isinstance(rules, dict):
        raise SourceAuthorityError('Настройки скоринга должны быть JSON-объектом')
    return rules
