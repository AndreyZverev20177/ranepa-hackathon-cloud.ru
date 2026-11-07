"""Conservative event identity checks for semantic clusters (stage 04).

These rules detect *explicit contradictions*, not external truth. Missing facts
are treated as unknown; fuzzy semantic similarity alone is not fact checking.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Iterable

from .preprocessing import PreparedPublication, normalized

_OUTAGE = re.compile(r'\b(?:сбой|сбо[еяим]|авари\w*|недоступ\w*|отключ\w*|отказ(?:ал[аио]?|ы|а)?|перестал[аи]?\s+работать|ошибк[аи]\s+50[234]|outage|downtime)\b', re.I)
_BREACH = re.compile(r'\b(?:утечк\w*|взлом\w*|компромет\w*|похищен\w*\s+данн\w*|data\s+breach)\b', re.I)
_FINANCE = re.compile(r'\b(?:выручк\w*|финансов\w*\s+(?:итог\w*|результат\w*)|отч[её]тност\w*|прибыл\w*|мсфо)\b', re.I)
_RELEASE = re.compile(r'\b(?:запустил\w*|запуск\w*|представил\w*|релиз\w*|анонс\w*|открыл\w*\s+доступ)\b', re.I)
_DENIAL = re.compile(r'\b(?:опроверг\w*|отрица\w*|ложн\w*\s+сообщени\w*|не\s+(?:было|происходил\w*|подтвердил\w*))\b', re.I)
_YEAR = re.compile(r'(?<!\d)(20\d{2})(?!\d)')
_QUARTER = re.compile(r'\b([1-4]|IV|III|II|I|перв\w*|втор\w*|трет\w*|четв[её]рт\w*)[- ]?(?:й|го)?\s+квартал\w*', re.I)
_HALF = re.compile(r'\b([12]|II|I|перв\w*|втор\w*)[- ]?(?:е|го|й)?\s+полугоди\w*', re.I)

_SERVICES: dict[str, tuple[str, ...]] = {
    'object_storage': (r'\bs3\b', r'объектн\w*\s+хранилищ\w*', r'object\s+storage'),
    'virtual_machines': (r'виртуальн\w*\s+машин\w*', r'virtual\s+machines?'),
    'kubernetes': (r'\bkubernetes\b', r'\bk8s\b'),
    'dns': (r'\bdns\b',),
    'api': (r'\bapi\b', r'\bапи\b'),
    'ai_factory': (r'\bai\s+factory\b', r'\bevolution\s+ai\b'),
}


@dataclass(frozen=True, slots=True)
class EventFacts:
    kind: str
    period: str | None
    services: frozenset[str]
    denied: bool
    published_at_utc: datetime


def _period(text: str) -> str | None:
    """Nearest explicitly mentioned fiscal period and year, without guessing."""
    quarter = _QUARTER.search(text)
    half = _HALF.search(text)
    period_match = min((x for x in (quarter, half) if x is not None),
                       key=lambda x: x.start(), default=None)
    if period_match is None:
        return None
    years = list(_YEAR.finditer(text))
    closest = min(years, key=lambda y: abs(y.start() - period_match.end()), default=None)
    year = closest.group(1) if closest and abs(closest.start() - period_match.end()) <= 90 else '?'
    token = period_match.group(1).casefold()
    if period_match.re is _QUARTER:
        num = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4'}.get(token)
        if num is None:
            num = ('1' if token.startswith('перв') else '2' if token.startswith('втор')
                   else '3' if token.startswith('трет') else '4' if token.startswith('четв')
                   else token)
        return f'{year}:Q{num}'
    number = '2' if token in ('2', 'ii') or token.startswith('втор') else '1'
    return f'{year}:H{number}'


def extract_event_facts(pub: PreparedPublication) -> EventFacts:
    text = normalized(' '.join(x for x in (pub.title or '', pub.clean_text) if x))
    # Only one explicit dominant class, otherwise remain unspecified.
    # A financial results *announcement* is finance, not an unrelated release.
    # Two different strong event classes make identity uncertain.
    has_outage = bool(_OUTAGE.search(text))
    has_breach = bool(_BREACH.search(text))
    has_finance = bool(_FINANCE.search(text))
    if sum((has_outage, has_breach, has_finance)) > 1:
        kind = 'unknown'
    elif has_outage:
        kind = 'outage'
    elif has_breach:
        kind = 'breach'
    elif has_finance:
        kind = 'finance'
    elif _RELEASE.search(text):
        kind = 'release'
    else:
        kind = 'unknown'
    services = frozenset(name for name, patterns in _SERVICES.items()
                         if any(re.search(pattern, text, re.I) for pattern in patterns))
    return EventFacts(kind, _period(text) if kind == 'finance' else None,
                      services, bool(_DENIAL.search(text)), pub.published_at_utc)


def identity_conflicts(a: EventFacts, b: EventFacts, *, max_gap_days: int = 7,
                       incident_gap_hours: int = 36) -> tuple[str, ...]:
    """Return explicit blockers. We never invent missing facts."""
    reasons: list[str] = []
    gap_hours = abs((a.published_at_utc - b.published_at_utc).total_seconds()) / 3600
    if gap_hours > max_gap_days * 24:
        reasons.append('outside_time_window')
    if a.kind != 'unknown' and b.kind != 'unknown' and a.kind != b.kind:
        reasons.append('different_event_type')
    if a.kind in ('outage', 'breach') or b.kind in ('outage', 'breach'):
        if gap_hours > incident_gap_hours:
            reasons.append('independent_incident_time')
        if a.services and b.services and a.services.isdisjoint(b.services):
            reasons.append('different_incident_service')
        if a.denied != b.denied:
            reasons.append('contradictory_incident_status')
    if a.kind == b.kind == 'finance' and a.period and b.period:
        year_a, period_a = a.period.split(':', 1)
        year_b, period_b = b.period.split(':', 1)
        if period_a != period_b or (year_a != '?' and year_b != '?' and year_a != year_b):
            reasons.append('different_reporting_period')
    return tuple(reasons)
