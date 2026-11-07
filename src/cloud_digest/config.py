"""Константы baseline-v1. Не меняйте значения при рефакторинге."""
from datetime import timedelta, timezone

REQUIRED = ('message_id', 'published_at', 'source_type', 'source_name', 'url', 'text')
STRINGS = ('author_name', 'author_id', 'language', 'geo', 'ba_topic_hit', 'title')
METRICS = ('audience', 'views', 'likes', 'comments', 'reposts')
KNOWN = set(REQUIRED + STRINGS + METRICS + ('ba_sentiment',))
MSK = timezone(timedelta(hours=3))
EMPTY = 'Прямого или косвенного упоминания Cloud.ru не найдено'
SYNTHETIC = 'СИНТЕТИЧЕСКИЙ ПРИМЕР — ВЫМЫСЕЛ'
BA = {'negative': 'negative', '-': 'negative', '−': 'negative', 'neutral': 'neutral', '0': 'neutral', 'positive': 'positive', '+': 'positive'}


