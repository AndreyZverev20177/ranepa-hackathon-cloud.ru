"""Stage 04 synthetic regression tests. No model download or internet access."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud_digest.clustering import SemanticClusterConfig, cluster_semantically
from cloud_digest.digest import digest
from cloud_digest.embeddings import EmbeddedPublication, EmbeddingService
from cloud_digest.event_facts import extract_event_facts, identity_conflicts
from cloud_digest.internet_candidates import (InternetCandidateError, fetch_internet_candidates,
                                               google_news_rss_url)
from cloud_digest.candidate_cli import build_candidate_report, main as candidate_main
from cloud_digest.preprocessing import prepare_publications


def row(mid, text, *, time='2026-08-31T09:00:00+03:00', title=None):
    return dict(message_id=mid, published_at=time, source_name='Тестовое СМИ',
                source_type='Онлайн-СМИ', url=f'https://example.org/{mid}',
                title=title, text=text, audience=None, views=None, likes=None,
                comments=None, reposts=None, ba_sentiment=None, ba_topic_hit=None)


def clust(rows, vecs, **kwargs):
    pubs = prepare_publications(rows)
    embedded = [EmbeddedPublication(p.message_id, '', tuple(v) if v is not None else None, False)
                for p, v in zip(pubs, vecs)]
    return cluster_semantically(pubs, embedded, SemanticClusterConfig(**kwargs))


class StaticEncoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return [next(vector for needle, vector in self.vectors.items() if needle in text)
                for text in texts]


class SemanticClusteringTests(unittest.TestCase):
    def test_high_similarity_merges_finance_paraphrases(self):
        rows = [row('a', 'Cloud.ru сообщил, что выручка от ИИ за I полугодие 2026 увеличилась.'),
                row('b', 'Доходы Cloud.ru от решений ИИ за первое полугодие 2026 существенно выросли.')]
        result = clust(rows, [(1, 0), (0.94, 0.34)])
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].message_ids, ('a', 'b'))

    def test_borderline_pair_requires_review(self):
        rows = [row('a', 'Cloud.ru объявил об обновлении виртуальной платформы'),
                row('b', 'Компания Cloud.ru представила новую версию платформы')]
        result = clust(rows, [(1, 0), (0.83, 0.558)])
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(result.review_pairs[0].reason, 'borderline_similarity')

    def test_below_low_not_reviewed(self):
        rows = [row('a', 'Cloud.ru представил новый функционал'),
                row('b', 'Cloud.ru выпустил другие обновления')]
        result = clust(rows, [(1, 0), (0.1, 0.995)])
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(result.review_pairs, ())

    def test_finance_announcement_and_written_quarter_period(self):
        rows = [row('a', 'Cloud.ru представил финансовые результаты за второй квартал 2026 года.'),
                row('b', 'Cloud.ru сообщил о выручке за I квартал 2026 года.')]
        result = clust(rows, [(1,0),(1,0)])
        self.assertEqual(len(result.clusters), 2)
        self.assertIn('different_reporting_period', result.review_pairs[0].reason)

    def test_financial_period_blocked_at_similarity_one(self):
        rows = [row('a', 'Выручка Cloud.ru за I полугодие 2026 составила 10 млрд рублей'),
                row('b', 'Выручка Cloud.ru за II полугодие 2026 составила 10 млрд рублей')]
        result = clust(rows, [(1, 0), (1, 0)])
        self.assertEqual(len(result.clusters), 2)
        self.assertIn('different_reporting_period', result.review_pairs[0].reason)

    def test_distinct_outage_services_blocked(self):
        rows = [row('a', 'В Cloud.ru произошёл сбой объектного хранилища S3'),
                row('b', 'В Cloud.ru произошёл сбой виртуальных машин')]
        result = clust(rows, [(1, 0), (1, 0)])
        self.assertEqual(len(result.clusters), 2)
        self.assertIn('different_incident_service', result.review_pairs[0].reason)

    def test_distinct_incident_time_blocked(self):
        rows = [row('a', 'Сбой API в Cloud.ru: пользователи не получают ответы'),
                row('b', 'Сбой API в Cloud.ru: пользователи не получают ответы',
                    time='2026-09-02T10:00:00+03:00')]
        result = clust(rows, [(1, 0), (1, 0)])
        self.assertEqual(len(result.clusters), 2)
        self.assertIn('independent_incident_time', result.review_pairs[0].reason)

    def test_outage_confirmation_and_denial_block(self):
        rows = [row('a', 'Cloud.ru подтвердил сбой и недоступность API для клиентов'),
                row('b', 'Cloud.ru опроверг сообщения о сбое и недоступности API')]
        result = clust(rows, [(1, 0), (1, 0)])
        self.assertEqual(len(result.clusters), 2)
        self.assertIn('contradictory_incident_status', result.review_pairs[0].reason)

    def test_different_event_types_blocked(self):
        rows = [row('a', 'Cloud.ru сообщил об увеличении выручки за первое полугодие'),
                row('b', 'Cloud.ru сообщил о запуске новой облачной платформы')]
        result = clust(rows, [(1, 0), (1, 0)])
        self.assertEqual(len(result.clusters), 2)
        self.assertIn('different_event_type', result.review_pairs[0].reason)

    def test_complete_link_prevents_transitive_merge(self):
        rows = [row('a', 'Cloud.ru платформа получила обновление пакета Alpha'),
                row('b', 'Cloud.ru платформа получила обновление пакета Beta'),
                row('c', 'Cloud.ru платформа получила обновление пакета Gamma')]
        # cos(25deg)>0.88, cos(50deg)<0.78
        import math
        angles = [math.radians(d) for d in (0, 25, 50)]
        result = clust(rows, [(math.cos(a), math.sin(a)) for a in angles])
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(result.clusters[0].message_ids, ('a', 'b'))
        self.assertEqual(result.clusters[1].message_ids, ('c',))

    def test_without_vector_only_exact_text(self):
        rows = [row('a', 'Cloud.ru сообщил о новых сервисах для бизнеса'),
                row('b', 'Cloud.ru сообщил о новых сервисах для бизнеса'),
                row('c', 'Cloud.ru сообщил о новых продуктах для бизнеса')]
        result = clust(rows, [None, None, None])
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(result.skipped_without_vector, 3)

    def test_stable_order_across_input_shuffle(self):
        rows = [row('a', 'Cloud.ru обновил технологическую платформу'),
                row('b', 'Cloud.ru объявил об обновлении инфраструктуры')]
        a = clust(rows, [(1, 0), (0.95, 0.312)])
        b = clust(rows[::-1], [(0.95, 0.312), (1, 0)])
        self.assertEqual(a.clusters, b.clusters)

    def test_invalid_config(self):
        for kwargs in (dict(low_threshold=.9, high_threshold=.8),
                       dict(low_threshold=float('nan')), dict(max_gap_days=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SemanticClusterConfig(**kwargs)

    def test_semantic_digest_optin_changes_clusters_not_scores(self):
        rows = [row('a', 'Cloud.ru сообщил о выручке за первое полугодие 2026 года'),
                row('b', 'Cloud.ru опубликовал финансовые итоги за I полугодие 2026 года')]
        encoder = StaticEncoder({'сообщил': [1,0], 'опубликовал':[.95,.312]})
        service = EmbeddingService(encoder=encoder)
        before = digest(rows, date(2026,8,31))
        after = digest(rows, date(2026,8,31), semantic=True, embedding_service=service)
        self.assertEqual(before['rule_version'], 'baseline-v1')
        self.assertEqual(after['rule_version'], 'semantic-clustering-v1')
        self.assertEqual(before['summary_counts']['clusters'], 2)
        self.assertEqual(after['summary_counts']['clusters'], 1)
        self.assertEqual(after['items'][0]['cluster_size'], 2)
        self.assertEqual(after['items'][0]['importance_score'], before['items'][0]['importance_score'])
        self.assertEqual(after['semantic_clustering']['clusters'][0]['message_ids'], ['a','b'])

    def test_review_queue_updates_cluster_id(self):
        rows = [row('a', 'Cloud.ru сообщил о выручке за первое полугодие 2026 года'),
                row('b', 'Cloud.ru опубликовал финансовые итоги за I полугодие 2026 года')]
        rows[1]['ba_sentiment'] = 'negative'
        service = EmbeddingService(encoder=StaticEncoder({'сообщил':[1,0], 'опубликовал':[1,0]}))
        report = digest(rows, date(2026, 8, 31), semantic=True, embedding_service=service)
        self.assertEqual(report['review_queue'][0]['cluster_id'], report['items'][0]['cluster_id'])


class InternetCandidatesTests(unittest.TestCase):
    FEED = b'''<?xml version="1.0"?><rss version="2.0"><channel><title>News</title>
    <item><title>Cloud.ru \xd0\xbe\xd0\xb1\xd0\xbd\xd0\xbe\xd0\xb2\xd0\xb8\xd0\xbb \xd0\xbf\xd0\xbb\xd0\xb0\xd1\x82\xd1\x84\xd0\xbe\xd1\x80\xd0\xbc\xd1\x83</title>
    <description>Cloud.ru \xd1\x81\xd0\xbe\xd0\xbe\xd0\xb1\xd1\x89\xd0\xb8\xd0\xbb \xd0\xbe \xd0\xbd\xd0\xbe\xd0\xb2\xd1\x8b\xd1\x85 \xd1\x84\xd1\x83\xd0\xbd\xd0\xba\xd1\x86\xd0\xb8\xd1\x8f\xd1\x85</description>
    <link>https://news.google.com/articles/abc</link><pubDate>Mon, 31 Aug 2026 09:00:00 GMT</pubDate><source>Example News</source></item>
    <item><title>Another brand</title><link>https://news.google.com/articles/def</link>
    <pubDate>Mon, 31 Aug 2026 09:00:00 GMT</pubDate></item>
    </channel></rss>'''

    def test_google_rss_candidate_metadata_and_filter(self):
        url = google_news_rss_url('"Cloud.ru"', day=date(2026,8,31))
        self.assertIn('after%3A2026-08-31', url)
        rows = fetch_internet_candidates([url], day=date(2026,8,31), fetcher=lambda _: self.FEED)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['source_name'], 'Example News')
        self.assertIsNone(rows[0]['audience'])
        self.assertTrue(rows[0]['internet_candidate'])
        self.assertFalse(rows[0]['verified_full_text'])

    def test_internet_no_automatic_scoring(self):
        local = [row('a', 'Cloud.ru сообщил о новых функциях платформы')]
        internet = fetch_internet_candidates([google_news_rss_url('Cloud.ru')], day=date(2026,8,31),
                                             fetcher=lambda _: self.FEED)
        service = EmbeddingService(encoder=StaticEncoder({'сообщил':[1,0], 'обновил':[1,0]}))
        report = build_candidate_report(local, date(2026,8,31), service, internet_rows=internet)
        self.assertEqual(report['counts']['internet_candidates'], 1)
        self.assertTrue(any(m['review_only'] for g in report['clusters'] for m in g['members']))
        self.assertNotIn('importance_score', json.dumps(report))

    def test_network_only_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inp, out = folder/'source.json', folder/'output.json'
            inp.write_text(json.dumps([row('a','Cloud.ru запустил платформу')], ensure_ascii=False), encoding='utf-8')
            with patch('cloud_digest.candidate_cli.fetch_internet_candidates') as fetch:
                with patch('cloud_digest.candidate_cli.EmbeddingService', return_value=EmbeddingService(
                    encoder=StaticEncoder({'Cloud.ru':[1,0]}))):
                    self.assertEqual(candidate_main(['--input',str(inp),'--date','2026-08-31','--output',str(out)]), 0)
                fetch.assert_not_called()

    def test_internet_feed_url_allowlist_and_xml_safety(self):
        for url in ('http://habr.com/rss', 'https://127.0.0.1/rss', 'https://evil.test/rss'):
            with self.assertRaises(InternetCandidateError):
                fetch_internet_candidates([url], day=date(2026,8,31), fetcher=lambda _: self.FEED)
        with self.assertRaises(InternetCandidateError):
            fetch_internet_candidates([google_news_rss_url('Cloud.ru')], day=date(2026,8,31),
                                      fetcher=lambda _: b'<!DOCTYPE rss><rss/>')


if __name__ == '__main__':
    unittest.main()
