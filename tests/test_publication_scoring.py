"""СИНТЕТИЧЕСКИЕ ТЕСТЫ — не измерение достоверности источников."""
import contextlib
from datetime import date
import io
import json
from pathlib import Path
import tempfile
import unittest

from cloud_digest.cli import main
from cloud_digest.scoring import PublicationScoreWeights, score_publication_v2, card
from cloud_digest.source_authority import (
    SourceAuthorityRegistry, SourceAuthorityError, TOP25_AUTHORITY, load_scoring_rules,
)
from cloud_digest.digest import digest

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / 'config' / 'publication_scoring_v2.json'
TOP25 = ROOT / 'partner_data' / 'top25_sources.json'
DAY = date(2026, 10, 27)


def record(identifier='1', *, url='https://t.me/test/100', source_type='Мессенджеры каналы',
           audience=None, views=None, text='Cloud.ru объявил о новом сервисе без подробностей'):
    return {
        'message_id': identifier, 'published_at': '2026-10-27T12:00:00+03:00',
        'source_type': source_type, 'source_name': 'Тестовый источник', 'url': url,
        'text': text, 'audience': audience, 'views': views, 'ba_sentiment': None,
        'ba_topic_hit': None,
    }


class AuthorityAndPublicationScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_scoring_rules(RULES)
        cls.registry = SourceAuthorityRegistry.from_json(TOP25, rules=cls.rules)
        cls.weights = PublicationScoreWeights.from_settings(cls.rules['weights'])

    def test_top25_exactly_matches_ranked_rows(self):
        entries = json.loads(TOP25.read_text(encoding='utf-8'))
        self.assertEqual(len(entries), 25)
        self.assertEqual(len(TOP25_AUTHORITY), 25)
        for rank, source in enumerate(entries, 1):
            actual = self.registry.resolve('https://' + source['source'] + '/news', source['source_type'])
            self.assertEqual(actual.rank, rank)
            self.assertEqual(actual.media_weight, source['media_weight'])
            self.assertEqual(actual.basis, 'top25_heuristic')

    def test_platform_aliases_and_source_types(self):
        channel = self.registry.resolve('https://telegram.me/channel/12', 'Мессенджеры каналы')
        chat = self.registry.resolve('https://t.me/channel/12', 'Мессенджеры чаты')
        self.assertEqual(channel.rank, 1)
        self.assertEqual(chat.rank, 6)
        self.assertNotEqual(channel.score, chat.score)
        self.assertTrue(channel.platform_level)

    def test_high_authority_zero_history_is_not_zero(self):
        result = self.registry.resolve('https://www.forbes.ru/news/1', 'Онлайн-СМИ')
        self.assertEqual(result.rank, 20)
        self.assertEqual(result.media_weight, 0)
        self.assertGreaterEqual(result.score, 80)
        item = card(record(url='https://forbes.ru/123', source_type='Онлайн-СМИ'), ('direct', 'Cloud.ru'),
                    scoring='publication-v2', source_registry=self.registry)
        self.assertGreater(item['importance_score'], 0)
        self.assertEqual(item['source_top25_media_weight'], 0)

    def test_unknown_source_type_fallback(self):
        source = self.registry.resolve('https://newspaper.invalid/1', 'Онлайн-СМИ')
        self.assertIsNone(source.rank)
        self.assertIsNone(source.media_weight)
        self.assertEqual(source.score, self.rules['fallback_by_type']['Онлайн-СМИ'])
        self.assertEqual(source.basis, 'type_heuristic')

    def test_known_external_domain_fallback(self):
        source = self.registry.resolve('https://www.rbc.ru/technology/1', 'Онлайн-СМИ')
        self.assertIsNone(source.rank)
        self.assertEqual(source.basis, 'domain_heuristic')
        self.assertEqual(source.score, self.rules['fallback_by_domain']['rbc.ru'])
        fake = self.registry.resolve('https://fake-rbc.ru/1', 'Онлайн-СМИ')
        self.assertEqual(fake.basis, 'type_heuristic')

    def test_large_audience_outranks_small_on_same_platform(self):
        weak = record(audience=1000, views=100)
        strong = record('2', audience=1_400_000, views=100_000)
        score1, _, _ = score_publication_v2(weak, False, False, self.registry, self.weights)
        score2, _, _ = score_publication_v2(strong, False, False, self.registry, self.weights)
        self.assertGreater(score2, score1)

    def test_score_breakdown_and_cap(self):
        item = record(audience=20_000_000, views=20_000_000)
        score, explanation, meta = score_publication_v2(item, True, True, self.registry, self.weights)
        self.assertEqual(score, sum(meta['publication_score_breakdown'].values()))
        self.assertLessEqual(score, 100)
        self.assertTrue(any('authority_basis' in detail for detail in explanation))
        self.assertEqual(meta['publication_score'], score)

    def test_missing_metrics_reported_explicitly(self):
        item = card(record(), ('direct', 'Cloud.ru'), scoring='publication-v2', source_registry=self.registry)
        self.assertIn('audience: нет данных', item['importance_explanation'])
        self.assertIn('views: нет данных', item['importance_explanation'])
        self.assertEqual(item['source_authority_scope'], 'platform')

    def test_baseline_card_unmodified(self):
        row = record(audience=10000, views=1000)
        expected = card(row, ('direct', 'Cloud.ru'))
        self.assertEqual(expected['importance_score'], 30)
        self.assertEqual(expected['importance_explanation'], ['audience >= 10000: +20', 'views >= 1000: +10'])
        self.assertNotIn('source_authority_score', expected)
        self.assertEqual(expected, card(row, ('direct', 'Cloud.ru'), scoring='baseline'))

    def test_digest_v2_has_separate_version_and_baseline_stays_old(self):
        rows = [record()]
        base = digest(rows, DAY)
        new = digest(rows, DAY, scoring='publication-v2', source_registry=self.registry,
                     scoring_weights=self.weights)
        self.assertEqual(base['rule_version'], 'baseline-v1')
        self.assertEqual(new['rule_version'], 'publication-scoring-v2')
        self.assertNotIn('source_top25_rank', base['items'][0])
        self.assertIn('source_top25_rank', new['items'][0])

    def test_bad_top25_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'top25_sources.json'
            path.write_text('[{}]', encoding='utf-8')
            with self.assertRaises(SourceAuthorityError):
                SourceAuthorityRegistry.from_json(path, rules=self.rules)

    def test_bad_authority_override_rejected(self):
        rules = dict(self.rules)
        rules['top25_authority'] = list(rules['top25_authority'])
        rules['top25_authority'][0] = {**rules['top25_authority'][0], 'authority': 150}
        with self.assertRaises(SourceAuthorityError):
            SourceAuthorityRegistry.from_json(TOP25, rules=rules)

    def test_invalid_weight_config_rejected(self):
        invalid = dict(self.rules['weights'])
        invalid['negative'] = 60
        with self.assertRaises(ValueError):
            PublicationScoreWeights.from_settings(invalid)
        invalid['negative'] = True
        with self.assertRaises(ValueError):
            PublicationScoreWeights.from_settings(invalid)

    def test_cli_scoring_enabled_only_by_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            input_file = d / 'input.json'
            input_file.write_text(json.dumps([record()], ensure_ascii=False), encoding='utf-8')
            outdir = d / 'artifacts'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(['--input', str(input_file), '--date', str(DAY), '--output-dir', str(outdir),
                             '--scoring', 'publication-v2', '--sources-file', str(TOP25),
                             '--scoring-rules', str(RULES)])
            self.assertEqual(code, 0)
            output = json.loads((outdir / f'digest-{DAY}.json').read_text(encoding='utf-8'))
            self.assertEqual(output['rule_version'], 'publication-scoring-v2')

    def test_cli_missing_source_file_is_error_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            inp = d / 'input.json'
            inp.write_text(json.dumps([record()]), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(['--input', str(inp), '--date', str(DAY), '--output-dir', str(d / 'out'),
                             '--scoring', 'publication-v2', '--scoring-rules', str(RULES),
                             '--sources-file', str(d / 'missing.json')])
            self.assertEqual(code, 2)
            self.assertFalse((d / 'out').exists())

    def test_historical_weight_not_mislabelled_as_authority(self):
        high_history = self.registry.resolve('https://t.me/a', 'Мессенджеры каналы')
        low_history = self.registry.resolve('https://forbes.ru/a', 'Онлайн-СМИ')
        self.assertGreater(high_history.media_weight, low_history.media_weight)
        self.assertLess(high_history.score, low_history.score)


if __name__ == '__main__':
    unittest.main()
