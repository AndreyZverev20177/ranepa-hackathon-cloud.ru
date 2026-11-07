"""Stage 03 tests. Model inference is faked; no Internet / GPU is required."""
import contextlib
from datetime import date
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud_digest.embedding_cli import build_embedding_export, main
from cloud_digest.embeddings import (
    EmbeddingConfig, EmbeddingError, EmbeddingService, cosine_similarity,
)
from cloud_digest.preprocessing import prepare_publications


def row(message_id, text, *, day='2026-08-31T09:00:00+03:00'):
    return {
        'message_id': message_id, 'published_at': day,
        'source_name': 'Источник', 'source_type': 'СМИ',
        'url': 'https://example.org/' + message_id,
        'text': text, 'audience': 1000, 'views': 100,
        'likes': 0, 'comments': 0, 'reposts': 0,
        'ba_sentiment': 'neutral', 'ba_topic_hit': None,
    }


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[len(text) % 13 + 1, sum(map(ord, text)) % 17 + 1, 2.0] for text in texts]


class EmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_prepare_preserves_text_and_skip_short(self):
        first = row('a', 'Cloud.ru сообщил о запуске новых функций облачной платформы')
        second = row('b', 'https://cloud.ru/no-text')
        pubs = prepare_publications([first, second])
        fake = FakeEncoder()
        actual = EmbeddingService(EmbeddingConfig(), encoder=fake).encode(pubs)
        self.assertEqual(len(actual), 2)
        self.assertIsNotNone(actual[0].vector)
        self.assertIsNone(actual[1].vector)
        self.assertEqual(actual[1].skip_reason, 'insufficient_semantic_text')
        self.assertEqual(pubs[0].raw_text, first['text'])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(fake.calls[0][0]), 1)
        self.assertNotIn('https://', fake.calls[0][0][0])

    def test_duplicate_text_in_same_batch_only_encoded_once(self):
        records = [
            row('a', 'Cloud.ru представил новые решения для малого бизнеса'),
            row('b', 'Cloud.ru представил новые решения для малого бизнеса'),
        ]
        fake = FakeEncoder()
        out = EmbeddingService(encoder=fake).encode(prepare_publications(records))
        self.assertEqual(len(fake.calls[0][0]), 1)
        self.assertEqual(out[0].vector, out[1].vector)
        self.assertFalse(out[0].from_cache)
        self.assertTrue(out[1].from_cache)

    def test_disk_cache_reused_without_loading_model(self):
        docs = prepare_publications([row('a', 'Cloud.ru сообщил о новых возможностях платформы')])
        cfg = EmbeddingConfig(cache_dir=self.root / 'cache')
        first = EmbeddingService(cfg, encoder=FakeEncoder()).encode(docs)
        second = EmbeddingService(cfg).encode(docs)
        self.assertEqual(first[0].vector, second[0].vector)
        self.assertTrue(second[0].from_cache)
        self.assertEqual(len(list((self.root / 'cache').rglob('*.json'))), 1)

    def test_revision_and_model_change_invalidate_cache(self):
        docs = prepare_publications([row('a', 'Cloud.ru сообщил о новых возможностях платформы')])
        a = EmbeddingConfig(model_name='demo', revision='rev-a', cache_dir=self.root / 'cache')
        b = EmbeddingConfig(model_name='demo', revision='rev-b', cache_dir=self.root / 'cache')
        first = EmbeddingService(a, encoder=FakeEncoder()).encode(docs)
        fake = FakeEncoder()
        second = EmbeddingService(b, encoder=fake).encode(docs)
        self.assertFalse(first[0].from_cache)
        self.assertFalse(second[0].from_cache)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(list((self.root / 'cache').rglob('*.json'))), 2)

    def test_corrupted_cache_recomputed(self):
        docs = prepare_publications([row('a', 'Cloud.ru сообщил о новых возможностях платформы')])
        cfg = EmbeddingConfig(cache_dir=self.root / 'cache')
        EmbeddingService(cfg, encoder=FakeEncoder()).encode(docs)
        cache_file = next((self.root / 'cache').rglob('*.json'))
        cache_file.write_text('broken json', encoding='utf-8')
        fake = FakeEncoder()
        result = EmbeddingService(cfg, encoder=fake).encode(docs)
        self.assertFalse(result[0].from_cache)
        self.assertEqual(len(fake.calls), 1)

    def test_batch_size_controls_encoder_invocations(self):
        docs = prepare_publications([
            row(str(i), f'Cloud.ru сообщил о новой уникальной услуге номер {i}')
            for i in range(5)
        ])
        fake = FakeEncoder()
        actual = EmbeddingService(EmbeddingConfig(batch_size=2), encoder=fake).encode(docs)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(len(actual), 5)
        for _, kwargs in fake.calls:
            self.assertTrue(kwargs['normalize_embeddings'])
            self.assertTrue(kwargs['convert_to_numpy'])

    def test_cosine_identity_and_orthogonal_and_errors(self):
        self.assertAlmostEqual(cosine_similarity((3.0, 4.0), (3.0, 4.0)), 1.0)
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (0.0, 1.0)), 0.0)
        with self.assertRaises(EmbeddingError):
            cosine_similarity((1.0,), (1.0, 2.0))
        with self.assertRaises(EmbeddingError):
            cosine_similarity((0.0, 0.0), (1.0, 1.0))
        with self.assertRaises(EmbeddingError):
            cosine_similarity((float('nan'), 0.0), (1.0, 0.0))

    def test_bad_config_and_zero_vector(self):
        for bad in (0, -3, True):
            with self.subTest(bad=bad), self.assertRaises(EmbeddingError):
                EmbeddingConfig(batch_size=bad)
        class BadEncoder:
            def encode(self, texts, **kwargs):
                return [[0, 0, 0] for _ in texts]
        docs = prepare_publications([row('a', 'Cloud.ru сообщил о новых возможностях платформы')])
        with self.assertRaises(EmbeddingError):
            EmbeddingService(encoder=BadEncoder()).encode(docs)

    def test_empty_input_does_not_load_model(self):
        results = EmbeddingService(EmbeddingConfig()).encode([])
        self.assertEqual(results, [])

    def test_export_filter_date_and_brand(self):
        rows = [
            row('a', 'Cloud.ru представил новую версию облачной платформы'),
            row('b', 'Другой провайдер представил новую версию облачной платформы'),
            row('c', 'Cloud.ru сообщил о запуске новой платформы для компаний', day='2026-09-01T09:00:00+03:00'),
        ]
        service = EmbeddingService(encoder=FakeEncoder())
        result = build_embedding_export(rows, date(2026, 8, 31), service)
        self.assertEqual(result['counts']['input_records'], 3)
        self.assertEqual(result['counts']['day_records'], 2)
        self.assertEqual(result['counts']['selected_records'], 1)
        self.assertEqual(len(result['items']), 1)
        self.assertEqual(result['items'][0]['message_id'], 'a')
        self.assertNotIn('raw_text', result['items'][0])
        self.assertEqual(build_embedding_export(rows, date(2026, 8, 31), service, brand_only=False)['counts']['selected_records'], 2)

    def test_cli_creates_embedding_json_with_fake_encoder(self):
        source = self.root / 'input.json'
        output = self.root / 'vectors.json'
        rows = [row('a', 'Cloud.ru объявил о новых продуктах для бизнеса'),
                row('b', 'Компания представила новые продукты для бизнеса')]
        source.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        from cloud_digest.embedding_cli import EmbeddingService as RealService

        def fake_service(config):
            return RealService(config, encoder=FakeEncoder())

        with patch('cloud_digest.embedding_cli.EmbeddingService', side_effect=fake_service):
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(['--input', str(source), '--date', '2026-08-31',
                           '--output', str(output), '--cache-dir', str(self.root / 'cache')])
        self.assertEqual(rc, 0)
        payload = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(payload['embedding_schema'], 'cloud-digest-embeddings-v1')
        self.assertEqual(payload['counts']['selected_records'], 1)
        self.assertEqual(payload['counts']['embedded_records'], 1)
        self.assertEqual(len(payload['items'][0]['vector']), 3)
        self.assertEqual(len(json.loads(source.read_text(encoding='utf-8'))), 2)

    def test_cli_does_not_overwrite_input_or_baseline(self):
        source = self.root / 'input.json'
        source.write_text(json.dumps([row('a', 'Cloud.ru сообщил о запуске новой платформы для компаний')]), encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(['--input', str(source), '--date', '2026-08-31',
                         '--output', str(source)])
        self.assertEqual(code, 2)
        self.assertEqual(len(json.loads(source.read_text(encoding='utf-8'))), 1)


if __name__ == '__main__':
    unittest.main()
