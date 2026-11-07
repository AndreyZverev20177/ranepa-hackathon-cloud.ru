"""Этап 2: подготовка данных, не изменяющая поведение baseline-v1."""
import json
from datetime import date, timezone
from pathlib import Path
import tempfile
import unittest

from cloud_digest.digest import digest
from cloud_digest.partner import adapt
from cloud_digest.preprocessing import (
    PreparedPublication, clean_text, embedded_urls, prepare_publication,
    prepare_publications, semantic_text,
)
from cloud_digest.validation import InputError, load, validate_records


def row(**overrides):
    result = dict(
        message_id='x', published_at='2026-08-31T12:00:00+03:00',
        source_type='СМИ', source_name='Пример', url='https://example.org/news',
        text='Cloud.ru: выручка 19,3 млрд рублей', audience=10000,
        views=1500, likes=15, comments=2, reposts=4,
    )
    result.update(overrides)
    return result


class PrepareTests(unittest.TestCase):
    def test_html_and_readable_facts(self):
        raw = ('<h2>Cloud.ru: результаты</h2><p>Выручка — 19,3 млрд &amp; 51,3%.'
               ' <a href="https://example.org">Отчёт</a></p>'
               '<script>скрытый код</script><style>.ad{display:none}</style>')
        cleaned = clean_text(raw)
        self.assertIn('Cloud.ru: результаты', cleaned)
        self.assertIn('19,3 млрд & 51,3%', cleaned)
        self.assertIn('Отчёт', cleaned)
        self.assertNotIn('<h2>', cleaned)
        self.assertNotIn('скрытый код', cleaned)
        self.assertNotIn('display:none', cleaned)

    def test_urls_preserved_separately_and_removed_only_for_semantics(self):
        raw = ('Cloud.ru рост на 51,3%.\nhttps://cloud.ru/a?utm_source=spam\n'
               'Читать далее...\nhttps://cloud.ru/a?utm_source=spam')
        clean = clean_text(raw)
        self.assertIn('https://cloud.ru/a', clean)
        self.assertNotIn('Читать далее', clean)
        self.assertEqual(embedded_urls(raw), ('https://cloud.ru/a?utm_source=spam',))
        semantic = semantic_text(raw)
        self.assertIn('51,3%', semantic)
        self.assertNotIn('http', semantic)
        self.assertNotIn('utm_source', semantic)

    def test_prepared_publication_fields_and_raw_preserved(self):
        raw = 'Cloud.ru\u200b сообщает о 503. https://cloud.ru/news?id=7'
        source = row(text=raw, title='Новости облачного провайдера')
        prepared = prepare_publication(source)
        self.assertIsInstance(prepared, PreparedPublication)
        self.assertEqual(prepared.message_id, 'x')
        self.assertEqual(prepared.raw_text, raw)
        self.assertEqual(source['text'], raw)
        self.assertEqual(prepared.title, 'Новости облачного провайдера')
        self.assertEqual(prepared.audience, 10000)
        self.assertEqual(prepared.views, 1500)
        self.assertEqual(prepared.published_at_utc.tzinfo, timezone.utc)
        self.assertIn('503', prepared.semantic_text)
        self.assertIn('Cloud.ru сообщает', prepared.clean_text)
        self.assertEqual(prepare_publications([source]), [prepared])

    def test_inferred_title_does_not_reintroduce_tracking_url(self):
        prepared = prepare_publication(row(text='Cloud.ru рост 51,3% https://cloud.ru/x?utm_source=1'))
        self.assertNotIn('https://', prepared.semantic_text)
        self.assertIn('51,3%', prepared.semantic_text)

    def test_url_only_not_semantically_ready(self):
        prepared = prepare_publication(row(text='https://cloud.ru/news/123', title=None))
        self.assertFalse(prepared.semantic_ready)
        self.assertEqual(prepared.embedded_urls, ('https://cloud.ru/news/123',))
        self.assertEqual(prepared.semantic_text, '')

    def test_validated_json_csv_and_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            json_path = folder / 'input.json'
            json_path.write_text(json.dumps([row(title='Отдельный заголовок')], ensure_ascii=False), encoding='utf-8')
            records, warnings = load(json_path)
            self.assertEqual(warnings, [])
            self.assertEqual(records[0]['title'], 'Отдельный заголовок')
            self.assertEqual(prepare_publication(records[0]).title, 'Отдельный заголовок')

    def test_validate_partner_style_rows_as_well(self):
        records, extras = validate_records([row(unknown='hidden')])
        self.assertEqual(extras, ['unknown'])
        self.assertNotIn('unknown', records[0])
        with self.assertRaisesRegex(InputError, 'audience'):
            validate_records([row(audience=-1)])
        with self.assertRaisesRegex(InputError, 'message_id'):
            validate_records([row(), row()])
        with self.assertRaisesRegex(InputError, 'title'):
            validate_records([row(title=123)])

    def test_partner_adapter_keeps_title_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'partner.json'
            payload = [{'id': 'x', 'timeCreate': 1788177600, 'url': 'https://example.org/a',
                        'text': 'Cloud.ru объявил о запуске', 'title': 'Запуск платформы',
                        'hubtype': 'СМИ', 'authorObject': {'fullname': 'Медиа'}, 'audienceCount': 20}]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            adapted = adapt(path)
            self.assertEqual(adapted[0]['title'], 'Запуск платформы')
            self.assertEqual(adapted[0]['audience'], 20)
            self.assertEqual(prepare_publication(adapted[0]).title, 'Запуск платформы')
            payload[0]['authorObject'] = ['not a mapping']
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(InputError, 'authorObject'):
                adapt(path)

    def test_partner_rejects_fabricated_identity_and_bad_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'partner.json'
            payload = {'timeCreate': 1788177600, 'url': 'https://example.org/a',
                       'text': 'Cloud.ru новость', 'authorObject': {}}
            for field, value, expected in [('id', None, 'id'), ('id', 'x', None),
                                           ('audienceCount', -7, 'audienceCount'),
                                           ('audienceCount', 8, None)]:
                payload[field] = value
                path.write_text(json.dumps([payload], ensure_ascii=False), encoding='utf-8')
                if expected:
                    with self.assertRaisesRegex(InputError, expected):
                        adapt(path)
                else:
                    adapt(path)
            path.write_text('[broken', encoding='utf-8')
            with self.assertRaisesRegex(InputError, 'JSON'):
                adapt(path)

    def test_baseline_still_uses_original_and_unchanged_score(self):
        record = row(text='Cloud.ru: отчёт', views=1000)
        result = digest(validate_records([record])[0], date(2026, 8, 31))
        self.assertEqual(result['items'][0]['importance_score'], 30)
        self.assertEqual(result['items'][0]['summary_160_300'], record['text'])
        self.assertEqual(result['rule_version'], 'baseline-v1')


if __name__ == '__main__':
    unittest.main()
