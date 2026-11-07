"""СИНТЕТИЧЕСКИЙ ПРИМЕР — ВЫМЫСЕЛ. Тесты поведения, не оценка качества."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from cloud_digest.cli import main
from cloud_digest.core import EMPTY, SYNTHETIC, InputError, brand, digest, load, render

DAY = date(2026, 10, 27)


def row(identifier='a', text='Cloud.ru: синтетический текст', **values):
    return dict(message_id=identifier, published_at='2026-10-27T12:00:00+03:00',
                source_type='synthetic', source_name=SYNTHETIC, url='https://example.invalid/' + identifier,
                text=text, **values)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, records, name='input.json'):
        path = self.root / name
        path.write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8-sig')
        return path

    def calculate(self, records, top=30):
        return digest(load(self.write(records))[0], DAY, top)

    def cli(self, source, *extra):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(['--input', str(source), '--date', str(DAY), '--output-dir', str(self.root / 'results'), *extra])
        return code, output.getvalue(), error.getvalue()

    def test_csv_json_bom_null_determinism(self):
        records = [row(audience=None, views=1000, ba_sentiment='0', language=None)]
        json_rows, _ = load(self.write(records))
        path = self.root / 'input.csv'
        with path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        csv_rows, _ = load(path)
        self.assertEqual(json_rows, csv_rows)
        self.assertEqual(digest(json_rows, DAY), digest(csv_rows, DAY))
        self.assertEqual(digest(json_rows, DAY), digest(json_rows, DAY))

    def test_msk_boundaries(self):
        records = [row(str(i), f'Cloud.ru {i}') for i in range(4)]
        for record, time in zip(records, ['2026-10-26T20:59:59Z', '2026-10-26T21:00:00Z',
                                          '2026-10-28T01:59:59+05:00', '2026-10-27T21:00:00Z']):
            record['published_at'] = time
        self.assertEqual({x['message_id'] for x in self.calculate(records)['items']}, {'1', '2'})

    def test_invalid_fields_no_result(self):
        for field, value in [('published_at', '2026-10-27T12:00:00'), ('text', ''), ('url', '/relative'),
                             ('audience', -1), ('views', 0.5), ('likes', True), ('ba_topic_hit', True),
                             ('ba_sentiment', False), ('ba_sentiment', 0.0), ('author_id', 1)]:
            with self.subTest(field=field, value=value):
                record = row()
                record[field] = value
                code, _, error = self.cli(self.write([record]))
                self.assertEqual(code, 2)
                self.assertIn(field, error)
                if record['text']:
                    self.assertNotIn(record['text'], error)
                self.assertFalse((self.root / 'results').exists())

    def test_missing_duplicate_id_and_invalid_outside_day(self):
        missing = row()
        del missing['message_id']
        outside = row('b')
        outside.update(published_at='2020-01-01T00:00:00Z', audience=-1)
        for records in [[missing], [row(), row()], [row(), outside]]:
            self.assertEqual(self.cli(self.write(records))[0], 2)

    def test_malformed_files(self):
        for name, content in [('bad.json', ''), ('bad.json', '{}'), ('bad.json', '[bad'),
                              ('bad.csv', 'message_id,message_id\na,b'),
                              ('bad.csv', ','.join(row()) + '\na,b'),
                              ('bad.csv', ','.join(row()) + '\n"unterminated')]:
            with self.subTest(content=content):
                path = self.root / name
                path.write_text(content, encoding='utf-8')
                self.assertEqual(self.cli(path)[0], 2)

    def test_empty_cases(self):
        other_day = row()
        other_day['published_at'] = '2026-10-26T00:00:00Z'
        for records in [[], [other_day], [row(text='Чужой бренд')]]:
            result = self.calculate(records)
            self.assertEqual(result['empty_message'], EMPTY)
            self.assertEqual(result['items'], [])
            self.assertEqual(self.cli(self.write(records))[0], 0)
        path = self.root / 'empty.csv'
        path.write_text(','.join(row()) + '\n', encoding='utf-8')
        self.assertEqual(load(path)[0], [])

    def test_brand_boundaries_and_exact_span(self):
        for text in ['notcloud.ru', 'cloud.ru.evil', 'cloud.russian', 'x-cloud.ru', 'xsbercloud']:
            self.assertIsNone(brand(text), text)
        self.assertEqual(brand('ß СберКлауд и Cloud.RU.'), ('direct', 'Cloud.RU'))
        self.assertEqual(brand('Заметка про Cloud.ru.'), ('direct', 'Cloud.ru'))
        self.assertIsNone(brand('notcloud.ru/cloud.ru.evil'))
        self.assertEqual(brand('СберКлауд'), ('historical', 'СберКлауд'))
        self.assertEqual(brand('https://cloud.ru/path'), ('direct', 'cloud.ru'))

    def test_review_reasons_and_confidence(self):
        result = self.calculate([row('a', 'SberCloud: отлично, но сбой', ba_sentiment='+'),
                                 row('b', 'Чужой бренд', ba_topic_hit='cloud'),
                                 row('c', 'Cloud.ru ' + 'обычный текст ' * 20)])
        item = next(item for item in result['items'] if item['message_id'] == 'a')
        self.assertEqual(set(item['review_reasons']), {'historical_brand', 'mixed_sentiment', 'ba_sentiment_mismatch', 'short_text'})
        self.assertTrue(item['mixed'])
        self.assertEqual(item['confidence'], 0.4)
        self.assertEqual(result['review_queue'][1]['reasons'], ['topic_hit_without_brand'])
        clean = next(item for item in result['items'] if item['message_id'] == 'c')
        self.assertEqual(clean['confidence'], 0.8)
        self.assertFalse(clean['human_review_needed'])

    def test_duplicates_representative_and_own_review(self):
        first = row('a', 'Cloud.ru текст', ba_sentiment='+')
        second = row('b', ' cloud.RU   текст ', audience=10000)
        second['published_at'] = '2026-10-27T08:00:00Z'
        result = self.calculate([first, second])
        item = result['items'][0]
        self.assertEqual(item['message_id'], 'b')
        self.assertEqual(item['cluster_size'], 2)
        self.assertEqual(item['importance_score'], 20)
        self.assertEqual(len(item['cluster_id']), 64)
        self.assertIn('ba_sentiment_mismatch', result['review_queue'][0]['reasons'])
        self.assertNotIn('ba_sentiment_mismatch', result['review_queue'][1]['reasons'])

    def test_top_n_and_counts(self):
        records = [row(f'{i:02}', f'Cloud.ru текст {i}') for i in range(35)]
        result = self.calculate(records)
        self.assertEqual(len(result['items']), 30)
        self.assertEqual(result['items'][0]['message_id'], '00')
        self.assertEqual(result['items'][-1]['rank_in_digest'], 30)
        self.assertEqual(result, self.calculate(list(reversed(records))))
        counts = result['summary_counts']
        self.assertEqual(counts['clusters'], counts['selected_items'] + counts['excluded_top_n'])
        self.assertEqual(counts['brand_candidates'], counts['clusters'] + counts['duplicates_removed'])
        self.assertEqual(counts['day_records'], counts['brand_candidates'] + counts['excluded_no_brand'])

    def test_selection_before_sentiment_display(self):
        result = self.calculate([row('a', 'Cloud.ru спасибо', audience=10000, views=1000),
                                 row('b', 'Cloud.ru текст'), row('c', 'Cloud.ru сбой')], 2)
        self.assertEqual([item['message_id'] for item in result['items']], ['c', 'a'])

    def test_risk_outside_top_n(self):
        result = self.calculate([row('a', 'Cloud.ru авария A', audience=10000, views=1000),
                                 row('b', 'Cloud.ru утечка B')], 1)
        self.assertTrue(result['risk_alert'])
        self.assertEqual(result['risk_alert_details']['message_ids'], ['a', 'b'])
        self.assertEqual([item['message_id'] for item in result['review_queue']], ['a', 'b'])
        self.assertEqual(result['items'][0]['recommended_action'], 'escalate_pr')
        self.assertEqual(result['items'][0]['importance_score'], 100)

    def test_excerpt_and_unknown_metrics(self):
        short = 'Cloud.ru коротко'
        result = self.calculate([row('a', short), row('b', 'Cloud.ru ' + 'я' * 400)])
        self.assertEqual(result['items'][0]['summary_160_300'], short)
        self.assertEqual(len(result['items'][1]['summary_160_300']), 300)
        self.assertIsNone(result['items'][0]['audience'])
        self.assertIn('audience: нет данных', result['items'][0]['importance_explanation'])
        self.assertEqual(result['items'][0]['importance_score'], 0)
        self.assertEqual(result['items'][0]['recommended_action'], 'watch')

    def test_unknown_fields_and_ba_zero(self):
        records, warnings = load(self.write([row(secret='not-forwarded', ba_sentiment=0)]))
        self.assertEqual(warnings, ['secret'])
        self.assertNotIn('secret', records[0])
        self.assertEqual(digest(records, DAY)['items'][0]['ba_sentiment'], 0)

    def test_io_errors_and_no_input_overwrite(self):
        self.assertEqual(self.cli(self.root / 'missing.json')[0], 1)
        (self.root / 'results').write_text('block directory')
        code, _, error = self.cli(self.write([row()]))
        self.assertEqual(code, 1)
        self.assertIn('старыми или частичными', error)
        source = self.write([row()], f'digest-{DAY}.json')
        before = source.read_bytes()
        self.assertEqual(self.cli(source, '--output-dir', str(self.root))[0], 2)
        self.assertEqual(source.read_bytes(), before)

    def test_cli_replay_synthetic_and_validation_preserves_old(self):
        source = self.write([row()])
        self.assertEqual(self.cli(source, '--synthetic')[0], 0)
        output = self.root / 'results' / f'digest-{DAY}.json'
        before = output.read_bytes()
        self.assertTrue(json.loads(before)['synthetic'])
        text = output.with_suffix('.txt').read_text(encoding='utf-8')
        self.assertIn(SYNTHETIC, text)
        self.assertIn('нет данных', text)
        self.assertEqual(self.cli(source, '--synthetic')[0], 0)
        self.assertEqual(output.read_bytes(), before)
        source.write_text('invalid')
        self.assertEqual(self.cli(source)[0], 2)
        self.assertEqual(output.read_bytes(), before)

    def test_invalid_cli_date_and_limit(self):
        source = self.write([])
        for args in [('--date', '20261027'), ('--date', '2026-02-30'), ('--top-n', '0'), ('--top-n', '31')]:
            self.assertEqual(self.cli(source, *args)[0], 2)

    def test_dropped_limit_and_empty_review_text(self):
        result = self.calculate([row(str(i), 'без бренда', ba_topic_hit='запрос') for i in range(15)])
        self.assertEqual(len(result['dropped_examples']), 10)
        self.assertEqual(result['summary_counts']['review_records'], 15)
        self.assertIn(EMPTY, render(result))
        self.assertIn('review_records: 15', render(result))


if __name__ == '__main__':
    unittest.main()
