"""Тесты загрузки пользовательских датасетов и CLI-интеграции без Streamlit."""
from pathlib import Path
import csv
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from cloud_digest.cli import main
from cloud_digest.dataset_upload import detect_dataset, save_uploaded_dataset


class DatasetUploadTests(unittest.TestCase):
    def test_partner_export_detection(self):
        raw = json.dumps([{'id': 7, 'timeCreate': 1780000000, 'url': 'https://example.org/a'}]).encode()
        self.assertEqual(detect_dataset(raw, 'own_data.json'), 'partner')

    def test_normalized_json_detection(self):
        raw = (ROOT / 'demo' / 'messages.json').read_bytes()
        self.assertEqual(detect_dataset(raw, 'my_feed.json'), 'normalized')

    def test_normalized_csv_detection(self):
        raw = (ROOT / 'demo' / 'messages.csv').read_bytes()
        self.assertEqual(detect_dataset(raw, 'my_feed.csv'), 'normalized')

    def test_reject_digest_as_input(self):
        raw = json.dumps({'items': [], 'digest_date': '2026-08-31'}).encode()
        with self.assertRaisesRegex(ValueError, 'готовый дайджест'):
            detect_dataset(raw, 'digest.json')

    def test_reject_incompatible_json(self):
        with self.assertRaisesRegex(ValueError, 'Неизвестная JSON-схема'):
            detect_dataset(b'[{}]', 'unknown.json')

    def test_reject_empty_array(self):
        with self.assertRaisesRegex(ValueError, 'непустой'):
            detect_dataset(b'[]', 'empty.json')

    def test_reject_csv_without_required_fields(self):
        with self.assertRaisesRegex(ValueError, 'CSV: нужны столбцы'):
            detect_dataset(b'id,text\n1,hello', 'dataset.csv')

    def test_reject_utf8_errors(self):
        with self.assertRaisesRegex(ValueError, 'UTF-8'):
            detect_dataset(b'\xff\xfe', 'bad.json')

    def test_reject_large_file(self):
        with self.assertRaisesRegex(ValueError, '30 МБ'):
            detect_dataset(b'0' * (30 * 1024 * 1024 + 1), 'big.json')

    def test_save_uses_hash_not_filename(self):
        raw = (ROOT / 'demo' / 'messages.json').read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            path1, kind1 = save_uploaded_dataset(raw, '../../private.json', Path(temp))
            path2, kind2 = save_uploaded_dataset(raw, 'other.json', Path(temp))
            self.assertEqual((path1, kind1), (path2, kind2))
            self.assertEqual(path1.parent, Path(temp))
            self.assertEqual(path1.read_bytes(), raw)
            self.assertNotIn('private', path1.name)

    def test_uploaded_normalized_json_runs_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            source, kind = save_uploaded_dataset((ROOT / 'demo' / 'messages.json').read_bytes(),
                                                 'uploaded.json', Path(temp) / 'uploads')
            self.assertEqual(kind, 'normalized')
            import contextlib
            previous = Path.cwd()
            try:
                os.chdir(ROOT)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = main(['--input', str(source), '--date', '2026-10-27',
                                   '--output-dir', str(Path(temp) / 'out'), '--scoring', 'story-v1',
                                   '--synthetic'])
            finally:
                os.chdir(previous)
            self.assertEqual(result, 0)
            self.assertTrue((Path(temp) / 'out' / 'digest-2026-10-27.json').is_file())
            self.assertTrue((Path(temp) / 'out' / 'digest-2026-10-27.txt').is_file())


if __name__ == '__main__':
    unittest.main()
