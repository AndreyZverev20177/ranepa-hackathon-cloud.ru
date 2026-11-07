"""Проверки локальных редакторских правок без зависимости от Streamlit."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cloud_digest.editor_review import (
    blank_store, effective_status, export_editorial_digest,
    load_store, review_key, save_review, validate_review,
)


class EditorReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'artifacts' / 'editor_reviews.json'
        self.day = '2026-08-31'
        self.example = {
            'digest_date': self.day,
            'rule_version': 'story-scoring-v1',
            'items': [
                {'message_id': '101', 'cluster_id': 'story-a', 'importance_score': 62,
                 'sentiment': 'negative', 'brand_link': 'direct', 'human_review_needed': True,
                 'story_card': {'description_excerpt': 'Основная новость', 'title': 'Сбой'}},
                {'message_id': '102', 'cluster_id': 'story-b', 'importance_score': 48,
                 'sentiment': 'neutral', 'brand_link': 'direct', 'human_review_needed': False,
                 'summary_160_300': 'Другая новость'},
            ],
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_blank_store_when_missing(self):
        self.assertEqual(load_store(self.path), blank_store())

    def test_invalid_store_refuses_overwrite(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            save_review(self.path, self.day, 'story-a', {'status': 'approved'})
        self.assertEqual(self.path.read_text(encoding='utf-8'), '{broken')

    def test_roundtrip_and_timestamp(self):
        saved = save_review(self.path, self.day, 'story-a', {'status': 'approved', 'tone': 'neutral', 'summary': 'Перепроверено'})
        stored = load_store(self.path)['reviews'][review_key(self.day, 'story-a')]
        self.assertEqual(saved, stored)
        self.assertEqual(stored['status'], 'approved')
        self.assertEqual(stored['tone'], 'neutral')
        self.assertIn('+00:00', stored['updated_at'])
        self.assertTrue(self.path.is_file())

    def test_second_save_preserves_original_created_at(self):
        a = save_review(self.path, self.day, 'story-a', {'status': 'approved'})
        b = save_review(self.path, self.day, 'story-a', {'status': 'excluded'})
        self.assertEqual(a['created_at'], b['created_at'])
        self.assertEqual(b['status'], 'excluded')

    def test_multiple_days_do_not_collide(self):
        save_review(self.path, '2026-08-30', 'same', {'status': 'approved'})
        save_review(self.path, '2026-08-31', 'same', {'status': 'excluded'})
        self.assertEqual(len(load_store(self.path)['reviews']), 2)

    def test_auto_review_flag(self):
        self.assertEqual(effective_status(self.example['items'][0], {}), 'needs_review')
        self.assertEqual(effective_status(self.example['items'][1], {}), 'pending')

    def test_manual_status_overrides_flag(self):
        self.assertEqual(effective_status(self.example['items'][0], {'status': 'approved'}), 'approved')

    def test_invalid_status_and_tone(self):
        with self.assertRaises(ValueError):
            validate_review({'status': 'superuser'})
        with self.assertRaises(ValueError):
            validate_review({'tone': 'unknown'})

    def test_invalid_long_text(self):
        with self.assertRaises(ValueError):
            validate_review({'notes': 'x' * 2001})
        with self.assertRaises(ValueError):
            validate_review({'summary': 'x' * 1001})

    def test_cluster_requests_require_target(self):
        with self.assertRaises(ValueError):
            validate_review({'cluster_action': 'check_merge'})
        with self.assertRaises(ValueError):
            validate_review({'cluster_action': 'check_split'})
        value = validate_review({'cluster_action': 'check_merge', 'cluster_target': 'story-b'})
        self.assertEqual(value['cluster_target'], 'story-b')

    def test_export_keeps_score_and_model_fields_unchanged(self):
        original = copy.deepcopy(self.example)
        save_review(self.path, self.day, 'story-a', {'status': 'approved', 'tone': 'neutral', 'brand_link': 'none', 'summary': 'Исправлено'})
        exported = export_editorial_digest(self.example, load_store(self.path))
        self.assertEqual(self.example, original)
        updated = exported['items'][0]
        self.assertEqual(updated['importance_score'], 62)
        self.assertEqual(updated['sentiment'], 'negative')
        self.assertEqual(updated['editorial']['final_tone'], 'neutral')
        self.assertEqual(updated['editorial']['final_brand_link'], 'none')
        self.assertEqual(updated['editorial']['summary'], 'Исправлено')
        self.assertFalse(updated['editorial']['score_recalculated'])

    def test_excluded_is_removed_only_from_editorial_items(self):
        save_review(self.path, self.day, 'story-b', {'status': 'excluded'})
        exported = export_editorial_digest(self.example, load_store(self.path))
        self.assertEqual(len(exported['items']), 1)
        self.assertEqual(exported['excluded_by_editor'][0]['cluster_id'], 'story-b')
        self.assertEqual(len(self.example['items']), 2)

    def test_serialize_editorial_export(self):
        exported = export_editorial_digest(self.example, blank_store())
        self.assertEqual(json.loads(json.dumps(exported, ensure_ascii=False))['counts']['visible_stories'], 2)
        self.assertIn('Brand Analytics', exported['editorial_disclaimer'])


if __name__ == '__main__':
    unittest.main()
