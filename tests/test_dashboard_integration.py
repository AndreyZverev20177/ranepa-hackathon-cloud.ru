"""Контракт CLI -> story_card -> редакторский экспорт, без веб-сервера."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cloud_digest.cli import main
from cloud_digest.editor_review import export_editorial_digest, load_store, save_review


class DashboardPipelineTests(unittest.TestCase):
    def test_demo_story_pipeline_review_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / 'output'
            import os
            prev = Path.cwd()
            try:
                os.chdir(ROOT)  # config/ and top25_sources.json are resolved relative to repo
                exit_code = main([
                    '--input', str(ROOT / 'demo' / 'messages.json'),
                    '--date', '2026-10-27', '--output-dir', str(result),
                    '--scoring', 'story-v1', '--synthetic',
                ])
            finally:
                os.chdir(prev)
            self.assertEqual(exit_code, 0)
            source = json.loads((result / 'digest-2026-10-27.json').read_text(encoding='utf-8'))
            self.assertEqual(source['rule_version'], 'story-scoring-v1')
            self.assertTrue(source['items'])
            self.assertTrue(all('story_card' in x for x in source['items']))
            first = source['items'][0]
            original = first['importance_score']
            store_path = Path(tmp) / 'reviews.json'
            save_review(store_path, '2026-10-27', first['cluster_id'], {
                'status': 'approved', 'tone': 'neutral', 'summary': 'Редакторская выжимка',
            })
            reviewed = export_editorial_digest(source, load_store(store_path))
            self.assertEqual(reviewed['counts']['visible_stories'], len(source['items']))
            first_after = next(i for i in reviewed['items'] if i['cluster_id'] == first['cluster_id'])
            self.assertEqual(first_after['importance_score'], original)
            self.assertEqual(first_after['editorial']['summary'], 'Редакторская выжимка')
            self.assertEqual(first_after['editorial']['status'], 'approved')

if __name__ == '__main__':
    unittest.main()
