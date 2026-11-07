"""Stage 07: одна сюжетная карточка и объяснимая хронология.

Синтетические данные для тестирования структуры, не проверка истинности СМИ.
"""
from __future__ import annotations

from datetime import date
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from cloud_digest.cli import main
from cloud_digest.digest import digest, render
from cloud_digest.embeddings import EmbeddingService
from cloud_digest.scoring import PublicationScoreWeights
from cloud_digest.source_authority import SourceAuthorityRegistry, load_scoring_rules
from cloud_digest.story_scoring import load_story_weights
from cloud_digest.story_view import build_story_card

ROOT = Path(__file__).resolve().parents[1]
DAY = date(2026, 8, 31)


def row(message_id: str, *, text: str = 'Cloud.ru обновил возможности сервиса',
        published: str = '2026-08-31T09:00:00+03:00', audience: int | None = 100,
        source_name: str = 'Первый канал', url: str = 'https://t.me/first_channel/1',
        views: int | None = 50, source_type: str = 'Мессенджеры каналы') -> dict:
    return {
        'message_id': message_id, 'text': text, 'published_at': published,
        'audience': audience, 'views': views, 'source_name': source_name,
        'source_type': source_type, 'url': url, 'ba_sentiment': None,
        'ba_topic_hit': None,
    }


class StoryDigestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rules = load_scoring_rules(ROOT / 'config/publication_scoring_v2.json')
        cls.registry = SourceAuthorityRegistry.from_json(
            ROOT / 'partner_data/top25_sources.json', rules=rules
        )
        cls.weights = PublicationScoreWeights.from_settings(rules['weights'])
        cls.story_weights = load_story_weights(ROOT / 'config/story_scoring_v1.json')

    def report(self, items: list[dict], *, scoring: str = 'story-v1', top_n: int = 30, **kwargs) -> dict:
        if scoring == 'baseline':
            return digest(items, DAY, top_n=top_n)
        return digest(items, DAY, top_n=top_n, scoring=scoring,
                      source_registry=self.registry, scoring_weights=self.weights,
                      story_weights=self.story_weights, **kwargs)

    @staticmethod
    def late_story() -> list[dict]:
        text = 'Cloud.ru представил новый сервис для заказчиков'
        return [
            row('a', text=text, published='2026-08-31T12:00:00+03:00',
                source_name='Маленький канал', audience=200,
                url='https://t.me/first_channel/1'),
            row('b', text=text, published='2026-08-31T15:00:00+03:00',
                source_name='РБК', audience=1_200_000, views=100_000,
                source_type='Онлайн-СМИ', url='https://www.rbc.ru/tech/1'),
        ]

    def test_one_card_for_story_and_main_source_is_late_publisher(self) -> None:
        report = self.report(self.late_story())
        self.assertEqual(len(report['items']), 1)
        item = report['items'][0]
        view = item['story_card']
        self.assertEqual(view['schema_version'], 'story-card-v1')
        self.assertEqual(view['main_source']['message_id'], 'b')
        self.assertEqual(view['main_source']['url'], 'https://www.rbc.ru/tech/1')
        self.assertEqual(view['main_source']['publication_score'], item['publication_score'])
        self.assertEqual(view['priority']['story_score'], item['importance_score'])
        self.assertEqual(view['spread']['publication_count'], 2)
        self.assertEqual(len(view['timeline']), 2)
        self.assertEqual(item['story_publication_ids'], ['a', 'b'])

    def test_full_timeline_is_time_sorted_with_source_provenance(self) -> None:
        raw = self.late_story()
        raw[0]['published_at'] = '2026-08-31T09:00:00+00:00'
        raw[1]['published_at'] = '2026-08-31T15:00:00+03:00'
        report = self.report(list(reversed(raw)))
        timeline = report['items'][0]['story_card']['timeline']
        self.assertEqual([item['message_id'] for item in timeline], ['a', 'b'])
        self.assertTrue(timeline[0]['is_first'])
        self.assertFalse(timeline[0]['is_main_source'])
        self.assertTrue(timeline[1]['is_main_source'])
        self.assertTrue(all(item['url'].startswith('https://') for item in timeline))

    def test_score_breakdown_is_exact_not_recalculated(self) -> None:
        item = self.report(self.late_story())['items'][0]
        view = item['story_card']['priority']
        self.assertEqual(
            {c['code']: c['points'] for c in view['contributions']},
            item['story_score_breakdown'],
        )
        self.assertEqual(sum(c['points'] for c in view['contributions']),
                         item['story_score_uncapped'])
        self.assertEqual(view['story_score'], min(100, view['score_before_cap']))
        self.assertEqual(view['publication_score_base'],
                         item['story_score_breakdown']['best_publication'])

    def test_history_and_facts_are_present_in_txt(self) -> None:
        raw = [
            row('a', text='Cloud.ru: выручка 37,6 млрд рублей'),
            row('b', text='Cloud.ru: выручка 37,6 млрд рублей. Инвестиции составили 8,5 млрд рублей',
                published='2026-08-31T11:00:00+03:00',
                url='https://www.rbc.ru/news/123', source_name='РБК',
                source_type='Онлайн-СМИ'),
        ]
        # Близкие по смыслу строки могут остаться разными кластерами при baseline,
        # поэтому история здесь проверяется при одинаковых текстах, а факты отдельно.
        report = self.report([raw[0], {**raw[1], 'text': raw[0]['text']}])
        txt = render(report)
        self.assertIn('История публикаций (по времени):', txt)
        self.assertIn('Главный источник:', txt)
        self.assertIn('Почему такой приоритет:', txt)
        self.assertIn('https://www.rbc.ru/news/123', txt)
        self.assertIn('Story Score:', txt)
        self.assertIn('не уникальная аудитория', txt)
        self.assertNotIn('story_score_breakdown:', txt)
        self.assertNotIn('dropped_examples:', txt)

    def test_new_fact_signals_follow_publication_in_timeline(self) -> None:
        class ConstantEncoder:
            def encode(self, texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

        first = row('one', text='Выручка Cloud.ru за первое полугодие 2026 составила 37,6 млрд рублей')
        second = row(
            'two', text='Cloud.ru: доходы от ИИ достигли 19,3 млрд рублей за первое полугодие 2026 года',
            url='https://rbc.ru/news/test', source_name='РБК', source_type='Онлайн-СМИ',
            published='2026-08-31T12:00:00+03:00',
        )
        report = self.report([first, second], semantic=True,
                             embedding_service=EmbeddingService(encoder=ConstantEncoder()))
        self.assertEqual(len(report['items']), 1)
        view = report['items'][0]['story_card']
        self.assertEqual(len(view['timeline']), 2)
        self.assertTrue(view['timeline'][1]['new_fact_signals'])
        self.assertIn('не проверены', render(report))

    def test_review_and_crisis_are_visible(self) -> None:
        data = [row('a', text='Cloud.ru недоступен из-за аварии')]
        report = self.report(data)
        view = report['items'][0]['story_card']
        self.assertTrue(view['review']['required'])
        self.assertTrue(view['review']['contains_crisis'])
        self.assertIn('Требуется проверка:', render(report))

    def test_missing_data_is_not_fake_coverage(self) -> None:
        report = self.report([row('a', audience=None, views=None)])
        spread = report['items'][0]['story_card']['spread']
        self.assertEqual(spread['max_reported_audience'], 0)
        self.assertEqual(spread['additional_audience_proxy'], 0)
        self.assertEqual(spread['audience_measure'], 'potential_audience_proxy_not_unique_reach')
        self.assertNotIn('unique_reach', report['items'][0]['story_card'])

    def test_title_derived_from_text_or_labeled_fallback(self) -> None:
        item = self.report([row('a', text='Cloud.ru: опубликован технический отчёт')])['items'][0]
        self.assertEqual(item['story_card']['title'], 'Cloud.ru: опубликован технический отчёт')
        self.assertEqual(item['story_card']['title_source'], 'explicit_or_first_line_of_main_publication')
        long_text = 'Cloud.ru сообщил о результатах работы. ' * 12
        long_title = self.report([row('long', text=long_text)])['items'][0]['story_card']
        self.assertLessEqual(len(long_title['title']), 140)
        self.assertEqual(long_title['title_source'], 'truncated_first_line_excerpt')
        with self.assertRaises(ValueError):
            build_story_card(item, row('other'))

    def test_empty_story_digest_txt(self) -> None:
        result = self.report([row('a', text='совершенно другая компания')])
        self.assertEqual(result['items'], [])
        self.assertIn('не найдено', render(result))

    def test_old_modes_are_unaffected(self) -> None:
        inputs = self.late_story()
        old = self.report(inputs, scoring='baseline')
        pub = self.report(inputs, scoring='publication-v2')
        self.assertFalse(any('story_card' in entry for entry in old['items']))
        self.assertFalse(any('story_card' in entry for entry in pub['items']))
        self.assertIn('summary_counts:', render(old))
        self.assertIn('summary_counts:', render(pub))

    def test_one_card_with_ten_reposts_has_full_timeline(self) -> None:
        items = [row(str(i), url=f'https://t.me/channel_{i}/1') for i in range(10)]
        report = self.report(items)
        self.assertEqual(len(report['items']), 1)
        self.assertEqual(report['items'][0]['story_card']['spread']['publication_count'], 10)
        self.assertEqual(len(report['items'][0]['story_card']['timeline']), 10)

    def test_cli_writes_structured_json_and_readable_text(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.json'
            source.write_text(json.dumps(self.late_story(), ensure_ascii=False), encoding='utf-8')
            target = root / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([
                    '--input', str(source), '--date', str(DAY),
                    '--output-dir', str(target), '--scoring', 'story-v1',
                    '--sources-file', str(ROOT / 'partner_data/top25_sources.json'),
                    '--scoring-rules', str(ROOT / 'config/publication_scoring_v2.json'),
                    '--story-rules', str(ROOT / 'config/story_scoring_v1.json'),
                ])
            self.assertEqual(code, 0)
            json_data = json.loads((target / f'digest-{DAY}.json').read_text(encoding='utf-8'))
            txt_data = (target / f'digest-{DAY}.txt').read_text(encoding='utf-8')
            self.assertEqual(len(json_data['items']), 1)
            self.assertIn('main_source', json_data['items'][0]['story_card'])
            self.assertIn('История публикаций', txt_data)
            self.assertIn('РБК', txt_data)


if __name__ == '__main__':
    unittest.main()
