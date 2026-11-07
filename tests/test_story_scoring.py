"""Проверки stage-06 на синтетических публикациях; не оценка качества на реальных СМИ."""
from datetime import date
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from cloud_digest.cli import main
from cloud_digest.digest import digest
from cloud_digest.embeddings import EmbeddingService
from cloud_digest.scoring import card, PublicationScoreWeights
from cloud_digest.source_authority import SourceAuthorityRegistry, load_scoring_rules
from cloud_digest.story_scoring import (
    StoryScoreWeights, extract_claims, score_story, _source_id, load_story_weights,
)

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / 'config/publication_scoring_v2.json'
STORY_RULES = ROOT / 'config/story_scoring_v1.json'
TOP25 = ROOT / 'partner_data/top25_sources.json'
DAY = date(2026, 8, 31)


def row(mid: str, *, url: str = 'https://t.me/local_channel/1',
        published: str = '2026-08-31T10:00:00+03:00', audience: int | None = None,
        source_name: str = 'Локальный канал', source_type: str = 'Мессенджеры каналы',
        text: str = 'Cloud.ru рассказал об изменениях сервиса', views: int | None = None) -> dict:
    return {'message_id': mid, 'url': url, 'published_at': published,
            'source_name': source_name, 'source_type': source_type, 'audience': audience,
            'views': views, 'text': text, 'ba_sentiment': None, 'ba_topic_hit': None}


class StoryScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rules = load_scoring_rules(RULES)
        cls.registry = SourceAuthorityRegistry.from_json(TOP25, rules=rules)
        cls.pub_weights = PublicationScoreWeights.from_settings(rules['weights'])
        cls.story_weights = load_story_weights(STORY_RULES)

    def calculate(self, records, **kwargs):
        return digest(records, DAY, scoring='story-v1', source_registry=self.registry,
                      scoring_weights=self.pub_weights, story_weights=self.story_weights, **kwargs)

    def score_group(self, records):
        cards = [card(x, ('direct', 'Cloud.ru'), scoring='publication-v2',
                      source_registry=self.registry, scoring_weights=self.pub_weights) for x in records]
        return score_story(cards, {x['message_id']: x for x in records}, self.story_weights)

    def test_singleton_score_matches_publication(self):
        item = row('one', audience=12300, views=1000)
        report = self.calculate([item])
        card_ = report['items'][0]
        self.assertEqual(card_['story_score'], card_['publication_score'])
        self.assertEqual(card_['importance_score'], card_['story_score'])
        self.assertEqual(card_['story_score_breakdown']['distribution'], 0)
        self.assertEqual(card_['story_score_breakdown']['additional_reach_proxy'], 0)
        self.assertEqual(card_['distinct_outlets'], 1)
        self.assertEqual(card_['first_published_at'], item['published_at'])

    def test_late_major_publication_becomes_representative_and_boosts_score(self):
        first = row('early', audience=800, url='https://t.me/small/10',
                    text='Cloud.ru сообщил об обновлении сервиса')
        late = row('late', audience=1_500_000, published='2026-08-31T17:00:00+03:00',
                   url='https://rbc.ru/technology/1', source_type='Онлайн-СМИ',
                   source_name='РБК', text=first['text'])
        alone = self.calculate([first])['items'][0]['story_score']
        result = self.calculate([first, late])
        self.assertEqual(result['summary_counts']['clusters'], 1)
        item = result['items'][0]
        self.assertGreater(item['story_score'], alone)
        self.assertEqual(item['message_id'], 'late')
        self.assertEqual(item['best_publication_id'], 'late')
        self.assertEqual(item['story_publication_ids'], ['early', 'late'])
        self.assertEqual(item['cluster_size'], 2)
        self.assertEqual(item['first_published_at'], first['published_at'])
        self.assertEqual(item['last_published_at'], late['published_at'])
        self.assertEqual(item['distinct_outlets'], 2)
        self.assertIn('early', [x['message_id'] for x in result['items'][0]['story_publications']])

    def test_no_inflation_for_reposts_in_same_telegram_channel(self):
        items = [row('a', url='https://t.me/channel_one/1', audience=400_000),
                 row('b', url='https://telegram.me/channel_one/2', audience=400_000,
                     published='2026-08-31T12:00:00+03:00')]
        one = self.score_group(items[:1])
        both = self.score_group(items)
        self.assertEqual(both.score, one.score)
        self.assertEqual(both.metadata['distinct_outlets'], 1)
        self.assertEqual(both.metadata['additional_audience_proxy'], 0)
        self.assertEqual(both.breakdown['distribution'], 0)
        self.assertEqual(both.breakdown['new_fact_signals'], 0)

    def test_vk_wall_and_video_same_publisher(self):
        self.assertEqual(_source_id(row('a', url='https://vk.com/wall-226874221_1163'))[0],
                         _source_id(row('b', url='https://vk.com/video-226874221_456239646'))[0])

    def test_distinct_channels_bonus_capped(self):
        items = [row(str(i), url=f'https://t.me/channel{i}/1', audience=10_000,
                     published=f'2026-08-31T{10+i:02d}:00:00+03:00') for i in range(7)]
        result = self.score_group(items)
        self.assertEqual(result.metadata['distinct_outlets'], 7)
        self.assertEqual(result.breakdown['distribution'], self.story_weights.distribution_cap)
        self.assertLessEqual(result.breakdown['additional_reach_proxy'], self.story_weights.reach_cap)
        self.assertLessEqual(result.metadata['additional_audience_proxy'], 2500)
        self.assertEqual(result.metadata['audience_measure'], 'potential_audience_proxy_not_unique_reach')

    def test_credible_publisher_bonus_not_platform_publisher(self):
        first = row('a', url='https://t.me/channelone/1', audience=123)
        one = row('b', url='https://forbes.ru/news/1', source_type='Онлайн-СМИ',
                  source_name='Forbes', audience=2000)
        second = row('c', url='https://rbc.ru/news/1', source_type='Онлайн-СМИ',
                     source_name='РБК', audience=2000)
        report_one = self.score_group([first, one])
        report_two = self.score_group([first, one, second])
        self.assertEqual(report_one.metadata['distinct_high_authority_domains'], 1)
        self.assertEqual(report_one.breakdown['additional_high_authority_domains'], 0)
        self.assertEqual(report_two.metadata['distinct_high_authority_domains'], 2)
        self.assertEqual(report_two.breakdown['additional_high_authority_domains'], 4)
        another = self.score_group([first, row('d', url='https://t.me/channeltwo/1')])
        self.assertEqual(another.breakdown['additional_high_authority_domains'], 0)

    def test_domain_reposts_do_not_add_high_authority_domains_twice(self):
        first = row('a', url='https://www.forbes.ru/a', source_type='Онлайн-СМИ')
        later = row('b', url='https://forbes.ru/b', source_type='Онлайн-СМИ')
        result = self.score_group([first, later])
        self.assertEqual(result.metadata['distinct_outlets'], 1)
        self.assertEqual(result.breakdown['additional_high_authority_domains'], 0)

    def test_claim_extraction_equivalence_and_ai_qualifier(self):
        self.assertEqual(extract_claims('Выручка Cloud.ru составила 37,6 млрд руб.'),
                         extract_claims('Cloud.ru: выручка составила 37600 млн рублей'))
        claims = extract_claims('Выручка Cloud.ru 37,6 млрд, доходы от ИИ 19,3 млрд')
        self.assertEqual(len(claims), 2)
        self.assertIn('ai_revenue', claims)
        self.assertFalse(extract_claims('Cloud.ru: в 2026 году прошли 3 мероприятия.'))

    def test_new_claim_from_later_article_counts_once(self):
        rows = [row('a', text='Выручка Cloud.ru составила 37,6 млрд рублей'),
                row('b', url='https://rbc.ru/a', source_type='Онлайн-СМИ',
                    published='2026-08-31T12:00:00+03:00',
                    text='Выручка Cloud.ru составила 37600 млн. Инвестиции составили 8,5 млрд рублей'),
                row('c', url='https://t.me/channel2/2', published='2026-08-31T15:00:00+03:00',
                    text='Инвестиции Cloud.ru составили 8500 млн рублей')]
        result = self.score_group(rows)
        self.assertEqual(len(result.metadata['novel_fact_signals']), 1)
        self.assertEqual(result.breakdown['new_fact_signals'], 3)

    def test_conflicting_numeric_claim_is_review_not_new_fact(self):
        rows = [row('a', text='Выручка Cloud.ru составила 37,6 млрд рублей'),
                row('b', published='2026-08-31T13:00:00+03:00',
                    url='https://rbc.ru/news/1', source_type='Онлайн-СМИ',
                    text='Выручка Cloud.ru составила 39,9 млрд рублей')]
        score = self.score_group(rows)
        self.assertEqual(score.breakdown['new_fact_signals'], 0)
        self.assertIn('conflicting_numeric_claims', score.review_reasons)
        report = self.calculate(rows)
        # Одинаковые тексты не требуются: в старом baseline это две отдельные темы.
        self.assertEqual(report['summary_counts']['clusters'], 2)

    def test_conflicting_claim_review_inside_one_cluster_direct(self):
        rows = [row('a', text='Выручка Cloud.ru составила 37,6 млрд рублей'),
                row('b', url='https://rbc.ru/news', source_type='Онлайн-СМИ',
                    published='2026-08-31T13:00:00+03:00',
                    text='Выручка Cloud.ru составила 39,9 млрд рублей')]
        items = [card(x, ('direct','Cloud.ru'), scoring='publication-v2', source_registry=self.registry) for x in rows]
        assessment = score_story(items, {x['message_id']: x for x in rows})
        self.assertEqual(len(assessment.metadata['conflicting_claim_signals']), 1)
        self.assertIn('conflicting_numeric_claims', assessment.review_reasons)

    def test_crisis_flag_survives_later_stronger_neutral_representative(self):
        # Один и тот же инцидент; флаг эскалации сохраняется у сюжета.
        first = row('a', url='https://t.me/x/1', text='Cloud.ru недоступ', audience=100)
        second = row('b', url='https://rbc.ru/a', source_type='Онлайн-СМИ',
                     text='Cloud.ru недоступ', audience=1_000_000,
                     published='2026-08-31T17:00:00+03:00')
        report = self.calculate([first, second])
        self.assertTrue(report['items'][0]['story_contains_crisis'] or report['risk_alert'])
        self.assertEqual(report['items'][0]['recommended_action'], 'escalate_pr')

    def test_order_determinism_and_story_rank(self):
        items = [row('early', text='Cloud.ru объявил о продукте', audience=200),
                 row('late', url='https://rbc.ru/news', source_type='Онлайн-СМИ',
                     text='Cloud.ru объявил о продукте', audience=200000,
                     published='2026-08-31T17:00:00+03:00'),
                 row('separate', text='Cloud.ru другое событие', audience=1200)]
        self.assertEqual(self.calculate(items), self.calculate(list(reversed(items))))
        values = [i['story_score'] for i in self.calculate(items)['items']]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_existing_publication_v2_is_not_modified(self):
        items = [row('a'), row('b', audience=1_000_000)]
        before = digest(items, DAY, scoring='publication-v2', source_registry=self.registry,
                        scoring_weights=self.pub_weights)
        after = digest(items, DAY, scoring='publication-v2', source_registry=self.registry,
                       scoring_weights=self.pub_weights, story_weights=self.story_weights)
        self.assertEqual(before, after)
        self.assertNotIn('story_scoring', before)

    def test_invalid_config_raises(self):
        raw = json.loads(STORY_RULES.read_text(encoding='utf-8'))['weights']
        with self.assertRaises(ValueError):
            StoryScoreWeights.from_settings({**raw, 'reach_cap': -1})
        with self.assertRaises(ValueError):
            StoryScoreWeights.from_settings({**raw, 'authority_threshold': True})
        with self.assertRaises(ValueError):
            StoryScoreWeights.from_settings({**raw, 'extra_parameter': 1})
        with self.assertRaises(ValueError):
            score_story([], {})

    def test_semantic_story_joins_paraphrases_and_adds_new_information(self):
        class ConstantEncoder:
            def encode(self, texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

        items = [row('a', url='https://t.me/first/1',
                     text='Выручка Cloud.ru за первое полугодие 2026 составила 37,6 млрд рублей'),
                 row('b', url='https://rbc.ru/news', source_type='Онлайн-СМИ',
                     published='2026-08-31T16:00:00+03:00',
                     text='Cloud.ru сообщила: доходы от ИИ достигли 19,3 млрд рублей за первое полугодие 2026 года')]
        report = self.calculate(items, semantic=True, embedding_service=EmbeddingService(encoder=ConstantEncoder()))
        self.assertEqual(report['summary_counts']['clusters'], 1)
        self.assertEqual(report['rule_version'], 'semantic-clustering-v1+story-scoring-v1')
        self.assertEqual(report['items'][0]['cluster_size'], 2)
        self.assertEqual(report['items'][0]['story_score_breakdown']['new_fact_signals'], 3)
        self.assertEqual(len(report['semantic_clustering']['clusters']), 1)

    def test_semantic_cluster_conflicting_values_enters_review_queue(self):
        class ConstantEncoder:
            def encode(self, texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

        items = [row('a', url='https://t.me/channel_a/1',
                     text='Выручка Cloud.ru за первое полугодие 2026 составила 37,6 млрд рублей'),
                 row('b', url='https://rbc.ru/a', source_type='Онлайн-СМИ',
                     published='2026-08-31T17:00:00+03:00',
                     text='Выручка Cloud.ru за первое полугодие 2026 составила 39,9 млрд рублей')]
        report = self.calculate(items, semantic=True, embedding_service=EmbeddingService(encoder=ConstantEncoder()))
        self.assertEqual(report['summary_counts']['clusters'], 1)
        self.assertEqual(report['story_scoring']['review_clusters_with_claim_conflicts'], 1)
        item = report['items'][0]
        self.assertTrue(item['human_review_needed'])
        self.assertIn('conflicting_numeric_claims', item['review_reasons'])
        self.assertIn('conflicting_numeric_claims', next(q['reasons'] for q in report['review_queue']
                                                        if q['message_id'] == item['message_id']))
        self.assertEqual(item['story_score_breakdown']['new_fact_signals'], 0)

    def test_risk_from_member_aggregated_when_representative_is_neutral(self):
        # Искусственно снижаем вес кризиса, чтобы нейтральный крупный издатель
        # стал представителем; проверяем агрегацию риска, а не веса stage-05.
        class ConstantEncoder:
            def encode(self, texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

        weights = PublicationScoreWeights(negative=0, crisis=0, audience=35,
                                          views=15, authority=45, historical_media_weight=5)
        items = [row('a', url='https://t.me/small/1', audience=100,
                     text='Cloud.ru недоступ API для пользователей. Утром произошла авария инфраструктуры'),
                 row('b', url='https://forbes.ru/news', source_type='Онлайн-СМИ',
                     audience=1000000, views=100000,
                     published='2026-08-31T17:00:00+03:00',
                     text='Cloud.ru сообщил о восстановлении работоспособности сервисов для клиентов')]
        report = digest(items, DAY, scoring='story-v1', source_registry=self.registry,
                        scoring_weights=weights, story_weights=self.story_weights,
                        semantic=True, embedding_service=EmbeddingService(encoder=ConstantEncoder()))
        self.assertEqual(report['summary_counts']['clusters'], 1)
        item = report['items'][0]
        self.assertEqual(item['message_id'], 'b')
        self.assertTrue(item['story_contains_crisis'])
        self.assertEqual(item['recommended_action'], 'escalate_pr')
        self.assertIn('story_contains_crisis', item['review_reasons'])

    def test_baseline_outage_cluster_not_rewarded_when_services_conflict(self):
        items = [row('a', url='https://t.me/channel_a/1',
                     text='В Cloud.ru произошёл сбой объектного хранилища S3'),
                 row('b', url='https://rbc.ru/a', source_type='Онлайн-СМИ',
                     published='2026-08-31T16:00:00+03:00',
                     text='В Cloud.ru произошёл сбой виртуальных машин')]
        report = self.calculate(items)
        self.assertEqual(report['summary_counts']['clusters'], 1)
        item = report['items'][0]
        self.assertEqual(item['story_score'], item['story_score_breakdown']['best_publication'])
        self.assertEqual(item['story_score_breakdown']['distribution'], 0)
        self.assertIn('incompatible_event_members', item['review_reasons'])
        self.assertEqual(report['story_scoring']['review_clusters_with_event_conflicts'], 1)

    def test_later_explicit_service_is_new_information(self):
        items = [row('a', text='Cloud.ru объявил об обновлении платформы'),
                 row('b', url='https://forbes.ru/a', source_type='Онлайн-СМИ',
                     published='2026-08-31T15:00:00+03:00',
                     text='Cloud.ru обновил платформу с поддержкой Kubernetes')]
        result = self.score_group(items)
        self.assertEqual(result.breakdown['new_fact_signals'], 3)
        self.assertEqual(result.metadata['novel_fact_signals'][0]['metric'], 'service:kubernetes')

    def test_incompatible_story_can_not_collect_new_fact_bonus(self):
        items = [row('a', text='Сбой Cloud.ru в хранилище S3'),
                 row('b', url='https://rbc.ru/news', source_type='Онлайн-СМИ',
                     published='2026-08-31T15:00:00+03:00',
                     text='Сбой Cloud.ru виртуальных машин')]
        result = self.score_group(items)
        self.assertEqual(result.breakdown['new_fact_signals'], 0)
        self.assertEqual(result.breakdown['additional_reach_proxy'], 0)
        self.assertEqual(result.breakdown['distribution'], 0)
        self.assertTrue(result.metadata['incompatible_member_pairs'])

    def test_cli_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp, outdir = root/'input.json', root/'out'
            inp.write_text(json.dumps([row('a'), row('b', url='https://rbc.ru/a',
                                                     source_type='Онлайн-СМИ')], ensure_ascii=False), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(['--input', str(inp), '--date', '2026-08-31', '--output-dir', str(outdir),
                             '--scoring', 'story-v1', '--story-rules', str(STORY_RULES),
                             '--sources-file', str(TOP25), '--scoring-rules', str(RULES)])
            self.assertEqual(code, 0)
            result = json.loads((outdir/'digest-2026-08-31.json').read_text(encoding='utf-8'))
            self.assertEqual(result['rule_version'], 'story-scoring-v1')
            self.assertIn('story_score', result['items'][0])
            self.assertEqual(len(result['items']), 1)

    def test_missing_story_config_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp, outdir = root/'input.json', root/'out'
            inp.write_text(json.dumps([row('a')], ensure_ascii=False), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(['--input', str(inp), '--date', '2026-08-31', '--output-dir', str(outdir),
                             '--scoring', 'story-v1', '--story-rules', str(root/'missing.json'),
                             '--sources-file', str(TOP25), '--scoring-rules', str(RULES)])
            self.assertEqual(code, 2)
            self.assertFalse(outdir.exists())


if __name__ == '__main__':
    unittest.main()
