"""Stage 08 quality metrics; synthetic labels, no network or model downloads."""
from __future__ import annotations

from datetime import date
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud_digest.evaluation import (
    Label, EvaluationError, read_gold, make_gold_template, cluster_metrics,
    mapping_from_groups, ranking_metrics, choose_threshold, require_dev_test,
    late_repost_audit, review_metrics,
)
from cloud_digest.evaluate_cli import _grid, build_evaluation, main as evaluation_main
from cloud_digest.clustering import ReviewPair, SemanticClusteringResult
from cloud_digest.source_authority import SourceAuthorityRegistry, load_scoring_rules
from cloud_digest.scoring import PublicationScoreWeights
from cloud_digest.story_scoring import load_story_weights

ROOT = Path(__file__).resolve().parents[1]
DAY = '2026-08-31'


def row(mid, text='Cloud.ru представил новый сервис', *, published='2026-08-31T10:00:00+03:00',
        url='https://t.me/small/1', audience=200, source_type='Мессенджеры каналы', source_name='Small'):
    return {'message_id': mid, 'text': text, 'published_at': published, 'url': url,
            'source_type': source_type, 'source_name': source_name,
            'audience': audience, 'views': None, 'ba_sentiment': None, 'ba_topic_hit': None}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = {
            mid: Label(mid, story, relevance, False, split)
            for mid, story, relevance, split in [
                ('a','A',3,'dev'), ('b','A',3,'dev'), ('c','B',1,'dev'),
                ('d','C',2,'test'), ('e','C',2,'test'), ('f','D',1,'test'),
            ]
        }

    def test_pairwise_perfect_and_bcubed(self):
        predictions = dict(a='1', b='1', c='2', d='3', e='3', f='4')
        result = cluster_metrics(predictions, self.labels, 'dev')
        self.assertEqual(result['pairwise']['tp'], 1)
        self.assertEqual(result['pairwise']['tn'], 2)
        self.assertEqual(result['pairwise']['precision'], 1)
        self.assertEqual(result['pairwise']['recall'], 1)
        self.assertEqual(result['b_cubed']['f1'], 1)

    def test_false_merge_and_missed_pair(self):
        pred = dict(a='a', b='b', c='a', d='x', e='x', f='y')
        result = cluster_metrics(pred, self.labels, 'dev')
        self.assertEqual(result['pairwise']['tp'], 0)
        self.assertEqual(result['pairwise']['fp'], 1)
        self.assertEqual(result['pairwise']['fn'], 1)
        self.assertEqual(result['pairwise']['precision'], 0)
        self.assertEqual(result['pairwise']['recall'], 0)
        self.assertLess(result['b_cubed']['f1'], 1)

    def test_undefined_recall_without_positive_pairs(self):
        labels = {'x': Label('x','X',1,False,'dev'), 'y': Label('y','Y',0,False,'dev')}
        self.assertIsNone(cluster_metrics({'x':'1','y':'2'}, labels, 'dev')['pairwise']['recall'])

    def test_mapping_rejects_duplicate_message_and_group(self):
        with self.assertRaises(EvaluationError):
            mapping_from_groups([('one',['a']), ('two',['a'])])
        with self.assertRaises(EvaluationError):
            mapping_from_groups([('one',['a']), ('one',['b'])])

    def test_ranking_avoids_double_credit_and_penalizes_merges(self):
        result = {'items': [{'cluster_id':'c1'}, {'cluster_id':'c2'}, {'cluster_id':'c3'}]}
        members = {'c1':['a','b'], 'c2':['c'], 'c3':['a']}
        quality = ranking_metrics(result, members, self.labels, 'dev', k=3)
        self.assertEqual(quality['recall_at_k'], 1)
        self.assertEqual(quality['judged_cards_in_top_k'], 3)
        self.assertAlmostEqual(quality['ndcg_at_k_judged_only'], 1)
        mixed = {'c1':['a','c'], 'c2':['b'], 'c3':[]}
        quality2 = ranking_metrics(result, mixed, self.labels, 'dev', k=3)
        self.assertEqual(quality2['ambiguous_mixed_story_cards'], 1)
        self.assertEqual(quality2['found_relevant_stories'], 1)
        self.assertLess(quality2['ndcg_at_k_judged_only'], 1)

    def test_critical_recall(self):
        labels = dict(self.labels)
        labels['a'] = Label('a','A',3,True,'dev')
        result = {'items':[{'cluster_id':'c2'}, {'cluster_id':'c1'}]}
        rank = ranking_metrics(result, {'c1':['a','b'], 'c2':['c']}, labels, 'dev')
        self.assertEqual(rank['critical_recall_at_k'], 1)

    def test_read_gold_unreviewed_template(self):
        values = make_gold_template([row('a'), row('b', text='Без Cloud.ru')], DAY)
        self.assertEqual(len(values['annotations']), 2)
        self.assertIsNone(values['annotations'][0]['story_id'])
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'gold.json'
            file.write_text(json.dumps(values, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(EvaluationError,'Нет заполненной'):
                read_gold(file, day=DAY, available_ids={'a','b'})
            values['annotations'][0].update(story_id='S', relevance=2, critical=False, split='dev')
            file.write_text(json.dumps(values, ensure_ascii=False), encoding='utf-8')
            loaded, coverage = read_gold(file, day=DAY, available_ids={'a','b'})
            self.assertEqual(len(loaded), 1)
            self.assertEqual(coverage['unreviewed'], 1)

    def test_gold_cannot_leak_across_dev_and_test(self):
        raw = {'schema':'cloud-digest-gold-v1','date_msk':DAY,'annotations':[
            {'message_id':'a','story_id':'same','relevance':2,'critical':False,'split':'dev'},
            {'message_id':'b','story_id':'same','relevance':2,'critical':False,'split':'test'}]}
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'labels.json'
            file.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaisesRegex(EvaluationError,'один сюжет'):
                read_gold(file, day=DAY, available_ids={'a','b'})

    def test_invalid_partial_or_outside_ids(self):
        raw = {'schema':'cloud-digest-gold-v1','date_msk':DAY,'annotations':[
            {'message_id':'a','story_id':'same','relevance':None,'critical':False,'split':'dev'}]}
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'labels.json'
            file.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaises(EvaluationError):
                read_gold(file, day=DAY, available_ids={'a'})
            raw['annotations'][0].update(relevance=1)
            file.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaises(EvaluationError):
                read_gold(file, day=DAY, available_ids=set())

    def test_require_split_pair_coverage(self):
        require_dev_test(self.labels)
        with self.assertRaisesRegex(EvaluationError,'test'):
            require_dev_test({k:v for k,v in self.labels.items() if v.split == 'dev'})

    def test_review_recovery(self):
        clusters = SemanticClusteringResult((), (ReviewPair('a','b',.8,'borderline_similarity'),),1,1,0,0)
        recovered = review_metrics(clusters, {'a':'1','b':'2','c':'3'}, self.labels, 'dev')
        self.assertEqual(recovered['missed_positive_pair_review_recall'], 1)
        self.assertEqual(recovered['precision_of_review_candidates'], 1)

    def test_calibration_has_precision_gate(self):
        base = {'pairwise': {'precision': .91, 'f1':.91}}
        good = {'pairwise': {'precision': .98, 'f1':.89}}
        rows = [
            {'high':.80,'low':.70,'dev_clusters':base,'dev_review':{'missed_positive_pair_review_recall':.9,'total_review_pairs':20}},
            {'high':.90,'low':.78,'dev_clusters':good,'dev_review':{'missed_positive_pair_review_recall':.6,'total_review_pairs':10}},
        ]
        self.assertEqual(choose_threshold(rows, self.labels, min_precision=.95)['selected'], {'high':.90,'low':.78})
        self.assertIsNone(choose_threshold(rows, self.labels, min_precision=.99)['selected'])

    def test_threshold_parser(self):
        self.assertEqual(_grid('0.88,0.82,0.88'), [.82,.88])
        for value in ('abc','.7,NaN','.7,1.2',''):
            with self.assertRaises(EvaluationError):
                _grid(value)

    def test_late_large_publisher_increases_story(self):
        rules = load_scoring_rules(ROOT/'config/publication_scoring_v2.json')
        registry = SourceAuthorityRegistry.from_json(ROOT/'partner_data/top25_sources.json', rules=rules)
        pubweights = PublicationScoreWeights.from_settings(rules['weights'])
        storyweights = load_story_weights(ROOT/'config/story_scoring_v1.json')
        input_ = [row('early'), row('later', published='2026-08-31T17:00:00+03:00',
                                   url='https://www.rbc.ru/tech/123', source_type='Онлайн-СМИ',
                                   source_name='РБК', audience=1_000_000)]
        labels = {'early':Label('early','z',2,False,'dev'), 'later':Label('later','z',2,False,'dev')}
        checks = late_repost_audit(input_, labels, registry, pubweights, storyweights)
        self.assertEqual(checks['eligible_labeled_stories'], 1)
        self.assertEqual(checks['score_increased'], 1)
        self.assertEqual(checks['main_source_changed_to_later'], 1)

    def test_cli_template_no_model(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            inp=root/'input.json'; out=root/'gold.json'
            inp.write_text(json.dumps([row('a'), row('b')]), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                code=evaluation_main(['--input',str(inp),'--date',DAY,'--output',str(out),'--make-template'])
            self.assertEqual(code,0)
            self.assertEqual(len(json.loads(out.read_text(encoding='utf-8'))['annotations']),2)
            self.assertFalse((root/'embedding_cache').exists())

    def test_calibration_semantic_mock_encoder_without_download(self):
        # Four distinct manually labeled stories; different stories in dev/test.
        from cloud_digest.embeddings import EmbeddedPublication
        from hashlib import sha256
        stories = {
            'a1': ('Cloud.ru годовой отчёт: 37 млрд выручки', (1.0,0.,0.,0.)),
            'a2': ('Cloud.ru отчитался о выручке 37 млрд', (0.99,0.01,0.,0.)),
            'b1': ('Cloud.ru исследовал методы сжатия LLM', (0.,1.0,0.,0.)),
            'b2': ('Cloud.ru изучил сжатие больших моделей', (0.,0.99,0.01,0.)),
            'c1': ('Cloud.ru открыл новую школу', (0.,0.,1.0,0.)),
            'c2': ('Cloud.ru запустил новую школу', (0.,0.,0.99,0.01)),
            'd1': ('Cloud.ru провёл митап инженеров', (0.,0.,0.,1.)),
            'd2': ('Cloud.ru провел встречу инженеров', (0.01,0.,0.,0.99)),
        }
        class StubService:
            def __init__(self, config):
                self.config = config
            def encode(self, prepared):
                return [EmbeddedPublication(p.message_id, sha256(p.semantic_text.encode()).hexdigest(),
                                            stories[p.message_id][1], False) for p in prepared]
        items = [row(mid, text=text, published=f'2026-08-31T{10+i}:00:00+03:00',
                     url=f'https://t.me/source{i}/1') for i,(mid,(text,_)) in enumerate(stories.items())]
        labels = [dict(message_id=mid, story_id=mid[0], relevance=2, critical=False,
                       split='dev' if mid[0] in 'ab' else 'test') for mid in stories]
        with tempfile.TemporaryDirectory() as td:
            gold=Path(td)/'gold.json'
            gold.write_text(json.dumps({'schema':'cloud-digest-gold-v1','date_msk':DAY,
                                        'review_status':'verified','annotations':labels}), encoding='utf-8')
            with patch('cloud_digest.evaluate_cli.EmbeddingService',StubService):
                result=build_evaluation(items, DAY, gold, run_semantic=True, calibrate=True,
                                        high_grid=[.85,.9],low_grid=[.72,.78], min_precision=.95,
                                        model_name='dummy',model_revision=None,device=None,cache_dir=Path(td),
                                        sources_file=ROOT/'partner_data/top25_sources.json',
                                        scoring_rules_file=ROOT/'config/publication_scoring_v2.json',
                                        story_rules_file=ROOT/'config/story_scoring_v1.json')
        self.assertTrue(result['calibration']['test_not_used_for_selection'])
        self.assertEqual(result['calibration']['decision']['selected'], {'high':.9,'low':.78})
        self.assertEqual(result['semantic']['clustering']['test']['pairwise']['f1'],1.0)
        self.assertEqual(result['semantic']['clustering']['dev']['pairwise']['f1'],1.0)

    def test_draft_gold_cannot_calibrate(self):
        with tempfile.TemporaryDirectory() as td:
            gold=Path(td)/'gold.json'
            gold.write_text(json.dumps({'schema':'cloud-digest-gold-v1','date_msk':DAY,
                'review_status':'draft','annotations':[{'message_id':'a','story_id':'x',
                'relevance':2,'critical':False,'split':'dev'}]}),encoding='utf-8')
            with self.assertRaisesRegex(EvaluationError,'review_status=verified'):
                build_evaluation([row('a')], DAY, gold, run_semantic=True, calibrate=True,
                                 high_grid=[.88], low_grid=[.78], min_precision=.9,
                                 model_name='stub', model_revision=None, device=None, cache_dir=Path(td),
                                 sources_file=ROOT/'partner_data/top25_sources.json',
                                 scoring_rules_file=ROOT/'config/publication_scoring_v2.json',
                                 story_rules_file=ROOT/'config/story_scoring_v1.json')

    def test_cli_report_with_manual_labels_no_model(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            inp=root/'input.json'; out=root/'report.json';gold=root/'gold.json'
            records = [row('a'), row('b'), row('c',text='Cloud.ru предоставил облако в компании'),
                       row('d',text='Cloud.ru повысил скорость API')]
            inp.write_text(json.dumps(records), encoding='utf-8')
            labels = [{'message_id':mid,'story_id':story,'relevance':rel,'critical':False,'split':split}
                      for mid,story,rel,split in [('a','x',2,'dev'),('b','x',2,'dev'),
                                                  ('c','y',1,'test'),('d','z',0,'test')]]
            gold.write_text(json.dumps({'schema':'cloud-digest-gold-v1','date_msk':DAY,'annotations':labels}), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                code=evaluation_main(['--input',str(inp),'--date',DAY,'--gold',str(gold),
                                      '--output',str(out),'--sources-file',str(ROOT/'partner_data/top25_sources.json'),
                                      '--scoring-rules',str(ROOT/'config/publication_scoring_v2.json'),
                                      '--story-rules',str(ROOT/'config/story_scoring_v1.json')])
            self.assertEqual(code,0)
            report=json.loads(out.read_text(encoding='utf-8'))
            self.assertIn('baseline',report)
            self.assertNotIn('semantic',report)
            self.assertTrue(report['quality_is_from_human_gold_only'])


if __name__ == '__main__':
    unittest.main()
