"""Local, deterministic aggregate audit. Never emits source text, URLs or IDs."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

MSK = timezone(timedelta(hours=3))


def counts(values):
    return dict(sorted(Counter(str(v) for v in values).items()))


def missing(value):
    return value is None or value == '' or value == [] or value == {}


def profile(rows):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    return {
        'rows': len(rows),
        'fields': {k: {'types': counts(type(r.get(k)).__name__ for r in rows),
                       'missing_or_empty': sum(missing(r.get(k)) for r in rows)} for k in fields},
    }


def csv_comparison(path, rows):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        csv_rows = list(reader)
        header = reader.fieldnames
    mismatch = Counter()
    for left, right in zip(csv_rows, rows):
        for key, value in right.items():
            cell = left.get(key)
            equal = cell == str(value)
            if value is None:
                equal = cell == ''
            elif isinstance(value, bool):
                equal = str(cell).lower() == str(value).lower()
            elif isinstance(value, (int, float)):
                try:
                    equal = math.isclose(float(cell), value, rel_tol=1e-12, abs_tol=1e-9)
                except (TypeError, ValueError):
                    equal = False
            if not equal:
                mismatch[key] += 1
    return {'csv_rows': len(csv_rows), 'json_rows': len(rows),
            'same_fields': set(header or []) == set(rows[0]),
            'mismatches_by_field': dict(mismatch), 'compared_in_file_order': True,
            'numeric_tolerance': 'relative 1e-12, absolute 1e-9; null equals empty CSV cell'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.source_dir.iterdir())
    raw_path = next(p for p in paths if p.suffix == '.json' and p.name not in
                    {'brand_dictionary.json', 'golden_set_26_31_aug.json', 'top25_sources.json'})
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    raw = read(raw_path)
    gold = read(args.source_dir / 'golden_set_26_31_aug.json')
    sources = read(args.source_dir / 'top25_sources.json')
    dictionary = read(args.source_dir / 'brand_dictionary.json')
    by_id = {r['id']: r for r in raw}
    dates = lambda zone: counts(datetime.fromtimestamp(r['timeCreate'], zone).date().isoformat() for r in raw)
    normalized = lambda s: ' '.join(s.casefold().split())
    text_counts = Counter(normalized(r['text']) for r in raw)
    groups = defaultdict(list)
    for r in raw:
        groups[(r['hub'], r['hubtype'])].append(r)
    source_errors = Counter()
    metric_map = {'total_audience': 'audienceCount', 'total_views': 'viewsCount',
                  'total_likes': 'likesCount', 'total_reposts': 'repostsCount'}
    def number(value):
        return float(value) if value != '' and value is not None else 0
    for s in sources:
        group = groups.get((s['source'], s['source_type']), [])
        actual = {'messages': len(group)}
        actual.update({key: sum(number(r[field]) for r in group) for key, field in metric_map.items()})
        actual['total_engagement'] = sum(r['likesCount'] + r['commentsCount'] + r['repostsCount'] for r in group)
        if group:
            actual['avg_audience'] = actual['total_audience'] / len(group)
            actual['avg_engagement'] = actual['total_engagement'] / len(group)
        for key, value in actual.items():
            if not math.isclose(value, s[key], abs_tol=0.005, rel_tol=1e-9):
                source_errors[key] += 1
    joined = [(g, by_id[g['message_id']]) for g in gold if g['message_id'] in by_id]
    same = {key: sum(g[key] == r[field] for g, r in joined) for key, field in
            {'text': 'text', 'url': 'url', 'source': 'hub', 'source_type': 'hubtype',
             'audience': 'audienceCount', 'views': 'viewsCount'}.items()}
    time_delta = counts(int((datetime.strptime(g['date']+' '+g['time'], '%d.%m.%Y %H:%M') -
                            datetime.fromtimestamp(r['timeCreate'], MSK).replace(tzinfo=None, second=0)).total_seconds()/60)
                        for g, r in joined)
    summary_lengths = [len(g['summary_160_300']) for g in gold]
    match_span = sum(g['matched_span'].casefold() in g['text'].casefold() for g in gold if g['brand_link'] != 'none')
    raw_fields = ['text','title','url']
    privacy = {
        'raw_url_nonempty': sum(bool(r['url']) for r in raw),
        'author_fullname_nonempty': sum(bool(r['authorObject'].get('fullname')) for r in raw),
        'author_url_nonempty': sum(bool(r['authorObject'].get('url')) for r in raw),
        'author_hash_nonempty': sum(bool(r['authorObject'].get('hash')) for r in raw),
        'author_age_nonempty': sum(not missing(r['authorObject'].get('age')) for r in raw),
        'geo_nonempty': sum(any(not missing(r.get(k)) for k in ('city','region','country','geoObject')) for r in raw),
        'text_with_url_pattern': sum(bool(re.search(r'https?://|www\.', r['text'])) for r in raw),
        'text_with_email_pattern': sum(bool(re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', r['text'])) for r in raw),
        'text_with_phone_like_pattern': sum(bool(re.search(r'(?<!\w)(?:\+7|8)[\s()-]*\d(?:[\s()-]*\d){9}(?!\d)', r['text'])) for r in raw),
        'method_limit': 'Presence/regex screening only, not a guarantee of anonymization; no values emitted',
    }
    result = {
        'method': {'runtime': 'Python 3.11+ standard library', 'timezone': 'UTC and UTC+03:00',
                   'duplicate_text': 'casefold + collapse whitespace', 'confidential': True},
        'files': [{'name': p.name, 'bytes': p.stat().st_size,
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths if p.is_file()],
        'raw': {**profile(raw), 'days_msk': dates(MSK), 'days_utc': dates(timezone.utc),
                'duplicate_ids': len(raw)-len(by_id),
                'duplicate_urls': len(raw)-len({r['url'] for r in raw}),
                'normalized_text_groups': len(text_counts),
                'duplicate_text_excess': len(raw)-len(text_counts),
                'largest_text_group': max(text_counts.values()),
                'sentiment_ba': counts(r['toneMark'] for r in raw),
                'source_types': counts(r['hubtype'] for r in raw),
                'message_types': counts(r['type'] for r in raw),
                'language': counts(r['language'] for r in raw),
                'unique_source_names': len({r['hub'] for r in raw}),
                'unique_source_type_pairs': len(groups),
                'metrics_negative': {k: sum(number(r[k]) < 0 for r in raw) for k in
                                     ('audienceCount','viewsCount','likesCount','commentsCount','repostsCount')},
                'text_length_min': min(len(r['text']) for r in raw),
                'text_length_max': max(len(r['text']) for r in raw),
                'empty_text_with_nonempty_title':sum(not r['text'] and bool(r['title']) for r in raw)},
        'golden': {**profile(gold), 'csv_comparison': csv_comparison(args.source_dir/'golden_set_26_31_aug.csv',gold),
                   'joined_by_id': len(joined), 'duplicate_ids': len(gold)-len({g['message_id'] for g in gold}),
                   'equal_fields_to_raw': same, 'time_delta_minutes_vs_msk': time_delta,
                   'label_counts': {k: counts(g[k] for g in gold) for k in
                                    ('sentiment_correct','brand_link','recommended_action','human_review_needed','mixed')},
                   'unique_clusters': len({g['cluster_id'] for g in gold}),
                   'cluster_size_counts': counts(g['cluster_size'] for g in gold),
                   'rank_unique': len({g['rank_in_digest'] for g in gold}),
                   'rank_min_max': [min(g['rank_in_digest'] for g in gold),max(g['rank_in_digest'] for g in gold)],
                   'summary_length_min_max': [min(summary_lengths),max(summary_lengths)],
                   'summary_outside_160_300': sum(n<160 or n>300 for n in summary_lengths),
                   'matched_span_in_text_for_relevant': match_span,
                   'relevant_rows': sum(g['brand_link']!='none' for g in gold)},
        'top_sources': {**profile(sources), 'csv_comparison': csv_comparison(args.source_dir/'top25_sources.csv',sources),
                        'source_type_pairs_found_in_raw': sum((s['source'],s['source_type']) in groups for s in sources),
                        'duplicate_source_names': len(sources)-len({s['source'] for s in sources}),
                        'duplicate_source_type_pairs': len(sources)-len({(s['source'],s['source_type']) for s in sources}),
                        'covered_raw_records_by_source_type_pair': sum(len(groups.get((s['source'],s['source_type']),[])) for s in sources),
                        'messages_sum':sum(s['messages'] for s in sources),
                        'mismatches_vs_unfiltered_raw': dict(source_errors),
                        'aggregate_tolerance': 'absolute 0.005, relative 1e-9; missing views treated as zero for reconciliation only',
                        'media_weight_min_max':[min(s['media_weight'] for s in sources),max(s['media_weight'] for s in sources)]},
        'dictionary': {k: {'category': v.get('category'), 'pattern_count': len(v.get('patterns',[])),
                           'empty_patterns': sum(not x.strip() for x in v.get('patterns',[])),
                           'duplicate_casefold_patterns':len(v.get('patterns',[]))-len({x.casefold() for x in v.get('patterns',[])})}
                       for k,v in dictionary.items() if isinstance(v,dict) and 'patterns' in v},
        'dictionary_minus_word_count':len(dictionary['minus_words']),
        'privacy': privacy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'raw_records':len(raw),'golden_records':len(gold),'source_records':len(sources),
                      'output':str(args.output)},ensure_ascii=True))


if __name__ == '__main__':
    main()
