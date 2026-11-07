"""Streamlit-панель редактора: читает результат CLI, хранит правки локально.

Запуск из корня репозитория: python -m streamlit run ui/app.py
"""
from __future__ import annotations

from datetime import date
from hashlib import sha256
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit

import streamlit as st

from cloud_digest.dataset_upload import detect_dataset, save_uploaded_dataset
from cloud_digest.editor_review import (
    effective_status, export_editorial_digest, load_store, review_of, save_review,
)

ROOT = Path(__file__).resolve().parents[1]
EDITOR_STORE = ROOT / 'artifacts' / 'editor_reviews.json'
INPUT_DIR = ROOT / 'partner_data'
RESULTS_DIR = ROOT / 'results' / 'ui'
UPLOADS_DIR = RESULTS_DIR / 'uploads'

TONE_LABELS = {
    'negative': 'Негативная', 'neutral': 'Нейтральная',
    'positive': 'Позитивная', 'as_model': 'Не изменять',
}
STATUS_LABELS = {
    'pending': 'Не рассмотрено', 'needs_review': 'На проверке',
    'approved': 'Подтверждено', 'excluded': 'Исключено',
}
BRAND_LABELS = {
    'as_model': 'Не изменять', 'direct': 'Прямая связь',
    'historical': 'Историческое название', 'none': 'Нет связи',
}
CLUSTER_LABELS = {
    'none': 'Нет', 'check_merge': 'Предложить объединение',
    'check_split': 'Предложить разделение',
}


def safe_url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    parsed = urlsplit(value)
    if parsed.scheme in ('http', 'https') and parsed.netloc:
        return value
    return None


def discover_results() -> list[Path]:
    candidates: list[Path] = []
    for location in (ROOT / 'results', ROOT / 'partner_results'):
        if location.exists():
            candidates.extend(p for p in location.rglob('digest-*.json') if p.is_file())
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)[:150]


def load_digest(payload: str | bytes) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict) or not isinstance(value.get('items'), list) or not isinstance(value.get('digest_date'), str):
        raise ValueError('Выбранный JSON не является дайджестом Cloud Digest')
    return value


def story_of(item: dict[str, Any]) -> dict[str, Any]:
    """Story-v1 полноценен; для baseline показываем карточку с ограничениями."""
    card = item.get('story_card')
    if isinstance(card, dict):
        return card
    return {
        'title': item.get('summary_160_300', '').splitlines()[0][:140] or f"Публикация {item.get('message_id', '')}",
        'description_excerpt': item.get('summary_160_300', ''),
        'main_source': {
            'source_name': item.get('source_name'), 'url': item.get('url'),
            'published_at': item.get('published_at'), 'publication_score': item.get('importance_score'),
        },
        'priority': {'story_score': item.get('importance_score'), 'contributions': []},
        'spread': {'publication_count': item.get('cluster_size', 1), 'distinct_outlets': None},
        'timeline': [{
            'message_id': item.get('message_id'), 'source_name': item.get('source_name'),
            'published_at': item.get('published_at'), 'url': item.get('url'),
            'publication_score': item.get('importance_score'), 'is_main_source': True,
        }],
        'review': {'required': item.get('human_review_needed', False), 'reasons': item.get('review_reasons', [])},
    }


def run_pipeline(input_file: Path, day: date, mode: str, use_semantic: bool, *, partner_export: bool) -> tuple[bool, str, Path]:
    # Разделяем результаты разных датасетов, алгоритмов и режимов: не перезаписываем старый дайджест.
    file_hash = sha256(input_file.read_bytes()).hexdigest()[:12]
    output_dir = RESULTS_DIR / ('semantic' if use_semantic else 'baseline-clusters') / day.isoformat() / mode / file_hash
    args = [
        sys.executable, '-m', 'cloud_digest',
        '--input', str(input_file), '--date', day.isoformat(),
        '--output-dir', str(output_dir), '--scoring', mode,
    ]
    if partner_export:
        args.append('--partner-export')
    if use_semantic:
        args += ['--clustering', 'semantic', '--embedding-device', 'cpu']
    try:
        completed = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=1200, check=False)
    except subprocess.TimeoutExpired:
        return False, 'Обработка превысила 20 минут. Проверьте загрузку модели и ресурсы компьютера.', output_dir
    except OSError as exc:
        return False, f'Не удалось запустить Python: {exc}', output_dir
    message = (completed.stdout + '\n' + completed.stderr).strip()
    return completed.returncode == 0, message or f'Код завершения: {completed.returncode}', output_dir


st.set_page_config(page_title='Cloud Digest · Редактор', page_icon='☁️', layout='wide')
st.markdown('''
<style>
.block-container {padding-top: 1.5rem; max-width: 1380px;}
h1, h2, h3 {letter-spacing: -0.02em;}
[data-testid="stMetric"] {border: 1px solid #dae1ec; border-radius: 12px; padding: 14px;}
</style>
''', unsafe_allow_html=True)

st.title('Cloud Digest')
st.caption('Редакторская панель интеллектуального отбора · данные из Brand Analytics через локальную выгрузку')

def mark_active_result(kind: str) -> None:
    st.session_state['active_digest'] = kind


with st.sidebar:
    st.header('Данные')
    files = discover_results()
    file_options = {str(p.relative_to(ROOT)): p for p in files}
    selected_name = st.selectbox(
        'Открыть готовый дайджест', ['— Автоматически —', *file_options.keys()],
        on_change=mark_active_result, args=('chosen_digest',),
    )
    st.caption('Здесь нужен результат digest-*.json, а не исходная выгрузка Brand Analytics.')
    uploaded = st.file_uploader(
        'Загрузить готовый дайджест (.json)', type=['json'], key='digest_upload',
        on_change=mark_active_result, args=('uploaded_digest',),
    )
    st.divider()
    with st.expander('Сформировать новый дайджест', expanded=True):
        st.markdown('**Исходные публикации**')
        dataset_upload = st.file_uploader(
            'Загрузить свой датасет BA / Cloud Digest (.json, .csv)',
            type=['json', 'csv'], key='dataset_upload',
            help='BA JSON с полями id/timeCreate; либо Cloud Digest JSON/CSV с message_id, published_at и др.',
        )
        sources = sorted(INPUT_DIR.glob('*.json')) if INPUT_DIR.exists() else []
        sources = [p for p in sources if 'выгрузк' in p.name.casefold()]
        source = st.selectbox('Или выбрать локальную выгрузку', sources, format_func=lambda p: p.name) if sources else None
        day = st.date_input('Дата дайджеста', value=date(2026, 8, 31))
        mode = st.selectbox('Скоринг', ['story-v1', 'publication-v2', 'baseline'], index=0)
        semantic = st.toggle('Семантическая кластеризация (embeddings)', value=False,
                             help='Требует установки extras [embeddings] и при первом запуске скачивает модель.')
        if dataset_upload is not None:
            st.caption(f'Для обработки выбран загруженный файл: {dataset_upload.name}')
        if st.button('Обработать выгрузку', type='primary',
                     disabled=source is None and dataset_upload is None, use_container_width=True):
            try:
                if dataset_upload is not None:
                    source_path, input_kind = save_uploaded_dataset(
                        dataset_upload.getvalue(), dataset_upload.name, UPLOADS_DIR)
                    partner_export = input_kind == 'partner'
                else:
                    source_path = source
                    if source_path is None:
                        raise ValueError('Не выбран исходный датасет')
                    partner_export = detect_dataset(source_path.read_bytes(), source_path.name) == 'partner'
                with st.spinner('Запускаю Cloud Digest; это может занять время при загрузке модели…'):
                    ok, msg, folder = run_pipeline(source_path, day, mode, semantic,
                                                   partner_export=partner_export)
                if ok:
                    st.session_state['latest_result_path'] = str(folder / f'digest-{day.isoformat()}.json')
                    st.session_state['active_digest'] = 'generated'
                    st.success('Дайджест создан и открыт ниже. Для папки «Загрузки» нажми кнопку скачивания в основной области.')
                else:
                    st.error('Ошибка запуска. Вывод команды:')
                    st.code(msg or 'Нет подробностей')
            except (OSError, UnicodeError, ValueError) as exc:
                st.error(f'Ошибка входного датасета: {exc}')
    st.divider()
    st.caption('Файлы обработки сохраняются в results/ui/. Правки — в artifacts/editor_reviews.json. Прямой связи с API Brand Analytics пока нет.')

try:
    origin_path = None
    active = st.session_state.get('active_digest')
    latest_string = st.session_state.get('latest_result_path')
    latest_path = Path(latest_string) if latest_string else None
    if active == 'generated' and latest_path and latest_path.is_file():
        origin_path = latest_path
        digest = load_digest(latest_path.read_text(encoding='utf-8'))
        origin = str(latest_path.relative_to(ROOT))
    elif active == 'uploaded_digest' and uploaded is not None:
        if uploaded.size > 20 * 1024 * 1024:
            raise ValueError('Загружаемый дайджест не должен превышать 20 МБ')
        digest = load_digest(uploaded.getvalue())
        origin = uploaded.name
    elif active == 'chosen_digest' and selected_name in file_options:
        origin_path = file_options[selected_name]
        digest = load_digest(origin_path.read_text(encoding='utf-8'))
        origin = selected_name
    elif latest_path and latest_path.is_file():
        origin_path = latest_path
        digest = load_digest(latest_path.read_text(encoding='utf-8'))
        origin = str(latest_path.relative_to(ROOT))
    elif selected_name in file_options:
        origin_path = file_options[selected_name]
        digest = load_digest(origin_path.read_text(encoding='utf-8'))
        origin = selected_name
    elif uploaded is not None:
        digest = load_digest(uploaded.getvalue())
        origin = uploaded.name
    elif files:
        origin_path = files[0]
        digest = load_digest(origin_path.read_text(encoding='utf-8'))
        origin = str(origin_path.relative_to(ROOT))
    else:
        digest = None
        origin = ''
    store = load_store(EDITOR_STORE)
except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
    st.error(f'Ошибка данных: {exc}')
    st.stop()

if digest is None:
    st.info('Пока нет дайджестов. Запусти обработку в боковой панели или загрузи готовый JSON.')
    st.stop()

st.caption(f'Открыт: `{origin}` · дата {digest["digest_date"]} · алгоритм `{digest.get("rule_version", "не указан")}`')
# Браузер не позволяет серверу незаметно записывать файлы в «Загрузки».
# Пользователь сам нажимает «Скачать»; JSON и TXT уже лежат в results/ui/.
digest_day = digest['digest_date']
json_payload = json.dumps(digest, ensure_ascii=False, indent=2).encode('utf-8')
txt_path = origin_path.with_suffix('.txt') if origin_path is not None else None
txt_payload = txt_path.read_bytes() if txt_path is not None and txt_path.is_file() else None
file1, file2, file3 = st.columns(3)
with file1:
    st.download_button('Скачать JSON', data=json_payload, file_name=f'digest-{digest_day}.json',
                       mime='application/json', use_container_width=True)
if txt_payload is not None:
    with file2:
        st.download_button('Скачать TXT', data=txt_payload, file_name=f'digest-{digest_day}.txt',
                           mime='text/plain', use_container_width=True)
    with file3:
        bundle = BytesIO()
        with ZipFile(bundle, 'w', ZIP_DEFLATED) as archive:
            archive.writestr(f'digest-{digest_day}.json', json_payload)
            archive.writestr(f'digest-{digest_day}.txt', txt_payload)
        st.download_button('Скачать всё ZIP', data=bundle.getvalue(),
                           file_name=f'cloud-digest-{digest_day}.zip',
                           mime='application/zip', use_container_width=True)
st.caption('На компьютере сервера результат уже сохранён в results/ui/. В папку «Загрузки» браузер сохраняет его после нажатия кнопки.')
if not str(digest.get('rule_version', '')).endswith('story-scoring-v1'):
    st.warning('Это не Story Score: история распространения и подробная разбивка могут отсутствовать. Для полного интерфейса сформируй режим story-v1.')

items = [x for x in digest['items'] if isinstance(x, dict) and x.get('cluster_id')]
day_str = digest['digest_date']
count_reviews = sum(effective_status(x, review_of(store, day_str, x)) == 'needs_review' for x in items)
count_approved = sum(effective_status(x, review_of(store, day_str, x)) == 'approved' for x in items)
count_excluded = sum(effective_status(x, review_of(store, day_str, x)) == 'excluded' for x in items)

cols = st.columns(4)
for col, label, value in zip(cols, ('Сюжетов', 'На проверке', 'Подтверждено', 'Исключено'),
                              (len(items), count_reviews, count_approved, count_excluded)):
    col.metric(label, value)

st.divider()
search_col, status_col, score_col = st.columns([3, 2, 2])
with search_col:
    search = st.text_input('Поиск по сюжету или источнику', placeholder='Например, выручка Cloud.ru').casefold().strip()
with status_col:
    status_filter = st.selectbox('Статус', ['Все', *STATUS_LABELS.values()])
with score_col:
    score_min = st.slider('Минимальный балл', min_value=0, max_value=100, value=0)

filtered = []
for item in items:
    view = story_of(item)
    status = effective_status(item, review_of(store, day_str, item))
    name = str(view['title'])
    publisher = str(view['main_source'].get('source_name') or '')
    if search and search not in (name + ' ' + publisher).casefold():
        continue
    if status_filter != 'Все' and STATUS_LABELS[status] != status_filter:
        continue
    if int(item.get('importance_score', 0)) < score_min:
        continue
    filtered.append(item)

left, right = st.columns([1, 2], gap='large')
with left:
    st.subheader('Лента сюжетов')
    st.caption(f'Отображено: {len(filtered)} из {len(items)}')
    if not filtered:
        st.info('По заданным фильтрам сюжетов нет.')
        st.stop()
    options = {str(x['cluster_id']): x for x in filtered}
    chosen = st.selectbox(
        'Выбрать сюжет', list(options),
        format_func=lambda cid: f"{options[cid].get('importance_score', 0):>3}/100 · {story_of(options[cid])['title'][:85]}",
    )
    st.dataframe([
        {
            'Балл': x.get('importance_score'),
            'Сюжет': story_of(x)['title'][:100],
            'Публикаций': story_of(x)['spread'].get('publication_count'),
            'Статус': STATUS_LABELS[effective_status(x, review_of(store, day_str, x))],
        }
        for x in filtered
    ], hide_index=True, use_container_width=True)
    with st.expander(f"Очередь первичной проверки BA/Cloud Digest ({len(digest.get('review_queue', []))})"):
        st.caption('Включает спорные публикации, которые могли не попасть в top-N. Здесь показывается очередь движка, без редакторского фильтра.')
        queue_rows = digest.get('review_queue', [])
        if queue_rows:
            st.dataframe([{
                'ID': q.get('message_id'),
                'Кластер': str(q.get('cluster_id') or '')[:16],
                'Причины': ', '.join(q.get('reasons', [])),
                'Источник': safe_url(q.get('url')),
            } for q in queue_rows], hide_index=True, use_container_width=True,
                column_config={'Источник': st.column_config.LinkColumn('Источник')})
        else:
            st.info('Очередь пуста.')

item = options[chosen]
view = story_of(item)
review = review_of(store, day_str, item)
status = effective_status(item, review)

with right:
    head1, head2 = st.columns([4, 1])
    with head1:
        st.subheader(view['title'])
    with head2:
        st.metric('Story Score', f"{item.get('importance_score', 0)}/100")
    st.caption(f"Главный источник: {view['main_source'].get('source_name', '—')} · публикаций: {view['spread'].get('publication_count', 1)} · {STATUS_LABELS[status]}")
    st.write(review.get('summary') or view.get('description_excerpt') or 'Фрагмент отсутствует')
    url = safe_url(view['main_source'].get('url'))
    if url:
        st.link_button('Открыть публикацию', url)
    tab_priority, tab_timeline, tab_editor = st.tabs(['Почему такой приоритет', 'История сюжета', 'Правка редактора'])
    with tab_priority:
        priority = view.get('priority', {})
        contributions = priority.get('contributions', [])
        if contributions:
            st.dataframe([{'Компонент': c.get('label', c.get('code')), 'Баллы': c.get('points', 0)}
                          for c in contributions], hide_index=True, use_container_width=True)
        else:
            st.write('Подробная разбивка доступна в режиме story-v1.')
        st.caption('Охват — оценочный показатель, не подтверждённая уникальная аудитория.')
        st.write('**Тональность:** BA —', TONE_LABELS.get(item.get('ba_sentiment'), 'нет данных'),
                 '· Cloud Digest —', TONE_LABELS.get(item.get('sentiment'), 'нет данных'))
        reasons = view.get('review', {}).get('reasons', [])
        if reasons:
            st.warning('Причины редакторской проверки: ' + ', '.join(map(str, reasons)))
    with tab_timeline:
        timeline = view.get('timeline', [])
        if timeline:
            st.dataframe([{
                'Дата / время': p.get('published_at'),
                'Источник': p.get('source_name'),
                'Publication Score': p.get('publication_score'),
                'Аудитория': p.get('audience'),
                'Ссылка': safe_url(p.get('url')),
            } for p in timeline], hide_index=True, use_container_width=True,
                         column_config={'Ссылка': st.column_config.LinkColumn('Ссылка')})
        else:
            st.info('Для этого сюжета история распространения отсутствует.')
    with tab_editor:
        st.caption('Правки сохраняются локально и отражаются в редакторском экспорте; исходные баллы не пересчитываются.')
        with st.form(f"review-{day_str}-{chosen}"):
            status_choices = list(STATUS_LABELS)
            chosen_status = st.selectbox('Решение редактора', status_choices,
                                         index=status_choices.index(review.get('status', 'pending')), format_func=STATUS_LABELS.get)
            tone_choices = list(TONE_LABELS)
            chosen_tone = st.selectbox('Исправление тональности', tone_choices,
                                       index=tone_choices.index(review.get('tone', 'as_model')), format_func=TONE_LABELS.get)
            brand_choices = list(BRAND_LABELS)
            chosen_brand = st.selectbox('Связь с брендом', brand_choices,
                                        index=brand_choices.index(review.get('brand_link', 'as_model')), format_func=BRAND_LABELS.get)
            summary = st.text_area('Редакторская выжимка', value=review.get('summary', ''),
                                   placeholder='Пустое поле сохраняет исходный фрагмент', max_chars=1000)
            notes = st.text_area('Комментарий / причина решения', value=review.get('notes', ''), max_chars=2000)
            with st.expander('Предложить исправление кластеризации'):
                action_choices = list(CLUSTER_LABELS)
                action = st.selectbox('Действие', action_choices,
                                      index=action_choices.index(review.get('cluster_action', 'none')), format_func=CLUSTER_LABELS.get)
                cluster_target = st.text_input('cluster_id для объединения', value=review.get('cluster_target', ''))
                split_ids = st.text_input('ID отделяемых публикаций (через запятую)', value=review.get('split_message_ids', ''))
                st.caption('Это заявка на исправление. Она НЕ меняет кластеры и Story Score автоматически.')
            submitted = st.form_submit_button('Сохранить решение', type='primary')
        if submitted:
            try:
                save_review(EDITOR_STORE, day_str, chosen, {
                    'status': chosen_status, 'tone': chosen_tone,
                    'brand_link': chosen_brand, 'summary': summary, 'notes': notes,
                    'cluster_action': action, 'cluster_target': cluster_target,
                    'split_message_ids': split_ids,
                })
            except (OSError, ValueError) as exc:
                st.error(f'Не удалось сохранить: {exc}')
            else:
                st.success('Решение сохранено локально.')
                st.rerun()

st.divider()
export = export_editorial_digest(digest, store)
st.download_button('Скачать редакторский JSON', data=json.dumps(export, ensure_ascii=False, indent=2).encode('utf-8'),
                   file_name=f'editorial-digest-{day_str}.json', mime='application/json')
st.caption('Исключённые сюжеты отсутствуют в items, но перечислены в excluded_by_editor. Кластеризация, скоринг и тональность BA не меняются автоматически. Для обратной записи в Brand Analytics потребуется согласованный API и права.')
