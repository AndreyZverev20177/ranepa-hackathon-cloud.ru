import argparse
import json
import sys
from pathlib import Path

from .core import InputError, digest, load, parse_date, render
from .partner import adapt
from .clustering import SemanticClusterConfig
from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingConfig, EmbeddingService, EmbeddingError
from .source_authority import SourceAuthorityRegistry, SourceAuthorityError, load_scoring_rules
from .scoring import PublicationScoreWeights
from .story_scoring import StoryScoreWeights, load_story_weights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Локальный baseline дайджеста; качество на реальных данных не измерено')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--partner-export', action='store_true', help='Input is original partner JSON export')
    parser.add_argument('--synthetic', action='store_true', help='Явно пометить результат как вымысел')
    parser.add_argument('--scoring', choices=('baseline', 'publication-v2', 'story-v1'), default='baseline',
                        help='baseline / publication-v2 / story-v1 (по всем публикациям кластера)')
    parser.add_argument('--sources-file', type=Path, default=Path('partner_data/top25_sources.json'),
                        help='Путь к исходному top25_sources.json (нужен для publication-v2)')
    parser.add_argument('--scoring-rules', type=Path, default=Path('config/publication_scoring_v2.json'),
                        help='Настройки авторитетности и весов скоринга publication-v2')
    parser.add_argument('--story-rules', type=Path, default=Path('config/story_scoring_v1.json'),
                        help='Настройки Story Score (только для --scoring story-v1)')
    parser.add_argument('--clustering', choices=('baseline', 'semantic'), default='baseline',
                        help='По умолчанию старый baseline; semantic включает embeddings')
    parser.add_argument('--similarity-high', type=float, default=0.88)
    parser.add_argument('--similarity-low', type=float, default=0.78)
    parser.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument('--embedding-revision', default=None)
    parser.add_argument('--embedding-device', default=None)
    parser.add_argument('--embedding-cache-dir', type=Path, default=Path('artifacts/embedding_cache'))
    args = parser.parse_args(argv)
    writing = False
    try:
        day = parse_date(args.date)
        if not 1 <= args.top_n <= 30:
            raise InputError('top-n: требуется целое 1–30')
        targets = [args.output_dir / f'digest-{day}.{extension}' for extension in ('json', 'txt')]
        if any(args.input.resolve() == target.resolve() or (target.exists() and args.input.exists() and target.samefile(args.input)) for target in targets):
            raise InputError('Вход не может совпадать с выходным файлом')
        rows, warnings = (adapt(args.input), []) if args.partner_export else load(args.input)
        if warnings:
            print('Предупреждение: неизвестные поля проигнорированы: ' + ', '.join(warnings), file=sys.stderr)
        semantic = args.clustering == 'semantic'
        service = EmbeddingService(EmbeddingConfig(
            model_name=args.embedding_model, revision=args.embedding_revision,
            device=args.embedding_device, cache_dir=args.embedding_cache_dir,
        )) if semantic else None
        cluster_cfg = SemanticClusterConfig(high_threshold=args.similarity_high,
                                           low_threshold=args.similarity_low) if semantic else None
        rules = load_scoring_rules(args.scoring_rules) if args.scoring in ('publication-v2', 'story-v1') else None
        registry = SourceAuthorityRegistry.from_json(args.sources_file, rules=rules) if rules is not None else None
        scoring_weights = PublicationScoreWeights.from_settings(rules['weights']) if rules is not None else None
        story_weights = load_story_weights(args.story_rules) if args.scoring == 'story-v1' else None
        result = digest(rows, day, args.top_n, args.synthetic,
                        semantic=semantic, embedding_service=service, clustering_config=cluster_cfg,
                        scoring=args.scoring, source_registry=registry, scoring_weights=scoring_weights,
                        story_weights=story_weights)
        payloads = [json.dumps(result, ensure_ascii=False, indent=2) + '\n', render(result)]
        writing = True
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for target, payload in zip(targets, payloads):
            target.write_text(payload, encoding='utf-8')
        print(f'Карточек: {len(result["items"])}; файлы: {targets[0]}, {targets[1]}')
        return 0
    except (InputError, EmbeddingError, SourceAuthorityError, ValueError) as exc:
        print(f'Ошибка входа: {exc}', file=sys.stderr)
        return 2
    except Exception:
        # Do not expose original text, file contents or provider metadata in errors.
        suffix = ' Запись не завершена; файлы могут быть старыми или частичными.' if writing else ''
        print('Ошибка чтения/записи или внутренняя ошибка.' + suffix, file=sys.stderr)
        return 1
