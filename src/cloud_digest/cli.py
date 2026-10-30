import argparse
import json
import sys
from pathlib import Path

from .core import InputError, digest, load, parse_date, render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Локальный baseline дайджеста; качество на реальных данных не измерено')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--synthetic', action='store_true', help='Явно пометить результат как вымысел')
    args = parser.parse_args(argv)
    writing = False
    try:
        day = parse_date(args.date)
        if not 1 <= args.top_n <= 30:
            raise InputError('top-n: требуется целое 1–30')
        targets = [args.output_dir / f'digest-{day}.{extension}' for extension in ('json', 'txt')]
        if any(args.input.resolve() == target.resolve() or (target.exists() and args.input.exists() and target.samefile(args.input)) for target in targets):
            raise InputError('Вход не может совпадать с выходным файлом')
        rows, warnings = load(args.input)
        if warnings:
            print('Предупреждение: неизвестные поля проигнорированы: ' + ', '.join(warnings), file=sys.stderr)
        result = digest(rows, day, args.top_n, args.synthetic)
        payloads = [json.dumps(result, ensure_ascii=False, indent=2) + '\n', render(result)]
        writing = True
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for target, payload in zip(targets, payloads):
            target.write_text(payload, encoding='utf-8')
        print(f'Карточек: {len(result["items"])}; файлы: {targets[0]}, {targets[1]}')
        return 0
    except InputError as exc:
        print(f'Ошибка входа: {exc}', file=sys.stderr)
        return 2
    except Exception:
        # Do not expose original text, file contents or provider metadata in errors.
        suffix = ' Запись не завершена; файлы могут быть старыми или частичными.' if writing else ''
        print('Ошибка чтения/записи или внутренняя ошибка.' + suffix, file=sys.stderr)
        return 1
