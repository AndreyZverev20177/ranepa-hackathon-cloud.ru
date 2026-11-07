#!/usr/bin/env python3
"""Windows-friendly launcher for the existing baseline-vs-semantic evaluation.

No scoring or clustering implementation is changed by this launcher.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the existing Cloud Digest baseline vs. semantic comparison"
    )
    parser.add_argument(
        "gold", nargs="?", default="gold/2026-08-31_auto_DRAFT.json",
        help="Path to draft or verified gold file (relative to repository root)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the commands and validate paths without loading the model",
    )
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    input_candidates = sorted((repo / "partner_data").glob("*26-31.08.json"))
    if len(input_candidates) != 1:
        print(
            "[ERROR] Expected exactly one source dataset matching "
            "partner_data/*26-31.08.json; found: " + str(len(input_candidates)),
            file=sys.stderr,
        )
        for candidate in input_candidates:
            print(" -", candidate, file=sys.stderr)
        return 2
    source_data = input_candidates[0]
    gold = Path(args.gold)
    gold = gold if gold.is_absolute() else repo / gold
    if not gold.is_file():
        print(f"[ERROR] Gold file not found: {gold}", file=sys.stderr)
        return 2

    report_script = repo / "tools" / "compare_report.py"
    if not report_script.is_file():
        print(
            "[ERROR] tools/compare_report.py is missing. "
            "Extract the previous comparison patch first.",
            file=sys.stderr,
        )
        return 2

    output_dir = repo / "results" / "model_comparison"
    output_json = output_dir / "evaluation_fixed_thresholds.json"
    output_prefix = output_dir / "baseline_vs_semantic"

    evaluate_command = [
        sys.executable, "-m", "cloud_digest.evaluate_cli",
        "--input", str(source_data), "--partner-export",
        "--date", "2026-08-31", "--gold", str(gold),
        "--semantic", "--high-grid", "0.88", "--low-grid", "0.78",
        "--embedding-device", "cpu", "--output", str(output_json),
    ]
    report_command = [
        sys.executable, str(report_script), "--report", str(output_json),
        "--output-prefix", str(output_prefix),
    ]

    print(f"[INFO] Python: {sys.executable}", flush=True)
    print(f"[INFO] Dataset: {source_data}", flush=True)
    print(f"[INFO] Gold: {gold}", flush=True)
    if args.dry_run:
        print("[DRY RUN] Commands to execute:")
        print(subprocess.list2cmdline(evaluate_command))
        print(subprocess.list2cmdline(report_command))
        return 0

    print("[CHECK] Importing sentence_transformers...", flush=True)
    import_check = subprocess.run(
        [sys.executable, "-c", "from sentence_transformers import SentenceTransformer"],
        cwd=repo,
        check=False,
    )
    if import_check.returncode != 0:
        print(
            '[ERROR] sentence_transformers failed to import in this environment.\n'
            'Check the traceback above. To install in .venv, run:\n'
            '  .venv\\Scripts\\python.exe -m pip install -e ".[embeddings]"',
            file=sys.stderr,
        )
        return import_check.returncode

    output_dir.mkdir(parents=True, exist_ok=True)
    print("[RUN] Evaluating baseline and real semantic embeddings...", flush=True)
    result = subprocess.run(evaluate_command, cwd=repo, check=False)
    if result.returncode != 0:
        print("[ERROR] Evaluation failed. See the original error above.", file=sys.stderr)
        return result.returncode
    if not output_json.is_file():
        print("[ERROR] Evaluation returned success but output JSON was not created.", file=sys.stderr)
        return 2
    print("[RUN] Building CSV and Markdown comparison...", flush=True)
    result = subprocess.run(report_command, cwd=repo, check=False)
    if result.returncode != 0:
        print("[ERROR] Comparison report generation failed.", file=sys.stderr)
        return result.returncode
    print("[OK] Open:", output_prefix.with_suffix(".md"))
    print("[OK] Open:", output_prefix.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
