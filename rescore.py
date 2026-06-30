#!/usr/bin/env python3
"""Delete existing scores in a problem folder and re-run the judge evals.

Finds every *run workspace* (a directory that contains a ``_meta.json``) under a
target folder, removes its ``_score*.json`` files, and re-scores it with the
judge ensemble configured in ``evaluation/.env`` (JUDGE_MODELS + OpenRouter, or
the single JUDGE_MODEL_NAME judge). This only re-runs scoring; it does not re-run
the research agent.

Examples:
    # Re-score every run under a batch / problem directory
    ./rescore.py workspaces/cli_runs/cli_20260629_ab12cd34

    # Re-score all Astronomy_000 runs sitting directly under workspaces/
    ./rescore.py workspaces --task Astronomy_000

    # See what would happen without touching anything
    ./rescore.py workspaces --task Astronomy_000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Load judge credentials BEFORE importing evaluation.* (config reads env at import).
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / "evaluation" / ".env")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.score import score_workspace_ensemble  # noqa: E402


def find_run_workspaces(root: Path, task_prefix: str | None) -> list[Path]:
    """Return run-workspace dirs (those containing _meta.json) under ``root``.

    ``root`` itself counts if it is a run workspace. When ``task_prefix`` is set,
    only runs whose _meta.json task_id matches (or, failing that, whose directory
    name starts with the prefix) are returned.
    """
    candidates: list[Path] = []
    if (root / "_meta.json").is_file():
        candidates.append(root)
    candidates.extend(sorted(p.parent for p in root.rglob("_meta.json") if p.parent != root))

    if not task_prefix:
        return candidates

    matched: list[Path] = []
    for ws in candidates:
        task_id = ""
        try:
            task_id = json.loads((ws / "_meta.json").read_text(encoding="utf-8")).get("task_id", "")
        except Exception:
            pass
        if task_id.startswith(task_prefix) or ws.name.startswith(task_prefix):
            matched.append(ws)
    return matched


def delete_scores(workspace: Path) -> list[str]:
    removed = []
    for score_file in sorted(workspace.glob("_score*.json")):
        score_file.unlink()
        removed.append(score_file.name)
    return removed


def rescore_one(workspace: Path, *, dry_run: bool) -> tuple[Path, list[str], dict]:
    removed = [] if dry_run else delete_scores(workspace)
    if dry_run:
        removed = [f.name for f in sorted(workspace.glob("_score*.json"))]
        return workspace, removed, {"dry_run": True}
    result = score_workspace_ensemble(workspace)
    return workspace, removed, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Problem/batch folder (or a single run workspace) to re-score.")
    parser.add_argument("--task", default=None, help="Only re-score runs whose task_id / dir name starts with this prefix.")
    parser.add_argument("--workers", type=int, default=1, help="Number of runs to score concurrently (default: 1).")
    parser.add_argument("--dry-run", action="store_true", help="List runs and score files that would be deleted; change nothing.")
    args = parser.parse_args(argv)

    root = Path(args.folder).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    runs = find_run_workspaces(root, args.task)
    if not runs:
        print(f"No run workspaces (dirs with _meta.json) found under {root}"
              + (f" matching task prefix '{args.task}'" if args.task else ""), file=sys.stderr)
        return 1

    print(f"Found {len(runs)} run workspace(s) to re-score under {root}"
          + (f" (task prefix '{args.task}')" if args.task else ""))
    if args.dry_run:
        for ws in runs:
            existing = [f.name for f in sorted(ws.glob("_score*.json"))]
            print(f"  [dry-run] {ws.relative_to(root) if ws != root else ws.name}: "
                  f"would delete {existing or '(none)'} and re-score")
        return 0

    failures = 0
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(rescore_one, ws, dry_run=False): ws for ws in runs}
        for i, future in enumerate(as_completed(futures), 1):
            ws = futures[future]
            label = ws.relative_to(root) if ws != root else ws.name
            try:
                _, removed, result = future.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[{i}/{len(runs)}] {label}: ERROR {type(exc).__name__}: {exc}")
                continue
            if isinstance(result, dict) and result.get("error"):
                failures += 1
                print(f"[{i}/{len(runs)}] {label}: deleted {removed or '(none)'} -> SCORING FAILED: {result['error']}")
            else:
                total = result.get("total_score") if isinstance(result, dict) else None
                std = result.get("total_score_std") if isinstance(result, dict) else None
                std_note = f" (std {std})" if std is not None else ""
                print(f"[{i}/{len(runs)}] {label}: deleted {removed or '(none)'} -> total_score={total}{std_note}")

    print(f"\nDone. {len(runs) - failures}/{len(runs)} re-scored successfully.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
