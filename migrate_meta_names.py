#!/usr/bin/env python3
"""One-shot normalization of run metadata naming drift.

Rewrites ``agent_name`` and ``model`` to their canonical forms (see
``evaluation/canonical_names.py``) in every ``_meta.json`` / ``_score.json`` under
a workspace tree, repairs the ``run_id`` field to match its containing folder
(folder name is the authoritative run id used by ``utils._is_run_workspace``),
backfills a missing ``_score.json`` ``model`` from the meta, and removes the
stray nested ``workspaces_orig_scorer/workspaces_orig_scorer/`` duplicate tree.

Defaults to a dry run. Examples:
    ./migrate_meta_names.py                 # dry-run over workspaces/
    ./migrate_meta_names.py --apply         # write changes to workspaces/
    ./migrate_meta_names.py --all --apply   # also the alt-scorer trees
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.canonical_names import canonical_agent, canonical_model  # noqa: E402

DEFAULT_TREE = REPO_ROOT / "workspaces"
ALT_TREES = [
    REPO_ROOT / "workspaces_multi-scorer_highModels",
    REPO_ROOT / "workspaces_orig_scorer",
]
NESTED_DUP = REPO_ROOT / "workspaces_orig_scorer" / "workspaces_orig_scorer"


def find_workspaces(root: Path) -> list[Path]:
    """Run-workspace dirs (those containing _meta.json) under ``root``."""
    out = []
    if (root / "_meta.json").is_file():
        out.append(root)
    out.extend(sorted(p.parent for p in root.rglob("_meta.json") if p.parent != root))
    return out


def _planned_changes(path: Path, folder_name: str) -> dict:
    """Return {field: (old, new)} for the JSON file at ``path`` (empty if none)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    changes = {}

    if "agent_name" in data:
        new = canonical_agent(data["agent_name"])
        if new != data["agent_name"]:
            changes["agent_name"] = (data["agent_name"], new)

    # model lives in _meta.json only; _score.json never carries one.
    if data.get("model"):
        new = canonical_model(data["model"])
        if new != data["model"]:
            changes["model"] = (data["model"], new)

    if "run_id" in data and data["run_id"] != folder_name:
        changes["run_id"] = (data["run_id"], folder_name)

    return changes


def _apply(path: Path, changes: dict) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for field, (_old, new) in changes.items():
        data[field] = new
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def process_tree(root: Path, *, apply: bool) -> int:
    n_changed = 0
    for ws in find_workspaces(root):
        folder_name = ws.name
        for fname in ("_meta.json", "_score.json"):
            fpath = ws / fname
            if not fpath.is_file():
                continue
            changes = _planned_changes(fpath, folder_name)
            if not changes:
                continue
            n_changed += 1
            rel = fpath.relative_to(REPO_ROOT)
            for field, (old, new) in changes.items():
                print(f"  {'WRITE' if apply else 'plan '} {rel}: {field}: {old!r} -> {new!r}")
            if apply:
                _apply(fpath, changes)
    return n_changed


def handle_nested_dup(*, apply: bool) -> None:
    if not NESTED_DUP.is_dir():
        return
    parent = NESTED_DUP.parent
    nested_runs = [d.name for d in NESTED_DUP.iterdir() if d.is_dir()]
    missing = [name for name in nested_runs if not (parent / name).is_dir()]
    if missing:
        print(f"  WARN  {NESTED_DUP.relative_to(REPO_ROOT)} has run dirs not present in "
              f"its parent ({missing}); NOT removing. Resolve manually.")
        return
    print(f"  {'REMOVE' if apply else 'plan  remove'} duplicate tree "
          f"{NESTED_DUP.relative_to(REPO_ROOT)} ({len(nested_runs)} run dirs, all mirrored in parent)")
    if apply:
        shutil.rmtree(NESTED_DUP)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    parser.add_argument("--all", action="store_true", help="Also process the alt-scorer workspace trees.")
    args = parser.parse_args(argv)

    trees = [DEFAULT_TREE] + (ALT_TREES if args.all else [])
    mode = "APPLY" if args.apply else "DRY-RUN"
    total = 0
    for tree in trees:
        if not tree.is_dir():
            continue
        print(f"[{mode}] {tree.relative_to(REPO_ROOT)}")
        total += process_tree(tree, apply=args.apply)

    print(f"[{mode}] nested duplicate check")
    handle_nested_dup(apply=args.apply)

    print(f"\n{mode}: {total} file(s) "
          f"{'updated' if args.apply else 'would change'}.")
    if not args.apply:
        print("Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
