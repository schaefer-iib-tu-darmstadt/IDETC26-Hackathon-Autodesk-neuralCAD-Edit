"""Re-score the files in ``submission/`` with the repository's own metric functions.

This is the independent check behind ``submission/scores.json``: it ignores every rating
stored in the database and recomputes all three metrics straight from the submitted
``.stl`` files, using the exact functions ``benchmark_evals/edit.py`` calls. Ground truth
and the start model still come from the database, because that is where they live.

Read-only — it never writes to the database, unlike ``run_all_benchmarks.py``.

    uv run python src/scripts/verify_submission.py --config src/config/edit_192_external.json

Chamfer samples 10 000 points per mesh without a seed, so its deltas carry ~0.004 of
sampling noise; the two voxel metrics are deterministic and must match to ~1e-9.
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.utils.db import DatabaseManager
from src.utils.evals_diff import diff_f1, volumetric_f1
from src.utils.evals_feature_geometric import chamfer_similarity_norm
from src.utils.process_config import load_config

METRICS = ("chamfer_similarity_norm", "volumetric_f1", "diff_f1")
# Chamfer is stochastic (10k unseeded samples); the voxel metrics are not.
TOLERANCE = {"chamfer_similarity_norm": 0.01, "volumetric_f1": 1e-9, "diff_f1": 1e-9}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config/edit_192_external.json")
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N requests (smoke test).")
    parser.add_argument("--save-json", default=None, help="Write the recomputed rows here.")
    return parser.parse_args()


def first(value):
    return value[0] if isinstance(value, list) else value


def collect_tasks(db: DatabaseManager, cad_dir: Path) -> list[dict]:
    """One row per request: submitted mesh plus the GT and start meshes to score against."""
    tasks = []
    for request in db.requests.find({"request_type": "edit"}):
        request_id = request["_id"]
        predicted = cad_dir / request_id / f"{request_id}.stl"
        start_brep = db.breps.find_one({"_id": request["brep_start"]})
        gt_edit = db.edits.find_one({"request": request_id, "user": request["user"]})
        gt_brep = db.breps.find_one({"_id": gt_edit["brep_end"]}) if gt_edit else None
        tasks.append({
            "request_id": request_id,
            "difficulty": request.get("difficulty"),
            "predicted": str(predicted.resolve()) if predicted.is_file() else None,
            "gt": first(gt_brep.get("stl")) if gt_brep else None,
            "start": first(start_brep.get("stl")) if start_brep else None,
        })
    return sorted(tasks, key=lambda row: row["request_id"])


def score(task: dict, db: DatabaseManager) -> dict:
    predicted, gt, start = task["predicted"], task["gt"], task["start"]
    if not (predicted and gt):
        return {**task, **{key: None for key in METRICS}}
    return {
        **task,
        "chamfer_similarity_norm": float(chamfer_similarity_norm(gt, predicted, db=db)),
        "volumetric_f1": float(volumetric_f1(gt, predicted, db=db)),
        "diff_f1": float(diff_f1(gt, predicted, db=db, start_rel=start)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    db = DatabaseManager(config)

    submission = Path(args.submission)
    claimed_path = submission / "scores.json"
    claimed_doc = json.loads(claimed_path.read_text(encoding="utf-8"))
    claimed = {row["request_id"]: row for row in claimed_doc["per_task"]}

    tasks = collect_tasks(db, submission / "cad")
    if args.limit:
        tasks = tasks[: args.limit]
    missing = [row["request_id"] for row in tasks if not row["predicted"]]
    print(f"requests in db: {len(tasks)}   submitted meshes: {len(tasks) - len(missing)}")
    if missing:
        print(f"  MISSING from {submission}/cad: {missing}")
    if not args.limit:
        unexpected = sorted(set(claimed) - {row["request_id"] for row in tasks})
        if unexpected:
            print(f"  in scores.json but not in the db: {unexpected}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda task: score(task, db), tasks))

    print(f"\n{'request_id':38s} " + " ".join(f"{key:>26s}" for key in METRICS))
    failures = []
    for row in rows:
        cells = []
        for key in METRICS:
            got = row[key]
            want = claimed.get(row["request_id"], {}).get(key)
            if got is None or want is None:
                cells.append(f"{'--':>26s}")
                failures.append((row["request_id"], key, want, got))
                continue
            delta = abs(got - want)
            flag = " " if delta <= TOLERANCE[key] else "!"
            cells.append(f"{got:11.6f} ({delta:+.1e}){flag}")
            if flag == "!":
                failures.append((row["request_id"], key, want, got))
        print(f"{row['request_id']:38s} " + " ".join(cells))

    print()
    if args.limit:
        print(f"(--limit {args.limit}: per-task deltas above are meaningful, the aggregate is not)")
    else:
        for key in METRICS:
            values = [row[key] for row in rows if row[key] is not None]
            mean = sum(values) / len(values) if values else float("nan")
            want = claimed_doc["metrics"][key]
            print(f"{key:26s} recomputed={mean:.6f}  submitted={want:.6f}  delta={mean - want:+.2e}")

    if failures:
        print(f"\nFAIL: {len(failures)} metric(s) outside tolerance")
        for request_id, key, want, got in failures[:20]:
            print(f"  {request_id} {key}: submitted={want} recomputed={got}")
    else:
        print("\nOK: every recomputed score matches submission/scores.json within tolerance")

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.save_json}")


if __name__ == "__main__":
    main()
