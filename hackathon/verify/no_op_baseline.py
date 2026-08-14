"""No-op baseline for hackathon metrics.

No-operation = unedited start model

Read-only: it opens the mongita DB, resolves STL paths itself and calls the
repo's own metric functions. It writes nothing into the database.

    uv run python hackathon/verify/no_op_baseline.py --config src/config/edit_192_external.json

"""

import argparse
import json
import os.path as osp

import numpy as np
import open3d as o3d

from src.utils.db import DatabaseManager
from src.utils.evals_diff import diff_f1, volumetric_f1
from src.utils.evals_feature_geometric import chamfer_similarity_norm
from src.utils.process_config import load_config

METRICS = ("chamfer_similarity_norm", "volume_f1", "diff_f1")


def stl_rel(db, brep_id):
    """Relative STL path of a brep, or None if absent on disk."""
    if not brep_id:
        return None
    brep = db.breps.find_one({"_id": brep_id})
    if not brep:
        return None
    val = brep.get("stl")
    if not val:
        return None
    rel = val[0] if isinstance(val, list) else val
    return rel if osp.exists(osp.join(db.root_dir, rel)) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the main db config file")
    parser.add_argument("--out", default="hackathon/verify/no_op_baseline.json")
    parser.add_argument("--seed", type=int, default=0,
                        help="Pins the chamfer point sampling; the official pipeline does not.")
    args = parser.parse_args()

    o3d.utility.random.seed(args.seed)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    db = DatabaseManager(load_config(args.config))
    requests = sorted(db.requests.find({"request_type": "edit"}), key=lambda r: r["_id"])

    records = []
    for i, req in enumerate(requests, 1):
        start_rel = stl_rel(db, req.get("brep_start"))
        gt_edit = db.edits.find_one({"request": req["_id"], "user": req["user"]})
        gt_rel = stl_rel(db, gt_edit["brep_end"]) if gt_edit else None

        if gt_rel is None or start_rel is None:
            print(f"[{i}/{len(requests)}] {req['_id']}: SKIP (no gt or start stl)", flush=True)
            continue

        record = {"request": req["_id"], "difficulty": req.get("difficulty")}
        for name, fn in (
            ("chamfer_similarity_norm", lambda: chamfer_similarity_norm(gt_rel, start_rel, db)),
            ("volume_f1", lambda: volumetric_f1(gt_rel, start_rel, db)),
            ("diff_f1", lambda: diff_f1(gt_rel, start_rel, db, start_rel=start_rel)),
        ):
            try:
                value = float(fn())
            except Exception as exc:  # a metric failure is a 0, not a crashed run
                print(f"    !! {name}: {type(exc).__name__}: {exc}", flush=True)
                value = float("nan")
            record[name] = 0.0 if value != value else value

        records.append(record)
        print(f"[{i}/{len(requests)}] {req['_id']} ({req.get('difficulty')}) done", flush=True)

    with open(args.out, "w") as f:
        json.dump(records, f, indent=1)

    if records:
        print(f"\nno-op over {len(records)} requests:")
        for metric in METRICS:
            print(f"  {metric:<26}{np.mean([r[metric] for r in records]):.3f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
