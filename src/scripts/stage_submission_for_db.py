"""Stage the files in ``submission/cad/`` for the official ingest + evaluation pipeline.

The submission ships one flat pair per request::

    submission/cad/<request_id>/<request_id>.step
    submission/cad/<request_id>/<request_id>.stl

``build_instructions_db.py`` crawls for a very different shape (see ``example_data/``)::

    <edit_id>/brep_end/<end_time>/{settings.json, tmp.step, tmp.stl, tmp_<view>.png}

This script writes that tree under ``data/model_outputs/`` for a **separate user id**, so the
submitted artifacts can be re-ingested and re-scored end to end without disturbing the run they
came from. It also writes a derived config listing the new user in ``benchmark_eval_users``,
because both the metric computation and the plots skip any user missing from that list.

The point of the exercise is the round trip: by default the ``.stl`` is *not* copied, so
``cadquery_convert.py`` re-tessellates it from the submitted STEP and the scores then rest on
the STEP alone. Pass ``--copy-stl`` to score the exact meshes the submission ships instead.
The two differ on requests whose ground-truth edit is voxel-identical to the start model: there
the GT diff is empty, so ``diff_f1`` is 1.0 only if the prediction's diff is *exactly* empty,
which a fresh triangulation does not reproduce.

    uv run python src/scripts/stage_submission_for_db.py --config src/config/edit_192_external.json

Then follow the printed commands. ``--purge`` removes the staged tree, the derived config and
every database record and brep file belonging to the staging user, so the round trip can be
repeated from a clean state.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.utils.db import DatabaseManager
from src.utils.process_config import load_config

# Any base works; it only has to be stable across runs and distinct per request, because
# DatabaseManager.get_brep_id_str builds the brep id as f"{user}_{end_time}".
END_TIME_BASE = 1800000000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config/edit_192_external.json")
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--user-id", default="submission-roundtrip",
                        help="User id the staged edits are ingested under.")
    parser.add_argument("--harness-dir", default="data/model_outputs/cadquery_script",
                        help="Parent directory; must sit inside a config models_dir path.")
    parser.add_argument("--copy-stl", action="store_true",
                        help="Copy the submitted .stl instead of re-tessellating from the STEP.")
    parser.add_argument("--full-eval-users", action="store_true",
                        help="Keep the parent config's whole benchmark_eval_users list and only "
                             "append the staging user. Off by default: a freshly downloaded "
                             "dataset carries none of the three hackathon metrics, so scoring "
                             "every listed user costs hundreds of computations before the one "
                             "number the round trip is about appears.")
    parser.add_argument("--purge", action="store_true",
                        help="Remove the staged tree, the derived config and all db records.")
    return parser.parse_args()


def derived_config_path(config_path: Path, user_id: str) -> Path:
    return config_path.with_name(f"{config_path.stem}_{user_id.replace('-', '_')}.json")


def write_derived_config(config_path: Path, user_id: str, full_eval_users: bool) -> Path:
    """Same config, but with ``user_id`` in benchmark_eval_users.

    Nothing outside that list is scored or plotted, so the staging user has to be added. By
    default the list is also cut down to the staging user plus the two human wildcards: "gt
    human" resolves the reference edit, "other human" draws the dashed human ceiling, and every
    other entry would only add computations the round trip does not ask about.
    """
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if full_eval_users:
        users = raw["benchmark_eval_users"]["edit"]
        if user_id not in users:
            users.append(user_id)
    else:
        raw["benchmark_eval_users"]["edit"] = ["gt human", "other human", user_id]
    raw["_note_roundtrip"] = (
        f"Derived from {config_path.name} by stage_submission_for_db.py: benchmark_eval_users "
        f"names '{user_id}', the submission re-ingested from its own STEP files. Use the "
        "original config to regenerate the official plots."
    )
    target = derived_config_path(config_path, user_id)
    target.write_text(json.dumps(raw, indent=4) + "\n", encoding="utf-8")
    return target


def stage(args: argparse.Namespace, cad_dir: Path, staging_root: Path) -> list[str]:
    request_ids = sorted(path.name for path in cad_dir.iterdir() if path.is_dir())
    if not request_ids:
        raise SystemExit(f"No request folders under {cad_dir}")

    for index, request_id in enumerate(request_ids):
        step_source = cad_dir / request_id / f"{request_id}.step"
        if not step_source.is_file() or step_source.stat().st_size == 0:
            raise SystemExit(f"Missing or empty STEP: {step_source}")

        end_time = END_TIME_BASE + index
        edit_dir = staging_root / f"{args.user_id}_{request_id}" / "brep_end" / f"{end_time}"
        edit_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(step_source, edit_dir / "tmp.step")

        stl_source = cad_dir / request_id / f"{request_id}.stl"
        if args.copy_stl:
            if not stl_source.is_file():
                raise SystemExit(f"--copy-stl given but missing: {stl_source}")
            shutil.copy2(stl_source, edit_dir / "tmp.stl")
        else:
            # Leave it absent so cadquery_convert.py rebuilds it from the STEP.
            (edit_dir / "tmp.stl").unlink(missing_ok=True)

        settings = {
            "edit_request_id": request_id,
            "edit_id": f"{args.user_id}_{request_id}",
            "userId": args.user_id,
            # Must stay False: display_rating_results folds every is_human user into
            # "other human", which would corrupt the human ceiling.
            "isHuman": False,
            "start_time": end_time,
            "end_time": end_time,
            "fileName": None,
            "failed_run": False,
            "source_step": step_source.as_posix(),
            "stl_origin": ("submitted verbatim" if args.copy_stl
                           else "re-tessellated from the submitted STEP"),
        }
        (edit_dir / "settings.json").write_text(
            json.dumps(settings, indent=2) + "\n", encoding="utf-8"
        )
    return request_ids


def purge(config: dict, args: argparse.Namespace, staging_dir: Path, config_path: Path) -> None:
    db = DatabaseManager(config)
    edits = list(db.edits.find({"user": args.user_id}))
    brep_dir = Path(db.root_dir) / db.brep_dir

    removed_files = 0
    for edit in edits:
        db.ratings.delete_many({"edit": edit["_id"]})
        brep_id = edit.get("brep_end")
        if not brep_id:
            continue
        # insert_brep never overwrites an existing copy, so a stale file would silently
        # survive a re-ingest under the same brep id.
        for path in brep_dir.glob(f"{brep_id}*"):
            path.unlink()
            removed_files += 1
        db.breps.delete_one({"_id": brep_id})

    db.edits.delete_many({"user": args.user_id})
    db.users.delete_many({"_id": args.user_id})
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    derived = derived_config_path(config_path, args.user_id)
    derived.unlink(missing_ok=True)
    print(f"purged user '{args.user_id}': {len(edits)} edits, {removed_files} brep files, "
          f"staging tree and {derived.name}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(str(config_path))
    staging_root = Path(args.harness_dir) / args.user_id / "val_edit_text"

    if args.purge:
        purge(config, args, staging_root.parent, config_path)
        return

    models_dirs = [Path(path).resolve() for path in config["models_dir"]["paths"]]
    if not any(staging_root.resolve().is_relative_to(path) for path in models_dirs):
        raise SystemExit(
            f"{staging_root} is outside every config models_dir ({models_dirs}); "
            "build_instructions_db.py would never find it."
        )

    request_ids = stage(args, Path(args.submission) / "cad", staging_root)
    derived = write_derived_config(config_path, args.user_id, args.full_eval_users)

    stl = ("copied verbatim from the submission" if args.copy_stl
           else "rebuilt from the STEP by cadquery_convert.py")
    print(f"staged {len(request_ids)} requests as user '{args.user_id}'")
    print(f"  tree   {staging_root.as_posix()}")
    print(f"  config {derived.as_posix()}")
    print(f"  stl    {stl}")
    print(f"  scored {', '.join(json.loads(derived.read_text(encoding='utf-8'))['benchmark_eval_users']['edit'])}")
    print("\nNext, in order:")
    print(f"  uv run python src/scripts_preprocess/cadquery_convert.py {staging_root.as_posix()} --no-views")
    print(f"  uv run python src/scripts/build_instructions_db.py --config {derived.as_posix()}")
    print(f"  uv run python src/scripts/run_all_benchmarks.py --config {derived.as_posix()}")
    print("\nUndo with --purge. To regenerate the official plots afterwards:")
    print(f"  uv run python src/scripts/run_all_benchmarks.py --config {config_path.as_posix()}")


if __name__ == "__main__":
    main()
