"""Regression guards for the checkpoint substrate (`cad_git`, `cad_tools`, `cad_trace`).

Each test fails against the code as it stood before the fix; the docstring names what breaks.
"""

import os.path as osp
import shutil
import threading
import time
from pathlib import Path

import pytest

from src.agents.cad_git import CadRepo, _ship_start_stl, assess_commit
from src.agents.cad_tools import load_shape, summarise_shape


# --- the input fallback has to carry the dataset's own mesh -------------------------------------

def test_ship_start_stl_copies_the_dataset_mesh(tmp_path, steps):
    """`diff_f1` is an F1 between two voxel diffs, so an empty ground-truth diff is matched only by
    a prediction whose diff is EXACTLY empty. Letting the scorer re-tessellate our STEP breaks that
    on the 3 of 48 requests where changing nothing is the correct answer -- measured 1.0 -> 0.0."""
    start_step = str(tmp_path / "start.step")
    shutil.copy(steps["origin"], start_step)
    start_stl = tmp_path / "start.stl"
    start_stl.write_text("solid dataset_mesh\nendsolid dataset_mesh\n", encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()
    _ship_start_stl(start_step, str(out))

    shipped = out / "tmp.stl"
    assert shipped.exists(), "the fallback shipped no .stl at all"
    assert shipped.read_text(encoding="utf-8") == start_stl.read_text(encoding="utf-8"), \
        "the mesh must be the dataset's, verbatim -- a re-export is what costs the point"


def test_ship_start_stl_is_silent_when_no_mesh_sits_next_to_the_step(tmp_path, steps):
    """3 of the 438 start STEPs have no sibling .stl; that must degrade, not raise."""
    start_step = str(tmp_path / "lonely.step")
    shutil.copy(steps["origin"], start_step)
    out = tmp_path / "out"
    out.mkdir()
    _ship_start_stl(start_step, str(out))
    assert not (out / "tmp.stl").exists()


# --- ADK runs the tool calls of one turn concurrently -------------------------------------------

def test_concurrent_commits_get_distinct_commit_numbers(repo, steps, monkeypatch):
    """ADK dispatches each function call of a turn as its own task (`flows/llm_flows/functions.py`)
    and runs sync tools in a thread pool. Without a lock both calls read `len(self.commits)` before
    either appends, take the same `n`, and write the same `c{n:03d}.step`."""
    original = repo._summarise

    def slow_summarise(path):
        # Widens the window between reading `n` and appending, so the race is deterministic
        # rather than a coin flip.
        result = original(path)
        time.sleep(0.15)
        return result

    monkeypatch.setattr(repo, "_summarise", slow_summarise)

    results = {}

    def submit(key, step):
        results[key] = repo.try_commit(step, attempt=1, render=False)

    threads = [threading.Thread(target=submit, args=("a", steps["edited"])),
               threading.Thread(target=submit, args=("b", steps["moved"]))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(repo.commits) == 2
    assert sorted(c.n for c in repo.commits) == [1, 2]
    assert len({c.step_path for c in repo.commits}) == 2, "two commits shared one STEP file"


# --- a renderer failure must not swallow the commit ---------------------------------------------

def test_a_throwing_renderer_does_not_lose_the_commit(repo, steps, monkeypatch):
    """`c{n:03d}.step` is written before the render. If the render raised, the commit was never
    appended, so the next attempt computed the same `n` and silently overwrote the file."""
    def boom(*args, **kwargs):
        raise OSError("renderer exploded")

    monkeypatch.setattr(repo, "_render_diff", boom)

    with pytest.raises(OSError):
        repo.try_commit(steps["edited"], attempt=1, render=True)

    assert len(repo.commits) == 1, "the commit was dropped along with the render"
    assert repo.head is not None and repo.head.n == 1
    assert osp.exists(repo.commits[0].step_path)


# --- degraded inputs must degrade the run, not end it -------------------------------------------

def test_repo_without_an_origin_still_commits(tmp_path, steps):
    """A dataset row with no start STEP used to reach `assess_commit(None, ...)` and die in
    `preserved_ratio` with an AttributeError."""
    repo = CadRepo(origin_step=None, work_dir=str(tmp_path / "repo"))
    r = repo.try_commit(steps["edited"], attempt=1, render=False)
    assert r["committed"] is True
    assert r["head_advanced"] is True


def test_assess_commit_without_an_origin_lets_the_candidate_through(steps):
    summary = summarise_shape(load_shape(steps["edited"]))
    advance, warnings, metrics, notes = assess_commit(None, summary)
    assert advance is True
    assert warnings == [] and notes == []
    assert metrics["preserved_ratio"] == 1.0


def test_a_corrupt_origin_degrades_instead_of_aborting(tmp_path):
    """The constructor summarised the origin unguarded, so an unreadable STEP killed the task
    before the agent had seen the request."""
    bad = tmp_path / "broken.step"
    bad.write_text("this is not a STEP file", encoding="utf-8")
    repo = CadRepo(origin_step=str(bad), work_dir=str(tmp_path / "repo"))
    assert repo.origin_summary is None
    assert repo.origin_error, "the failure has to be recorded, not just swallowed"


# --- the fingerprint fast path must not change what it reports ----------------------------------

def test_derived_bbox_matches_the_exact_one(steps):
    """The union of the exact per-solid boxes IS the exact box of the shape; if that ever stops
    holding, the change gate and `assess_damage` start reading a different geometry. The slow
    `-m slow` sweep checks the same identity over every start model of the dataset."""
    shape = load_shape(steps["origin"])
    exact = summarise_shape(shape, max_faces=400)
    derived = summarise_shape(shape, max_faces=400, derive_bbox_from_solids=True)
    assert derived["bbox"] == exact["bbox"]
    assert derived == exact


# --- tracing is a diagnostic, never a reason to lose a task -------------------------------------

def test_tracer_disables_itself_instead_of_raising(tmp_path):
    """The constructor opened the file unguarded; on a path too long for Windows it took the whole
    task with it."""
    from src.agents.cad_trace import CadRunTracer

    target = tmp_path / "a_directory"
    target.mkdir()
    tracer = CadRunTracer(path=str(target))     # a directory: open() must fail
    assert tracer.disabled is True
    tracer._emit("anything", x=1)               # must not raise
    summary = tracer.summary()
    assert summary["trace_file"] is None
    assert summary["trace_error"]


# --- the STL only belongs to an export that really IS the input ---------------------------------

def test_final_step_returns_the_origin_only_when_nothing_was_committed(repo, steps):
    """The agent decides whether to ship the dataset's `.stl` by comparing the exported path
    against the input, not by reading the label -- `fallback_head_after_selection_error` also
    starts with "fallback" but ships a real edit, and pairing that STEP with the start mesh would
    hand the scorer the unedited geometry as our answer."""
    source, label, _ = repo.final_step()
    assert osp.abspath(source) == osp.abspath(steps["origin"])
    assert label.startswith("fallback")

    repo.try_commit(steps["edited"], attempt=1, render=False)
    source, label, _ = repo.final_step()
    assert osp.abspath(source) != osp.abspath(steps["origin"])
    assert not label.startswith("fallback")
