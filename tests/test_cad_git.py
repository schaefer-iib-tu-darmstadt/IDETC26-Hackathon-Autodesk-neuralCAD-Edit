"""The Git4CAD gates. Every gate was written in response to a measured failure, so each test
names the failure it guards against.
"""

import os.path as osp
import cadquery as cq

import pytest
from src.agents.cad_git import CadRepo

from conftest import DATASET_INVALID_START


# --- the origin gate ------------------------------------------------------------------------

def test_submitting_the_origin_is_refused(repo, steps):
    """Guards: a script that cancels every edit would otherwise commit, become HEAD, win the
    later-wins tie-break and export the untouched input for diff_f1 0.0."""
    r = repo.try_commit(steps["origin"], attempt=1, render=False)
    assert r.get("reverted_to_origin") is True
    assert r["committed"] is False


def test_undoing_back_to_the_origin_after_a_commit_is_refused(repo, steps):
    """Guards: the parent comparison alone cannot see this -- it is the silent-zero path."""
    repo.try_commit(steps["edited"], attempt=1, render=False)
    r = repo.try_commit(steps["origin"], attempt=2, render=False)
    assert r.get("reverted_to_origin") is True


# --- a normal edit --------------------------------------------------------------------------

def test_a_real_edit_commits_and_moves_head(repo, steps):
    """Guards: checkpoint mode -- the next script must build on this state."""
    r = repo.try_commit(steps["edited"], attempt=1, render=False)
    assert r["committed"] and r["head_advanced"]
    assert repo.head_step == repo.commits[0].step_path


def test_resubmitting_head_is_refused(repo, steps):
    """Guards: wasted turns."""
    repo.try_commit(steps["edited"], attempt=1, render=False)
    r = repo.try_commit(steps["edited"], attempt=2, render=False)
    assert "identical to HEAD" in (r.get("reason") or "")


# --- the assembly-drop gate -----------------------------------------------------------------

def test_dropping_the_assembly_does_not_move_head(repo, steps):
    """Guards: a wreck must not become the base of the next script."""
    repo.try_commit(steps["edited"], attempt=1, render=False)
    r = repo.try_commit(steps["dropped"], attempt=2, render=False)
    assert r["committed"] is True
    assert r["head_advanced"] is False


def test_the_dropped_candidate_is_still_recorded(repo, steps):
    """Guards: every candidate stays available for export, however bad it is as a base."""
    repo.try_commit(steps["edited"], attempt=1, render=False)
    repo.try_commit(steps["dropped"], attempt=2, render=False)
    assert len(repo.commits) == 2


def test_an_identical_repeat_is_reported_as_a_duplicate(repo, steps):
    """Guards: ten identical commits on the first probe run cost ~118 s of render each."""
    repo.try_commit(steps["edited"], attempt=1, render=False)
    repo.try_commit(steps["dropped"], attempt=2, render=False)
    r = repo.try_commit(steps["dropped"], attempt=3, render=False)
    assert r.get("duplicate_of") == "c2"
    assert repo.duplicates == 1


def test_rotated_regular_cutout_is_not_reported_as_a_duplicate(tmp_path):
    """Guards: rotating a symmetric hex cut changed Dev16 diff-F1 by +0.4053."""
    plate = cq.Workplane("XY").box(40, 40, 5)

    def cutout(angle):
        cutter = (
            cq.Workplane("XY")
            .polygon(6, 20)
            .extrude(10, both=True)
            .rotate((0, 0, 0), (0, 0, 1), angle)
        )
        return plate.cut(cutter)

    origin = tmp_path / "origin.step"
    first = tmp_path / "hex_0.step"
    rotated = tmp_path / "hex_30.step"
    cq.exporters.export(plate, str(origin))
    cq.exporters.export(cutout(0), str(first))
    cq.exporters.export(cutout(30), str(rotated))

    local_repo = CadRepo(str(origin), str(tmp_path / "repo"))
    assert local_repo.try_commit(str(first), attempt=1, render=False)["committed"] is True
    result = local_repo.try_commit(str(rotated), attempt=2, render=False)

    assert result["committed"] is True
    assert result.get("duplicate_of") is None
    assert len(local_repo.commits) == 2


def test_a_deliberate_component_removal_does_move_head(repo, steps):
    """Guards: three requests ask for exactly this; a bare solid-count veto would block them
    permanently, because the target state has fewer solids than the input."""
    r = repo.try_commit(steps["merged"], attempt=1, render=False)
    assert r["committed"] and r["head_advanced"]


def test_a_global_transform_is_not_called_damage(repo, steps):
    """Guards: 'Scale the part 10x' matches no face at all, so preserved_ratio reads 0 %."""
    r = repo.try_commit(steps["moved"], attempt=1, render=False)
    assert r["committed"] and r["head_advanced"]
    assert not r.get("problems")
    assert r["preserved_ratio"] == 0.0      # the very reason a bare ratio veto would be wrong


# --- revert ---------------------------------------------------------------------------------

def test_revert_onto_a_refused_commit_is_denied(history_repo):
    """Guards: revert must not be a back door for installing a rejected base."""
    r = history_repo.revert(to=2)
    assert r["ok"] is False


def test_revert_skips_the_refused_commit_and_lands_on_c1(history_repo):
    """Guards: `head.n - 1` would land on c2, which the kernel refused as a base."""
    r = history_repo.revert()
    assert r["head"] == "c1"


def test_revert_cannot_move_forward(history_repo):
    """Guards: that is a redo, not a revert."""
    history_repo.revert()
    r = history_repo.revert(to=3)
    assert r["ok"] is False


def test_revert_note_names_commits_not_a_python_list(history_repo):
    """Guards: the note goes to the model, and `c{dropped}` rendered a list as `c[3]`."""
    r = history_repo.revert()
    assert "c[" not in (r.get("note") or "")
    assert "c3" in (r.get("note") or "")


def test_export_ignores_reverted_commits(history_repo):
    """Guards: shipping the state the agent explicitly rejected."""
    history_repo.revert()
    _, _, report = history_repo.final_step()
    assert report["chosen_commit"] == 1


# --- inherited invalidity (needs the dataset) -----------------------------------------------

@pytest.mark.dataset
@pytest.mark.skipif(not osp.exists(DATASET_INVALID_START),
                    reason=f"{DATASET_INVALID_START} not present")
def test_the_known_invalid_start_model_is_detected_as_invalid(tmp_path):
    """Guards: 1 of its 20 solids is broken before any edit, so an absolute validity gate would
    veto every attempt on this task and export the input for diff_f1 0.0."""
    from src.agents.cad_git import CadRepo

    repo = CadRepo(origin_step=DATASET_INVALID_START, work_dir=str(tmp_path / "repo"))
    assert repo.origin_summary["is_valid"] is False
    assert repo.origin_summary["invalid_solids"] == [0]
