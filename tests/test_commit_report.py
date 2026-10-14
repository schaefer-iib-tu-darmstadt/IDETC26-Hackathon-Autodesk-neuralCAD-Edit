"""What a commit reports back to the agent: the faces the edit created or removed, the preserved
ratio the gate scores, and the solid roster of an assembly.

`apply_edit` forwards `changed_faces` to the model after every accepted commit, so this is the
agent's only structural view of what its own script just did. The assertions pin two properties:
the delta is measured against HEAD (each step reports its own contribution, never earlier work),
and the report and the gate read the same face matcher, so they can never disagree about what was
touched.
"""

import pytest

from src.agents.cad_git import face_delta, preserved_ratio, solid_roster
from src.agents.cad_tools import load_shape, summarise_shape


# -- the per-commit face delta --------------------------------------------------------------------

def test_a_commit_reports_the_faces_the_edit_created(repo, steps):
    """The boss fused onto the plate: its wall, its top, and the host face it now sits in."""
    delta = repo.try_commit(steps["edited"], attempt=1, render=False)["changed_faces"]
    assert delta["faces_added"] == 3
    assert delta["faces_removed"] == 1                  # the plate's top face, before the imprint
    cylinders = [row for row in delta["new_faces"] if row["type"] == "CYLINDER"]
    assert [row["radius"] for row in cylinders] == [pytest.approx(4.0)]


def test_the_delta_is_measured_against_head_not_against_the_origin(repo, steps):
    """Each step reports its own contribution; otherwise a late commit re-announces earlier work."""
    repo.try_commit(steps["edited"], attempt=1, render=False)
    second = repo.try_commit(steps["edited2"], attempt=2, render=False)
    # The second boss only -- the first one is part of HEAD by now and is not re-reported.
    assert second["changed_faces"]["faces_added"] == 3
    assert [row["center"][1] for row in second["changed_faces"]["new_faces"]
            if row["type"] == "CYLINDER"] == [pytest.approx(-15.0)]


def test_a_removal_is_reported_as_removed_faces(repo, steps):
    report = repo.try_commit(steps["merged"], attempt=1, render=False)
    assert report["changed_faces"]["faces_removed"] == 6   # the deleted lug
    assert report["changed_faces"]["faces_added"] == 0


def test_the_delta_and_preserved_ratio_agree_on_what_was_touched(steps):
    """Both read one matcher, so the gate and the report can never tell different stories."""
    origin = summarise_shape(load_shape(steps["origin"]), max_faces=400)
    edited = summarise_shape(load_shape(steps["edited"]), max_faces=400)
    n_origin = len(origin["faces_listed"])
    delta = face_delta(origin, edited)
    assert preserved_ratio(origin, edited) == pytest.approx(
        (n_origin - delta["faces_removed"]) / n_origin
    )


# -- the solid roster ------------------------------------------------------------------------------

def test_an_assembly_commit_restates_the_solid_roster(repo, steps):
    roster = repo.try_commit(steps["edited"], attempt=1, render=False)["solids"]
    assert [entry["index"] for entry in roster] == [0, 1, 2, 3]


def test_a_single_solid_part_gets_no_roster(tmp_path, steps):
    """Nothing to disambiguate, so the room in the report is not worth spending."""
    import cadquery as cq

    path = str(tmp_path / "single.step")
    cq.exporters.export(cq.Workplane("XY").box(10, 10, 10).val(), path, exportType="STEP")
    assert solid_roster(summarise_shape(load_shape(path))) is None


def test_the_roster_is_dropped_when_it_would_be_longer_than_the_report(steps):
    origin = summarise_shape(load_shape(steps["origin"]), max_faces=400)
    assert solid_roster(origin) is not None
    origin["solids"] = origin["solids"] * 5              # 20 solids, past the cap
    assert solid_roster(origin) is None
