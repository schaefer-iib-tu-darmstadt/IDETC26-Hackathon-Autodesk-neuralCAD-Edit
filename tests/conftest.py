"""Shared fixtures and paths for the test suite.

Geometry is built with CadQuery rather than read from a recorded run, so the suite keeps working
after output folders are cleaned. Building and exporting it costs a few seconds, hence session
scope; every test that mutates a repo gets its own `CadRepo` over the same immutable STEP files.

Every path below is anchored on this file, so the suite runs from any working directory;
`pyproject.toml` puts the repository root on `sys.path` for the `src.*` imports.
"""

import os.path as osp
from pathlib import Path

import cadquery as cq
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = str(ROOT / "data" / "edit_192_external")
DATASET_DIR = osp.join(DATASET_ROOT, "breps")
# One of the three start models that fail `isValid()` before any edit -- exactly one broken solid
# out of 20. The relative-validity gate has no synthetic equivalent: CadQuery will not build a
# broken solid on demand.
DATASET_INVALID_START = osp.join(DATASET_DIR, "SUJ2G2UMJQR7PMBX_1759203649.19873.step")
CONFIG_PATH = str(ROOT / "src" / "config" / "edit_192_external.json")
RENDERER = str(ROOT / "src" / "agents" / "cad_render.py")
EDIT_HARNESS = str(ROOT / "src" / "harnesses" / "cadquery_script.py")


def _export(shape, path):
    cq.exporters.export(shape, path, exportType="STEP")
    return path


@pytest.fixture(scope="session")
def steps(tmp_path_factory):
    """A four-solid 'assembly' plus the states an agent typically produces from it.

    The lugs sit ON the base, inside its footprint, on purpose: removing one then leaves the
    bounding box essentially intact, which is what a real component-removal request looks like.
    A part placed off to the side would shrink the box far enough to trip `assess_damage`, and the
    test would be asserting against the calibrated damage threshold rather than the solid gate.
    """
    d = str(tmp_path_factory.mktemp("fixtures"))

    base = cq.Workplane("XY").box(40, 40, 10)
    lugs = [cq.Workplane("XY").transformed(offset=(x, 0, 7)).box(6, 6, 4).val()
            for x in (-12, 0, 12)]
    origin = cq.Compound.makeCompound([base.val()] + lugs)

    # A normal edit: a boss on the base, every other solid kept.
    boss = cq.Workplane("XY").transformed(offset=(0, 15, 5)).circle(4).extrude(6)
    edited = cq.Compound.makeCompound([base.val().fuse(boss.val())] + lugs)

    # A second, distinct state built on the first.
    boss2 = cq.Workplane("XY").transformed(offset=(0, -15, 5)).circle(4).extrude(6)
    edited2 = cq.Compound.makeCompound([base.val().fuse(boss.val()).fuse(boss2.val())] + lugs)

    # The assembly-drop failure: one solid selected, the rest thrown away.
    dropped = cq.Compound.makeCompound([lugs[0]])

    # A legitimate component removal: one lug deleted, everything else untouched.
    merged = cq.Compound.makeCompound([base.val()] + lugs[1:])

    # A global transform: every face moves, no face matches, topology unchanged.
    moved = origin.translate((0, 0, 250))

    return {
        "origin": _export(origin, osp.join(d, "origin.step")),
        "edited": _export(edited, osp.join(d, "edited.step")),
        "edited2": _export(edited2, osp.join(d, "edited2.step")),
        "dropped": _export(dropped, osp.join(d, "dropped.step")),
        "merged": _export(merged, osp.join(d, "merged.step")),
        "moved": _export(moved, osp.join(d, "moved.step")),
    }


@pytest.fixture
def repo(steps, tmp_path):
    """A fresh repo over the shared origin."""
    from src.agents.cad_git import CadRepo

    return CadRepo(origin_step=steps["origin"], work_dir=str(tmp_path / "repo"))


@pytest.fixture
def history_repo(repo, steps):
    """A repo whose history is c1 accepted, c2 refused as a base, c3 accepted -- HEAD on c3.

    The refused commit has to sit BEHIND HEAD, or "one step back" is trivially the origin and the
    skip-the-refused-commit behaviour is never exercised.
    """
    repo.try_commit(steps["edited"], attempt=1, render=False)     # c1, HEAD
    repo.try_commit(steps["dropped"], attempt=2, render=False)    # c2, refused as a base
    repo.try_commit(steps["edited2"], attempt=3, render=False)    # c3, HEAD
    assert [c.n for c in repo.commits] == [1, 2, 3]
    assert repo.head.n == 3
    return repo
