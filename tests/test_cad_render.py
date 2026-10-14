"""The render worker, end to end as a subprocess.

The agent only reaches the renderer with `features.diff_render` or `features.edit_render` switched
on (both off in the shipped config), so nothing else in the suite exercises `_render_diff` or
`cad_render.py`. That matters because a renderer that is broken for the whole run (no OpenGL,
unreadable STEP) must be distinguishable from "this edit produced no visible diff".
"""

import os.path as osp
import subprocess
import sys

from conftest import RENDERER
from src.agents.cad_git import CadRepo, _child_env


def _run(args):
    return subprocess.run([sys.executable, RENDERER] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600, env=_child_env())


def test_diff_render_writes_images_and_reports_volumes(tmp_path, steps):
    """The green/red channel the model reads, plus the numbers that tell a failed boolean from a
    genuine no-op -- an image with no colour looks the same either way."""
    out = tmp_path / "render"
    proc = _run(["--old", steps["origin"], "--new", steps["edited"],
                 "--output_dir", str(out), "--views", "iso,front", "--prefix", "diff"])
    assert proc.returncode == 0, proc.stderr

    stats = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
                 and line.split("=", 1)[0] in ("removed_volume", "added_volume"))
    assert float(stats["added_volume"]) > 0, "the boss added material and must show as green"
    assert float(stats["removed_volume"]) == 0.0, "nothing was cut away"

    for view in ("iso", "front"):
        png = out / f"diff_{view}.png"
        assert png.exists() and png.stat().st_size > 0


def test_an_unreadable_step_exits_non_zero(tmp_path):
    """The only channel the parent can read without parsing prose."""
    bad = tmp_path / "broken.step"
    bad.write_text("not a STEP file", encoding="utf-8")
    proc = _run(["--new", str(bad), "--output_dir", str(tmp_path / "out"), "--views", "iso"])
    assert proc.returncode != 0


def test_an_unknown_view_is_refused_not_silently_aliased(tmp_path, steps):
    """A camera name with no defined meaning must not write an iso image under the requested
    filename -- the model would then place features from the wrong camera."""
    out = tmp_path / "render"
    proc = _run(["--new", steps["origin"], "--output_dir", str(out),
                 "--views", "front_left", "--prefix", "view"])
    assert "unknown_view=front_left" in proc.stdout
    assert not (out / "view_front_left.png").exists()
    assert proc.returncode != 0, "not one view came out, and the parent has to be able to see that"


def test_a_known_alias_renders_under_the_requested_name(tmp_path, steps):
    """`ALIASES` exists because the model asks for the dataset's own view names: 'toprightiso' cost
    a real image on the probe5 run. The camera is resolved through the alias, but the file keeps the
    name that was asked for, or `CadRepo._render_diff` would not find it again."""
    out = tmp_path / "render"
    proc = _run(["--new", steps["origin"], "--output_dir", str(out),
                 "--views", "toprightiso", "--prefix", "view"])
    assert proc.returncode == 0, proc.stderr
    assert (out / "view_toprightiso.png").exists()


def test_repo_render_surfaces_a_renderer_failure(tmp_path, steps, monkeypatch):
    """A broken renderer has to reach `settings.json` instead of looking like an empty diff."""
    repo = CadRepo(origin_step=steps["origin"], work_dir=str(tmp_path / "repo"))
    monkeypatch.setattr(repo, "_renderer_cmd",
                        lambda *a, **k: [sys.executable, "-c", "import sys; sys.exit(9)"])
    images = repo._render_diff(_dummy(repo, steps))
    assert images == []
    assert "9" in (repo.last_render.get("error") or ""), repo.last_render


def _dummy(repo, steps):
    repo.try_commit(steps["edited"], attempt=1, render=False)
    return repo.commits[0]
