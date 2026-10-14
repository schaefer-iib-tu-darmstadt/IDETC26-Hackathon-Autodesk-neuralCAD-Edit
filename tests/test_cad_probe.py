"""Tests for the v4 stdout-only probe worker."""

from pathlib import Path

from src.agents.cad_tools import run_probe_script


def test_probe_returns_stdout_without_geometry_artifacts(steps, tmp_path):
    work_dir = tmp_path / "probe"
    script = """
import cadquery as cq

def my_cad_function(args):
    shape = cq.importers.importStep(args['input_file'])
    print(f'faces={len(shape.faces().vals())}')
    return shape
"""
    report = run_probe_script(script, steps["origin"], str(work_dir), timeout=30)

    assert report["status"] == "ok"
    assert report["returned_model"] is True
    assert "faces=" in report["stdout"]
    assert not (work_dir / "tmp.step").exists()
    assert not (work_dir / "tmp.png").exists()
    assert not (work_dir / "tmp.svg").exists()
    assert (work_dir / "probe.py").exists()


def test_probe_allows_print_only_function(steps, tmp_path):
    script = """
def my_cad_function(args):
    print('answer=42')
"""
    report = run_probe_script(script, steps["origin"], str(tmp_path / "probe"), timeout=30)
    assert report["status"] == "ok"
    assert report["returned_model"] is False
    assert "answer=42" in report["stdout"]


def test_probe_auto_wraps_top_level_analysis(steps, tmp_path):
    script = """
import cadquery as cq

shape = cq.importers.importStep(args['input_file'])
print(f'solids={len(shape.solids().vals())}')
"""
    report = run_probe_script(script, steps["origin"], str(tmp_path / "auto_probe"), timeout=30)

    assert report["status"] == "ok"
    assert report["returned_model"] is False
    assert "probe_auto_wrapped=1" in report["stdout"]
    assert "solids=4" in report["stdout"]
