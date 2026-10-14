"""Cross-cutting regressions for the v4 runner: config contract, seed, runtime accounting, and the
render-free edit worker it relies on."""

import json
import os.path as osp

from conftest import CONFIG_PATH, EDIT_HARNESS
from src.agents.cad_agent_v4 import CadAgentRunnerV4, _runtime_breakdown
from src.agents.cad_tools import run_cadquery_script


def test_v4_config_uses_one_fixed_call_budget():
    with open(CONFIG_PATH, encoding="utf-8") as stream:
        config = json.load(stream)
    model = config["benchmark_models"]["gemini-3.7-flash_adk-agent-v4"]
    assert model["function"] == "agentic_cadquery_v4"
    assert model["max_turns"] == 10
    assert "query_budget" not in model["features"]
    assert model["features"]["max_probe_calls"] == 2
    assert model["features"]["transport_retry_base_seconds"] == 15
    assert model["features"]["edit_render"] is False
    assert model["features"]["diff_render"] is False


def test_v4_forwards_reproducibility_seed():
    runner = CadAgentRunnerV4({
        "model": "gemini-test",
        "seed": 1234,
        "temperature": 0,
    })
    assert runner._generation_config().seed == 1234


def test_v4_runtime_excludes_failed_transport_and_backoff():
    runtime = _runtime_breakdown(400.0, 305.0, 15.0)
    assert runtime == {
        "wall_time_seconds": 400.0,
        "transport_failed_request_seconds": 305.0,
        "transport_backoff_seconds": 15.0,
        "transport_overhead_seconds": 320.0,
        "effective_vlm_time_seconds": 80.0,
    }


def test_edit_worker_can_skip_render_without_skipping_step(steps, tmp_path):
    script = """
import cadquery as cq

def my_cad_function(args):
    shape = cq.importers.importStep(args['input_file'])
    return shape.translate((0, 0, 1))
"""
    work_dir = tmp_path / "edit"
    report, artifacts = run_cadquery_script(
        script,
        steps["origin"],
        str(work_dir),
        EDIT_HARNESS,
        summarise=False,
        render=False,
    )
    assert report["status"] == "ok"
    assert artifacts["step"] is not None
    assert osp.exists(artifacts["step"])
    assert artifacts["png"] is None
    assert not (work_dir / "tmp.png").exists()
