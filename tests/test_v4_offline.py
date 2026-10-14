"""The whole agent, end to end, with a scripted model and no API.

Each scenario drives the real `CadAgentRunnerV4` -- real ADK runner, real tool dispatch, real
CadQuery subprocesses, real checkpoints and contracts -- through a fixed sequence of model turns,
and pins what the run produces: the termination reason, which checkpoint is delivered, the
DeltaCard verdicts, the tool-call log, the files on disk, and what the model itself was shown.

This is the regression net for the agent's behaviour. The geometry is a CadQuery-built plate with
a boss, so no dataset is needed. Values such as the 0.95 contract score or the -226.19 mm^3 volume
delta are what the current code measures on that fixture; a change in them is a change in the
agent, not noise (the voxel-free contract metrics are deterministic).

The only thing replaced is the model object (`CadAgentRunnerV4._model`); see `fake_llm.py`.
"""

import json
import os
import os.path as osp
from pathlib import Path

import cadquery as cq
import pytest
from fake_llm import ScriptedLlm

from src.agents.cad_agent_base import NUDGE, RESUME
from src.agents.cad_agent_v4 import SYSTEM_PROMPT, CadAgentRunnerV4
from src.agents.cad_git import CadRepo

REQUEST = "Cut a 6 mm through hole at the centre of the top face of the plate."
SENTINEL_STL = "solid dataset_mesh\nendsolid dataset_mesh\n"

CONFIG = {
    "model": "gemini-3.7-flash",
    "thinking_level": "medium",
    "seed": 20260818,
    "temperature": 0,
    "1m_token_cost_input": 0.75,
    "1m_token_cost_output": 3.75,
    "max_turns": 10,
    "damaged_candidate_policy": "fallback_to_input",
    "features": {
        "views_start": 7,
        "views_feedback": "iso",
        "edit_render": False,
        "diff_render": False,
        "max_probe_calls": 2,
        "probe_timeout": 30,
        "transport_retry_base_seconds": 1,      # keep the one retry scenario fast
        "render_timeout": 300,
        "trace": True,
    },
}

HOLE_SCRIPT = """import cadquery as cq

def my_cad_function(args):
    model = cq.importers.importStep(args['input_file'])
    cutter = cq.Workplane("XY").circle(3.0).extrude(20, both=True)
    return model.cut(cutter)
"""

BUMP_SCRIPT = """import cadquery as cq

def my_cad_function(args):
    model = cq.importers.importStep(args['input_file'])
    bump = cq.Workplane("XY").transformed(offset=(-12, 8, 6)).box(4, 4, 4)
    return model.union(bump)
"""

IDENTITY_SCRIPT = """import cadquery as cq

def my_cad_function(args):
    return cq.importers.importStep(args['input_file'])
"""

RAISING_SCRIPT = """import cadquery as cq

def my_cad_function(args):
    raise RuntimeError("selector matched nothing")
"""

PROBE_SCRIPT = """import cadquery as cq
shape = cq.importers.importStep(args['input_file'])
print('faces', len(shape.faces().vals()))
"""

SPEC = {
    "operation": "remove",
    "target_description": "the large top planar face of the base plate; cut a 6 mm through hole",
    "assertions": [
        {"metric": "volume_delta", "operator": "lt", "value": 0, "required": True},
        {"metric": "solid_delta", "operator": "eq", "value": 0, "required": True},
    ],
    "preservation": {"keep_solid_count": True, "allow_global_transform": False, "required": True},
    "completion_criteria": [
        {"metric": "face_type_delta:CYLINDER", "operator": "ge", "value": 1, "required": True},
    ],
}

EXPECTATION = {
    "operation": "remove",
    "expected_changes": [
        {"metric": "volume_delta", "operator": "between", "value": [-230.0, -220.0],
         "tolerance": 0.5, "required": True},
        {"metric": "face_type_delta:CYLINDER", "operator": "eq", "value": 1, "required": True},
        {"metric": "circle_radius_count_delta:3.0", "operator": "eq", "value": 2, "required": False},
    ],
    "forbidden_changes": [
        {"metric": "solid_delta", "operator": "nonzero", "required": True},
        {"metric": "bbox_z_delta", "operator": "nonzero", "required": True},
    ],
    "preservation": {"min_preserved_ratio": 0.6, "keep_solid_count": True},
}

GROUND_TOP = [("ground", {"spec": SPEC, "entity": "faces", "kind": "PLANE", "limit": 12})]
GROUND_HEAD = [("ground", {"spec": {}, "entity": "faces", "kind": "CYLINDER", "limit": 6})]


def pick_top_face(responses):
    """From a `ground` answer, the handle of the largest upward-facing planar face."""
    rows = responses[0]["rows"]
    top = [r for r in rows if r.get("type") == "PLANE" and r.get("normal") == [0.0, 0.0, 1.0]]
    top.sort(key=lambda r: -r["area"])
    return top[0]["handle"]


def edit(script, handle=None, expectation=EXPECTATION):
    """An apply_edit turn; with `handle=None` the target is picked from the previous ground."""
    if handle is None:
        return lambda rs: [("apply_edit", {"script": script, "target_handles": pick_top_face(rs),
                                           "expectation": expectation})]
    return [("apply_edit", {"script": script, "target_handles": handle, "expectation": expectation})]


@pytest.fixture(scope="module")
def origin(tmp_path_factory):
    """A 40x30x8 plate with a r=5 boss, plus a sentinel .stl next to it (what the dataset ships)."""
    d = tmp_path_factory.mktemp("offline")
    base = cq.Workplane("XY").box(40, 30, 8)
    boss = cq.Workplane("XY").transformed(offset=(10, 0, 4)).circle(5).extrude(6)
    path = str(d / "origin.step")
    cq.exporters.export(base.union(boss), path, exportType="STEP")
    (d / "origin.stl").write_text(SENTINEL_STL, encoding="utf-8")
    return path


def run(turns, origin, out_dir, monkeypatch, max_turns=10, request=REQUEST):
    fake = ScriptedLlm(script=list(turns))
    # The one and only patch: the model object. Everything else is the shipped code path, and
    # because `_model` never runs, no credentials are read.
    monkeypatch.setattr(CadAgentRunnerV4, "_model", lambda self: fake)
    result = CadAgentRunnerV4(config=json.loads(json.dumps(CONFIG))).run_edit(
        instruction_text="Edit the existing model according to the provided instructions.",
        task_info_dict={"request_text": request, "brep_start_path_step": origin},
        output_dir=str(out_dir),
        max_turns=max_turns,
        harness_script_file=None,
    )
    return result, fake


def files_under(d):
    return sorted(osp.relpath(osp.join(r, f), d).replace("\\", "/")
                  for r, _, fs in os.walk(d) for f in fs)


def tool_names(result):
    return [e["tool"] for e in result["tool_call_log"]]


def responses_after_turn(fake, k):
    """The tool responses the model received after its k-th turn (1-based)."""
    return fake.seen[k]["responses"]


# -- the intended path ------------------------------------------------------------------------

def test_ground_edit_verify_finish_delivers_the_verified_head(origin, tmp_path, monkeypatch):
    turns = [
        GROUND_TOP,
        edit(HOLE_SCRIPT),
        GROUND_HEAD,
        [("finish", {"reason": "Hole cut and verified against the DeltaCard."})],
        "The 6 mm hole is cut through the plate centre. DONE",
    ]
    result, fake = run(turns, origin, tmp_path, monkeypatch)

    assert result["termination_reason"] == "agent_completed"
    assert result["final_source"] == "contract_commit_c1"
    assert result["retention"]["selection_reason"] == "v4_verified_head"
    assert result["retention"]["chosen_commit"] == 1
    assert result["finish_requested"] is True
    assert result["shipped_unmodified_input"] is False
    assert (result["num_iterations"], result["num_scripts_run"], result["num_ground_calls"],
            result["num_probe_calls"], result["num_commits"]) == (5, 1, 2, 0, 1)
    assert result["contract_scores"] == {"c1": 0.95}
    assert result["contract_hard_pass"] == {"c1": True}
    assert result["token_counts"] == {"input_tokens": 5000, "cached_input_tokens": 0,
                                      "output_tokens": 250, "thinking_tokens": 100,
                                      "cost_estimate": 0.00506}
    assert result["run_trace"]["model_calls"] == 5 and result["run_trace"]["tool_calls"] == 4

    log = result["tool_call_log"]
    assert tool_names(result) == ["ground", "apply_edit", "ground", "finish"]
    assert log[0]["matched"] == 7 and log[0]["revision"] == "origin"
    assert log[1] == {"tool": "apply_edit", "attempt": 1, "commit": 1, "head_advanced": True,
                      "contract_score": 0.95, "contract_hard_pass": True, "preserved_ratio": 0.75}
    assert log[2]["matched"] == 2 and log[2]["revision"] == "c1"

    # The exported geometry is the hole-cut plate, byte-equal to the checkpoint.
    summary = CadRepo._summarise(str(tmp_path / "tmp.step"))
    assert summary["volume"] == pytest.approx(9845.0442, abs=1e-3)
    assert summary["counts"]["faces"] == 9
    assert summary["counts"]["by_face_type"] == {"PLANE": 7, "CYLINDER": 2}
    assert (tmp_path / "tmp.step").read_bytes() == \
        (tmp_path / "iteration_output" / "repo" / "c001.step").read_bytes()
    assert files_under(tmp_path) == [
        "iteration_output/attempt_001/script.py", "iteration_output/attempt_001/stdout.txt",
        "iteration_output/attempt_001/tmp.step", "iteration_output/repo/c001.step",
        "iteration_output/run_trace.jsonl", "tmp.step",
    ]

    # What the model was shown: tool surface, prompt, generation config -- on every call.
    for call in fake.seen:
        assert call["tools"] == ["apply_edit", "finish", "ground", "probe"]
        assert call["labels"] == {"adk_agent_name": "cad_editor_v4"}
        assert call["system_instruction"].startswith(SYSTEM_PROMPT)
        gen = call["generation"]
        assert (gen["temperature"], gen["max_output_tokens"], gen["seed"]) == (0.0, 32768, 20260818)
        assert "medium" in gen["thinking_level"].lower() and gen["include_thoughts"] is True
        assert len(gen["safety"]) == 4 and all("BLOCK_NONE" in t for _, t in gen["safety"])
    first = fake.seen[0]["last_text"][0]
    assert first.startswith(f"Edit request: {REQUEST}")
    assert "Initial geometry summary:" in first and "Begin with `ground`." in first
    # The apply_edit response the model saw is the compact DeltaCard view, not the repo report.
    card = responses_after_turn(fake, 2)[0]
    assert card["committed"] and card["head"] == "c1"
    assert card["delta_card"]["hard_pass"] is True
    assert card["delta_card"]["observed_delta"]["volume"] == pytest.approx(-226.1947, abs=1e-3)
    assert card["changed_faces"]["faces_added"] == 3 and card["changed_faces"]["faces_removed"] == 2
    assert responses_after_turn(fake, 4)[0]["selected_candidate"] == "c1"


def test_done_without_finish_still_delivers_but_is_not_completed(origin, tmp_path, monkeypatch):
    """`finish` is what marks a run completed; a bare DONE after a commit is only 'stopped with
    result', and the contract rank (not the verified-head rule) picks the delivery."""
    turns = [GROUND_TOP, edit(HOLE_SCRIPT), "Hole cut. DONE"]
    result, _ = run(turns, origin, tmp_path, monkeypatch)
    assert result["termination_reason"] == "agent_stopped_with_result"
    assert result["finish_requested"] is False
    assert result["retention"]["selection_reason"] == "v4_contract_rank"
    assert result["final_source"] == "contract_commit_c1"


def test_a_done_marker_without_finish_counts_as_completed(origin, tmp_path, monkeypatch):
    """The fallback for a model that commits and then stops talking without calling `finish`:
    one of the `DONE_MARKERS` in its last text. Delivery still goes through the contract rank."""
    turns = [GROUND_TOP, edit(HOLE_SCRIPT), "Hole cut. EDIT_COMPLETE"]
    result, _ = run(turns, origin, tmp_path, monkeypatch)
    assert result["termination_reason"] == "agent_completed"
    assert result["finish_requested"] is False
    assert result["retention"]["selection_reason"] == "v4_contract_rank"
    assert result["final_source"] == "contract_commit_c1"


# -- everything that refuses ------------------------------------------------------------------

def test_failed_contract_reverts_and_every_refusal_is_logged(origin, tmp_path, monkeypatch):
    turns = [
        GROUND_TOP,
        # Adds material while the contract demands a cut -> committed, contract fails, HEAD reverts.
        edit(BUMP_SCRIPT),
        # A handle from a revision that is not HEAD.
        edit(HOLE_SCRIPT, handle="c1:F0"),
        # Prose where assertion objects are required.
        edit(HOLE_SCRIPT, handle="", expectation={"operation": "remove",
                                                  "expected_changes": ["cut a hole in the plate"]}),
        [("probe", {"script": PROBE_SCRIPT})],
        [("probe", {"script": PROBE_SCRIPT})],
        [("probe", {"script": PROBE_SCRIPT})],          # third one is over budget
        [("finish", {})],
        "I could not complete the request.",
    ]
    result, fake = run(turns, origin, tmp_path, monkeypatch)

    assert result["termination_reason"] == "agent_stopped_with_result"
    assert result["finish_requested"] is False
    assert result["contract_hard_pass"] == {"c1": False}
    assert result["contract_scores"] == {"c1": 0.625}
    assert (result["num_scripts_run"], result["num_probe_calls"], result["num_ground_calls"]) == (1, 2, 1)
    assert tool_names(result) == ["ground", "apply_edit", "apply_edit", "apply_edit",
                                  "probe", "probe", "probe", "finish"]
    log = result["tool_call_log"]
    assert log[1]["commit"] == 1 and log[1]["head_advanced"] is False
    assert log[2]["refused"] == "stale_handle"
    assert log[3]["refused"] == "expectation_schema"
    assert log[4]["status"] == "ok" and log[5]["status"] == "ok"
    assert log[6]["refused"] == "probe_budget_exhausted"
    assert log[7]["refused"] == "no_verified_head"

    bump, stale, schema, probe1, _, probe3, finish = [
        responses_after_turn(fake, k)[0] for k in range(2, 9)
    ]
    assert bump["committed"] is True and bump["head_advanced"] is False and bump["head"] == "origin"
    assert "required assertion failed: volume_delta" in bump["delta_card"]["failures"]
    assert stale["handle_errors"][0]["error"] == "stale_handle"
    assert "prose" in schema["reason"]
    assert "faces 8" in probe1["stdout"] and probe1["returned_model"] is False
    assert probe3["budget_exhausted"] is True
    assert finish["ok"] is False

    # A reverted commit is still the best thing on record and is what gets delivered.
    assert result["final_source"] == "commit_c1"
    assert result["retention"]["selection_reason"] == "v4_fallback_ranked_winner_only_reverted_left"
    assert CadRepo._summarise(str(tmp_path / "tmp.step"))["volume"] == pytest.approx(10135.2389, abs=1e-3)
    assert not (tmp_path / "tmp.stl").exists()
    # The stale handle never reached the worker: one attempt directory, two probe directories.
    assert not (tmp_path / "iteration_output" / "attempt_002").exists()
    assert (tmp_path / "iteration_output" / "probe_002" / "stdout.txt").exists()


def test_worker_failure_gate_refusal_and_post_ground_rule(origin, tmp_path, monkeypatch):
    """A script that raises, a script that changes nothing, and a `finish` before the committed
    HEAD was grounded again: each is refused with its own reason and costs what it should."""
    # HEAD stays the origin through the two refusals, so the top-face handle from the first
    # ground (origin:F2 on this fixture) remains valid for all three edits.
    turns = [
        GROUND_TOP,
        edit(RAISING_SCRIPT, handle="origin:F2"),    # worker error: no STEP, nothing committed
        edit(IDENTITY_SCRIPT, handle="origin:F2"),   # exported, but identical to the input: refused
        edit(HOLE_SCRIPT, handle="origin:F2"),       # the real edit -> c1
        [("finish", {})],                            # c1 has not been grounded since its commit
        GROUND_HEAD,
        [("finish", {"reason": "verified"})],
        "DONE",
    ]
    result, fake = run(turns, origin, tmp_path, monkeypatch)

    assert tool_names(result) == ["ground", "apply_edit", "apply_edit", "apply_edit",
                                  "finish", "ground", "finish"]
    log = result["tool_call_log"]
    assert log[1]["attempt"] == 1 and log[1]["committed"] is False and log[1]["status"] != "ok"
    assert log[2]["attempt"] == 2 and log[2]["committed"] is False
    assert "ORIGINAL input" in log[2]["reason"]
    assert log[3]["commit"] == 1 and log[3]["head_advanced"] is True
    assert log[4]["refused"] == "post_ground_required"
    assert log[6]["reason"] == "verified"

    raised, identical, _, early_finish = [responses_after_turn(fake, k)[0] for k in range(2, 6)]
    assert raised["committed"] is False and "selector matched nothing" in raised["stdout"]
    assert identical["committed"] is False and "cancelled out" in identical["reason"]
    assert early_finish["ok"] is False and "has not been grounded" in early_finish["reason"]

    assert result["num_scripts_run"] == 3 and result["num_commits"] == 1
    assert result["termination_reason"] == "agent_completed"
    assert result["final_source"] == "contract_commit_c1"
    attempts = tmp_path / "iteration_output"
    assert not (attempts / "attempt_001" / "tmp.step").exists()      # the script raised
    assert (attempts / "attempt_002" / "tmp.step").exists()          # exported, then refused
    assert [p.name for p in sorted((attempts / "repo").iterdir())] == ["c001.step"]


# -- nothing committed ----------------------------------------------------------------------

def test_a_model_that_never_edits_gets_nudged_then_the_input_ships(origin, tmp_path, monkeypatch):
    turns = ["I will think about it first.", "Still considering the geometry.", "No edit."]
    result, fake = run(turns, origin, tmp_path, monkeypatch)

    assert result["termination_reason"] == "no_valid_edit"
    # Nudge exhaustion is recorded under the loop's default label, not as its own value.
    assert result["agent_stop_signal"] == "max_turns_reached"
    assert result["final_source"] == "fallback_unmodified_input"
    assert result["shipped_unmodified_input"] is True
    assert result["num_iterations"] == 3 and result["tool_call_log"] == []
    assert fake.seen[1]["last_text"] == [NUDGE] and fake.seen[2]["last_text"] == [NUDGE]
    # The dataset's own mesh travels with the unmodified STEP (what diff_f1 needs on a no-op GT).
    assert (tmp_path / "tmp.stl").read_text(encoding="utf-8") == SENTINEL_STL
    assert (tmp_path / "tmp.step").read_bytes() == Path(origin).read_bytes()


def test_the_call_budget_is_enforced_by_adk_and_mapped_to_max_turns(origin, tmp_path, monkeypatch):
    turns = [[("ground", {"spec": SPEC, "entity": "faces"})]] * 6
    result, fake = run(turns, origin, tmp_path, monkeypatch, max_turns=2)

    assert result["max_llm_calls"] == 2
    assert result["agent_stop_signal"] == "max_turns_reached"
    assert result["termination_reason"] == "no_valid_edit"
    assert result["agent_error"] is None
    assert result["num_iterations"] == 2 and result["num_ground_calls"] == 2
    assert [e["matched"] for e in result["tool_call_log"]] == [8, 8]
    assert (tmp_path / "tmp.stl").exists()
    assert len(fake.script) == 4


# -- the model call fails ------------------------------------------------------------------

def test_a_transient_transport_error_is_retried_with_the_resume_message(origin, tmp_path, monkeypatch):
    turns = [
        GROUND_TOP,
        RuntimeError("503 UNAVAILABLE: The model is overloaded."),
        edit(HOLE_SCRIPT, handle="origin:F2"),
        GROUND_HEAD,
        [("finish", {"reason": "done"})],
        "DONE",
    ]
    result, fake = run(turns, origin, tmp_path, monkeypatch)

    assert result["transport_retries"] == 1
    assert result["run_trace"]["model_errors"] == 1
    assert fake.seen[2]["last_text"] == [RESUME]
    assert result["termination_reason"] == "agent_completed"
    assert result["final_source"] == "contract_commit_c1"
    assert result["retention"]["selection_reason"] == "v4_verified_head"
    assert result["num_iterations"] == 5           # the failed call carries no usage


def test_a_transport_error_after_finish_keeps_the_completed_outcome(origin, tmp_path, monkeypatch):
    """`finish` already chose the delivery; the model's closing statement failing in transport
    must not turn a finished task into an error or a retry."""
    turns = [
        GROUND_TOP,
        edit(HOLE_SCRIPT),
        GROUND_HEAD,
        [("finish", {"reason": "done"})],
        RuntimeError("503 UNAVAILABLE: The model is overloaded."),
    ]
    result, _ = run(turns, origin, tmp_path, monkeypatch)
    assert result["termination_reason"] == "agent_completed_after_transport_error"
    assert result["agent_error"].startswith("RuntimeError: 503")
    assert result["transport_retries"] == 0
    assert result["final_source"] == "contract_commit_c1"
    assert result["retention"]["selection_reason"] == "v4_verified_head"


def test_a_bug_in_our_own_code_ends_the_run_but_keeps_the_commit(origin, tmp_path, monkeypatch):
    turns = [
        GROUND_TOP,
        edit(HOLE_SCRIPT),
        TypeError("our own bug: NoneType is not subscriptable"),
        "never reached",
    ]
    result, fake = run(turns, origin, tmp_path, monkeypatch)

    assert result["termination_reason"] == "error_after_commit"
    assert result["agent_error"].startswith("TypeError: our own bug")
    assert result["transport_retries"] == 0
    assert result["final_source"] == "contract_commit_c1"
    assert result["retention"]["selection_reason"] == "v4_contract_rank"
    assert len(fake.script) == 1


def test_a_bug_before_any_commit_ships_the_input_as_an_error(origin, tmp_path, monkeypatch):
    """The error label survives the input fallback: a run that crashed is not a 'no valid edit'."""
    result, _ = run([TypeError("bug on the first call")], origin, tmp_path, monkeypatch)
    assert result["termination_reason"] == "error_before_commit"
    assert result["agent_error"].startswith("TypeError")
    assert result["shipped_unmodified_input"] is True
    assert (tmp_path / "tmp.stl").read_text(encoding="utf-8") == SENTINEL_STL
    assert result["tool_call_log"] == [] and result["num_iterations"] == 0
