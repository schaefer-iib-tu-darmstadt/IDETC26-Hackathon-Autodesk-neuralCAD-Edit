"""CAD Agent v4 -- grounded delta contracts over a compact checkpoint substrate.

An earlier version of this agent had seven tools and a separate lookup allowance, and still lost
tasks to the same pattern: it explored, edited something plausible, declared success, and shipped
geometry that did not answer the request. More tools did not fix that, because the missing piece
was never capability -- it was that nothing ever checked the edit against a statement of what the
edit was supposed to do.

So this version exposes four operations and makes the second one mandatory:

    ground      -- ask the B-Rep graph which entities the request refers to, and commit to an
                   EditSpec: the operation, the target handles, the measurable deltas, and what
                   must survive untouched
    probe       -- run a read-only script; no STEP export, no render, no commit
    apply_edit  -- run a script against HEAD, then accept it only if the resulting geometry
                   satisfies both the EditSpec and the per-step expectation declared with it
    finish      -- stop, subject to the same contract

Two consequences worth naming, both deliberate:

* **One budget.** Every model call counts the same, whether it grounds, probes or edits. An earlier
  version gave lookups their own allowance and the agent simply spent it; making the cost uniform
  is what makes `ground` compete with `apply_edit` on merit.
* **No ground truth anywhere.** The contract is checked between two summaries of the agent's own
  geometry, so it behaves identically on a scored request and on an unseen part.

The model object, generation config, prompt helpers and the fixed between-turn messages live in
`cad_agent_base`; checkpointing and the validity/damage gates in `cad_git`; the B-Rep graph in
`cad_graph`; the contract machinery in `cad_contract`. This module owns the four tools, the message
loop around them, and the choice of what to deliver.
"""

from __future__ import annotations

import asyncio
import json
import os
import os.path as osp
import shutil
import threading
import time

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

from src.agents import cad_tools
from src.agents.cad_agent_base import (
    MAX_NUDGES,
    MAX_TRANSPORT_RETRIES,
    NUDGE,
    RESUME,
    CadAgentRunnerBase,
    DONE_MARKERS,
    _for_prompt,
    _image_part,
    _is_transient,
    _write_stdout,
)
from src.agents.cad_contract import (
    choose_delivery_commit,
    evaluate_contract,
    expectation_schema_issue,
    infer_request_intent,
    normalise_edit_spec,
    normalise_expectation,
    operation_preflight_issue,
    script_preflight_issue,
    target_preflight_issue,
)
from src.agents.cad_git import RENDER_TIMEOUT, CadRepo, _ship_start_stl
from src.agents.cad_graph import BRepGraph
from src.agents.cad_trace import CadRunTracer


APP_NAME = "neuralcad_edit_v4"

def _runtime_breakdown(wall_seconds, failed_request_seconds, backoff_seconds):
    """Separate quota/transport waiting from effective agent runtime."""
    wall = max(0.0, float(wall_seconds))
    failed = max(0.0, float(failed_request_seconds))
    backoff = max(0.0, float(backoff_seconds))
    overhead = min(wall, failed + backoff)
    return {
        "wall_time_seconds": round(wall, 2),
        "transport_failed_request_seconds": round(failed, 2),
        "transport_backoff_seconds": round(backoff, 2),
        "transport_overhead_seconds": round(overhead, 2),
        "effective_vlm_time_seconds": round(max(0.0, wall - overhead), 2),
    }


SYSTEM_PROMPT = """You are an expert autonomous CAD editing agent. Modify the existing STEP model with CadQuery to satisfy the request precisely while preserving everything outside the requested region.

## Workflow: GROUND -> APPLY EDIT -> VERIFY DELTA -> FINISH OR CORRECT

1. Your first action should be `ground`. Give it a compact EditSpec with:
   - operation: add, remove, modify, transform, copy, pattern, assembly, composite, or edge_finish
   - target_description
   - deterministic assertions when the request states measurable outcomes
   - preservation rules
2. `ground` returns a small B-Rep neighbourhood. Handles such as `c2:F17` are valid ONLY for that HEAD revision.
   Ground again after EVERY accepted commit: `finish` refuses a HEAD that has not been post-grounded and verified.
   Solid rows may include `axial_feature_spans`, grouped by x/y/z, analytic type, span,
   radius, and count. A multi-solid query also returns `mating_feature_pairs`: use its member
   handles and independent spans before probing. Never share an engagement depth or start plane
   across mating solids unless their grounded spans are equal.
   For a profile substitution, make the first query `entity="solids", limit=4` and inspect
   `mating_feature_pairs` before any face refinement or probe.
3. Use `probe` only when the graph cannot answer a numerical assumption. It is stdout-only and
   budgeted per task. Never use it as an edit.
   For a planar section use `cq.Workplane(plane).newObject([solid]).section()`; CadQuery Shape/Solid has no `.section()` method.
4. `apply_edit` runs one logically testable feature on CURRENT HEAD. Pass the grounded handles and a StepExpectation describing the delta you expect.
   `expected_changes` and `forbidden_changes` must contain assertion OBJECTS, never prose strings.
   Every script must import only `args['input_file']`; never hardcode a path.
5. Read the DeltaCard. Correct failed required assertions or target warnings. Do not repeat an unchanged selector.
6. Post-ground the accepted HEAD, inspect its DeltaCard and per-solid magnitude, then call `finish` and state DONE.

## Contract schema

Assertions use `{metric, operator, value, tolerance, required}`. Supported metrics:
`volume_delta`, `volume_ratio`, `solid_delta`, `face_delta`, `edge_delta`,
`bbox_x_delta`, `bbox_y_delta`, `bbox_z_delta`, `center_x_delta`, `center_y_delta`,
`center_z_delta`, `face_type_delta:CYLINDER`, and
`circle_radius_count_delta:5.0`. Operators: eq, ne, gt, ge, lt, le, between, nonzero.
Set assertion `required: true` only for an explicit request constraint. Preservation supports
`min_preserved_ratio`, `keep_solid_count`, `allow_global_transform`, and
`max_translation_ratio`, and `required`. Numeric thresholds are hard constraints only with
`required: true`; supplied boolean preservation rules are enforced directly.
Literal intent rule: "add", "create", "copy", "duplicate", or "insert" means the
existing instance remains unless the request also explicitly says remove, replace, move, or
reposition. Never satisfy a creation request by merely relocating an existing component.

## CadQuery rules

- Define exactly `my_cad_function(args)` and return a CadQuery Workplane, Shape, Compound, or Assembly.
- Import CURRENT HEAD with `cq.importers.importStep(args['input_file'])`.
- Perform an actual edit; do not return a print-only or unchanged model from `apply_edit`.
- Preserve all assembly components unless removal or merge is explicitly requested.
- Prefer analytic selectors using surface type, radius, centre, normal, axis and local adjacency.
- For profile edits on X/Y/Z axes, construct an explicit `cq.Plane(origin=..., xDir=..., normal=...)` per solid. Named `Workplane("XZ")`/`Workplane("YZ")` normals can invert bounded coordinates.
- A grounded selector's index refers to the exact current STEP snapshot. Verify type and measure in code before applying a destructive operation.
- For multi-part requests, make one accepted commit per independent feature. HEAD already contains previous accepted features.
- For fillets/chamfers on imported geometry, select only the intended edge group and use a fallback strategy if the kernel rejects the operation.
- For transforms, make the anchor frame explicit. Scaling about the world origin is wrong unless the request explicitly names it.
- Never rebuild a complex imported part from scratch when a local boolean, transform, copy, or edge operation can preserve it.
"""


def _split_handles(value):
    if not value:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


class CadAgentRunnerV4(CadAgentRunnerBase):
    """Runs one grounded-contract edit request to completion."""

    def _generation_config(self):
        config = super()._generation_config()
        if self.config.get("seed") is not None:
            config.seed = int(self.config["seed"])
        return config

    async def _run_edit(self, instruction_text, task_info_dict, output_dir, max_turns,
                        harness_script_file):
        os.makedirs(output_dir, exist_ok=True)
        iteration_dir = osp.join(output_dir, "iteration_output")
        os.makedirs(iteration_dir, exist_ok=True)
        if harness_script_file is None:
            harness_script_file = osp.join(
                osp.dirname(osp.dirname(osp.abspath(__file__))), "harnesses", "cadquery_script.py"
            )

        input_step = (task_info_dict.get("brep_start_path_step")
                      or task_info_dict.get("brep_start_path_stp"))
        input_step = osp.abspath(input_step) if input_step else None
        request = (task_info_dict.get("request_text")
                   or task_info_dict.get("request_transcript") or instruction_text)
        repo = CadRepo(
            origin_step=input_step,
            work_dir=osp.join(iteration_dir, "repo"),
            render_views=self._feat("views_feedback", "iso"),
            render_timeout=int(self._feat("render_timeout", RENDER_TIMEOUT)),
        )
        baseline = repo.origin_summary
        views = cad_tools.find_start_views(input_step) if input_step else []
        max_probe_calls = max(0, int(self._feat("max_probe_calls", 2)))
        total_budget = max(1, int(max_turns))
        retry_base_seconds = max(1.0, float(self._feat("transport_retry_base_seconds", 15.0)))
        state = {
            "intent": infer_request_intent(request),
            "attempts": 0,
            "probes": 0,
            "grounds": 0,
            "log": [],
            "spec": normalise_edit_spec({}),
            "spec_defined": False,
            "cards": {},
            "finish_requested": False,
            "finish_reason": "",
            "grounded_revisions": set(),
        }
        request_probe_limit = min(max_probe_calls, 1) if state["intent"]["profile_substitution"] else max_probe_calls
        counter_lock = threading.Lock()
        graph_cache = {}

        def current_revision():
            return f"c{repo.head.n}" if repo.head else "origin"

        def current_graph():
            revision = current_revision()
            if revision not in graph_cache:
                graph_cache[revision] = BRepGraph(repo.head_step, revision=revision)
            return graph_cache[revision]

        def ground(spec: dict, entity: str = "faces", kind: str = "", solid: int = -1,
                   near: str = "", radius: float = -1.0, min_measure: float = 0.0,
                   limit: int = 12, seed_handles: str = "", depth: int = 1) -> dict:
            """Ground the request in a compact local B-Rep graph of CURRENT HEAD.

            Args:
                spec: EditSpec dict. On the first call include operation, target_description,
                    assertions, preservation, and completion_criteria. Later calls may pass {}.
                entity: faces, edges or solids.
                kind: Optional analytic type such as PLANE, CYLINDER, CIRCLE or LINE.
                solid: Optional solid index; -1 searches all solids.
                near: Optional x,y,z or x,y,z,radius position filter.
                radius: Optional analytic radius; -1 disables the filter.
                min_measure: Minimum face area, edge length, or solid volume.
                limit: Maximum candidate rows, capped at 24.
                seed_handles: Optional comma-separated current handles for local graph traversal.
                depth: Local graph traversal depth, 0 to 3.
            """
            with counter_lock:
                state["grounds"] += 1
            if spec:
                state["spec"] = normalise_edit_spec(spec)
                state["spec_defined"] = True
            graph = current_graph()
            result = graph.query(
                entity=entity,
                kind=kind,
                solid=solid,
                near=near,
                radius=radius,
                min_measure=min_measure,
                limit=limit,
                seed_handles=seed_handles,
                depth=depth,
            )
            if (
                state["intent"]["profile_substitution"]
                and state["grounds"] == 1
                and "mating_feature_pairs" not in result
                and len(graph.solids) > 1
            ):
                result["mating_feature_pairs"] = graph.mating_feature_pairs()
                result["mating_pair_note"] = (
                    "Profile pre-grounding: compare each solid member span independently."
                )
                result["profile_grounding_note"] = (
                    "These pairs were injected because the first query was not entity='solids'. "
                    "Use their local handles before probing or refining faces."
                )
            state["grounded_revisions"].add(graph.revision)
            result["overview"] = graph.overview()
            result["edit_spec"] = state["spec"]
            result["request_intent"] = state["intent"]
            result["probe_budget"] = {"limit": request_probe_limit, "used": state["probes"]}
            if repo.head and repo.head.n in state["cards"]:
                result["latest_delta_card"] = state["cards"][repo.head.n].as_dict()
                result["verification"] = (
                    "This HEAD is now post-grounded; inspect the DeltaCard before finish."
                )
            if not state["spec_defined"]:
                result["warning"] = "No EditSpec is stored yet; include it in the next ground call."
            state["log"].append({
                "tool": "ground",
                "revision": graph.revision,
                "entity": entity,
                "matched": result.get("matched"),
            })
            return result

        def probe(script: str) -> dict:
            """Run a stdout-only read-only script against CURRENT HEAD; no STEP or image is made.

            Top-level analysis code is accepted and automatically wrapped with args available.
            """
            with counter_lock:
                if state["probes"] >= request_probe_limit:
                    state["log"].append({"tool": "probe", "refused": "probe_budget_exhausted"})
                    return {
                        "budget_exhausted": True,
                        "reason": f"The probe limit of {request_probe_limit} is exhausted. Use the grounded graph or edit now.",
                    }
                state["probes"] += 1
                number = state["probes"]
            work_dir = osp.join(iteration_dir, f"probe_{number:03d}")
            report = cad_tools.run_probe_script(
                script,
                repo.head_step or "",
                work_dir,
                timeout=int(self._feat("probe_timeout", 30)),
            )
            _write_stdout(work_dir, report.get("stdout", ""))
            state["log"].append({
                "tool": "probe",
                "probe": number,
                "status": report.get("status"),
                "returned_model": report.get("returned_model"),
            })
            return report

        def apply_edit(script: str, target_handles: str = "", expectation: dict = None) -> dict:
            """Apply one edit to CURRENT HEAD and return a deterministic DeltaCard.

            Args:
                script: Complete Python source defining my_cad_function(args).
                target_handles: Comma-separated handles returned by ground for this HEAD revision.
                expectation: StepExpectation dict with operation, expected_changes,
                    forbidden_changes, and preservation.
            """
            handles = _split_handles(target_handles)
            raw_expectation = expectation or {}
            schema_issue = expectation_schema_issue(raw_expectation)
            if schema_issue:
                state["log"].append({"tool": "apply_edit", "refused": "expectation_schema"})
                return {
                    "committed": False,
                    "reason": schema_issue,
                    "correction": "Replace prose with supported metric assertion objects.",
                }
            expectation = normalise_expectation(raw_expectation)
            before_graph = current_graph()
            intent_issue = operation_preflight_issue(request, state["spec"], expectation)
            if intent_issue:
                state["log"].append({
                    "tool": "apply_edit",
                    "refused": "intent_conflict",
                    "reason": intent_issue,
                })
                return {
                    "committed": False,
                    "reason": intent_issue,
                    "correction": "Ground source/anchor handles if needed, then add a structurally new feature.",
                }
            grounding_issue = target_preflight_issue(state["intent"], handles)
            if grounding_issue:
                state["log"].append({
                    "tool": "apply_edit",
                    "refused": "target_grounding",
                    "reason": grounding_issue,
                })
                return {
                    "committed": False,
                    "reason": grounding_issue,
                    "correction": "Ground the local profile faces/edges, then retry the edit.",
                }
            stale = [before_graph.resolve(handle) for handle in handles]
            plane_issue = script_preflight_issue(script, state["intent"])
            if plane_issue:
                state["log"].append({
                    "tool": "apply_edit",
                    "refused": "ambiguous_workplane",
                    "reason": plane_issue,
                })
                return {
                    "committed": False,
                    "reason": plane_issue,
                    "correction": "Use explicit cq.Plane frames and each mating member's own grounded span.",
                }
            stale = [row for row in stale if row.get("error")]
            if stale:
                state["log"].append({"tool": "apply_edit", "refused": "stale_handle"})
                return {"committed": False, "reason": "One or more target handles are stale or invalid.", "handle_errors": stale}

            with counter_lock:
                state["attempts"] += 1
                number = state["attempts"]
            work_dir = osp.join(iteration_dir, f"attempt_{number:03d}")
            revision_before = before_graph.revision
            summary_before = repo.parent_summary
            head_before = repo.head.n if repo.head else 0
            print(f"[cad-agent-v4] apply_edit -> attempt {number:03d} on {revision_before}", flush=True)
            report, artifacts = cad_tools.run_cadquery_script(
                script=script,
                input_file=repo.head_step or "",
                work_dir=work_dir,
                harness_script_file=harness_script_file,
                baseline=None,
                summarise=False,
                render=bool(self._feat("edit_render", False)),
            )
            _write_stdout(work_dir, report.get("stdout", ""))
            if report.get("status") != "ok" or not artifacts.get("step"):
                out = {
                    "committed": False,
                    "stdout": report.get("stdout", "")[-2500:],
                    "reason": report.get("note") or report.get("error") or "The script produced no STEP file.",
                }
                state["log"].append({"tool": "apply_edit", "attempt": number, "committed": False, "status": report.get("status")})
                return out

            commit_report = repo.try_commit(
                artifacts["step"],
                script_path=artifacts.get("script"),
                attempt=number,
                png=artifacts.get("png"),
                render=bool(self._feat("diff_render", False)),
            )
            if not commit_report.get("committed"):
                commit_report["stdout"] = report.get("stdout", "")[-1500:]
                state["log"].append({"tool": "apply_edit", "attempt": number, "committed": False, "reason": commit_report.get("reason")})
                return commit_report

            commit = repo.commits[-1]
            revision_after = f"c{commit.n}"
            candidate_graph = BRepGraph(commit.step_path, revision=revision_after)
            graph_cache[revision_after] = candidate_graph
            target_change = before_graph.target_change_report(handles, candidate_graph)
            card = evaluate_contract(
                state["spec"],
                expectation,
                summary_before,
                commit.summary,
                commit_report,
                target_change=target_change,
                revision_before=revision_before,
                revision_after=revision_after,
                require_structural_delta=state["intent"]["requires_structural_delta"],
                request_intent=state["intent"],
            )
            state["cards"][commit.n] = card

            if not card.hard_pass and commit_report.get("head_advanced"):
                repo.revert(to=head_before)
                commit_report["head_advanced"] = False
                commit_report["contract_rejected_as_head"] = True

            compact = {
                "committed": True,
                "commit": commit_report.get("commit"),
                "head_advanced": commit_report.get("head_advanced"),
                "head": f"c{repo.head.n}" if repo.head else "origin",
                "delta_card": card.as_dict(),
                "changed_faces": commit_report.get("changed_faces"),
                "problems": commit_report.get("problems"),
                "kernel_notes": commit_report.get("kernel_notes"),
                "stdout": report.get("stdout", "")[-1200:],
            }
            state["log"].append({
                "tool": "apply_edit",
                "attempt": number,
                "commit": commit.n,
                "head_advanced": compact["head_advanced"],
                "contract_score": card.contract_score,
                "contract_hard_pass": card.hard_pass,
                "preserved_ratio": commit.preserved_ratio,
            })
            return compact

        def finish(reason: str = "All requested features are complete and verified.") -> dict:
            """Finish only a safe HEAD that was inspected after its commit."""
            if repo.head is None:
                note = "No contract-passing HEAD exists. Correct the last DeltaCard and apply a new edit."
                state["log"].append({"tool": "finish", "refused": "no_verified_head"})
                return {
                    "ok": False,
                    "reason": note,
                    "correction": "Do not finish or state DONE; revise the rejected edit.",
                }
            revision = f"c{repo.head.n}"
            card = state["cards"].get(repo.head.n)
            if not card or not card.hard_pass:
                note = "The current HEAD has no passing DeltaCard."
                state["log"].append({"tool": "finish", "refused": "contract_failed"})
                return {
                    "ok": False,
                    "reason": note,
                    "failures": card.failures if card else ["missing DeltaCard"],
                    "correction": "Correct the failed contract before finishing.",
                }
            if revision not in state["grounded_revisions"]:
                note = f"{revision} has not been grounded after its commit."
                state["log"].append({"tool": "finish", "refused": "post_ground_required"})
                return {
                    "ok": False,
                    "reason": note,
                    "correction": "Call ground on CURRENT HEAD, inspect latest_delta_card, then finish.",
                }
            state["finish_requested"] = True
            state["finish_reason"] = str(reason)[:500]
            state["log"].append({"tool": "finish", "reason": state["finish_reason"]})
            choice = choose_delivery_commit(
                repo.commits, state["cards"], head=repo.head, finish_requested=True
            )
            return {
                "ok": bool(choice),
                "selected_candidate": f"c{choice.n}" if choice else "original input",
                "note": "Return a short final statement containing DONE.",
            }

        agent = LlmAgent(
            name="cad_editor_v4",
            model=self._model(),
            instruction=SYSTEM_PROMPT,
            tools=[ground, probe, apply_edit, finish],
            generate_content_config=self._generation_config(),
        )
        tracer = None
        if self._feat("trace", True):
            tracer = CadRunTracer(
                path=osp.join(iteration_dir, "run_trace.jsonl"),
                extra={
                    "request": str(request)[:300],
                    "model": self.config.get("model"),
                    "max_llm_calls": total_budget,
                    "seed": self.config.get("seed"),
                    "origin": input_step,
                },
            )
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME, plugins=[tracer] if tracer else None)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id="harness")

        profile_instruction = (
            "Profile substitution detected: first ground entity='solids' with limit=4 and inspect "
            "mating_feature_pairs before any probe. Use a separate explicit cq.Plane and grounded "
            "span for each solid.\n"
            if state["intent"]["profile_substitution"] else ""
        )
        parts = [types.Part(text=(
            f"Edit request: {request}\n\n"
            "Read the model only through args['input_file']; it always points at CURRENT HEAD.\n"
            f"Initial geometry summary:\n{json.dumps(_for_prompt(baseline), indent=1)}\n\n"
            f"{profile_instruction}"
            "Begin with `ground`. Store a concise EditSpec, query only the likely target entity, "
            "then apply the first real edit. Treat add/create/copy/duplicate/insert literally: "
            "preserve existing instances unless this request also explicitly asks to move, "
            "replace, or remove them."
        ))]
        if views and int(self._feat("views_start", 7)) > 0:
            shown = views[:int(self._feat("views_start", 7))]
            parts.append(types.Part(text=f"{len(shown)} canonical renders of the original model follow."))
            for name, path in shown:
                parts.append(types.Part(text=f"View: {name}"))
                parts.append(_image_part(path))
        message = types.Content(role="user", parts=parts)

        totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        trace = []
        calls_used = 0
        nudges = 0
        transport_retries = 0
        transport_failed_request_seconds = 0.0
        transport_backoff_seconds = 0.0
        termination = "max_turns_reached"
        agent_stop_signal = termination
        agent_error = None
        t0 = time.perf_counter()
        print(f"[cad-agent-v4] {request[:100]} | {len(views)} views | max {total_budget} total calls", flush=True)

        while calls_used < total_budget:
            last_text = ""
            turn_started = time.perf_counter()
            try:
                async for event in runner.run_async(
                    user_id="harness",
                    session_id=session.id,
                    new_message=message,
                    run_config=RunConfig(max_llm_calls=total_budget - calls_used),
                ):
                    usage = event.usage_metadata
                    if usage:
                        calls_used += 1
                        used = {
                            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                            "cached_input_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
                            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                            "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
                        }
                        for key, value in used.items():
                            totals[key] += value
                        event_parts = (event.content.parts if event.content else None) or []
                        trace.append({
                            "turn": len(trace) + 1,
                            "seconds": round(time.perf_counter() - turn_started, 2),
                            "tools_called": [part.function_call.name for part in event_parts if part.function_call],
                            **used,
                        })
                        turn_started = time.perf_counter()
                    if event.content and event.author != "user":
                        last_text = " ".join(
                            part.text for part in (event.content.parts or [])
                            if part.text and not getattr(part, "thought", False)
                        ) or last_text
            except Exception as exc:
                transient = _is_transient(exc)
                if transient:
                    # turn_started is reset after every successful usage event, so this excludes
                    # only the failed request phase, not earlier successful model/tool work.
                    transport_failed_request_seconds += max(0.0, time.perf_counter() - turn_started)
                if state["finish_requested"] and repo.commits:
                    agent_error = f"{type(exc).__name__}: {exc}"
                    termination = "agent_completed_after_transport_error"
                    break
                if "LlmCallsLimit" in type(exc).__name__ or "max_llm_calls" in str(exc):
                    termination = "max_turns_reached"
                    break
                if transient and transport_retries < MAX_TRANSPORT_RETRIES and calls_used < total_budget:
                    transport_retries += 1
                    delay = min(
                        60.0,
                        retry_base_seconds * (2 ** (transport_retries - 1)),
                    )
                    backoff_started = time.perf_counter()
                    await asyncio.sleep(delay)
                    transport_backoff_seconds += max(0.0, time.perf_counter() - backoff_started)
                    message = types.Content(role="user", parts=[types.Part(text=RESUME)])
                    continue
                agent_error = f"{type(exc).__name__}: {exc}"
                termination = "error_after_commit" if repo.commits else "error_before_commit"
                break

            if state["finish_requested"] or repo.commits:
                termination = (
                    "agent_completed"
                    if state["finish_requested"] or any(marker in last_text.upper() for marker in DONE_MARKERS)
                    else "agent_stopped_with_result"
                )
                break
            if nudges >= MAX_NUDGES:
                break
            nudges += 1
            message = types.Content(role="user", parts=[types.Part(text=NUDGE)])

        agent_stop_signal = termination
        contract_choice = choose_delivery_commit(
            repo.commits, state["cards"], head=repo.head,
            finish_requested=state["finish_requested"],
        )
        verified_head_selected = bool(
            state["finish_requested"] and contract_choice and repo.head
            and contract_choice.n == repo.head.n
        )
        retention = {
            "selection_reason": (
                "v4_verified_head" if verified_head_selected
                else "v4_contract_rank" if contract_choice
                else "no_contract_candidate"
            ),
            "chosen_commit": contract_choice.n if contract_choice else None,
            "head_commit": repo.head.n if repo.head else None,
            "num_commits": len(repo.commits),
            "num_head_advances": sum(1 for commit in repo.commits if commit.is_head_candidate),
        }
        if contract_choice and contract_choice.is_valid and not contract_choice.damage.get("damaged"):
            source_step = contract_choice.step_path
            final_source = f"contract_commit_c{contract_choice.n}"
        else:
            source_step, final_source, fallback_retention = repo.final_step(
                self.config.get("damaged_candidate_policy", "fallback_to_input")
            )
            retention.update(fallback_retention)
            retention["selection_reason"] = f"v4_fallback_{fallback_retention.get('selection_reason')}"

        if not (source_step and osp.exists(source_step)) and input_step and osp.exists(input_step):
            source_step = input_step
            final_source = "fallback_unmodified_input_no_source"
        final_step = osp.join(output_dir, "tmp.step")
        if source_step and osp.exists(source_step):
            shutil.copy(source_step, final_step)
        ships_input = bool(input_step and source_step and osp.abspath(source_step) == osp.abspath(input_step))
        if ships_input:
            _ship_start_stl(input_step, output_dir)
            if not termination.startswith("error_"):
                termination = "no_valid_edit"

        chosen = next((commit for commit in repo.commits if commit.n == retention.get("chosen_commit")), None)
        if chosen and chosen.png and osp.exists(chosen.png):
            shutil.copy(chosen.png, osp.join(output_dir, "tmp.png"))

        cost = (
            totals["input_tokens"] * float(self.config.get("1m_token_cost_input", 0.0))
            + (totals["output_tokens"] + totals["thinking_tokens"])
            * float(self.config.get("1m_token_cost_output", 0.0))
        ) / 1e6
        print(
            f"[cad-agent-v4] done: {termination} | result={final_source} | "
            f"{state['attempts']} edits | {state['grounds']} grounds | {state['probes']} probes | ${cost:.4f}",
            flush=True,
        )
        runtime = _runtime_breakdown(
            time.perf_counter() - t0,
            transport_failed_request_seconds,
            transport_backoff_seconds,
        )
        return {
            "token_counts": {**totals, "cost_estimate": round(cost, 5)},
            "num_iterations": len(trace),
            "num_scripts_run": state["attempts"],
            "num_ground_calls": state["grounds"],
            "num_probe_calls": state["probes"],
            "max_probe_calls": request_probe_limit,
            "max_llm_calls": total_budget,
            "num_commits": len(repo.commits),
            "contract_scores": {f"c{number}": card.contract_score for number, card in state["cards"].items()},
            "contract_hard_pass": {f"c{number}": card.hard_pass for number, card in state["cards"].items()},
            "request_intent": state["intent"],
            "edit_spec": state["spec"],
            "finish_requested": state["finish_requested"],
            "finish_reason": state["finish_reason"],
            "origin_is_valid": bool((repo.origin_summary or {}).get("is_valid")),
            "origin_error": repo.origin_error,
            "agent_error": agent_error,
            "agent_stop_signal": agent_stop_signal,
            "shipped_unmodified_input": ships_input,
            "transport_retries": transport_retries,
            **runtime,
            "run_trace": tracer.summary() if tracer else None,
            "termination_reason": termination,
            "final_source": final_source,
            "retention": retention,
            # Backwards-compatible field consumed by the evaluator; now intentionally quota-free.
            "total_vlm_time_seconds": runtime["effective_vlm_time_seconds"],
            "execution_trace": trace,
            "tool_call_log": state["log"],
        }
