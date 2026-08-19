"""CAD Agent v3 with Git4CAD checkpoints and visual diff feedback.

The agent edits the current verified HEAD, can inspect or dry-run geometry without mutating it,
and exports the best retained candidate after deterministic geometry checks.
"""

import asyncio
import json
import mimetypes
import os
import os.path as osp
import shutil
import threading
import time

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

from src.agents import cad_tools
from src.agents.cad_git import RENDER_TIMEOUT, CadRepo, _ship_start_stl
from src.agents.cad_trace import CadRunTracer

try:  # Credentials live in .env; do not rely on src.vlms.gemini having been imported first.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


APP_NAME = "neuralcad_edit_v3"

SAFETY_OFF = [
    types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for category in (
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    )
]
DONE_MARKERS = ("EDIT_COMPLETE", "TASK_COMPLETE", "FINAL_ANSWER")


def _image_part(path):
    """Load an image as a Gemini inline-data part."""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        return types.Part(inline_data=types.Blob(data=handle.read(), mime_type=mime))

SYSTEM_PROMPT = """You are an expert autonomous CAD engineering agent. You modify an existing 3D model (STEP file) with CadQuery to fulfil a natural-language edit request.


## You are editing under version control

Your work is committed like source code. `args['input_file']` ALWAYS points at the CURRENT state (HEAD), never at the original file. So:

- Your first script edits the original.
- Every later script edits the RESULT OF YOUR LAST ACCEPTED COMMIT. Do not redo earlier steps.
- A multi-part request is done one feature per script, not all at once.
- `revert` moves HEAD back if a commit went the wrong way.

## Every script has the same shape

```python
import cadquery as cq

def my_cad_function(args):
    model = cq.importers.importStep(args['input_file'])   # this is HEAD
    # apply ONE step of the edit
    return model
```

## Looking before you cut: `inspect` and `dry_run`

`run_script` is for CHANGING the model. Two separate tools are for finding things out, and they
are cheap: they never commit, never render, and never touch HEAD.

- **`inspect`** answers questions about the CURRENT state's B-Rep. Ask it for the faces of one
  solid, the cylindrical faces above some area, the circular edges of a given radius, everything
  near a point. This is how you turn "the four long edges of the blade" or "the fillet at the
  front center" into a selector you can actually write. It reports how many entities matched, so
  an empty answer means your filter is wrong, not that the feature is missing.
- **`dry_run`** executes a script and gives you back only its stdout. Use it to check an
  assumption before you spend an edit on it -- which way a rotation goes, what a selector picks
  up, whether your arithmetic lands where you think.

**Never use `run_script` to look at something.** A script that imports the model, prints and
returns it unchanged is rejected as "identical to HEAD", costs you an edit turn, and tells you
less than one `inspect` call would.

## The commit gate

Every result is recorded. What the gate decides is only whether HEAD MOVES TO IT, i.e. whether
your next script may build on it. HEAD stays put when:
- your edit made the model invalid (see the next section -- only if YOU broke it),
- solids were lost (you edited one component of an assembly and returned only that one),
- the part shrank drastically or most of the input's faces no longer exist (you rebuilt it
  instead of editing it).

Growing the model is never treated as damage. When HEAD stays put, read the reason and change the
APPROACH -- retrying the same operation with a slightly different parameter is not reliable. In
particular `.fillet()` and `.chamfer()` can return broken geometry WITHOUT raising, and radius ->
success is not monotonic. If an edge operation keeps failing, build the feature with a boolean
instead: construct a cutter or a boss solid and `.cut()` / `.union()` it.

Producing geometry you already produced is reported back to you as a duplicate and is not
committed again. Treat that as a hard signal: the approach is exhausted, move on or change it.

## Invalid input geometry

**Some of these STEP files are already invalid before you touch them.** When the report says the
result is invalid *as the input already was*, that defect is not yours and repairing it is not
part of the request. Do not reach for `ShapeFix_Shape`, `BRepTools_ReShape` or any other repair
machinery -- keep editing normally. Only an invalidity your own edit introduced is a problem, and
the report names that case explicitly.

## Multi-solid assemblies

```python
solids = model.val().Solids()
# edit the target solid, keep every other one unchanged, then reassemble ALL of them:
model = cq.Workplane().add(cq.Compound.makeCompound(edited_and_untouched_solids))
```

## Reading the commit report

`changed_faces` lists the faces your edit CREATED, with their type, area, centre and (for
cylinders and cones) radius and axis. That is the handle for the next step: if you cut a hole and
now have to chamfer its edge, the cylinder you need is already named there -- do not go looking
for it. `solids` restates the solid roster whenever the part is an assembly, so you can see at
once whether your edit renumbered or dropped a component.

## Reading the diff images

After each commit you get renders of the change AGAINST THE ORIGINAL INPUT:
- **red** = material your edit removed
- **green** = material your edit added
- **grey translucent** = the resulting part

Check the coloured region sits exactly where the request asks -- right face, right coordinate,
right size. If you cannot see the change because it is hidden inside the part, call `render_view`
with a different camera or a tighter zoom instead of guessing.

## Common CadQuery patterns

```python
# select circular edges by radius
class RadiusSelector(cq.Selector):
    def __init__(self, r, tol=0.02):
        self.r, self.tol = r, tol
    def filter(self, objectList):
        return [o for o in objectList
                if o.geomType() == 'CIRCLE' and abs(o.radius() - self.r) < self.tol]

model = model.faces(">Z").workplane().hole(diameter=1.7)      # hole through top face
model = model.faces(">Z").workplane().circle(5.0).extrude(10)  # boss on top face
cutter = cq.Workplane("XY").transformed(offset=(cx, cy, cz)).circle(r).extrude(h, both=True)
model = model.cut(cutter)
```

## Procedure

1. On your FIRST turn, list every distinct sub-requirement of the request as a numbered checklist.
   "Add a hole AND chamfer its edge" is TWO items.
2. If the request names a feature you cannot locate from the summary and the renders, call
   `inspect` for it. Otherwise go straight to `run_script` with a real edit.
3. After each commit, state in ONE short line which checklist items are now satisfied, judging
   from the diff images and `changed_faces`, and immediately run the next script.
4. Only when EVERY item is confirmed done, answer with EDIT_COMPLETE.

Your turn budget is limited and every turn spent describing an image is a turn not spent fixing
the model. Running a corrected script is almost always worth more than looking at another view --
call `render_view` only when you genuinely cannot tell from the diff images where your change
landed. `inspect` and `dry_run` come out of a separate, smaller budget; when it runs out you are
told so and can still edit. Keep your commentary to a sentence.

Do the minimum that satisfies the request. Everything outside the requested region must stay
exactly as it was -- that is scored separately from the edit itself.
"""

# Bugs in our own code, not transport trouble -- retrying them just burns the clock.
_PERMANENT_ERRORS = (TypeError, AttributeError, KeyError, IndexError, NameError,
                     ImportError, SyntaxError, ValueError)


def _is_transient(exc):
    """Return whether a transport-level failure is safe to retry."""
    if isinstance(exc, _PERMANENT_ERRORS):
        return False
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(k in blob for k in (
        "429", "500", "502", "503", "504", "resource_exhausted", "resource exhausted",
        "unavailable", "deadline", "timeout", "timed out", "connection", "reset",
        "internal error", "overloaded", "retry",
    ))


NUDGE = (
    "Nothing has been committed yet -- no valid, changed geometry exists. Write the edit as a "
    "CadQuery script and call run_script."
)
MAX_NUDGES = 2

RESUME = (
    "The previous request to you failed in transport, not in your work. Nothing you did was lost: "
    "the history and every checkpoint are intact. Continue from where you stopped."
)
# ADK can re-raise exhausted provider retries. Match status text because google-genai uses generic
# ServerError/ClientError classes rather than one exception type per HTTP status.
MAX_TRANSPORT_RETRIES = 3


# The repository summarises with max_faces=400 because preservation matching scores every face.
# The model does not need 400: on the dataset's largest part that is thousands of lines of JSON in
# the same message as the request and seven images. Measured on a multi-solid rotor request: the
# agent spent four of ten turns re-deriving by hand facts that sat in that wall of text.
# Trimming the face list for the prompt while the gate keeps the full one costs nothing.
PROMPT_MAX_FACES = 60

# Read-only lookups get their own allowance on top of `max_turns`, which stays the edit budget.
# The split is not cosmetic: ADK enforces `RunConfig.max_llm_calls`, never a tool-call count
# (`invocation_context.increment_llm_call_count`, reached from `base_llm_flow._call_llm_async`), so
# every tool round-trip is one LLM call. A lookup tool sharing the edit budget would just relabel
# the cost that motivated it -- the agent would still be paying an edit turn per question.
DEFAULT_QUERY_BUDGET = 6


def lookup_budget(features):
    """(inspect_on, dry_run_on, budget) for a `features` dict. Budget 0 disables both tools.

    Kept at module level so an ablation config can be checked without starting a run: the honest
    comparison against a lookup-free v3 is `query_budget: 0`, which must leave the run turn-for-turn
    identical to what it was before, not merely similar.
    """
    budget = int((features or {}).get("query_budget", DEFAULT_QUERY_BUDGET))
    if budget <= 0:
        return False, False, 0
    inspect_on = bool((features or {}).get("inspect", True))
    dry_run_on = bool((features or {}).get("dry_run", True))
    return inspect_on, dry_run_on, (budget if (inspect_on or dry_run_on) else 0)


def _write_stdout(work_dir, text):
    """Persist a script's stdout next to its source. Best-effort: diagnostics never fail a run."""
    if not text:
        return
    try:
        os.makedirs(work_dir, exist_ok=True)
        with open(osp.join(work_dir, "stdout.txt"), "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
    except OSError:
        pass


def _for_prompt(summary):
    """Model-facing view of a repository summary: same structure, shorter face list."""
    if not summary:
        return summary
    trimmed = dict(summary)
    faces = trimmed.get("faces_listed") or []
    if len(faces) > PROMPT_MAX_FACES:
        trimmed["faces_listed"] = faces[:PROMPT_MAX_FACES]
        trimmed["faces_truncated"] = (
            f"Showing the {PROMPT_MAX_FACES} largest of {len(faces)} faces, sorted by area. Every "
            f"face carries the index of the solid it belongs to, and `solids[]` lists the full "
            f"face-type histogram per solid, so a face missing here is still accounted for. Call "
            f"`inspect` to see any of the remaining {len(faces) - PROMPT_MAX_FACES}."
        )
    return trimmed


class CadAgentRunnerV3:
    """Runs one edit request to completion and writes the winning geometry to `output_dir`."""

    def __init__(self, config):
        self.config = config
        self.features = config.get("features") or {}

    def _feat(self, name, default):
        return self.features.get(name, default)

    def _model(self):
        timeout = float(self.config.get("timeout_seconds", 300.0))
        client_kwargs = {"http_options": types.HttpOptions(timeout=int(timeout * 1000))}

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            client_kwargs.update(
                vertexai=True,
                project=project,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            )
        elif os.environ.get("GEMINI_API_KEY"):
            client_kwargs["api_key"] = os.environ["GEMINI_API_KEY"]
        else:
            raise ValueError("Set GOOGLE_CLOUD_PROJECT (Vertex + ADC) or GEMINI_API_KEY.")

        return Gemini(model=self.config["model"], client_kwargs=client_kwargs)

    def _generation_config(self):
        kwargs = {
            "temperature": float(self.config.get("temperature", 0.0)),
            "max_output_tokens": int(self.config.get("max_output_tokens", 32768)),
            "safety_settings": SAFETY_OFF,
        }
        level = self.config.get("thinking_level")
        if level in ("low", "medium", "high"):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level,
                                                             include_thoughts=True)
        elif level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1,
                                                             include_thoughts=True)
        return types.GenerateContentConfig(**kwargs)

    def run_edit(self, instruction_text, task_info_dict, output_dir, max_turns=10,
                 harness_script_file=None):
        """Sync entry point for the harness, which runs tasks in threads."""
        return asyncio.run(self._run_edit(
            instruction_text, task_info_dict, output_dir, max_turns, harness_script_file
        ))

    async def _run_edit(self, instruction_text, task_info_dict, output_dir, max_turns,
                        harness_script_file):
        # 1. Workspace, baseline geometry, repository
        os.makedirs(output_dir, exist_ok=True)
        iteration_dir = osp.join(output_dir, "iteration_output")

        if harness_script_file is None:
            harness_script_file = osp.join(
                osp.dirname(osp.dirname(osp.abspath(__file__))), "harnesses", "cadquery_script.py"
            )

        input_step = (task_info_dict.get("brep_start_path_step")
                      or task_info_dict.get("brep_start_path_stp"))
        input_step = osp.abspath(input_step) if input_step else None
        request = (task_info_dict.get("request_text")
                   or task_info_dict.get("request_transcript") or instruction_text)

        feedback_views = self._feat("views_feedback", "iso,front,top")
        repo = CadRepo(origin_step=input_step, work_dir=osp.join(iteration_dir, "repo"),
                       render_views=feedback_views,
                       render_timeout=int(self._feat("render_timeout", RENDER_TIMEOUT)))
        baseline = repo.origin_summary
        views = cad_tools.find_start_views(input_step) if input_step else []

        state = {"attempts": 0, "queries": 0, "log": [], "renders": []}
        # ADK dispatches the function calls of one model turn concurrently, so two `run_script`
        # calls could otherwise draw the same attempt number and share one work_dir.
        counter_lock = threading.Lock()

        # 2. Tools
        def run_script(script: str) -> dict:
            """Run a CadQuery script on the CURRENT state (HEAD) and try to commit the result.

            Args:
                script: Complete Python source defining my_cad_function(args). Import the model
                    from args['input_file'], apply one step of the edit, and return it.
            """
            with counter_lock:
                state["attempts"] += 1
                n = state["attempts"]
            work_dir = osp.join(iteration_dir, f"attempt_{n:03d}")
            head_before = repo.head.n if repo.head else 0
            print(f"[cad-agent-v3] run_script -> attempt {n:03d} (on "
                  f"{'c%d' % head_before if head_before else 'origin'})", flush=True)

            report, artifacts = cad_tools.run_cadquery_script(
                script=script,
                input_file=repo.head_step or "",     # <- checkpoint: build on HEAD
                work_dir=work_dir,
                harness_script_file=harness_script_file,
                baseline=None,                       # the repo owns the gate now
                summarise=False,                     # `try_commit` fingerprints the same STEP
            )

            # The tool returns the tail of stdout to the model but nothing kept it, so a finished
            # run could not be asked afterwards what the model actually saw -- recovering a failed
            # script's traceback meant re-executing it locally.
            _write_stdout(work_dir, report.get("stdout", ""))

            if report.get("status") != "ok" or not artifacts.get("step"):
                state["log"].append({"attempt": n, "committed": False,
                                     "status": report.get("status")})
                # A timeout carries `error` and neither `note` nor `stdout`, so the generic default
                # would tell the model its script "produced no STEP file" and hide the one fact
                # that decides what to do next: it ran out of time rather than failing.
                return {"committed": False, "stdout": report.get("stdout", "")[-3000:],
                        "reason": report.get("note") or report.get("error")
                                  or "The script produced no STEP file.",
                        "git_log": repo.log()}

            result = repo.try_commit(artifacts["step"], script_path=artifacts.get("script"),
                                     attempt=n, png=artifacts.get("png"),
                                     render=bool(self._feat("diff_render", True)))
            result["stdout"] = report.get("stdout", "")[-2000:]

            state["log"].append({
                "attempt": n,
                "committed": bool(result.get("committed")),
                "commit": result.get("commit"),
                "duplicate_of": result.get("duplicate_of"),
                "head_advanced": result.get("head_advanced"),
                "is_valid": result.get("is_valid"),
                "validity_regressed": result.get("validity_regressed"),
                "damaged": result.get("damaged"),
                "preserved_ratio": result.get("preserved_ratio"),
                "delta_vs_origin": result.get("delta_vs_origin"),
            })

            # Attach the images of the commit THIS attempt produced -- including a candidate HEAD
            # refused, because seeing why it was refused is the point of the visual channel. Never
            # fall back to `commits[-1]`: on a duplicate or an unchanged result that would show an
            # older commit's diff as if it were the new one.
            if result.get("committed") and repo.commits:
                latest = repo.commits[-1]
                if latest.diff_images:
                    # ADK lifts types.Part out of a tool result into the FunctionResponse.
                    for view, path in latest.diff_images:
                        result[f"diff_{view}"] = _image_part(path)
                    state["renders"].append(latest.diff_images)
            return result

        def render_view(views: str, zoom: float = 0.0) -> dict:
            """Render the current state from specific camera angles when the change is unclear.

            Args:
                views: Comma-separated camera names from: iso, front, back, left, right, top,
                    bottom. Example: "front,right".
                zoom: Optional zoom factor, e.g. 2.0 to look closer at a small feature. 0 = auto.
            """
            print(f"[cad-agent-v3] render_view({views}, zoom={zoom})", flush=True)
            images = repo.render_head(views=views, zoom=zoom or None)
            if not images:
                reason = f"Could not render views '{views}'."
                err = (repo.last_render or {}).get("error")
                return {"ok": False, "reason": reason + (f" {err}." if err else "")}
            out = {"ok": True, "rendered": [v for v, _ in images],
                   "note": ("red = removed, green = added, grey = the current part."
                            if repo.head else
                            "Nothing is committed yet, so this is the unmodified input, "
                            "rendered plain -- there is no change to colour in yet.")}
            # Naming a view that did not come back matters: without this the model reasons about a
            # camera it never received, and a partial result is indistinguishable from a full one.
            missing = [v.strip() for v in views.split(",")
                       if v.strip() and v.strip() not in {v for v, _ in images}]
            if missing:
                out["not_rendered"] = missing
                out["valid_view_names"] = "iso, front, back, left, right, top, bottom"
            for view, path in images:
                out[f"view_{view}"] = _image_part(path)
            return out

        def revert(to_commit: int = 0) -> dict:
            """Move HEAD back to an earlier commit when the last change was wrong.

            Args:
                to_commit: Commit number to return to, e.g. 1 for c1. 0 means one step back.
            """
            print(f"[cad-agent-v3] revert(to={to_commit})", flush=True)
            res = repo.revert(to=to_commit or None)
            res["git_log"] = repo.log()
            state["log"].append({"attempt": state["attempts"], "revert_to": res.get("head")})
            return res

        def _take_query(tool_name):
            """Claim one slot of the lookup budget. Returns (refusal_or_None, slot_number).

            The budget is enforced here rather than through ADK because ADK counts LLM CALLS, not
            tool calls (`invocation_context.increment_llm_call_count`, reached from
            `base_llm_flow._call_llm_async`), and every tool round-trip is one LLM call. Without a
            separate allowance, a lookup tool would simply eat edit turns and relabel the problem
            instead of removing it -- so the runner is given `max_turns + query_budget` calls and
            this counter keeps the extra ones from being spent on more lookups.

            The slot number is handed back from inside the lock rather than re-read afterwards:
            ADK dispatches the function calls of one model turn concurrently, so two `dry_run`
            calls would otherwise be able to claim different slots and then write into the same
            directory.
            """
            with counter_lock:
                if state["queries"] >= query_budget:
                    state["log"].append({"tool": tool_name, "refused": "query_budget_exhausted"})
                    return {"budget_exhausted": True, "reason": (
                        f"The lookup budget of {query_budget} calls is used up. Everything left is "
                        f"edit turns -- write the script with what you already know.")}, 0
                state["queries"] += 1
                return None, state["queries"]

        def inspect(entity: str = "faces", solid: int = -1, kind: str = "",
                    min_area: float = 0.0, near: str = "", radius: float = -1.0,
                    limit: int = 40) -> dict:
            """Query the B-Rep of the CURRENT state (HEAD). Read-only: nothing is committed.

            Filters combine with AND; leave one out to ignore it. The answer reports how many
            entities matched in total, so an empty result means the filter is wrong.

            Args:
                entity: What to list: "faces", "edges" or "solids".
                solid: Restrict to one solid by index, as listed in `solids`. -1 = all solids.
                kind: Surface or curve type, e.g. "PLANE", "CYLINDER", "CONE", "TORUS", "BSPLINE"
                    for faces; "CIRCLE" or "LINE" for edges. Empty = any.
                min_area: Faces only. Ignore faces smaller than this. 0 = no limit.
                near: "x,y,z" or "x,y,z,radius" -- keep only entities whose centre lies within
                    radius of that point. Default radius is 10 percent of the bounding-box
                    diagonal. Empty = no position filter.
                radius: Keep only circular edges / cylindrical faces of this radius,
                    within 0.05 mm. -1 = no radius filter.
                limit: How many records to return, largest first. Max 60.
            """
            refusal, _ = _take_query("inspect")
            if refusal:
                return refusal
            print(f"[cad-agent-v3] inspect({entity}, solid={solid}, kind={kind or '-'}, "
                  f"min_area={min_area}, near={near or '-'}, radius={radius})", flush=True)
            result = repo.inspect(entity=entity, solid=solid, kind=kind, min_area=min_area,
                                  near=near, radius=radius, limit=limit)
            state["log"].append({"tool": "inspect", "entity": entity, "solid": solid,
                                 "kind": kind, "matched": result.get("matched"),
                                 "total": result.get("total")})
            return result

        def dry_run(script: str) -> dict:
            """Run a script against HEAD and return only its stdout. Nothing is committed.

            For checking an assumption before spending an edit on it: what a selector picks up,
            which way a rotation goes, whether a computed coordinate lands where you expect.
            Print what you want to know -- the geometry the script returns is discarded.

            Args:
                script: Complete Python source defining my_cad_function(args), same shape as for
                    run_script. Import the model from args['input_file'].
            """
            refusal, n = _take_query("dry_run")
            if refusal:
                return refusal
            work_dir = osp.join(iteration_dir, f"query_{n:03d}")
            print(f"[cad-agent-v3] dry_run -> query {n:03d}", flush=True)
            report, artifacts = cad_tools.run_cadquery_script(
                script=script,
                input_file=repo.head_step or "",
                work_dir=work_dir,
                harness_script_file=harness_script_file,
                baseline=None,
                summarise=False,
            )
            _write_stdout(work_dir, report.get("stdout", ""))
            returned_model = bool(artifacts.get("step"))
            state["log"].append({"tool": "dry_run", "query": n, "status": report.get("status"),
                                 "returned_model": returned_model})
            out = {
                # NOT `status == "ok"`: without a STEP the harness calls the run an error, and for
                # a lookup that prints and returns nothing the error is the expected outcome. What
                # the model needs to know is what came back, not how the edit path would grade it.
                "stdout": report.get("stdout", "")[-3000:],
                "returned_model": returned_model,
                "note": (
                    "The script ran and returned a model. Nothing was committed -- call "
                    "`run_script` with it if this is the edit you want."
                    if returned_model else
                    "The script returned no model, which is fine for a lookup. If it raised "
                    "instead, the traceback is in the stdout above. Nothing was committed."
                ),
            }
            # A timeout produces no stdout at all, so without this the answer is an empty string
            # and the model cannot tell a silent script from a killed one.
            if report.get("error"):
                out["error"] = report["error"]
            return out

        tools = [run_script]
        if self._feat("render_on_demand", True):
            tools.append(render_view)
        if self._feat("revert", True):
            tools.append(revert)
        inspect_on, dry_run_on, query_budget = lookup_budget(self.features)
        if inspect_on:
            tools.append(inspect)
        if dry_run_on:
            tools.append(dry_run)

        agent = LlmAgent(
            name="cad_editor_v3",
            model=self._model(),
            instruction=SYSTEM_PROMPT,
            tools=tools,
            generate_content_config=self._generation_config(),
        )
        tracer = None
        if self._feat("trace", True):
            tracer = CadRunTracer(
                path=osp.join(iteration_dir, "run_trace.jsonl"),
                extra={"request": str(request)[:300], "model": self.config.get("model"),
                       "max_turns": max_turns, "origin": input_step,
                       "origin_is_valid": bool((repo.origin_summary or {}).get("is_valid"))},
            )
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME,
                                plugins=[tracer] if tracer else None)
        session = await runner.session_service.create_session(app_name=APP_NAME,
                                                              user_id="harness")

        # 3. Initial multimodal turn
        parts = [types.Part(text=
            f"Edit request: {request}\n\n"
            # Deliberately NOT the absolute path: a model that hardcodes it would edit the ORIGIN
            # every turn instead of HEAD. The result would still differ from HEAD, so it would
            # commit and HEAD would advance -- silently dropping the previous feature each time.
            f"Read the model ONLY via args['input_file']; it always points at HEAD.\n"
            f"Always read it via args['input_file'] -- it tracks HEAD as you commit.\n\n"
            f"Input model geometry summary:\n{json.dumps(_for_prompt(baseline), indent=1)}\n\n"
            f"Start now:\n"
            f"1. List the distinct sub-requirements of the request as a numbered checklist.\n"
            f"2. Call `run_script` with a complete script implementing `my_cad_function(args)` "
            f"that performs the first (or only) step. Apply the edit directly; if you first need "
            f"to find a face or an edge the summary does not pin down, call `inspect` for it "
            f"rather than writing a script that only prints.\n"
        )]
        if views and self._feat("views_start", 7):
            keep = int(self._feat("views_start", 7))
            shown = views[:keep]
            parts.append(types.Part(text=
                f"Below are the {len(shown)} standard renders of the input model. Use them to "
                f"locate features and understand orientation."
            ))
            for name, path in shown:
                parts.append(types.Part(text=f"View: {name}"))
                parts.append(_image_part(path))

        message = types.Content(role="user", parts=parts)

        # 4. Drive the runner
        totals = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        trace = []
        termination = "max_turns_reached"
        calls_used = 0
        nudges = 0
        transport_retries = 0
        t0 = time.perf_counter()
        agent_error = None
        # `max_turns` stays the EDIT budget, so a v3 run with lookups disabled is turn-for-turn the
        # same run as before. The lookup allowance is added on top and `_take_query` keeps it from
        # being spent on anything else; unclaimed lookups roll over into edit turns, never the
        # other way round.
        total_budget = max_turns + query_budget
        print(f"[cad-agent-v3] {request[:100]} | {len(views)} views | max {max_turns} turns "
              f"+ {query_budget} lookups", flush=True)

        while calls_used < total_budget:
            t_turn = time.perf_counter()
            last_text = ""
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
                            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                            "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
                        }
                        for k, v in used.items():
                            totals[k] += v
                        event_parts = (event.content.parts if event.content else None) or []
                        trace.append({
                            "turn": len(trace) + 1,
                            "seconds": round(time.perf_counter() - t_turn, 2),
                            "tools_called": [p.function_call.name for p in event_parts
                                             if p.function_call],
                            **used,
                        })
                        t_turn = time.perf_counter()

                    if event.content and event.author != "user":
                        last_text = " ".join(
                            p.text for p in (event.content.parts or [])
                            if p.text and not getattr(p, "thought", False)
                        ) or last_text
            except Exception as e:
                if "LlmCallsLimit" in type(e).__name__ or "max_llm_calls" in str(e):
                    termination = "max_turns_reached"
                    break
                if (_is_transient(e) and transport_retries < MAX_TRANSPORT_RETRIES
                        and calls_used < total_budget):
                    transport_retries += 1
                    print(f"[cad-agent-v3] transport error ({type(e).__name__}), retry "
                          f"{transport_retries}/{MAX_TRANSPORT_RETRIES}", flush=True)
                    # The ADK session already holds the full history, so the original message must
                    # NOT be resent -- that would duplicate the seven start images in the context
                    # and re-bill them. A short continuation is enough to re-enter the loop.
                    await asyncio.sleep(2 ** transport_retries)
                    message = types.Content(role="user", parts=[types.Part(text=RESUME)])
                    continue
                # Never re-raise: `run_harness` would write `failed_run: true` with no geometry,
                # the task would aggregate as 0.0, and a plain resume would SKIP it because
                # settings.json exists. Every commit made so far is still exportable, so record
                # the failure and fall through to step 5. Realistic triggers over 48 tasks x 8
                # threads: a transient Vertex 429/503, an OSError from shutil/subprocess, or an
                # exception inside a plugin hook (ADK wraps and re-raises those).
                agent_error = f"{type(e).__name__}: {e}"
                print(f"[cad-agent-v3] run aborted after {calls_used} calls: {agent_error}",
                      flush=True)
                termination = ("error_after_commit" if repo.commits else "error_before_commit")
                break

            # Driven by any recorded candidate, not by HEAD: an invalid or damaged attempt is
            # still a fallback worth keeping, so the agent should keep working while that is all
            # it has.
            if repo.commits:
                termination = ("agent_completed"
                               if any(m in last_text.upper() for m in DONE_MARKERS)
                               else "agent_stopped_with_result")
                break
            if nudges >= MAX_NUDGES:
                break
            nudges += 1
            message = types.Content(role="user", parts=[types.Part(text=NUDGE)])

        # 5. Export the winning state (replay-calibrated selection, not simply HEAD).
        # Reached on every path, including an aborted run -- see the except clause above.
        final_step = osp.join(output_dir, "tmp.step")
        try:
            source_step, final_source, retention = repo.final_step(
                self.config.get("damaged_candidate_policy", "fallback_to_input")
            )
        except Exception as e:
            # Selection itself failing must still not cost the geometry: ship HEAD, or the input.
            agent_error = agent_error or f"select_final failed: {type(e).__name__}: {e}"
            source_step = repo.head_step
            final_source = "fallback_head_after_selection_error"
            retention = {"selection_reason": "selection_error", "num_commits": len(repo.commits)}
        # Last resort: the raw input. Shipping no geometry at all scores 0.0 on all three metrics
        # AND leaves the scorer with no mesh at all, so the input always beats nothing.
        if not (source_step and osp.exists(source_step)) and input_step and osp.exists(input_step):
            source_step = input_step
            final_source = "fallback_unmodified_input_no_source"
        if source_step and osp.exists(source_step):
            shutil.copy(source_step, final_step)

        # Decided on the exported PATH, never on the label: `fallback_head_after_selection_error`
        # also starts with "fallback" but ships a real edit, and pairing that STEP with the start
        # mesh would hand the scorer the unedited geometry as our answer.
        ships_input = bool(input_step and source_step
                           and osp.abspath(source_step) == osp.abspath(input_step))

        # Whenever the answer is "unchanged", the dataset's own `.stl` has to travel with the STEP.
        # `diff_f1` is an F1 between two voxel diffs, so an empty ground-truth diff is only matched
        # by a prediction whose diff is EXACTLY empty -- and the scorer re-tessellating our STEP
        # with CadQuery instead of using Fusion's mesh is enough to break that. Measured on
        # SUJ2G2UMJQR7PMBX_1759209917: same B-rep, 1.0 with the shipped STL, 0.0 re-exported.
        # It decides 3 of the 48 requests, where changing nothing is the correct answer.
        if ships_input:
            _ship_start_stl(input_step, output_dir)

        # `termination_reason` predicts score without ground truth, which only holds
        # while it describes the exported result rather than the agent's mood: a run that says
        # "completed" and then ships the unmodified input produced no edit, whatever it claimed.
        # The agent's own last word is kept separately so nothing is lost.
        agent_stop_signal = termination
        if ships_input and not termination.startswith("error_"):
            termination = "no_valid_edit"

        # A render next to the STEP, for human inspection during ingestion.
        chosen = next((c for c in repo.commits if c.n == retention.get("chosen_commit")), None)
        if chosen and chosen.diff_images:
            shutil.copy(chosen.diff_images[0][1], osp.join(output_dir, "tmp.png"))
        elif chosen and chosen.png and osp.exists(chosen.png):
            shutil.copy(chosen.png, osp.join(output_dir, "tmp.png"))

        cost = (totals["input_tokens"] * float(self.config.get("1m_token_cost_input", 0.0))
                + (totals["output_tokens"] + totals["thinking_tokens"])
                * float(self.config.get("1m_token_cost_output", 0.0))) / 1e6

        print(f"[cad-agent-v3] done: {termination} | result={final_source} | "
              f"{state['attempts']} scripts | {len(repo.commits)} commits | "
              f"{repo.rejected} not built on | {repo.duplicates} repeated | ${cost:.4f}",
              flush=True)

        return {
            "token_counts": {**totals, "cost_estimate": round(cost, 5)},
            "num_iterations": len(trace),
            "num_scripts_run": state["attempts"],
            # The re-check for the lookup tools: if they worked, the share of script executions
            # that change nothing falls while these rise. Recorded per run so the comparison needs
            # no re-derivation from the scripts on disk.
            "num_queries": state["queries"],
            "query_budget": query_budget,
            "num_commits": len(repo.commits),
            "num_duplicate_attempts": repo.duplicates,
            # A renderer broken for the whole run used to look exactly like "no edit produced a
            # visible diff". Counting it here makes the condition legible in settings.json.
            "num_render_errors": sum(1 for c in repo.commits if (c.render_stats or {}).get("error")),
            "origin_is_valid": retention.get("origin_is_valid"),
            "origin_error": repo.origin_error,
            "agent_error": agent_error,
            "agent_stop_signal": agent_stop_signal,
            "shipped_unmodified_input": ships_input,
            "transport_retries": transport_retries,
            "run_trace": tracer.summary() if tracer else None,
            "num_head_advances": retention.get("num_head_advances"),
            "preserved_ratio": chosen.preserved_ratio if chosen else None,
            "termination_reason": termination,
            "final_source": final_source,
            "retention": retention,
            "git_log": repo.log(),
            "total_vlm_time_seconds": round(time.perf_counter() - t0, 2),
            "execution_trace": trace,
            "tool_call_log": state["log"],
        }
