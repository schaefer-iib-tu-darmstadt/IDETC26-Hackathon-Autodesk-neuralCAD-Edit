"""Git4CAD -- version control over B-Rep states.

The stock harness rewrites the whole script every round and starts from the original STEP each
time, so an attempt is never a step on top of its predecessor. That makes "was my last change an
improvement?" unanswerable and lets a late bad attempt overwrite a good early one (measured:
diff_f1 peaks at 4-6 scripts and falls again at 7-10).

This module gives the agent the five operations that make an iterative edit legible:

    commit  -- a validated state, accepted only when the kernel gates pass
    diff    -- against the parent commit *and* against the origin (the scored comparison)
    revert  -- go back to an earlier commit
    log     -- a compact text history, so the context does not have to carry ten rounds of images
    HEAD    -- what the next script builds on

plus a pre-commit hook: a candidate that loses solids, destroys most of the untouched geometry, or
rewrites the part from scratch is rejected before it can become HEAD.

Validity is judged *relative to the input*: some dataset STEP files fail `isValid()` before any
edit, so only invalidity the edit itself introduced counts as a regression. HEAD stays a
conservative base for the next edit, while final delivery may still retain a candidate that merely
inherited the input's defect.

Imports no SDK -- pure geometry, unit-testable on its own.
"""

import os
import os.path as osp
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

from src.agents.cad_tools import _differs, assess_damage, load_shape, summarise_shape

# Face matching tolerances, relative to the origin's bounding-box diagonal / face area.
FACE_AREA_RTOL = 0.02
FACE_CENTRE_RTOL = 5e-3

# Default seconds before a diff render is killed. A single render of a complex part was measured at
# ~118 s, and `--workers N` puts N of them on the machine at once, so this is a config knob
# (`features.render_timeout`) rather than a constant.
RENDER_TIMEOUT = 300


def _rank(candidate):
    """Return the deterministic delivery rank for a retained candidate."""
    damage = candidate["damage"]
    return (
        1 if candidate["is_valid"] else 0,
        0 if damage["damaged"] else 1,
        0 if damage["solids_lost"] else 1,
        1 if candidate.get("is_valid_absolute", True) else 0,
        candidate["attempt"],
    )


def select_final(candidates, damaged_policy="fallback_to_input"):
    """Choose the retained candidate to export, or None for the input fallback."""
    if not candidates:
        return None, "no_candidate"

    winner = max(candidates, key=_rank)
    if damaged_policy == "export_anyway" or not winner["damage"]["damaged"]:
        return winner, "ranked_winner"

    undamaged = [candidate for candidate in candidates if not candidate["damage"]["damaged"]]
    if undamaged:
        return max(undamaged, key=_rank), "ranked_winner_after_damage_veto"
    return None, "all_candidates_damaged"


def _ship_start_stl(input_step, output_dir):
    """Copy the dataset STL when the unmodified input is selected for delivery."""
    start_stl = osp.splitext(input_step)[0] + ".stl"
    if osp.exists(start_stl):
        shutil.copy(start_stl, osp.join(output_dir, "tmp.stl"))


def _child_env():
    """Environment for the render subprocess, forced to UTF-8.

    The child prints paths and exception texts; on Windows it would otherwise encode them with the
    ANSI code page and die with `UnicodeEncodeError` on the first non-ASCII character, while the
    parent decodes as UTF-8.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _invalid_solids(shape):
    """Indices of the solids OCC rejects, or None when the check itself could not run.

    None and [] must stay distinguishable. Collapsing a failed check into "no broken solids" is
    wrong in both directions: on the candidate it would wave every genuinely broken result through
    as inherited, and on the origin it would flag every inherited defect as newly caused. Both
    misreadings were observed in testing and cost most of a task's turn budget each.
    """
    try:
        return [i for i, s in enumerate(shape.Solids()) if not s.isValid()]
    except Exception:
        return None


@dataclass
class Commit:
    """One accepted state of the model."""

    n: int
    step_path: str
    summary: dict
    script_path: Optional[str] = None
    attempt: int = 0
    delta_vs_parent: dict = field(default_factory=dict)
    delta_vs_origin: dict = field(default_factory=dict)
    preserved_ratio: float = 1.0
    is_valid: bool = True
    validity_regressed: bool = False
    damage: dict = field(default_factory=dict)
    is_head_candidate: bool = True
    abandoned: bool = False
    png: Optional[str] = None
    diff_images: list = field(default_factory=list)
    render_stats: dict = field(default_factory=dict)
    message: str = ""

    def log_line(self):
        d = self.delta_vs_origin
        flags = []
        if self.abandoned:
            flags.append("reverted")
        if self.validity_regressed:
            flags.append("INVALID (caused by this edit)")
        elif not self.is_valid:
            # Inherited from the input. Named explicitly so the history cannot be read as a defect
            # the agent introduced; read that way, the agent spends its whole budget on a repair
            # nothing asked for.
            flags.append("invalid, as the input already was")
        if self.damage.get("damaged"):
            flags.append("DAMAGED")
        if self.damage.get("solids_lost"):
            flags.append(f"-{self.damage['solids_lost']} solid")
        return (
            f"c{self.n}  vol {d.get('volume', 0.0):+.3f}  faces {d.get('faces', 0):+d}  "
            f"solids {d.get('solids', 0):+d}  preserved {self.preserved_ratio:.0%}"
            + (("  [" + ", ".join(flags) + "]") if flags else "")
            + (f"  {self.message}" if self.message else "")
        )

    def as_candidate(self):
        """Return the normalized candidate record used by final delivery selection.

        `is_valid` carries the *relative* verdict on purpose. On a task whose input is already
        invalid every candidate inherits that flaw, so ranking on the absolute value would push all
        of them into the last-resort tier and decide between them by nothing. The absolute value
        travels alongside for labelling only.
        """
        return {"attempt": self.attempt or self.n, "step": self.step_path, "png": self.png,
                "is_valid": not self.validity_regressed, "is_valid_absolute": self.is_valid,
                "damage": self.damage, "commit": self.n}


def _delta(before, after):
    return {
        "volume": round(after["volume"] - before["volume"], 4),
        "faces": after["counts"]["faces"] - before["counts"]["faces"],
        "solids": after["counts"]["solids"] - before["counts"]["solids"],
        "bbox_size": [round(a - b, 3) for a, b in
                      zip(after["bbox"]["size"], before["bbox"]["size"])],
    }


def _match_faces(before_summary, after_summary):
    """Greedy face correspondence between two summaries.

    Returns (matched, only_before, only_after). Faces are matched on (surface type, area, centre)
    because those survive the STEP round-trip exactly enough -- the measured drift is in volume,
    not in face count or placement.

    Both `preserved_ratio` and the per-commit face delta read this, so the two can never disagree
    about which faces the edit touched.
    """
    before = before_summary.get("faces_listed") or []
    after = after_summary.get("faces_listed") or []

    diag = max((sum(s ** 2 for s in before_summary["bbox"]["size"])) ** 0.5, 1e-9)
    centre_tol = diag * FACE_CENTRE_RTOL

    unmatched = list(after)
    matched, only_before = 0, []
    for face in before:
        for i, cand in enumerate(unmatched):
            if cand["type"] != face["type"]:
                continue
            denom = max(abs(face["area"]), 1e-9)
            if abs(cand["area"] - face["area"]) / denom > FACE_AREA_RTOL:
                continue
            dist = sum((a - b) ** 2 for a, b in zip(cand["center"], face["center"])) ** 0.5
            if dist > centre_tol:
                continue
            unmatched.pop(i)
            matched += 1
            break
        else:
            only_before.append(face)

    return matched, only_before, unmatched


def preserved_ratio(origin_summary, result_summary):
    """Fraction of the origin's faces still findable in the result.

    The B-Rep equivalent of an image editor's background-preservation score.
    """
    origin_faces = origin_summary.get("faces_listed") or []
    if not origin_faces:
        return 1.0
    matched, _, _ = _match_faces(origin_summary, result_summary)
    return matched / len(origin_faces)


# Half of the 48 requests are assemblies, and a solid index that shifts under an edit is one of the
# named failure modes of the task ("index ambiguity"). Re-stating the roster after every commit is
# cheap while it stays short; past this the shape is a mould or a rotor stack and `ground` is the
# right way in.
ROSTER_MAX_SOLIDS = 12


def solid_roster(summary):
    """Compact per-solid list for the commit report, or None when it would not be worth the room."""
    solids = (summary or {}).get("solids") or []
    if not solids or len(solids) > ROSTER_MAX_SOLIDS:
        return None
    return [{"index": s["index"], "volume": s["volume"], "center": s["center"],
             "faces": s.get("faces")} for s in solids]


# How many of the newly created faces travel back with each commit. The point of the delta is that
# it is SHORT -- a hole cut into a plate creates three faces, not four hundred -- so the cap is a
# guard against a rebuild-from-scratch flooding the context, not a working limit. Everything the
# cap hides is reachable through `ground`.
DELTA_MAX_FACES = 20


def face_delta(parent_summary, result_summary):
    """What this edit added to and removed from the B-Rep, as face records.

    The commit report used to carry only aggregate numbers -- volume, bbox, face and solid counts.
    That tells the agent THAT its edit landed, never WHERE, so the follow-up step ("now chamfer the
    edge of the hole you just cut") had no handle to select against: the only face index it ever
    received described the *origin*, and after its first commit it was blind to its own work.
    Measured cost of that blindness on a multi-solid rotor request: 4 of 10 turns went to throwaway
    scripts that imported the model, printed faces and returned it unchanged.

    Costs nothing in turns -- the summaries are already computed for the gate.

    A face the edit *reshaped* rather than created shows up on both sides, because `_match_faces`
    matches on area within `FACE_AREA_RTOL` and a boss imprinted on a plate changes that plate's
    area. That is the useful reading, not a defect: fusing a boss reports the cylinder, the boss
    top and the host face it now sits in, which is exactly the set the follow-up step selects
    against. Only faces that survived untouched are absent.
    """
    if not parent_summary or not result_summary:
        return None
    _, removed, added = _match_faces(parent_summary, result_summary)
    delta = {
        "faces_added": len(added),
        "faces_removed": len(removed),
        "new_faces": added[:DELTA_MAX_FACES],
    }
    if len(added) > DELTA_MAX_FACES:
        delta["new_faces_truncated"] = (
            f"{len(added)} faces are new; showing the first {DELTA_MAX_FACES}. Use `inspect` for "
            f"the rest."
        )
    return delta


def assess_commit(origin_summary, candidate_summary):
    """Judge a candidate. Returns (head_may_advance, warnings, metrics, notes).

    This decides only whether the candidate is a safe BASE for the next script -- never whether it
    is exported. Kernel verdict and model opinion stay separate instances by design: visual
    feedback is known to raise the invalid rate (CADFusion 4.84 % -> 6.20 %), so validity must not
    depend on what any model thinks it sees.
    """
    warnings = []
    notes = []
    if not origin_summary:
        # No origin to judge against (the dataset row carried no start STEP). Everything the gate
        # decides is a comparison, so with nothing to compare to the only honest verdict is "let it
        # through" -- refusing here would make HEAD unreachable for the whole task.
        return True, [], {"preserved_ratio": 1.0, "damaged": False, "damage_reasons": [],
                          "volume_ratio": None, "solids_lost": 0,
                          "is_valid": bool(candidate_summary.get("is_valid")),
                          "validity_regressed": False, "invalid_solids": None,
                          "dropped_assembly": False, "global_transform_suspected": False}, []

    ratio = preserved_ratio(origin_summary, candidate_summary)
    damage = assess_damage(origin_summary, candidate_summary)

    is_valid = bool(candidate_summary.get("is_valid"))
    origin_valid = bool(origin_summary.get("is_valid", True))
    origin_bad = origin_summary.get("invalid_solids")
    cand_bad = candidate_summary.get("invalid_solids")

    # A regression is a defect THIS edit introduced. Inheriting the input's own broken solid is not
    # one, and must not gate anything -- see the module header.
    #
    # Compared by identity, not cardinality: an edit that breaks solid #3 while a boolean rebuild
    # incidentally heals the input's broken #7 leaves the count unchanged, and a count test would
    # call that innocent and then tell the model its result is fine. Indices are stable enough for
    # this because both summaries come from the same import path.
    if is_valid:
        validity_regressed = False
    elif origin_valid:
        validity_regressed = True                      # input was clean, so this edit broke it
    elif origin_bad is None or cand_bad is None:
        # Check failed on one side. Be conservative about the candidate (assume it may have
        # regressed) only when we could not inspect the candidate itself; if only the origin is
        # unknown, trust its `is_valid` flag, which already says it was broken.
        validity_regressed = cand_bad is None
    else:
        validity_regressed = bool(set(cand_bad) - set(origin_bad))

    metrics = {
        "preserved_ratio": round(ratio, 4),
        "damaged": damage["damaged"],
        "damage_reasons": damage["reasons"],
        "volume_ratio": damage["volume_ratio"],
        "solids_lost": damage["solids_lost"],
        "is_valid": is_valid,
        "validity_regressed": validity_regressed,
        "invalid_solids": len(cand_bad) if cand_bad is not None else None,
    }

    if validity_regressed:
        warnings.append(
            "This edit made the model INVALID. CadQuery reports success for some broken results "
            "without raising, so this is not visible in stdout. HEAD did not move. Change the "
            "approach -- retrying the same operation with a slightly different value is not "
            "reliable."
        )
    elif not is_valid:
        notes.append(
            f"isValid() is False, but it already was for the unmodified input "
            f"({len(origin_bad) if origin_bad else 'some'} broken solid(s) of "
            f"{origin_summary['counts']['solids']}), and your "
            f"edit did not add any. This is a property of the source file. Do NOT try to repair "
            f"it: shape-fixing geometry you did not create wastes the turn budget and risks the "
            f"parts that are fine. Treat this result as usable and keep editing."
        )

    if damage["damaged"]:
        warnings.append(
            "Geometry was destroyed: " + "; ".join(damage["reasons"]) +
            ". That is a rebuilt or broken part, not an edit. Import the model from "
            "args['input_file'] and modify it."
        )

    # Solid loss alone is NOT the assembly-drop signature -- the ground truth loses a solid on one
    # of the 48 edits, and three requests ask for it outright ("merge the mirrored part", "remove
    # the collision", "change the plug"). Vetoing HEAD on the bare count would permanently block
    # those: the target state has fewer solids than the input, so no attempt could ever advance.
    # What distinguishes a dropped assembly from a deliberate merge is that the dropped one also
    # loses the untouched faces. Both signals together gate; either alone only informs.
    solids_before = origin_summary["counts"]["solids"]
    solids_after = candidate_summary["counts"]["solids"]
    dropped_assembly = bool(
        damage["solids_lost"] and solids_before > 1
        and solids_after * 2 < solids_before and ratio < 0.5
    )
    metrics["dropped_assembly"] = dropped_assembly

    if dropped_assembly:
        warnings.append(
            f"Component loss: {solids_before} solids went in, {solids_after} came out, and only "
            f"{ratio:.0%} of the input's faces survive. That is the signature of selecting one "
            f"solid of an assembly, editing it, and returning only that one. Reassemble every "
            f"solid before returning:\n"
            f"  solids = model.val().Solids()   # edit the target, keep the rest\n"
            f"  model = cq.Workplane().add(cq.Compound.makeCompound(all_solids))"
        )
    elif damage["solids_lost"]:
        notes.append(
            f"{damage['solids_lost']} solid(s) fewer than the input, but the surrounding geometry "
            f"is intact ({ratio:.0%} of faces preserved). If the request asked you to merge, "
            f"remove or replace a component, this is correct and HEAD moved here."
        )

    # A global transform matches no face at all -- `preserved_ratio` compares absolute areas and
    # absolute centre positions, so "Scale the part 10x" (request 16) and the move/shrink requests
    # score 0 % while being perfectly correct. Topology is the discriminator: a scale or a move
    # keeps every face and edge and only changes their size or position, whereas the destruction
    # this warning is for removes them. Suppress the warning when the counts are untouched.
    topology_intact = (
        candidate_summary["counts"]["faces"] == origin_summary["counts"]["faces"]
        and candidate_summary["counts"]["edges"] == origin_summary["counts"]["edges"]
        and solids_after == solids_before
    )
    metrics["global_transform_suspected"] = bool(ratio < 0.35 and topology_intact)

    if ratio < 0.35 and not damage["damaged"] and not topology_intact:
        warnings.append(
            f"Only {ratio:.0%} of the input's faces survive. Everything outside the region named "
            f"in the request must stay untouched."
        )
    elif metrics["global_transform_suspected"]:
        notes.append(
            f"Face matching finds only {ratio:.0%} of the input's faces, but the face, edge and "
            f"solid counts are unchanged -- that is what a scale, move or rotation of the whole "
            f"part looks like, not damage. If the request asked for a global transform this is "
            f"correct; if it did not, you have moved geometry that should have stayed put."
        )

    head_may_advance = (not validity_regressed and not damage["damaged"]
                        and not dropped_assembly)
    return head_may_advance, warnings, metrics, notes


class CadRepo:
    """Commit history over B-Rep states for one edit request."""

    def __init__(self, origin_step, work_dir, render_views="iso,front,top",
                 render_timeout=RENDER_TIMEOUT):
        self.origin_step = origin_step
        self.work_dir = work_dir
        self.render_views = render_views
        self.render_timeout = render_timeout
        self.commits: list[Commit] = []
        self.head: Optional[Commit] = None
        self.rejected = 0
        self.duplicates = 0
        self.last_render = {}
        self.origin_error = None
        # ADK runs the function calls of one model turn concurrently -- `asyncio.create_task` per
        # call in `flows/llm_flows/functions.py`, sync tools dispatched to a thread pool. Two
        # `apply_edit` calls in one turn would otherwise interleave inside `try_commit` and hand
        # both the same commit number.
        self._lock = threading.RLock()
        os.makedirs(work_dir, exist_ok=True)
        # After makedirs, and guarded: a corrupt origin STEP must degrade the run, not abort it
        # before the agent has seen the request.
        self.origin_summary = None
        if origin_step:
            try:
                self.origin_summary = self._summarise(origin_step)
            except Exception as e:
                self.origin_error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _summarise(step_path):
        """`summarise_shape` plus the per-solid validity the relative gate needs.

        Kept here rather than in `cad_tools.summarise_shape` so that function stays a plain,
        side-effect-free fingerprint with stable defaults.
        """
        shape = load_shape(step_path)
        # max_faces drives `faces_listed`, which preservation matching reads -- the default of 40
        # would silently score only the largest faces on a complex part.
        summary = summarise_shape(shape, max_faces=400, derive_bbox_from_solids=True,
                                  group_by_solid=True)
        summary["invalid_solids"] = _invalid_solids(shape)
        return summary

    # -- state ---------------------------------------------------------------------------------

    @property
    def head_step(self):
        """What the next script must build on: HEAD if there is one, else the original input."""
        return self.head.step_path if self.head else self.origin_step

    @property
    def parent_summary(self):
        return self.head.summary if self.head else self.origin_summary

    def log(self):
        if not self.commits:
            return "(no commits yet -- HEAD is the unmodified input)"
        lines = [c.log_line() + ("   <- HEAD" if self.head and c.n == self.head.n else "")
                 for c in self.commits]
        return "\n".join(lines)

    def _find_twin(self, summary):
        """An existing commit with geometrically indistinguishable content, or None.

        Reuses `_differs` rather than an equality test on numbers: a STEP round-trip moves the
        volume by up to 3.5e-4 relative, so an exact comparison would call every re-export a new
        state. `_differs` gates on topology first and uses relative tolerances.

        Known limit: two genuinely different edits with the same fingerprint -- a feature moved to
        a mirror-symmetric position keeps volume, counts, radii and centroid -- are reported as
        twins. The change gate has always had that blind spot, so the sensitivity is at least
        consistent across the harness, and the cost is bounded: the earlier commit with
        that fingerprint stays in the export pool, so no *scored* geometry is lost, only one of two
        states our own metrics cannot tell apart.
        """
        for c in self.commits:
            if not _differs(c.summary, summary):
                return c
        return None

    # -- operations ----------------------------------------------------------------------------

    def try_commit(self, step_path, script_path=None, attempt=0, png=None, render=True):
        """Record a candidate; advance HEAD only if it is a safe base for the next script.

        Every candidate is kept for the export decision -- a solid that is invalid or damaged is
        still better than nothing on the one task where it is the only attempt matching the
        request. It just does not get built on.

        Returns a report dict for the model. Diff images are attached by the caller as image
        parts, so this module stays free of any SDK type.
        """
        with self._lock:
            return self._commit(step_path, script_path=script_path, attempt=attempt, png=png,
                                render=render)

    def _commit(self, step_path, script_path=None, attempt=0, png=None, render=True):
        """`try_commit` with the lock already held."""
        try:
            summary = self._summarise(step_path)
        except Exception as e:
            return {"committed": False, "reason": f"The exported STEP could not be re-imported: {e}"}

        # Checked FIRST, and against the origin -- not the parent. History: the first checkpoint
        # version compared only against the parent, and that silently dropped the input comparison
        # the earlier flat harness had. Consequence: a script that undoes c1 back to the original state
        # differs from HEAD, so it commits, HEAD advances, and `_rank`'s later-wins tie-break
        # exports the untouched input for diff_f1 0.0 -- on 45 of the 48 requests. The origin is
        # the reference diff_f1 scores against, so it is the one comparison that may never lapse.
        if self.origin_summary and not _differs(self.origin_summary, summary):
            return {
                "committed": False,
                "reason": (
                    "The result is geometrically identical to the ORIGINAL input, so nothing was "
                    "committed. Whatever this script did, it cancelled out every edit made so far "
                    "-- exporting this would score zero. If you wanted to discard earlier work, "
                    "call `revert` instead; it keeps the history intact. Otherwise apply the "
                    "requested change to HEAD."
                ),
                "reverted_to_origin": True,
                "git_log": self.log(),
            }

        parent = self.parent_summary
        changed = _differs(parent, summary) if parent else True
        if not changed:
            where = f"c{self.head.n}" if self.head else "the unmodified input"
            return {
                "committed": False,
                "reason": (
                    f"The result is geometrically identical to HEAD ({where}), so nothing was "
                    f"committed and no images were rendered. Either the operation did not take "
                    f"effect -- a selector that matched nothing, a boolean against empty geometry "
                    f"-- or you re-applied an edit that is already part of HEAD. Remember that "
                    f"args['input_file'] already contains everything committed so far; do not "
                    f"redo earlier steps."
                ),
                "preserved_ratio": 1.0,
                "git_log": self.log(),
            }

        # Already produced this exact geometry before? Then re-rendering and re-committing it buys
        # nothing and hides the repetition from the agent. Measured in testing: ten of eleven
        # commits were the same state, costing ~118 s of diff render each while the history said
        # only "c1..c10" with no hint that they were identical.
        twin = self._find_twin(summary)
        if twin is not None:
            self.duplicates += 1
            return {
                "committed": False,
                "duplicate_of": f"c{twin.n}",
                "reason": (
                    f"This produces exactly the geometry already recorded as c{twin.n}"
                    + (" (which you reverted)" if twin.abandoned else "")
                    + ". Nothing was committed and no new images were rendered, because there is "
                    f"nothing new to look at. Repeating an approach that did not work will not "
                    f"start working. Either accept c{twin.n} and move to the next part of the "
                    f"request, or change the strategy -- a different selector, a different "
                    f"operation, or different values."
                ),
                "git_log": self.log(),
            }

        advance, warnings, metrics, notes = assess_commit(self.origin_summary, summary)

        # `assess_damage` only ever looks at the origin, so a commit that deletes exactly what its
        # predecessor built is invisible to it -- the solid count is back where it started.
        # Measured: c1 added the two requested ports (diff_f1 0.8432), c2 then removed the two
        # original ones (0.5969), and nothing anywhere said so. Kept as a note
        # rather than a gate: removing solids is a legitimate edit, and only the request can say
        # whether "create a port" means add one or move one.
        if parent is not None:
            lost_vs_parent = parent["counts"]["solids"] - summary["counts"]["solids"]
            if lost_vs_parent > 0 and self.head is not None:
                notes.append(
                    f"This removed {lost_vs_parent} solid(s) that were present in HEAD "
                    f"(c{self.head.n}). If that was intended, continue. If you were only adding a "
                    f"feature, you have undone earlier work -- HEAD already contains it, so do not "
                    f"rebuild or replace it."
                )

        n = len(self.commits) + 1
        kept = osp.join(self.work_dir, f"c{n:03d}.step")
        shutil.copy(step_path, kept)

        commit = Commit(
            n=n,
            step_path=kept,
            summary=summary,
            script_path=script_path,
            attempt=attempt or n,
            png=png,
            delta_vs_parent=_delta(parent, summary) if parent else {},
            delta_vs_origin=_delta(self.origin_summary, summary) if self.origin_summary else {},
            preserved_ratio=metrics["preserved_ratio"],
            is_valid=metrics["is_valid"],
            validity_regressed=metrics["validity_regressed"],
            damage={"damaged": metrics["damaged"], "reasons": metrics["damage_reasons"],
                    "solids_lost": metrics["solids_lost"],
                    "volume_ratio": metrics["volume_ratio"]},
            is_head_candidate=advance,
            message="" if advance else "not built on",
        )

        # Recorded BEFORE rendering. `c{n:03d}.step` is already on disk, so a renderer failure that
        # escaped `_render_diff` would leave the file without a commit -- the next attempt would
        # compute the same `n` and overwrite it.
        self.commits.append(commit)
        if advance:
            self.head = commit
        else:
            self.rejected += 1

        if render:
            commit.diff_images = self._render_diff(commit)
            if self.last_render:
                report_render = dict(self.last_render)
                if (report_render.get("removed_volume") == 0.0
                        and report_render.get("added_volume") == 0.0
                        and not report_render.get("error")):
                    report_render["warning"] = (
                        "The renderer found no material added or removed relative to the input, "
                        "even though the geometry did change. Either the boolean failed or the "
                        "difference is below the display threshold -- read the numbers, not the "
                        "colours."
                    )
                commit.render_stats = report_render

        report = {
            "committed": True,
            "commit": f"c{n}",
            "head_advanced": advance,
            "head": f"c{self.head.n}" if self.head else "original input",
            "delta_vs_parent": commit.delta_vs_parent,
            "delta_vs_origin": commit.delta_vs_origin,
            "result": {
                "volume": summary["volume"],
                "bbox": summary["bbox"],
                "counts": summary["counts"],
                "circular_edge_radii": summary["circular_edge_radii"],
            },
            # The B-Rep handle on what just happened, so the next step can select against the
            # geometry this one created instead of probing for it. See `face_delta`.
            "changed_faces": face_delta(parent, summary),
            "solids": solid_roster(summary),
            "is_valid": commit.is_valid,
            "validity_regressed": commit.validity_regressed,
            "preserved_ratio": metrics["preserved_ratio"],
            "damaged": metrics["damaged"],
            "git_log": self.log(),
        }
        if commit.render_stats:
            report["render_measurements"] = commit.render_stats
        if notes:
            report["kernel_notes"] = " ".join(notes)
        if warnings:
            report["problems"] = " ".join(warnings)
            report["note"] = (
                f"Recorded as c{n}, but HEAD stayed at "
                f"{'c%d' % self.head.n if self.head else 'the original input'} -- the next script "
                f"will build on that, not on this result. Fix the problems above."
            )
        else:
            report["note"] = (
                "Committed and HEAD moved here; your next script builds on this state. The images "
                "show the change AGAINST THE ORIGINAL INPUT: red = material removed, green = "
                "material added, grey translucent = the result. Check the coloured regions sit "
                "exactly where the request asks. A non-zero delta does not mean the edit is "
                "correct."
            )
        return report

    def revert(self, to=None):
        """Move HEAD back and abandon everything after it.

        Abandoning matters: without it the export selection would still rank a reverted commit as
        a candidate and could ship the very state the agent just rejected, which is what an early
        test run did.

        Two rules that are easy to get wrong, because commit numbers are dense over *all* commits
        including abandoned ones:

        * "one step back" means the previous commit that is both live and a legal HEAD -- not
          `head.n - 1`. That can land on a commit the agent already discarded and silently
          un-abandon it, or on one the kernel refused as a base (invalid, damaged, dropped
          assembly), which would make `revert` the one way to install a wreck as the foundation.
          `to=None` is the default path from the tool (`to_commit=0`), so this is the common case.
        * revert never moves forward. `to` above HEAD would resurrect a discarded later commit,
          which is a redo, not a revert.
        """
        with self._lock:
            return self._revert(to)

    def _revert(self, to=None):
        """`revert` with the lock already held."""
        if not self.commits:
            return {"ok": False, "reason": "No commits to revert."}

        eligible = [c.n for c in self.commits
                    if not c.abandoned and c.is_head_candidate
                    and (self.head is None or c.n < self.head.n)]
        if to is None:
            to = max(eligible, default=0)

        if to <= 0:
            for c in self.commits:
                c.abandoned = True
            self.head = None
            return {"ok": True, "head": "original input", "abandoned": [c.n for c in self.commits],
                    "log": self.log()}

        if self.head is not None and to > self.head.n:
            return {"ok": False,
                    "reason": (f"c{to} is ahead of HEAD (c{self.head.n}); revert only moves back. "
                               f"To rebuild a later state, write the script again.")}

        match = [c for c in self.commits if c.n == to and not c.abandoned]
        if not match:
            available = [c.n for c in self.commits if not c.abandoned and c.is_head_candidate]
            return {"ok": False,
                    "reason": f"No live commit c{to}. Available: {available or 'none'}"}
        if not match[0].is_head_candidate:
            return {"ok": False,
                    "reason": (f"c{to} was recorded but refused as a base ({match[0].log_line()}), "
                               f"so HEAD cannot move onto it -- the next script would build on "
                               f"broken geometry. Revert further back, or to 0 for the original "
                               f"input. It stays available for export either way.")}

        dropped = [c.n for c in self.commits if c.n > to and not c.abandoned]
        for c in self.commits:
            if c.n > to:
                c.abandoned = True
        self.head = match[0]
        dropped_names = ", ".join(f"c{n}" for n in dropped)
        return {"ok": True, "head": f"c{self.head.n}", "abandoned": dropped, "log": self.log(),
                "note": f"HEAD is c{to}; {dropped_names} will not be exported. Your next script "
                        f"builds on c{to}." if dropped else None}

    # -- rendering -----------------------------------------------------------------------------

    def _renderer_cmd(self, new_step, out_dir, names, prefix, zoom=None, old_step=None,
                      plain=False):
        cmd = [
            sys.executable,
            osp.join(osp.dirname(osp.abspath(__file__)), "cad_render.py"),
            "--new", new_step,
            "--output_dir", out_dir,
            "--views", ",".join(names),
            "--prefix", prefix,
        ]
        if old_step:
            cmd += ["--old", old_step]
        if plain:
            cmd.append("--plain")
        if zoom:
            cmd += ["--zoom", str(zoom)]
        return cmd

    def _run_renderer(self, cmd):
        """Spawn the render worker and record what it reported in `self.last_render`.

        A renderer that is broken for the whole run -- no OpenGL, an unreadable STEP, an import
        error -- used to be indistinguishable from "this edit produced no visible diff": the exit
        code and stderr were discarded and `last_render` stayed empty. Both are kept now so the
        condition shows up in `settings.json` instead of silently costing every task its images.
        """
        self.last_render = {}
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=self.render_timeout, env=_child_env())
        except subprocess.TimeoutExpired:
            # Not a return: views written before the kill are still valid.
            self.last_render = {"error": f"the renderer timed out after {self.render_timeout}s"}
            return
        except OSError as e:
            self.last_render = {"error": f"the renderer could not be started: {e}"}
            return

        for line in (proc.stdout or "").splitlines():
            key, _, value = line.partition("=")
            if key in ("removed_volume", "added_volume"):
                try:
                    self.last_render[key] = float(value)
                except ValueError:
                    pass
            elif key == "unknown_view":
                self.last_render.setdefault("unknown_views", []).append(value)

        if proc.returncode != 0:
            tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
            self.last_render["error"] = (
                f"the renderer exited with code {proc.returncode}" + (f": {tail}" if tail else "")
            )

    def _render_diff(self, commit, views=None, zoom=None, prefix=None):
        """Render the diff against the ORIGIN -- the comparison diff_f1 actually scores.

        Also records what the renderer measured (`self.last_render`), because a failed boolean and
        a genuine no-op produce the same picture: an image with no colour on it. Without the
        numbers the model cannot tell "my edit did not land" from "the diff could not be computed"
        and re-does work that already succeeded.
        """
        out_dir = osp.join(self.work_dir, f"c{commit.n:03d}_render")
        prefix = prefix or "diff"
        names = [v.strip() for v in (views or self.render_views).split(",") if v.strip()]
        # Clear the targets first: a failing re-render would otherwise leave the previous images in
        # place and they would be collected as if they were the new ones.
        for view in names:
            stale = osp.join(out_dir, f"{prefix}_{view}.png")
            if osp.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

        self._run_renderer(self._renderer_cmd(commit.step_path, out_dir, names, prefix, zoom=zoom,
                                              old_step=self.origin_step))

        if not osp.isdir(out_dir):
            return []
        found = []
        for view in names:
            path = osp.join(out_dir, f"{prefix}_{view}.png")
            if osp.exists(path) and osp.getsize(path) > 0:
                found.append((view, path))
        return found

    # -- finalisation --------------------------------------------------------------------------

    def final_step(self, damaged_policy="fallback_to_input"):
        """The state to export. Returns (path, source_label, report).

        HEAD is the conservative edit base; delivery selects from all retained candidates.
        """
        # A reverted commit is one the agent explicitly rejected; it only comes back into play
        # when nothing else survives, which still beats exporting the raw input.
        live = [c for c in self.commits if not c.abandoned]
        pool = live or self.commits
        candidates = [c.as_candidate() for c in pool]
        winner, reason = select_final(candidates, damaged_policy)
        if not live and self.commits:
            reason += "_only_reverted_left"

        report = {
            "selection_reason": reason,
            "num_commits": len(self.commits),
            "num_head_advances": sum(1 for c in self.commits if c.is_head_candidate),
            "num_duplicate_attempts": self.duplicates,
            "origin_is_valid": bool(self.origin_summary.get("is_valid")) if self.origin_summary
                               else None,
            "chosen_commit": winner["commit"] if winner else None,
            "head_commit": self.head.n if self.head else None,
            "overrode_head": bool(winner and self.head and winner["commit"] != self.head.n),
        }

        if winner:
            label = f"commit_c{winner['commit']}"
            if not winner["is_valid"]:
                label += "_invalid_last_resort"
            elif not winner.get("is_valid_absolute", True):
                # Invalid, but only as the input already was -- a normal result, not a fallback.
                label += "_inherited_invalid"
            if winner["damage"].get("damaged"):
                label += "_damaged"
            return winner["step"], label, report

        if self.origin_step and osp.exists(self.origin_step):
            label = ("fallback_input_all_candidates_damaged" if reason == "all_candidates_damaged"
                     else "fallback_unmodified_input")
            return self.origin_step, label, report
        return None, "none", report
