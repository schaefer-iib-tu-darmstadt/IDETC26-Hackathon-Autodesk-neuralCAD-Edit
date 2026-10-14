"""CAD Geometry Tools & Subprocess Execution Engine.

Provides deterministic, offline geometry inspection and code execution tools for the CAD Agent.
Extracts BRep properties (faces, radii, bounding boxes) and validates CadQuery script outputs.
"""

import glob
import math
import os
import os.path as osp
import subprocess
import sys

import cadquery as cq

# Dataset render naming, in the order we show them to the model.
VIEW_ORDER = ("toprightiso", "front", "right", "top", "left", "back", "bottom")


def find_start_views(step_path):
    """Return [(view_name, image_path)] for the renders shipped next to a STEP file."""
    if not step_path:
        return []

    stem = osp.splitext(step_path)[0]
    found = {}
    for path in glob.glob(f"{stem}_*.jpg") + glob.glob(f"{stem}_*.png"):
        view = osp.splitext(osp.basename(path))[0].rsplit("_", 1)[-1]
        found.setdefault(view, path)

    ordered = [(v, found.pop(v)) for v in VIEW_ORDER if v in found]
    ordered.extend(sorted(found.items()))
    return ordered


def _bbox_raw(shape):
    """The six unrounded bounds. Rounding happens once, in `_bbox_from_raw`."""
    bb = shape.BoundingBox()
    return (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)


def _bbox_from_raw(raw):
    lo, hi = raw[:3], raw[3:]
    return {
        "min": [round(v, 3) for v in lo],
        "max": [round(v, 3) for v in hi],
        # From the raw bounds, never from the rounded ones: a box spanning -15.9995..15.8995 has
        # size 31.899, but rounding first and subtracting after gives 31.900. That half-millimetre
        # of double rounding is exactly what the dataset-wide check caught.
        "size": [round(hi[i] - lo[i], 3) for i in range(3)],
    }


def _bbox_union_raw(raws):
    """Axis-aligned union of several raw bound tuples.

    The exact AABB of a union of solids IS the union of their exact AABBs, so this reproduces
    `_bbox(shape)` rather than approximating it -- but only while every face of the shape belongs
    to one of those solids, which is what `_solids_cover_shape` checks.
    """
    return tuple([min(r[i] for r in raws) for i in range(3)]
                 + [max(r[i] for r in raws) for i in range(3, 6)])


def _solids_cover_shape(faces, solids):
    """True when every face of the shape sits inside one of its solids.

    An imported STEP compound can carry loose faces, shells or wires that belong to no solid; their
    extent would be missing from a union of solid boxes. Counting faces is ~10 ms, against the 13 s
    the union saves, so the guard is free.
    """
    try:
        return len(faces) == sum(len(s.Faces()) for s in solids)
    except Exception:
        return False


def _bbox(shape):
    return _bbox_from_raw(_bbox_raw(shape))


def _principal_axis(solid):
    """Direction of the solid's long axis, from its inertia tensor.

    The axis with the smallest moment of inertia is the one mass sits closest to -- i.e. the
    direction the solid is longest in. Returned together with the moment ratio, because on a
    near-isotropic solid (a hub, a cube) the axis is arbitrary and the caller must be able to
    tell: ratio near 1 means "no meaningful long axis".

    The ratio is smallest/second-smallest moment, not smallest/largest: a flat square plate has
    one large moment and two equal small ones, so smallest/largest reports a confident 0.5 for an
    axis that is in fact arbitrary. Comparing the two smallest is what actually asks "is the long
    direction unique?".

    Returns (axis, ratio) or (None, None). Best-effort: a malformed solid must not take the
    summary down with it.
    """
    try:
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps

        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid.wrapped, props)
        pp = props.PrincipalProperties()
        moments = list(pp.Moments())
        axes = [pp.FirstAxisOfInertia(), pp.SecondAxisOfInertia(), pp.ThirdAxisOfInertia()]
        lo = min(range(3), key=lambda i: moments[i])
        d = axes[lo]
        vec = [d.X(), d.Y(), d.Z()]
        # Sign is arbitrary for an inertia axis; pin it so two runs of the same part agree.
        for c in vec:
            if abs(c) > 1e-9:
                if c < 0:
                    vec = [-v for v in vec]
                break
        mid = sorted(moments)[1]
        ratio = round(moments[lo] / mid, 3) if mid > 1e-12 else None
        return [round(v, 3) for v in vec], ratio
    except Exception:
        return None, None


def _angle_between(a, b):
    """Angle in degrees between two undirected axes, so always in [0, 90]."""
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return round(math.degrees(math.acos(max(-1.0, min(1.0, dot)))), 2)


def _face_record(index, face, solid=None):
    """Compact description of one face: type, size, placement, and analytic parameters.

    `solid` tags the face with the index of the solid it belongs to. Measured need: on the rotor
    task the agent spent two of ten turns writing probe scripts that walked `s1.Faces()` and
    printed exactly these fields, because a flat face list over the whole compound cannot answer
    "the faces of blade 1" -- the single question it asked most.
    """
    rec = {
        "id": f"F{index}",
        "type": face.geomType(),
        "area": round(face.Area(), 3),
        "center": [round(c, 3) for c in face.Center().toTuple()],
    }
    if solid is not None:
        rec["solid"] = solid

    if rec["type"] == "PLANE":
        try:
            rec["normal"] = [round(n, 3) for n in face.normalAt(face.Center()).toTuple()]
        except Exception:
            pass
        try:
            n_inner = len(face.innerWires())
            if n_inner:
                rec["inner_loops"] = n_inner
        except Exception:
            pass
    elif rec["type"] in ("CYLINDER", "CONE", "TORUS"):
        try:
            from OCP.BRepAdaptor import BRepAdaptor_Surface

            adaptor = BRepAdaptor_Surface(face.wrapped)
            if rec["type"] == "CYLINDER":
                surface = adaptor.Cylinder()
                rec["radius"] = round(float(surface.Radius()), 3)
            elif rec["type"] == "CONE":
                surface = adaptor.Cone()
                rec["radius"] = round(float(surface.RefRadius()), 3)
                rec["half_angle_deg"] = round(math.degrees(float(surface.SemiAngle())), 2)
            else:
                surface = adaptor.Torus()
                rec["radius"] = round(float(surface.MinorRadius()), 3)
                rec["major_radius"] = round(float(surface.MajorRadius()), 3)
            axis = surface.Axis().Direction()
            rec["axis"] = [round(axis.X(), 3), round(axis.Y(), 3), round(axis.Z(), 3)]
        except Exception:
            pass

    return rec


def _edge_record(index, edge, solid=None):
    """Compact description of one edge: type, length, midpoint, and analytic parameters.

    Circular edges carry their radius because that is what fillet and chamfer selection needs.
    `summarise_shape` reports only a histogram of the radii present -- how many edges of radius r
    exist, never *where* they are -- so an instruction like "add rounds to the four long edges of
    the blade" cannot be turned into a selector from the summary alone.
    """
    rec = {
        "id": f"E{index}",
        "type": edge.geomType(),
        "length": round(edge.Length(), 3),
        "center": [round(c, 3) for c in edge.Center().toTuple()],
    }
    if solid is not None:
        rec["solid"] = solid
    if rec["type"] == "CIRCLE":
        try:
            rec["radius"] = round(edge.radius(), 3)
        except Exception:
            pass
    return rec


def _parse_near(near):
    """Parse 'x,y,z' or 'x,y,z,r' into ((x, y, z), r or None). None for anything unusable."""
    if not near:
        return None
    try:
        nums = [float(v) for v in str(near).replace(";", ",").split(",") if v.strip() != ""]
    except ValueError:
        return None
    if len(nums) == 3:
        return (nums[0], nums[1], nums[2]), None
    if len(nums) == 4:
        return (nums[0], nums[1], nums[2]), nums[3]
    return None


def _diagonal_from_size(shape):
    """Bounding-box diagonal from the cheap box, used only as the default reach for `near`.

    Deliberately `BoundingBox()` and not `_bbox`: `AddOptimal_s` is exact and costs ~13 s on the
    largest part in the dataset, and nothing here needs that precision -- it sets a search radius.
    """
    try:
        bb = shape.BoundingBox()
        return (bb.xlen ** 2 + bb.ylen ** 2 + bb.zlen ** 2) ** 0.5
    except Exception:
        return 1.0


# Absolute, in millimetres. Radii are named in the requests in mm ("R=0.2mm"), and a relative
# tolerance would make a 0.2 mm round unfindable while matching a 30 mm bore to within 1.5 mm.
QUERY_RADIUS_TOL = 0.05


def summarise_shape(shape, max_faces=40, derive_bbox_from_solids=False, group_by_solid=False):
    """Numeric fingerprint of a shape. Used both as the agent's eyes and as the change gate.

    `derive_bbox_from_solids=True` builds the shape's bounding box from the per-solid boxes this
    function already computes for a multi-solid shape, instead of calling `BRepBndLib.AddOptimal_s`
    a second time on the whole compound. Measured on the dataset's largest part (58 solids,
    557 faces): 13.3 s saved of a 28 s call, with `min`, `max` and `size` identical to three
    decimals. Off by default; `CadRepo._summarise` opts in.

    `group_by_solid=True` adds the structure the agent kept re-deriving by hand: every face record
    carries the index of its solid, and every solid carries its face-type histogram, its inertia
    long axis and the angles to the other solids' axes. All three were measured needs on the rotor
    task, where four of ten turns went on probe scripts that computed exactly this -- one of them
    hand-deriving an angle from two printed cylinder centres that the axis vectors already
    determined. Additive only: no existing key changes, so `_differs`, `preserved_ratio` and
    `assess_damage` are unaffected. Off by default; `CadRepo._summarise` opts in.
    """
    faces = shape.Faces()
    edges = shape.Edges()
    solids = shape.Solids()

    face_types = {}
    for f in faces:
        face_types[f.geomType()] = face_types.get(f.geomType(), 0) + 1

    circle_radii = {}
    n_line = 0
    for e in edges:
        if e.geomType() == "CIRCLE":
            try:
                r = round(e.radius(), 3)
                circle_radii[r] = circle_radii.get(r, 0) + 1
            except Exception:
                pass
        elif e.geomType() == "LINE":
            n_line += 1

    # Walking solid by solid renumbers the ids, which is harmless -- they are opaque labels for the
    # model, and every consumer matches faces on (type, area, centre). `_solids_cover_shape` guards
    # the case where the compound carries loose faces that belong to no solid.
    if group_by_solid and solids and _solids_cover_shape(faces, solids):
        records, seq = [], 0
        for si, sol in enumerate(solids):
            for f in sol.Faces():
                records.append(_face_record(seq, f, solid=si))
                seq += 1
    else:
        records = [_face_record(i, f) for i, f in enumerate(faces)]
    records.sort(key=lambda r: r["area"], reverse=True)

    solids = shape.Solids()
    try:
        vol = round(float(shape.Volume()), 4)
    except Exception:
        vol = round(float(sum(s.Volume() for s in solids)), 4) if solids else 0.0

    try:
        com = [round(c, 3) for c in shape.Center().toTuple()]
    except Exception:
        com = [round(c, 3) for c in solids[0].Center().toTuple()] if solids else [0.0, 0.0, 0.0]

    # Per-solid boxes double as the shape's own box when the solids cover it -- see `_bbox_union`.
    solid_bboxes = None
    if len(solids) > 1 and derive_bbox_from_solids and _solids_cover_shape(faces, solids):
        raws = [_bbox_raw(s) for s in solids]
        solid_bboxes = [_bbox_from_raw(r) for r in raws]
        shape_bbox = _bbox_from_raw(_bbox_union_raw(raws))
    else:
        shape_bbox = _bbox(shape)

    summary = {
        "is_valid": bool(shape.isValid()),
        "volume": vol,
        "center_of_mass": com,
        "bbox": shape_bbox,
        "counts": {
            "solids": len(solids),
            "faces": len(faces),
            "edges": len(edges),
            "by_face_type": face_types,
        },
        "circular_edge_radii": [
            {"radius": r, "count": n} for r, n in sorted(circle_radii.items())
        ],
        "straight_edges": n_line,
        "faces_listed": records[:max_faces],
    }

    if len(records) > max_faces:
        summary["faces_truncated"] = (
            f"Showing the {max_faces} largest of {len(records)} faces, sorted by area. "
            f"Raise max_faces to see more."
        )

    if len(solids) > 1:
        boxes = solid_bboxes if solid_bboxes is not None else [_bbox(s) for s in solids]
        summary["solids"] = [
            {
                "index": i,
                "volume": round(float(s.Volume()), 4),
                "center": [round(c, 3) for c in s.Center().toTuple()],
                "bbox": boxes[i],
            }
            for i, s in enumerate(solids)
        ]
        if group_by_solid:
            _describe_solids(summary["solids"], solids, records)

    return summary


# Above this the long axis is not unique: the two smallest moments are within 10 % of each other,
# which is a cube, a square plate or a disc. Reporting an axis there would invite the model to
# rotate about a direction that is an artefact of the tessellation.
ISOTROPIC_MOMENT_RATIO = 0.9
# Pairwise angles are O(n^2) lines of JSON; past a handful of solids the table is longer than the
# rest of the summary and answers nothing the model asked for.
MAX_SOLIDS_FOR_ANGLES = 8


def _describe_solids(entries, solids, records):
    """Attach per-solid structure to `summary["solids"]` -- in place."""
    by_solid = {}
    for r in records:
        si = r.get("solid")
        if si is not None:
            by_solid.setdefault(si, {}).setdefault(r["type"], 0)
            by_solid[si][r["type"]] += 1

    for i, (entry, sol) in enumerate(zip(entries, solids)):
        if i in by_solid:
            entry["faces"] = sum(by_solid[i].values())
            entry["by_face_type"] = by_solid[i]
        axis, ratio = _principal_axis(sol)
        if axis is not None:
            entry["long_axis"] = axis
            entry["axis_moment_ratio"] = ratio
            if ratio is not None and ratio > ISOTROPIC_MOMENT_RATIO:
                entry["axis_note"] = "near-isotropic solid: long_axis is arbitrary"

    directional = [(e["index"], e["long_axis"]) for e in entries
                   if e.get("long_axis") and (e.get("axis_moment_ratio") or 1.0)
                   <= ISOTROPIC_MOMENT_RATIO]
    if 2 <= len(directional) <= MAX_SOLIDS_FOR_ANGLES:
        for pos, (i, ai) in enumerate(directional):
            others = {f"to_solid_{j}": _angle_between(ai, aj)
                      for j, aj in directional[pos + 1:]}
            if others:
                entries[i]["angle_deg"] = others


def load_shape(step_path):
    """Import a STEP file and return the top-level shape."""
    return cq.importers.importStep(step_path).val()


# A STEP round-trip is not bit-exact: OCC re-parameterises surfaces on export. Measured on two
# parts by importing and re-exporting unchanged (deterministic across repeats):
#   relative volume drift up to 3.5e-4, relative centre-of-mass drift up to 6.1e-5,
#   while face count, bounding box and the circular-edge-radius histogram were exactly stable.
# The thresholds below sit ~15x above the measured drift; the exact signals carry the gate.
VOLUME_REL_TOL = 5e-3
CENTROID_REL_TOL = 1e-3
BBOX_REL_TOL = 1e-4
FACE_LAYOUT_AREA_RTOL = 2e-4
FACE_LAYOUT_CENTRE_RTOL = 1e-5


def _face_layout_differs(before, after):
    """Detect feature placement changes hidden by aggregate shape statistics.

    A regular polygon rotated around its own centre preserves volume, topology, bounding box,
    centroid and radius histograms. Those were the complete fingerprint, so a 30 degree
    rotation of an inscribed hexagonal cutout was rejected as a duplicate even though that
    orientation changed Dev16 diff-F1 from 0.5714 to 0.9767. Face area and centre are already
    recorded for grounding and are stable to three decimals across a STEP round-trip; match them
    as an unordered set so OCC face enumeration order remains irrelevant.

    Truncated summaries deliberately fall back to the aggregate checks below. A partial face list
    cannot prove equality, and treating an omitted small face as a placement change would create
    false commits on the largest assemblies.
    """
    left = before.get("faces_listed") or []
    right = after.get("faces_listed") or []
    if not left or not right or before.get("faces_truncated") or after.get("faces_truncated"):
        return False
    if len(left) != len(right):
        return True

    diag = max((sum(s ** 2 for s in before["bbox"]["size"])) ** 0.5, 1e-9)
    centre_tol = max(0.002, diag * FACE_LAYOUT_CENTRE_RTOL)
    unmatched = list(right)
    for face in left:
        for index, candidate in enumerate(unmatched):
            if candidate.get("type") != face.get("type"):
                continue
            area_tol = max(
                0.002,
                FACE_LAYOUT_AREA_RTOL * max(abs(face.get("area", 0.0)), 1.0),
            )
            if abs(candidate.get("area", 0.0) - face.get("area", 0.0)) > area_tol:
                continue
            distance = sum(
                (a - b) ** 2
                for a, b in zip(candidate.get("center", ()), face.get("center", ()))
            ) ** 0.5
            if distance > centre_tol:
                continue
            unmatched.pop(index)
            break
        else:
            return True
    return False




def _differs(before, after):
    """True when `after` is geometrically distinguishable from `before`.

    Deliberately redundant: topology catches feature edits, the radius histogram catches pure
    dimension changes, and the centroid catches material that only moved.
    """
    if after["counts"]["faces"] != before["counts"]["faces"]:
        return True
    if after["counts"]["solids"] != before["counts"]["solids"]:
        return True
    if after["circular_edge_radii"] != before["circular_edge_radii"]:
        return True
    if _face_layout_differs(before, after):
        return True

    diag = max((sum(s ** 2 for s in before["bbox"]["size"])) ** 0.5, 1e-9)
    if any(abs(a - b) / diag > BBOX_REL_TOL
           for a, b in zip(after["bbox"]["size"], before["bbox"]["size"])):
        return True
    if any(abs(a - b) / diag > CENTROID_REL_TOL
           for a, b in zip(after["center_of_mass"], before["center_of_mass"])):
        return True

    return abs(after["volume"] - before["volume"]) / max(abs(before["volume"]), 1e-9) > VOLUME_REL_TOL


# Damage thresholds, calibrated against the 48 human ground-truth edits (2026-08-17). Measured by
# summarising every start STEP and its GT edit STEP and taking the observed range of the ratios:
#
#   volume ratio gt/start   min 0.4500  median 1.0058  max 970.1
#   bbox diagonal ratio     min 0.9773  median 1.0000  max  10.0
#   solid count delta       -1 once, 0 in 37 cases, +1..+4 in 10 cases
#
# Correct edits shrink almost nothing and may grow without bound, so only the shrinking side
# carries signal and only a one-sided veto is defensible. Both thresholds sit below the tightest
# human edit with margin, so no human-plausible edit trips them. Losing a solid happens once in
# the GT, which makes it a ranking penalty rather than a veto.
DAMAGE_VOLUME_RATIO = 0.35   # GT minimum 0.45
DAMAGE_DIAG_RATIO = 0.95     # GT minimum 0.9773


def _diagonal(summary):
    return max(sum(s ** 2 for s in summary["bbox"]["size"]) ** 0.5, 1e-9)


def assess_damage(before, after):
    """Judge whether `after` lost geometry relative to `before`, without access to ground truth.

    Used to stop a late catastrophic attempt from displacing an earlier clean one. Deliberately
    one-sided: growth is never damage, because human edits grow geometry by up to 970x in volume,
    while the tightest human edit shrinks the bounding box by only 2.3 %.

    `after` may be the trimmed result dict from `run_cadquery_script`; only volume, bbox size and
    solid count are read.
    """
    vol_before = max(abs(before["volume"]), 1e-9)
    volume_ratio = after["volume"] / vol_before
    diag_ratio = _diagonal(after) / _diagonal(before)
    solids_lost = max(0, before["counts"]["solids"] - after["counts"]["solids"])

    reasons = []
    if volume_ratio < DAMAGE_VOLUME_RATIO:
        reasons.append(
            f"volume dropped to {volume_ratio:.1%} of the input; no ground-truth edit in this "
            f"dataset keeps less than 45 %"
        )
    if diag_ratio < DAMAGE_DIAG_RATIO:
        reasons.append(
            f"bounding box diagonal shrank to {diag_ratio:.1%} of the input, so geometry is "
            f"missing at the extremities"
        )

    return {
        "damaged": bool(reasons),
        "reasons": reasons,
        "volume_ratio": round(volume_ratio, 4),
        "diagonal_ratio": round(diag_ratio, 4),
        "solids_lost": solids_lost,
    }


def run_cadquery_script(script, input_file, work_dir, harness_script_file, baseline=None,
                        timeout=120, summarise=True, render=True):
    """Execute a CadQuery script in a subprocess and measure what it produced.

    `summarise=False` skips step 4, which re-imports the exported STEP and fingerprints it. That
    fingerprint costs ~28 s on the largest part in the dataset (58 solids / 557 faces), almost all
    of it in `_bbox` -- `BRepBndLib.AddOptimal_s` is exact and expensive, and `summarise_shape`
    calls it once for the shape plus once per solid. The agent fingerprints the same file itself
    afterwards (`CadRepo._summarise`, which needs per-solid validity on top) and would otherwise
    pay it twice per attempt, so it passes `summarise=False`.

    Skipping it drops `is_valid` and `result` from the report and disables the `baseline` gate, so
    only pass False when the caller does its own inspection. The default keeps the full report for
    standalone use.
    """
    # 1. Prepare isolated workspace and write script file
    os.makedirs(work_dir, exist_ok=True)
    script_file = osp.join(work_dir, "script.py")
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script)

    # 2. Run harness script in subprocess
    cmd = [sys.executable, harness_script_file, "--output_dir", work_dir, "--function_file", script_file]
    if input_file:
        cmd += ["--input_file", input_file]
    if not render:
        cmd.append("--skip_render")

    step_path = osp.join(work_dir, "tmp.step")
    png_path = osp.join(work_dir, "tmp.png")
    artifacts = {"script": script_file, "step": None, "png": None, "stdout": ""}

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Execution exceeded {timeout}s and was killed."}, artifacts

    stdout = (proc.stdout or "") + (proc.stderr or "")
    artifacts["stdout"] = stdout
    report = {"stdout": stdout[-4000:]}

    # 3. Check for generated STEP artifact
    if not osp.exists(step_path) or osp.getsize(step_path) == 0:
        report["status"] = "error"
        report["step_written"] = False
        report["note"] = (
            "No STEP file was produced, so nothing was kept. Either the script raised, or "
            "my_cad_function returned None. Read the stdout above and fix the script."
        )
        return report, artifacts

    artifacts["step"] = step_path
    if osp.exists(png_path):
        artifacts["png"] = png_path

    # 4. Extract shape metrics & validate BRep
    if not summarise:
        report["status"] = "ok"
        report["step_written"] = True
        return report, artifacts

    try:
        result = summarise_shape(load_shape(step_path))
    except Exception as e:
        report["status"] = "error"
        report["step_written"] = True
        report["note"] = f"The exported STEP could not be re-imported: {e}"
        return report, artifacts

    report["status"] = "ok"
    report["step_written"] = True
    report["is_valid"] = result["is_valid"]
    report["result"] = {
        "volume": result["volume"],
        "bbox": result["bbox"],
        "center_of_mass": result["center_of_mass"],
        "counts": result["counts"],
        "circular_edge_radii": result["circular_edge_radii"],
    }

    # 5. Compute geometric delta vs baseline & evaluate change gate
    if baseline is not None:
        changed = _differs(baseline, result)
        report["changed_vs_input"] = changed
        report["delta"] = {
            "volume": round(result["volume"] - baseline["volume"], 4),
            "faces": result["counts"]["faces"] - baseline["counts"]["faces"],
            "solids": result["counts"]["solids"] - baseline["counts"]["solids"],
            "bbox_size": [round(a - b, 3) for a, b in
                          zip(result["bbox"]["size"], baseline["bbox"]["size"])],
        }
        report["kept_as_result"] = bool(result["is_valid"] and changed)

        if not result["is_valid"]:
            report["note"] = (
                "The solid is INVALID and was discarded. CadQuery reports success for some "
                "broken results without raising, so this is not visible in stdout. Change the "
                "approach or the parameters -- retrying the same operation with a slightly "
                "different value is not reliable."
            )
        elif not changed:
            report["note"] = (
                "Valid, but geometrically identical to the input, so it was not kept. This is "
                "the expected outcome for a probe script; if you meant to edit, the operation "
                "did not take effect."
            )
        else:
            report["note"] = (
                "Valid and different from the input -- kept as the current result. Now verify "
                "against the request: right location, right dimensions, rest of the part "
                "untouched. A non-zero delta does not mean the edit is correct."
            )

    return report, artifacts


def run_probe_script(script, input_file, work_dir, probe_harness_file=None, timeout=30):
    """Execute a read-only probe and return stdout without exporting or rendering geometry.

    This is intentionally a separate worker from ``run_cadquery_script``: the edit worker's job is
    to export geometry, and a flag that sometimes suppresses that would leave accidental export work
    easy to reintroduce. The probe's return value is ignored; only its stdout is useful.
    """
    os.makedirs(work_dir, exist_ok=True)
    script_file = osp.join(work_dir, "probe.py")
    with open(script_file, "w", encoding="utf-8") as stream:
        stream.write(script)

    if probe_harness_file is None:
        probe_harness_file = osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))), "harnesses", "cadquery_probe.py"
        )
    cmd = [
        sys.executable,
        probe_harness_file,
        "--output_dir", work_dir,
        "--function_file", script_file,
    ]
    if input_file:
        cmd += ["--input_file", input_file]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") + (exc.stderr or "")
        return {
            "status": "error",
            "error": f"Probe exceeded {timeout}s and was killed.",
            "stdout": stdout[-4000:],
            "returned_model": False,
        }

    stdout = (proc.stdout or "") + (proc.stderr or "")
    returned_model = "probe_returned_model=1" in stdout
    report = {
        "status": "ok" if proc.returncode == 0 else "error",
        "stdout": stdout[-4000:],
        "returned_model": returned_model,
    }
    if proc.returncode != 0:
        report["error"] = f"Probe worker exited with code {proc.returncode}."
    return report
