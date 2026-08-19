#!/usr/bin/env python3
"""Diff Render Worker (standalone subprocess).

Renders a colour-coded difference between two STEP states:
    red   = material removed by the edit
    green = material added by the edit
    grey  = the result itself, translucent, for context

Run as a subprocess, never imported into the agent process: the benchmark harness runs tasks
through a ThreadPoolExecutor and the VTK renderer underneath `cadquery.vis` is not thread-safe.

Usage:
    python cad_render.py --old a.step --new b.step --output_dir out [--views iso,front,top]
                         [--plain]        render `--new` alone, no diff overlay
"""

import argparse
import os
import os.path as osp
import sys

import cadquery as cq
from cadquery.vis import show

# Camera presets: (azimuth, elevation). `iso` matches the dataset's toprightiso convention.
VIEWS = {
    "iso": (45, 30),
    "front": (0, 0),
    "back": (180, 0),
    "left": (-90, 0),
    "right": (90, 0),
    "top": (0, 89),
    "bottom": (0, -89),
}

# The seven dataset start images are labelled with Fusion's names, and those labels are shown to
# the model at the start of every run -- so it learns that vocabulary from us and then asks for it
# back. Measured: a request for "bottom,toprightiso" returned one image instead of two and cost
# 27 s. Aliasing is the honest fix; renaming the start views would desynchronise
# them from the dataset filenames they are globbed from.
ALIASES = {
    "toprightiso": "iso",
    "isometric": "iso",
    "topright": "iso",
    "iso1": "iso",
}

COLOR_RESULT = (0.70, 0.71, 0.75, 0.30)
COLOR_REMOVED = (0.88, 0.13, 0.13, 1.00)
COLOR_ADDED = (0.10, 0.72, 0.24, 1.00)

# Below this volume a boolean residue is numerical noise from the STEP round-trip, not an edit.
MIN_DIFF_VOLUME = 1e-7


def _load(path):
    return cq.importers.importStep(path).val()


def _safe_volume(shape):
    try:
        return float(shape.Volume())
    except Exception:
        return 0.0


def build_diff_assembly(old_shape, new_shape):
    """Colour-coded assembly of the edit. Returns (assembly, stats)."""
    stats = {"removed_volume": 0.0, "added_volume": 0.0}

    try:
        removed = old_shape.cut(new_shape)
        stats["removed_volume"] = _safe_volume(removed)
    except Exception:
        removed = None
    try:
        added = new_shape.cut(old_shape)
        stats["added_volume"] = _safe_volume(added)
    except Exception:
        added = None

    asm = cq.Assembly()
    asm.add(new_shape, name="result", color=cq.Color(*COLOR_RESULT))
    if removed is not None and stats["removed_volume"] > MIN_DIFF_VOLUME:
        asm.add(removed, name="removed", color=cq.Color(*COLOR_REMOVED))
    if added is not None and stats["added_volume"] > MIN_DIFF_VOLUME:
        asm.add(added, name="added", color=cq.Color(*COLOR_ADDED))
    return asm, stats


def render(obj, out_path, view="iso", width=900, height=700, zoom=None):
    """Screenshot one camera angle. Returns True when a non-empty file was written."""
    # No silent fallback to iso: an unrecognised name -- a typo, or one the model invents such as
    # "isometric" or "front_left" -- would otherwise write an iso image under the requested
    # filename, and the model would reason about feature placement from the wrong camera.
    view = ALIASES.get(view, view)
    if view not in VIEWS:
        print(f"unknown_view={view}")
        return False
    azimuth, elevation = VIEWS[view]
    kwargs = dict(
        screenshot=out_path,
        interact=False,
        width=width,
        height=height,
        azimuth=azimuth,
        elevation=elevation,
        trihedron=False,
        gradient=False,
        bgcolor=(1.0, 1.0, 1.0),
    )
    if zoom:
        kwargs["zoom"] = float(zoom)
    try:
        show(obj, **kwargs)
    except Exception as e:  # a failed render must never take the edit down with it
        print(f"render failed for view '{view}': {e}")
        return False
    return osp.exists(out_path) and osp.getsize(out_path) > 0


def main():
    # The parent decodes our stdout as UTF-8 and also sets PYTHONUTF8 for us; this is the belt for
    # a direct CLI invocation on a Windows console, where a non-ASCII path in an error message
    # would otherwise raise UnicodeEncodeError and take the render down.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--old")
    ap.add_argument("--new", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--views", default="iso,front,top")
    ap.add_argument("--prefix", default="diff")
    ap.add_argument("--zoom", type=float, default=None)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--plain", action="store_true", help="Render --new alone, without a diff.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        new_shape = _load(args.new)
    except Exception as e:
        print(f"could not import --new {args.new}: {e}", file=sys.stderr)
        return 2

    if args.plain or not args.old or not osp.exists(args.old):
        target, stats = new_shape, {"removed_volume": 0.0, "added_volume": 0.0}
    else:
        try:
            target, stats = build_diff_assembly(_load(args.old), new_shape)
        except Exception as e:
            # The diff is a nicety; the result itself is still worth showing.
            print(f"diff against --old failed, rendering the result alone: {e}", file=sys.stderr)
            target, stats = new_shape, {"removed_volume": 0.0, "added_volume": 0.0}

    print(f"removed_volume={stats['removed_volume']:.6f}")
    print(f"added_volume={stats['added_volume']:.6f}")

    # A non-zero exit is the only channel the parent can read without parsing prose, so "not one
    # view came out" has to reach it -- otherwise a headless machine renders nothing all run and
    # every task merely looks like it had no visible diff.
    wanted = [v.strip() for v in args.views.split(",") if v.strip()]
    written = 0
    for view in wanted:
        path = osp.join(args.output_dir, f"{args.prefix}_{view}.png")
        if render(target, path, view=view, width=args.width, height=args.height, zoom=args.zoom):
            print(f"wrote {path}")
            written += 1

    return 0 if written or not wanted else 3


if __name__ == "__main__":
    sys.exit(main())
