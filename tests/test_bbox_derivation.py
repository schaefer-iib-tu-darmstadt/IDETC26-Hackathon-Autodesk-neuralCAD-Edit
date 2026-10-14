"""Validates the fast bounding-box path against the exact one over the real dataset.

`summarise_shape(derive_bbox_from_solids=True)` builds the shape's box from the per-solid boxes it
already computes, instead of calling `BRepBndLib.AddOptimal_s` a second time on the whole compound.
Measured on the largest part in the dataset (58 solids, 557 faces) that is 13.3 s of a 28 s call.

The identity is mathematically exact -- the AABB of a union is the union of the AABBs -- but it is
also load-bearing: `_differs`, `assess_damage` and `preserved_ratio` all read `bbox`, so a
divergence would move the gates rather than merely the numbers. Hence a check against every start
model in the benchmark, not against one part.

Slow by construction (it runs the expensive path once per part): ~6 minutes. Excluded by
default; run with `-m slow`.
"""

import os.path as osp

import pytest
from conftest import DATASET_ROOT

DATA = DATASET_ROOT


def _start_steps():
    import pandas as pd

    parquet = osp.join(DATA, "parquets", "val_edit_text.parquet")
    if not osp.exists(parquet):
        return []
    df = pd.read_parquet(parquet)
    out = []
    for entry in df["brep_start_path"]:
        # One row lists every export of the same B-rep (.f3d, .smt, .step, .stl); only the STEP is
        # what the harness ever hands to CadQuery.
        for path in (str(p) for arr in entry for p in arr):
            if path.lower().endswith(".step"):
                full = osp.join(DATA, path)
                if osp.exists(full):
                    out.append(full)
    return sorted(set(out))


@pytest.mark.slow
@pytest.mark.dataset
def test_derived_bbox_matches_the_exact_one_on_every_start_model():
    from src.agents.cad_tools import load_shape, summarise_shape

    steps = _start_steps()
    if not steps:
        pytest.skip(f"{DATA} not present")

    multi_solid = 0
    for path in steps:
        shape = load_shape(path)
        exact = summarise_shape(shape, max_faces=400)
        derived = summarise_shape(shape, max_faces=400, derive_bbox_from_solids=True)
        assert derived["bbox"] == exact["bbox"], f"bbox diverged on {osp.basename(path)}"
        assert derived == exact, f"summary diverged on {osp.basename(path)}"
        if exact["counts"]["solids"] > 1:
            multi_solid += 1

    # The fast path only engages on multi-solid shapes, so a run where none qualified would pass
    # while testing nothing.
    assert multi_solid > 0, "no multi-solid start model was exercised"
    print(f"\n{len(steps)} start models checked, {multi_solid} of them multi-solid")
