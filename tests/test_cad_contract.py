"""Tests for request-conditioned geometric delta contracts."""

from types import SimpleNamespace

from src.agents.cad_contract import (
    _metric_value,
    choose_delivery_commit,
    evaluate_contract,
    expectation_schema_issue,
    infer_request_intent,
    normalise_edit_spec,
    operation_preflight_issue,
    script_preflight_issue,
    target_preflight_issue,
)
from src.agents.cad_git import CadRepo


def _summary(path):
    return CadRepo._summarise(path)


def test_contract_metrics_read_actual_summary_schema():
    before = {
        "counts": {"by_face_type": {"CYLINDER": 2}},
        "circular_edge_radii": [{"radius": 0.75, "count": 4}],
    }
    after = {
        "counts": {"by_face_type": {"CYLINDER": 3}},
        "circular_edge_radii": [
            {"radius": 0.75, "count": 2},
            {"radius": 0.85, "count": 4},
        ],
    }

    assert _metric_value("face_type_delta:CYLINDER", before, after) == 1
    assert _metric_value("circle_radius_count_delta:0.75", before, after) == -2
    assert _metric_value("circle_radius_count_delta:0.85", before, after) == 4


def test_radius_contract_keeps_legacy_dictionary_compatibility():
    before = {"circular_edge_radii": {"0.85": 1}}
    after = {"circular_edge_radii": {"0.85": 3}}
    assert _metric_value("circle_radius_count_delta:0.85", before, after) == 2


def test_contract_accepts_expected_feature_delta(steps):
    before = _summary(steps["origin"])
    after = _summary(steps["edited"])
    spec = {
        "operation": "add",
        "target_description": "add one boss",
        "assertions": [
            {"metric": "volume_delta", "operator": "gt", "value": 0, "required": True},
            {"metric": "face_delta", "operator": "nonzero", "required": True},
        ],
        "preservation": {"keep_solid_count": True, "required": True},
    }
    card = evaluate_contract(
        spec,
        {},
        before,
        after,
        {"is_valid": True, "preserved_ratio": 0.95},
    )
    assert card.hard_pass is True
    assert card.contract_score > 0.7
    assert all(row["passed"] for row in card.assertions)


def test_contract_rejects_noop_for_required_change(steps):
    before = _summary(steps["origin"])
    spec = {
        "operation": "add",
        "assertions": [
            {"metric": "volume_delta", "operator": "gt", "value": 0, "required": True},
        ],
    }
    card = evaluate_contract(
        spec,
        {},
        before,
        before,
        {"is_valid": True, "preserved_ratio": 1.0},
    )
    assert card.hard_pass is False
    assert "volume_delta" in card.failures[0]


def test_contract_catches_forbidden_solid_loss(steps):
    before = _summary(steps["origin"])
    after = _summary(steps["merged"])
    card = evaluate_contract(
        {"operation": "modify", "preservation": {"keep_solid_count": True, "required": True}},
        {},
        before,
        after,
        {"is_valid": True, "preserved_ratio": 0.8},
    )
    assert card.hard_pass is False
    assert card.preservation["solids_lost"] == 1


def test_unknown_operations_are_bounded():
    spec = normalise_edit_spec({"operation": "invent-a-widget", "assertions": list(range(100))})
    assert spec["operation"] == "unknown"
    assert len(spec["assertions"]) == 20


def test_add_contract_rejects_transform_only_candidate(steps):
    before = _summary(steps["origin"])
    moved = _summary(steps["moved"])
    card = evaluate_contract(
        {"operation": "add"},
        {"operation": "add"},
        before,
        moved,
        {"is_valid": True, "preserved_ratio": 0.0},
    )
    assert card.hard_pass is False
    assert any("no structural delta" in failure for failure in card.failures)


def test_transform_contract_allows_topology_preserving_move(steps):
    before = _summary(steps["origin"])
    moved = _summary(steps["moved"])
    card = evaluate_contract(
        {"operation": "transform"},
        {"operation": "transform"},
        before,
        moved,
        {"is_valid": True, "preserved_ratio": 0.0},
    )
    assert card.hard_pass is True


def test_creation_request_preflight_refuses_relocation():
    issue = operation_preflight_issue(
        "Create an outlet port on the top right.",
        {"operation": "add"},
        {"operation": "transform"},
    )
    assert issue is not None
    assert "Preserve existing instances" in issue


def test_request_intent_overrides_underclassified_edit_spec(steps):
    intent = infer_request_intent("Please create two new ports.")
    assert intent["requires_structural_delta"] is True
    before = _summary(steps["origin"])
    moved = _summary(steps["moved"])
    card = evaluate_contract(
        {"operation": "modify"},
        {"operation": "modify"},
        before,
        moved,
        {"is_valid": True, "preserved_ratio": 0.0},
        require_structural_delta=intent["requires_structural_delta"],
    )
    assert card.hard_pass is False
    assert any("no structural delta" in failure for failure in card.failures)


def test_creation_request_can_include_explicit_move():
    issue = operation_preflight_issue(
        "Create a copy and move the existing port to the left.",
        {"operation": "composite"},
        {"operation": "transform"},
    )
    assert issue is None


def test_explicit_finish_prefers_verified_head_over_preservation_score():
    c1 = SimpleNamespace(
        n=1, abandoned=False, damage={}, validity_regressed=False,
        is_valid=True, preserved_ratio=0.99,
    )
    c2 = SimpleNamespace(
        n=2, abandoned=False, damage={}, validity_regressed=False,
        is_valid=True, preserved_ratio=0.80,
    )
    cards = {
        1: SimpleNamespace(hard_pass=True, contract_score=0.90),
        2: SimpleNamespace(hard_pass=True, contract_score=0.70),
    }
    chosen = choose_delivery_commit(
        [c1, c2], cards, head=c2, finish_requested=True,
    )
    assert chosen is c2

    fallback = choose_delivery_commit(
        [c1, c2], cards, head=c2, finish_requested=False,
    )
    assert fallback is c1


def test_expectation_schema_rejects_prose_assertions():
    issue = expectation_schema_issue({"expected_changes": ["replace the flower profile"]})
    assert issue is not None
    assert "assertion objects" in issue


def test_profile_substitution_requires_local_topology_handles():
    intent = infer_request_intent("Change the center flower profile into a hexagonal profile.")
    assert intent["profile_substitution"] is True
    assert target_preflight_issue(intent, ["origin:S0", "origin:S1"]) is not None
    assert target_preflight_issue(intent, ["origin:F12"]) is None


def test_profile_substitution_requires_explicit_plane_frame():
    intent = infer_request_intent("Change the flower profile into a hexagonal profile.")
    ambiguous = 'result = cq.Workplane("XZ").polygon(6, 10).extrude(4)'
    explicit = "plane = cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, 1, 0))"
    assert script_preflight_issue(ambiguous, intent) is not None
    assert script_preflight_issue(explicit, intent) is None


def _profile_summary(small_volume):
    return {
        "volume": small_volume + 900.0,
        "center_of_mass": [0.0, 0.0, 0.0],
        "bbox": {"size": [100.0, 20.0, 100.0]},
        "counts": {"solids": 2, "faces": 20, "edges": 40},
        "solids": [
            {"index": 0, "volume": small_volume},
            {"index": 1, "volume": 900.0},
        ],
    }


def test_dimension_free_profile_swap_rejects_extreme_per_solid_change():
    before = _profile_summary(100.0)
    after = _profile_summary(180.0)
    after["counts"] = {"solids": 2, "faces": 26, "edges": 52}
    intent = infer_request_intent("Change the flower profile into a hexagonal profile.")
    card = evaluate_contract(
        {"operation": "modify"},
        {"operation": "modify"},
        before,
        after,
        {"is_valid": True, "preserved_ratio": 0.9},
        request_intent=intent,
    )
    assert card.hard_pass is False
    assert card.observed_delta["solid_volume_changes"][0]["relative_delta"] == 0.8
    assert any("profile substitution" in failure for failure in card.failures)


def test_explicit_profile_dimension_disables_magnitude_prior():
    before = _profile_summary(100.0)
    after = _profile_summary(180.0)
    after["counts"] = {"solids": 2, "faces": 26, "edges": 52}
    intent = infer_request_intent("Change the profile into a 30 mm hexagonal profile.")
    card = evaluate_contract(
        {"operation": "modify"},
        {"operation": "modify"},
        before,
        after,
        {"is_valid": True, "preserved_ratio": 0.9},
        request_intent=intent,
    )
    assert card.hard_pass is True
