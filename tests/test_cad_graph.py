"""Tests for version-scoped B-Rep grounding handles."""

from src.agents.cad_graph import BRepGraph


def test_graph_exposes_local_adjacency_and_selector_recipes(steps):
    graph = BRepGraph(steps["origin"], revision="c0")
    overview = graph.overview()
    assert overview["counts"]["solids"] == 4
    assert overview["counts"]["faces"] > 0
    assert overview["counts"]["edges"] > 0

    result = graph.query(entity="faces", limit=3)
    assert result["revision"] == "c0"
    assert len(result["rows"]) == 3
    row = result["rows"][0]
    assert row["handle"].startswith("c0:F")
    assert row["selector"]["revision"] == "c0"
    assert row["edges"]

    local = graph.query(entity="edges", seed_handles=row["handle"], depth=1, limit=24)
    assert local["matched"] >= 1
    assert all(edge["handle"].startswith("c0:E") for edge in local["rows"])


def test_a_bad_query_is_answered_not_raised(steps):
    """The tool result is the only channel the model has; an exception would end the turn with
    nothing to read."""
    graph = BRepGraph(steps["origin"], revision="c0")
    unknown = graph.query(entity="holes")
    assert "error" in unknown and "faces, edges or solids" in unknown["error"]
    bad_near = graph.query(entity="faces", near="the top")
    assert bad_near["error"].startswith("Could not read near")


def test_a_malformed_handle_is_reported_not_reinterpreted(steps):
    """The revision prefix is checked first, so a handle without it reads as stale; a well-formed
    prefix with a bad entity letter or an out-of-range index is invalid. Neither raises."""
    graph = BRepGraph(steps["origin"], revision="c0")
    assert graph.resolve("nonsense")["error"] == "stale_handle"
    assert graph.resolve("c0:X99")["error"] == "invalid_handle"
    assert graph.resolve("c0:F999")["error"] == "invalid_handle"


def test_handles_expire_when_head_revision_changes(steps):
    before = BRepGraph(steps["origin"], revision="c0")
    after = BRepGraph(steps["edited"], revision="c1")
    handle = before.face_records[0]["handle"]
    assert after.resolve(handle)["error"] == "stale_handle"


def test_unchanged_snapshot_preserves_grounded_handle(steps):
    before = BRepGraph(steps["origin"], revision="origin")
    after = BRepGraph(steps["origin"], revision="same")
    handle = before.face_records[0]["handle"]
    report = before.target_change_report([handle], after)
    assert report["targets_total"] == 1
    assert report["targets_unchanged"] == [handle]


def test_solid_grounding_includes_grouped_axial_feature_spans(steps):
    graph = BRepGraph(steps["origin"], revision="origin")
    result = graph.query(entity="solids", limit=4)

    assert result["matched"] == 4
    assert isinstance(result["mating_feature_pairs"], list)
    assert "compare each member span" in result["mating_pair_note"]
    spans = result["rows"][0]["axial_feature_spans"]
    assert set(spans) == {"x", "y", "z"}
    planar = [group for groups in spans.values() for group in groups if group["type"] == "PLANE"]
    assert planar
    assert all(group["count"] >= 1 for group in planar)
    assert all(len(group["handles"]) <= 3 for group in planar)
