"""Executable, request-conditioned delta contracts for CAD Agent v4.

The agent is a poor judge of its own work: earlier iterations reported success on runs that had
changed nothing, and the harness could only answer "did the geometry move at all". A contract turns
the request into checkable arithmetic before any script runs -- which operation, on which grounded
handles, with which measurable delta, and what must stay untouched -- and every candidate is then
scored against it rather than against the agent's opinion.

Two properties matter more than the checks themselves:

* **No ground truth is consulted.** Every assertion compares two summaries of the agent's own
  geometry, so the contract works exactly as well at submission time as it does in evaluation.
* **The preflights run before execution.** A spec that names a workplane the part does not have, or
  a target handle from a stale revision, is refused while it still costs zero seconds of CadQuery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re


VALID_OPERATIONS = {
    "add", "remove", "modify", "transform", "copy", "pattern", "assembly", "composite",
    "edge_finish", "unknown",
}
VALID_COMPARATORS = {"eq", "ne", "gt", "ge", "lt", "le", "between", "nonzero"}
STRUCTURAL_OPERATIONS = {"add", "remove", "copy", "pattern", "edge_finish"}

_CREATE_RE = re.compile(
    r"\b(add|adds|added|adding|create|creates|created|creating|copy|copies|copied|copying|"
    r"duplicate|duplicates|duplicated|duplicating|insert|inserts|inserted|inserting)\b",
    re.IGNORECASE,
)
_RELOCATE_RE = re.compile(
    r"\b(move|moves|moved|moving|reposition|repositions|repositioned|repositioning|"
    r"relocate|relocates|relocated|relocating|translate|translates|translated|translating)\b",
    re.IGNORECASE,
)
_PROFILE_RE = re.compile(r"\b(profile|cross[- ]?section|sectional shape)\b", re.IGNORECASE)
_SUBSTITUTE_RE = re.compile(
    r"\b(change|convert|replace|turn|transform)\b.*\b(into|to|with)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?(?:\s*(?:mm|cm|m|in|inch|deg|°))?", re.IGNORECASE)
_NAMED_WORKPLANE_RE = re.compile(
    r"(?:cq\.)?Workplane\(\s*['\"](?:XY|XZ|YZ|front|back|left|right|top|bottom)['\"]",
    re.IGNORECASE,
)


def _dict(value):
    return dict(value) if isinstance(value, dict) else {}


def _list(value):
    return list(value) if isinstance(value, (list, tuple)) else []


def normalise_edit_spec(raw):
    """Return a bounded, serialisable task contract from a model-provided dict."""
    raw = _dict(raw)
    operation = str(raw.get("operation", "unknown")).strip().lower()
    if operation not in VALID_OPERATIONS:
        operation = "unknown"
    return {
        "operation": operation,
        "target_description": str(raw.get("target_description", ""))[:500],
        "target_region": _dict(raw.get("target_region")),
        "assertions": _list(raw.get("assertions"))[:20],
        "preservation": _dict(raw.get("preservation")),
        "completion_criteria": _list(raw.get("completion_criteria"))[:20],
    }


def normalise_expectation(raw):
    raw = _dict(raw)
    operation = str(raw.get("operation", "unknown")).strip().lower()
    if operation not in VALID_OPERATIONS:
        operation = "unknown"
    return {
        "operation": operation,
        "expected_changes": _list(raw.get("expected_changes"))[:12],
        "forbidden_changes": _list(raw.get("forbidden_changes"))[:12],
        "preservation": _dict(raw.get("preservation")),
    }

def expectation_schema_issue(raw):
    """Return a concise preflight error when prose was put into machine assertions."""
    raw = _dict(raw)
    for field_name in ("expected_changes", "forbidden_changes"):
        values = raw.get(field_name, [])
        if values in (None, []):
            continue
        if not isinstance(values, (list, tuple)):
            return f"StepExpectation.{field_name} must be a list of assertion objects."
        invalid = [index for index, value in enumerate(values) if not isinstance(value, dict)]
        if invalid:
            return (
                f"StepExpectation.{field_name} contains prose at indices {invalid}. "
                "Use assertion objects such as {metric: 'face_delta', operator: 'nonzero', "
                "required: true}; keep semantic prose in EditSpec.target_description."
            )
    return None


def infer_request_intent(request):
    """Extract only high-confidence lexical intent needed by deterministic safety gates."""
    text = str(request or "")
    explicitly_creates = bool(_CREATE_RE.search(text))
    explicitly_relocates = bool(_RELOCATE_RE.search(text))
    profile_substitution = bool(_PROFILE_RE.search(text) and _SUBSTITUTE_RE.search(text))
    return {
        "explicitly_creates": explicitly_creates,
        "explicitly_relocates": explicitly_relocates,
        "requires_structural_delta": explicitly_creates and not explicitly_relocates,
        "profile_substitution": profile_substitution,
        "has_explicit_dimension": bool(_NUMBER_RE.search(text)),
    }


def target_preflight_issue(intent, handles):
    """Require local topology handles for dimension-free profile substitutions."""
    handles = [str(handle) for handle in (handles or [])]
    if intent.get("profile_substitution") and not intent.get("has_explicit_dimension"):
        if not handles or all(":S" in handle for handle in handles):
            return (
                "Grounding is too broad for a local profile substitution. Select at least one "
                "feature face or edge handle (F/E), derive the existing mating envelope from its "
                "bounds or axial span, and preserve that envelope when changing the profile."
            )
    return None


def script_preflight_issue(script, intent):
    """Reject sign-ambiguous principal planes for dimension-free profile substitutions."""
    source = str(script or "")
    if (
        intent.get("profile_substitution")
        and not intent.get("has_explicit_dimension")
        and _NAMED_WORKPLANE_RE.search(source)
        and "cq.Plane(" not in source
    ):
        return (
            "Named principal workplanes have signed normals that can invert a coordinate-bounded "
            "profile edit. Construct an explicit cq.Plane(origin=..., xDir=..., normal=...) for "
            "each solid and extrude its independently grounded mating-feature span."
        )
    return None


def operation_preflight_issue(request, spec, expectation):
    """Reject an explicit create request that a step tries to satisfy by relocation alone.

    This is intentionally narrow. Words such as "create a hole" still allow a subtractive
    structural edit, while an explicit "move" in the same request permits a transform. The guard
    only blocks the measured high-cost ambiguity: moving an existing instance when the user asked
    to create, add, copy, duplicate, or insert one.
    """
    spec = normalise_edit_spec(spec)
    expectation = normalise_expectation(expectation)
    intent = infer_request_intent(request)
    step_operation = expectation["operation"]
    if intent["requires_structural_delta"] and step_operation == "transform":
        return (
            "Intent conflict: the request explicitly creates/adds geometry but this step declares "
            "a transform. Preserve existing instances and create the requested new feature; only "
            "relocate geometry when the request explicitly says move/reposition/relocate."
        )
    if spec["operation"] in {"add", "copy", "pattern"} and step_operation == "transform":
        return (
            f"Intent conflict: EditSpec operation '{spec['operation']}' cannot be completed by a "
            "transform-only step. Preserve the grounded source geometry and add the new result."
        )
    return None


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(metric, before, after):
    metric = str(metric or "").strip().lower()
    counts_before = before.get("counts") or {}
    counts_after = after.get("counts") or {}
    if metric == "volume_delta":
        return after.get("volume", 0.0) - before.get("volume", 0.0)
    if metric == "volume_ratio":
        return after.get("volume", 0.0) / max(abs(before.get("volume", 0.0)), 1e-9)
    if metric in ("solid_delta", "solids_delta"):
        return counts_after.get("solids", 0) - counts_before.get("solids", 0)
    if metric in ("face_delta", "faces_delta"):
        return counts_after.get("faces", 0) - counts_before.get("faces", 0)
    if metric in ("edge_delta", "edges_delta"):
        return counts_after.get("edges", 0) - counts_before.get("edges", 0)
    if metric.startswith("bbox_") and metric.endswith("_delta"):
        axis = {"x": 0, "y": 1, "z": 2}.get(metric[5:6])
        if axis is not None:
            return after.get("bbox", {}).get("size", [0, 0, 0])[axis] - before.get("bbox", {}).get("size", [0, 0, 0])[axis]
    if metric.startswith("center_") and metric.endswith("_delta"):
        axis = {"x": 0, "y": 1, "z": 2}.get(metric[7:8])
        if axis is not None:
            return after.get("center_of_mass", [0, 0, 0])[axis] - before.get("center_of_mass", [0, 0, 0])[axis]
    if metric.startswith("face_type_delta:"):
        kind = metric.split(":", 1)[1].upper()
        def face_type_count(summary):
            direct = summary.get("face_types")
            if isinstance(direct, dict):
                return direct.get(kind, 0)
            nested = (summary.get("counts") or {}).get("by_face_type") or {}
            return nested.get(kind, 0) if isinstance(nested, dict) else 0
        return face_type_count(after) - face_type_count(before)
    if metric.startswith("circle_radius_count_delta:"):
        radius = _number(metric.split(":", 1)[1])
        if radius is None:
            return None
        def count(summary):
            radii = summary.get("circular_edge_radii") or []
            if isinstance(radii, dict):
                entries = radii.items()
            elif isinstance(radii, list):
                entries = (
                    (entry.get("radius"), entry.get("count", 0))
                    for entry in radii
                    if isinstance(entry, dict)
                )
            else:
                entries = ()
            total = 0
            for key, amount in entries:
                numeric_radius = _number(key)
                numeric_amount = _number(amount)
                if numeric_radius is not None and abs(numeric_radius - radius) <= 0.05:
                    total += numeric_amount or 0
            return total
        return count(after) - count(before)
    return None


def _compare(actual, comparator, expected, tolerance=0.0):
    comparator = str(comparator or "eq").lower()
    if comparator not in VALID_COMPARATORS:
        return None
    if comparator == "nonzero":
        return abs(actual) > max(float(tolerance or 0.0), 1e-12)
    if comparator == "between":
        if not isinstance(expected, (list, tuple)) or len(expected) != 2:
            return None
        low, high = _number(expected[0]), _number(expected[1])
        return None if low is None or high is None else low - tolerance <= actual <= high + tolerance
    target = _number(expected)
    if target is None:
        return None
    tolerance = abs(float(tolerance or 0.0))
    if comparator == "eq":
        return abs(actual - target) <= tolerance
    if comparator == "ne":
        return abs(actual - target) > tolerance
    if comparator == "gt":
        return actual > target + tolerance
    if comparator == "ge":
        return actual >= target - tolerance
    if comparator == "lt":
        return actual < target - tolerance
    if comparator == "le":
        return actual <= target + tolerance
    return None


def evaluate_assertion(assertion, before, after, forbidden=False):
    assertion = _dict(assertion)
    metric = str(assertion.get("metric", ""))
    actual = _metric_value(metric, before, after)
    result = {
        "metric": metric,
        "operator": str(assertion.get("operator", "eq")),
        "expected": assertion.get("value"),
        "actual": None if actual is None else round(float(actual), 6),
        "required": bool(assertion.get("required", False)),
        "forbidden": bool(forbidden),
    }
    if actual is None:
        result.update(status="unknown", passed=False, note="Metric is not supported by the deterministic evaluator.")
        return result
    passed = _compare(
        actual,
        assertion.get("operator", "eq"),
        assertion.get("value"),
        _number(assertion.get("tolerance")) or 0.0,
    )
    if passed is None:
        result.update(status="unknown", passed=False, note="Comparator or expected value is invalid.")
        return result
    if forbidden:
        passed = not passed
    result.update(status="passed" if passed else "failed", passed=bool(passed))
    return result


def _translation_ratio(before, after):
    before_center = before.get("center_of_mass", [0.0, 0.0, 0.0])
    after_center = after.get("center_of_mass", [0.0, 0.0, 0.0])
    shift = math.sqrt(sum((a - b) ** 2 for a, b in zip(after_center, before_center)))
    diagonal = max(math.sqrt(sum(value ** 2 for value in before.get("bbox", {}).get("size", [0, 0, 0]))), 1e-9)
    return shift / diagonal


@dataclass
class DeltaCard:
    revision_before: str
    revision_after: str
    valid: bool
    hard_pass: bool
    contract_score: float
    assertions: list[dict] = field(default_factory=list)
    preservation: dict = field(default_factory=dict)
    target_change: dict = field(default_factory=dict)
    observed_delta: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    recommendation: str = ""

    def as_dict(self):
        return asdict(self)


def _solid_volume_changes(before, after):
    """Per-solid magnitude evidence when stable one-to-one pairing is available."""
    before_solids = before.get("solids") or []
    after_solids = after.get("solids") or []
    if not before_solids or len(before_solids) != len(after_solids):
        return []
    changes = []
    for left, right in zip(before_solids, after_solids):
        before_volume = float(left.get("volume", 0.0) or 0.0)
        after_volume = float(right.get("volume", 0.0) or 0.0)
        delta = after_volume - before_volume
        changes.append({
            "before_solid": left.get("index"),
            "after_solid": right.get("index"),
            "before_volume": round(before_volume, 4),
            "after_volume": round(after_volume, 4),
            "delta": round(delta, 4),
            "relative_delta": round(delta / max(abs(before_volume), 1e-9), 4),
        })
    return changes

def evaluate_contract(spec, expectation, before, after, commit_report,
                      target_change=None, revision_before="origin", revision_after="candidate",
                      require_structural_delta=False, request_intent=None):
    """Score one candidate against the task contract and this step's expectation.

    Two contracts are checked at once. `spec` is the task-level one, fixed when the request was
    grounded; `expectation` is what the agent predicted *this* script would do. Both must hold. The
    split is what makes a wrong prediction visible: a script can satisfy the task contract by
    accident while doing something else entirely, and the step expectation catches that.

    `before`/`after` are `summarise_shape` outputs, so every number here is deterministic and
    reproducible from the two STEP files alone.
    """
    spec = normalise_edit_spec(spec)
    expectation = normalise_expectation(expectation)
    target_change = _dict(target_change)
    request_intent = _dict(request_intent)
    assertions = []
    for assertion in spec["assertions"] + spec["completion_criteria"] + expectation["expected_changes"]:
        assertions.append(evaluate_assertion(assertion, before, after))
    for assertion in expectation["forbidden_changes"]:
        assertions.append(evaluate_assertion(assertion, before, after, forbidden=True))

    preservation_spec = dict(spec["preservation"])
    preservation_spec.update(expectation["preservation"])
    preserved_ratio = float(commit_report.get("preserved_ratio", 1.0) or 0.0)
    solids_lost = max(0, before.get("counts", {}).get("solids", 0) - after.get("counts", {}).get("solids", 0))
    preservation = {
        "preserved_ratio": round(preserved_ratio, 4),
        "solids_lost": solids_lost,
        "translation_ratio": round(_translation_ratio(before, after), 6),
    }
    observed = {
        "volume": round(after.get("volume", 0.0) - before.get("volume", 0.0), 4),
        "faces": after.get("counts", {}).get("faces", 0) - before.get("counts", {}).get("faces", 0),
        "edges": after.get("counts", {}).get("edges", 0) - before.get("counts", {}).get("edges", 0),
        "solids": after.get("counts", {}).get("solids", 0) - before.get("counts", {}).get("solids", 0),
        "bbox_size": [
            round(a - b, 3)
            for a, b in zip(after.get("bbox", {}).get("size", [0, 0, 0]), before.get("bbox", {}).get("size", [0, 0, 0]))
        ],
        "solid_volume_changes": _solid_volume_changes(before, after),
    }

    failures = []
    evaluated = [row for row in assertions if row["status"] != "unknown"]
    passed = [row for row in evaluated if row["passed"]]
    for row in evaluated:
        if row["required"] and not row["passed"]:
            failures.append(f"required assertion failed: {row['metric']}")

    effective_operation = spec["operation"]
    if effective_operation == "unknown":
        effective_operation = expectation["operation"]
    volume_tolerance = max(abs(float(before.get("volume", 0.0))) * 1e-7, 1e-6)
    has_structural_delta = bool(
        observed["faces"]
        or observed["edges"]
        or observed["solids"]
        or abs(float(observed["volume"])) > volume_tolerance
    )
    if (require_structural_delta or effective_operation in STRUCTURAL_OPERATIONS) and not has_structural_delta:
        failures.append(
            f"operation '{effective_operation}' produced no structural delta; the result looks "
            "like a relocation or no-op rather than the requested feature edit"
        )

    if (
        request_intent.get("profile_substitution")
        and not request_intent.get("has_explicit_dimension")
        and observed["solid_volume_changes"]
    ):
        largest = max(
            observed["solid_volume_changes"],
            key=lambda row: abs(row["relative_delta"]),
        )
        if abs(largest["relative_delta"]) > 0.65:
            failures.append(
                "dimension-free profile substitution changed solid "
                f"S{largest['before_solid']} volume by {largest['relative_delta']:+.1%}; "
                "preserve the existing mating envelope and derive the replacement profile size "
                "from grounded feature bounds instead of the whole solid"
            )

    minimum = _number(preservation_spec.get("min_preserved_ratio"))
    if minimum is not None:
        ok = preserved_ratio >= minimum
        preservation["min_preserved_ratio_passed"] = ok
        if bool(preservation_spec.get("required", False)) and not ok:
            failures.append(f"preserved ratio {preserved_ratio:.1%} is below {minimum:.1%}")
    if preservation_spec.get("keep_solid_count") is True and solids_lost:
        preservation["keep_solid_count_passed"] = False
        failures.append(f"lost {solids_lost} solid(s) although solid count is protected")
    elif preservation_spec.get("keep_solid_count") is True:
        preservation["keep_solid_count_passed"] = True
    if preservation_spec.get("allow_global_transform") is False:
        threshold = _number(preservation_spec.get("max_translation_ratio")) or 0.01
        ok = preservation["translation_ratio"] <= threshold
        preservation["global_position_passed"] = ok
        if not ok:
            failures.append("global position changed outside the allowed tolerance")

    if target_change.get("targets_total"):
        target_fraction = len(target_change.get("targets_changed", ())) / target_change["targets_total"]
    else:
        target_fraction = None
    target_change["changed_fraction"] = None if target_fraction is None else round(target_fraction, 4)
    reference_operation = effective_operation in {"add", "copy", "pattern"}
    if expectation.get("expected_changes") and target_fraction == 0 and not reference_operation:
        target_change["warning"] = "None of the grounded target handles changed."
    elif reference_operation and target_fraction is not None:
        target_change["role"] = "source_or_anchor"
        target_change["reference_preserved_fraction"] = round(1.0 - target_fraction, 4)

    base_score = len(passed) / len(evaluated) if evaluated else 0.5
    if target_fraction is None:
        target_bonus = 0.0
    else:
        target_bonus = 0.1 if reference_operation else 0.2 * target_fraction
    preservation_bonus = 0.2 * min(max(preserved_ratio, 0.0), 1.0)
    contract_score = max(0.0, min(1.0, 0.6 * base_score + target_bonus + preservation_bonus))
    if failures:
        recommendation = "Do not finish: " + "; ".join(failures)
    elif target_change.get("warning"):
        recommendation = "The edit is valid, but the grounded target appears unchanged; inspect the selector."
    elif evaluated and len(passed) < len(evaluated):
        recommendation = "Some expected delta assertions failed; correct only those mismatches."
    else:
        recommendation = "The deterministic delta contract is satisfied; finish if all request items are complete."

    return DeltaCard(
        revision_before=revision_before,
        revision_after=revision_after,
        valid=bool(commit_report.get("is_valid", after.get("is_valid", False))),
        hard_pass=not failures,
        contract_score=round(contract_score, 4),
        assertions=assertions,
        preservation=preservation,
        target_change=target_change,
        observed_delta=observed,
        failures=failures,
        recommendation=recommendation,
    )


def choose_contract_commit(commits, cards):
    """Select the strongest live commit using contracts before later-attempt tie breaking."""
    eligible = [commit for commit in commits if not commit.abandoned]
    if not eligible:
        return None

    def rank(commit):
        card = cards.get(commit.n)
        return (
            1 if card and card.hard_pass else 0,
            card.contract_score if card else 0.0,
            1 if commit.is_valid else 0,
            1 if not commit.damage.get("damaged") else 0,
            commit.preserved_ratio,
            commit.n,
        )

    return max(eligible, key=rank)


def choose_delivery_commit(commits, cards, head=None, finish_requested=False):
    """Prefer a verified HEAD when the agent explicitly declares the request complete."""
    if finish_requested and head is not None and not head.abandoned:
        card = cards.get(head.n)
        if (
            card
            and card.hard_pass
            and not head.damage.get("damaged")
            and not getattr(head, "validity_regressed", False)
        ):
            return head

    return choose_contract_commit(commits, cards)
