"""Version-scoped B-Rep graph and compact grounding queries for CAD Agent v4.

Half the requests name a feature by description -- "the four long edges of the blade", "the fillet
at the front center" -- and nothing in the pipeline maps such a phrase to a face. The agent has to
find it, and it used to do so by writing throwaway CadQuery scripts that printed face lists. This
module answers those questions directly and hands back a *handle* the agent can select against.

A handle looks like `origin:F12` or `c3:F12`: a revision plus an entity index. The revision prefix
is the load-bearing part. STEP export provides no persistent topological naming, so `F12` means
nothing beyond "the thirteenth face OCC enumerated in this particular file" -- after any edit the
same index can denote a different face. Rather than pretend otherwise, `resolve` refuses any handle
whose revision is not the current one and returns `stale_handle`. Selecting the wrong face then
fails loudly at lookup time instead of silently chamfering the wrong edge, and the recovery is one
cheap re-ground rather than a wasted edit turn.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math

from src.agents.cad_tools import (
    QUERY_RADIUS_TOL,
    _diagonal_from_size,
    _edge_record,
    _face_record,
    _parse_near,
    load_shape,
)


def _topology_key(entity):
    # Bucket key only. `hash` over an OCC TShape may collide, so every hit is confirmed with
    # `IsSame` in `_lookup_index`; the hash alone must never decide identity.
    return hash(entity.wrapped)


def _same(left, right):
    try:
        return bool(left.wrapped.IsSame(right.wrapped))
    except Exception:
        return False


def _lookup_index(entity, buckets, entities):
    for index in buckets.get(_topology_key(entity), []):
        if _same(entity, entities[index]):
            return index
    return None


def _distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _normalise_handles(handles):
    if not handles:
        return []
    if isinstance(handles, str):
        return [item.strip() for item in handles.split(",") if item.strip()]
    return [str(item).strip() for item in handles if str(item).strip()]


class BRepGraph:
    """A serialisable local view over one STEP snapshot."""

    def __init__(self, step_path, revision="origin"):
        self.step_path = step_path
        self.revision = str(revision or "origin")
        self.shape = load_shape(step_path)
        self.solids = list(self.shape.Solids())
        self.faces = list(self.shape.Faces())
        self.edges = list(self.shape.Edges())

        self._face_buckets = defaultdict(list)
        self._edge_buckets = defaultdict(list)
        for index, face in enumerate(self.faces):
            self._face_buckets[_topology_key(face)].append(index)
        for index, edge in enumerate(self.edges):
            self._edge_buckets[_topology_key(edge)].append(index)

        face_owners = self._owners("faces")
        edge_owners = self._owners("edges")
        self.face_records = []
        self.edge_records = []
        self.solid_records = []
        self.face_to_edges = defaultdict(set)
        self.edge_to_faces = defaultdict(set)

        for index, solid in enumerate(self.solids):
            self.solid_records.append({
                "id": f"S{index}",
                "handle": self.handle("S", index),
                "solid": index,
                "volume": round(float(solid.Volume()), 4),
                "center": [round(value, 3) for value in solid.Center().toTuple()],
                "faces": len(solid.Faces()),
                "is_valid": bool(solid.isValid()),
                "selector": {"entity": "solids", "index": index, "revision": self.revision},
            })

        for index, face in enumerate(self.faces):
            record = _face_record(index, face, solid=face_owners.get(index))
            try:
                bounds = face.BoundingBox()
                record["bounds"] = {
                    "min": [round(bounds.xmin, 3), round(bounds.ymin, 3), round(bounds.zmin, 3)],
                    "max": [round(bounds.xmax, 3), round(bounds.ymax, 3), round(bounds.zmax, 3)],
                }
            except Exception:
                pass
            record["handle"] = self.handle("F", index)
            record["selector"] = self._selector("faces", index, record)
            self.face_records.append(record)
            for edge in face.Edges():
                edge_index = _lookup_index(edge, self._edge_buckets, self.edges)
                if edge_index is None:
                    continue
                self.face_to_edges[index].add(edge_index)
                self.edge_to_faces[edge_index].add(index)

        for index, edge in enumerate(self.edges):
            record = _edge_record(index, edge, solid=edge_owners.get(index))
            record["handle"] = self.handle("E", index)
            record["selector"] = self._selector("edges", index, record)
            try:
                record["endpoints"] = [
                    [round(value, 3) for value in vertex.Center().toTuple()]
                    for vertex in edge.Vertices()
                ]
            except Exception:
                record["endpoints"] = []
            record["adjacent_faces"] = [
                self.handle("F", face_index)
                for face_index in sorted(self.edge_to_faces.get(index, ()))
            ]
            self.edge_records.append(record)

        for face_index, record in enumerate(self.face_records):
            record["edges"] = [
                self.handle("E", edge_index)
                for edge_index in sorted(self.face_to_edges.get(face_index, ()))
            ]
            neighbours = set()
            for edge_index in self.face_to_edges.get(face_index, ()):
                neighbours.update(self.edge_to_faces.get(edge_index, ()))
            neighbours.discard(face_index)
            record["adjacent_faces"] = [
                self.handle("F", neighbour) for neighbour in sorted(neighbours)
            ]

    def _owners(self, entity):
        entities = self.faces if entity == "faces" else self.edges
        buckets = self._face_buckets if entity == "faces" else self._edge_buckets
        owners = {}
        for solid_index, solid in enumerate(self.solids):
            members = solid.Faces() if entity == "faces" else solid.Edges()
            for member in members:
                index = _lookup_index(member, buckets, entities)
                if index is not None:
                    owners.setdefault(index, solid_index)
        return owners

    def _selector(self, entity, index, record):
        selector = {
            "entity": entity,
            "index": index,
            "revision": self.revision,
            "type": record.get("type"),
            "center": record.get("center"),
        }
        for key in ("area", "length", "radius", "axis", "normal", "solid"):
            if key in record:
                selector[key] = record[key]
        return selector

    def handle(self, prefix, index):
        """Revision-qualified id for one entity, e.g. `c3:F12`. See the module docstring."""
        return f"{self.revision}:{prefix}{index}"

    def records(self, entity):
        asked = (entity or "faces").strip().lower()
        asked = asked if asked.endswith("s") else asked + "s"
        if asked == "faces":
            return self.face_records
        if asked == "edges":
            return self.edge_records
        if asked == "solids":
            return self.solid_records
        raise ValueError(f"Unknown entity {entity!r}; expected faces, edges or solids.")

    def _axial_feature_spans(self, solid_index):
        """Group axis-aligned analytic faces into compact per-solid feature-depth evidence."""
        groups = {}
        for record in self.face_records:
            if record.get("solid") != solid_index:
                continue
            direction = record.get("normal") or record.get("axis")
            bounds = record.get("bounds")
            if not direction or not bounds:
                continue
            axis_index = max(range(3), key=lambda index: abs(direction[index]))
            if abs(direction[axis_index]) < 0.98:
                continue
            axis = "xyz"[axis_index]
            low = round(bounds["min"][axis_index], 3)
            high = round(bounds["max"][axis_index], 3)
            radius = record.get("radius")
            major_radius = record.get("major_radius")
            key = (
                axis,
                record.get("type"),
                low,
                high,
                None if radius is None else round(radius, 3),
                None if major_radius is None else round(major_radius, 3),
            )
            if key not in groups:
                groups[key] = {
                    "type": record.get("type"),
                    "span": [low, high],
                    "count": 0,
                    "area": 0.0,
                    "handles": [],
                }
                if radius is not None:
                    groups[key]["radius"] = round(radius, 3)
                if major_radius is not None:
                    groups[key]["major_radius"] = round(major_radius, 3)
            group = groups[key]
            group["count"] += 1
            group["area"] += float(record.get("area", 0.0))
            if len(group["handles"]) < 3:
                group["handles"].append(record["handle"])

        result = defaultdict(list)
        for key, group in groups.items():
            group["area"] = round(group["area"], 3)
            result[key[0]].append(group)
        for axis in result:
            result[axis].sort(
                key=lambda group: (
                    group["span"][0],
                    group["span"][1],
                    group["type"],
                    group.get("radius", -1.0),
                )
            )
            result[axis] = result[axis][:18]
        return dict(result)

    def mating_feature_pairs(self):
        """Repeated analytic feature spans that two or more solids share -- likely mating features.

        Half the dataset's requests are assemblies, where the interesting entity is usually not one
        face but the interface between components: a bolt circle, a pin and its bore, a shaft
        through two bearing seats. Equal-radius cylindrical spans on a common axis appearing in
        more than one solid identify exactly that, without any assembly metadata, which STEP import
        does not preserve. Capped at 12 so the answer stays readable in a prompt.
        """
        signatures = defaultdict(list)
        for solid_index in range(len(self.solids)):
            for axis, groups in self._axial_feature_spans(solid_index).items():
                for group in groups:
                    radius = group.get("radius")
                    if radius is None or group.get("count", 0) < 2:
                        continue
                    signature = (
                        axis,
                        group.get("type"),
                        round(float(radius), 2),
                        int(group.get("count", 0)),
                    )
                    signatures[signature].append({
                        "solid": solid_index,
                        "span": group["span"],
                        "handles": group.get("handles", []),
                    })

        pairs = []
        for (axis, kind, radius, count), members in signatures.items():
            if len({member["solid"] for member in members}) < 2:
                continue
            pairs.append({
                "axis": axis,
                "type": kind,
                "radius": radius,
                "count_per_solid": count,
                "members": members,
            })
        pairs.sort(key=lambda pair: (-pair["count_per_solid"], pair["axis"], pair["radius"]))
        return pairs[:12]

    def resolve(self, handle):
        """Look a handle up in this snapshot, or say precisely why it does not apply.

        A handle minted against an earlier revision is rejected rather than reinterpreted: the
        index would still be in range and would still name *a* face, just not the one the agent
        meant. `stale_handle` costs one re-ground; a silent mismatch costs an edit turn and can
        ship geometry nobody asked for.
        """
        handle = str(handle or "")
        prefix = f"{self.revision}:"
        if not handle.startswith(prefix):
            return {
                "error": "stale_handle",
                "handle": handle,
                "expected_revision": self.revision,
                "note": "Ground the current HEAD again before selecting an entity.",
            }
        local = handle[len(prefix):]
        if len(local) < 2 or local[0] not in "FES" or not local[1:].isdigit():
            return {"error": "invalid_handle", "handle": handle}
        entity = {"F": "faces", "E": "edges", "S": "solids"}[local[0]]
        index = int(local[1:])
        rows = self.records(entity)
        if index >= len(rows):
            return {"error": "invalid_handle", "handle": handle}
        return rows[index]

    def _neighbourhood(self, seed_handles, depth):
        wanted = set()
        queue = deque()
        for handle in _normalise_handles(seed_handles):
            record = self.resolve(handle)
            if "error" in record:
                continue
            queue.append((handle, 0))
            wanted.add(handle)

        while queue:
            handle, level = queue.popleft()
            if level >= depth:
                continue
            record = self.resolve(handle)
            linked = list(record.get("adjacent_faces", ())) + list(record.get("edges", ()))
            for neighbour in linked:
                if neighbour not in wanted:
                    wanted.add(neighbour)
                    queue.append((neighbour, level + 1))
        return wanted

    def query(self, entity="faces", kind="", solid=-1, near="", radius=-1.0,
              min_measure=0.0, limit=12, seed_handles=None, depth=1):
        """Return a compact, filterable local graph view."""
        try:
            rows = list(self.records(entity))
        except ValueError as exc:
            return {"error": str(exc)}
        kind = str(kind or "").strip().upper()
        spot = _parse_near(near)
        if near and spot is None:
            return {"error": f"Could not read near={near!r}; expected x,y,z or x,y,z,radius."}
        reach = None
        if spot is not None:
            reach = spot[1] if spot[1] is not None else 0.1 * _diagonal_from_size(self.shape)

        asks_for_solids = str(entity or "").strip().lower().rstrip("s") == "solid"
        has_specific_solid = solid is not None and int(solid) >= 0
        include_spans = asks_for_solids and (len(self.solids) <= 4 or has_specific_solid)

        neighbourhood = self._neighbourhood(seed_handles, max(0, min(int(depth), 3)))
        filtered = []
        for row in rows:
            if neighbourhood and row["handle"] not in neighbourhood:
                continue
            if solid is not None and int(solid) >= 0 and row.get("solid") != int(solid):
                continue
            if kind and row.get("type") != kind:
                continue
            measure = row.get("area", row.get("length", row.get("volume", 0.0)))
            if min_measure and measure < float(min_measure):
                continue
            if radius is not None and float(radius) >= 0:
                if row.get("radius") is None or abs(row["radius"] - float(radius)) > QUERY_RADIUS_TOL:
                    continue
            candidate = dict(row)
            if include_spans:
                candidate["axial_feature_spans"] = self._axial_feature_spans(row["solid"])
            if spot is not None:
                distance = _distance(candidate.get("center", (0, 0, 0)), spot[0])
                if distance > reach:
                    continue
                candidate["distance"] = round(distance, 3)
            filtered.append(candidate)

        filtered.sort(
            key=lambda row: row.get("distance", -row.get("area", -row.get("length", -row.get("volume", 0.0))))
        )
        cap = max(1, min(int(limit or 12), 24))
        result = {
            "revision": self.revision,
            "entity": (entity if str(entity).endswith("s") else f"{entity}s"),
            "matched": len(filtered),
            "rows": filtered[:cap],
            "handle_rule": "Handles are valid only for this revision and expire after HEAD advances.",
        }
        if include_spans and len(self.solids) > 1:
            result["mating_feature_pairs"] = self.mating_feature_pairs()
            result["mating_pair_note"] = (
                "Same analytic signature across solids; compare each member span independently."
            )
        if len(filtered) > cap:
            result["truncated"] = f"Showing {cap} of {len(filtered)} matches; narrow the query."
        if not filtered:
            result["note"] = "No entity matched; relax one filter instead of guessing a selector."
        return result

    def target_change_report(self, handles, after):
        """Which of the grounded targets actually changed between this snapshot and `after`.

        The counterpart to the contract's preservation clause: a target the agent declared it would
        edit must move, and everything it declared untouched must not. Matching is by type, centre,
        principal measure and radius rather than by index, because the index is meaningless across
        revisions. Tolerances are relative to the bounding-box diagonal so the check is
        scale-invariant, matching the metrics it is meant to protect.
        """
        changed, unchanged, errors = [], [], []
        diagonal = max(_diagonal_from_size(self.shape), 1e-9)
        for handle in _normalise_handles(handles):
            before = self.resolve(handle)
            if "error" in before:
                errors.append(before)
                continue
            candidates = after.records(before["selector"]["entity"])
            matched = False
            for candidate in candidates:
                if candidate.get("type") != before.get("type"):
                    continue
                if _distance(candidate.get("center", (0, 0, 0)), before.get("center", (0, 0, 0))) > 0.002 * diagonal:
                    continue
                before_measure = before.get("area", before.get("length", before.get("volume", 0.0)))
                after_measure = candidate.get("area", candidate.get("length", candidate.get("volume", 0.0)))
                if abs(after_measure - before_measure) > max(0.01, 0.02 * max(abs(before_measure), 1.0)):
                    continue
                if before.get("radius") is not None and abs(candidate.get("radius", -1) - before["radius"]) > QUERY_RADIUS_TOL:
                    continue
                matched = True
                break
            (unchanged if matched else changed).append(handle)
        return {
            "targets_total": len(changed) + len(unchanged),
            "targets_changed": changed,
            "targets_unchanged": unchanged,
            "errors": errors,
        }

    def overview(self):
        return {
            "revision": self.revision,
            "counts": {
                "solids": len(self.solid_records),
                "faces": len(self.face_records),
                "edges": len(self.edge_records),
            },
            "bbox": {
                "size": [
                    round(self.shape.BoundingBox().xlen, 3),
                    round(self.shape.BoundingBox().ylen, 3),
                    round(self.shape.BoundingBox().zlen, 3),
                ]
            },
            "handle_rule": "Ground again after every accepted commit.",
        }
