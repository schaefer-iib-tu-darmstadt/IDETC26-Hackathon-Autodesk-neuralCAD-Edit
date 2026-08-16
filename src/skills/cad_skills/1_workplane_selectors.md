# CAD Skill: Workplane Selectors & C++ Face Orientation

## Key Principle
Never calculate 3D transformation matrices or normal angles manually in Python. CadQuery's C++ OpenCASCADE kernel automatically calculates the local workplane coordinate system on ANY face in 1 millisecond.

## 1. Selecting Dominant Faces
```python
import cadquery as cq

# Select top-most planar face along Z and attach a cylinder
model = model.faces(">Z").workplane().circle(radius=9.0).extrude(distance=200.0)

# Select bottom-most planar face along Z
model = model.faces("<Z").workplane().circle(radius=5.0).extrude(distance=10.0)

# Select front face along Y (+Y) and cut a hole
model = model.faces(">Y").workplane().hole(diameter=1.7, depth=20.0)

# Select back face along Y (-Y)
model = model.faces("<Y").workplane().rect(20.0, 10.0).extrude(-5.0)
```

## 2. Selecting a Specific Face by Center Coordinate
If multiple faces exist on top, select the one closest to a specific (x, y, z):
```python
target_face = model.faces(lambda f: f.geomType() == "PLANE" and abs(f.Center().z - z_top) < 1.0 and abs(f.Center().x - x_target) < 5.0).val()
model = model.workplane(on=target_face).circle(radius).extrude(height)
```

## 3. Selecting Edges by Orientation & Type
* `|Z`: Edges parallel to Z-axis
* `|Y`: Edges parallel to Y-axis
* `|X`: Edges parallel to X-axis
* `#Z`: Edges perpendicular to Z-axis
* `%CIRCLE`: Circular edges only
* `>Z`: Top-most edges

Example:
```python
# Chamfer all vertical cylinder holes parallel to Y
model = model.edges("|Y and %CIRCLE").chamfer(0.2)
```
