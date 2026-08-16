# CAD Skill: Polar Patterns, Gears & Mirror Symmetry

## 1. Radial Polar Replication (8-tooth gears, rotor blades, bolt circles)
```python
import cadquery as cq

# Base single tooth / blade at angle 0
tooth = cq.Workplane("XY").transformed(offset=(r_center, 0, z_level)).rect(tooth_w, tooth_l).extrude(h)

# Replicate N instances evenly around center axis (cx, cy, cz)
num_instances = 8
angle_step = 360.0 / num_instances

combined_pattern = tooth
for i in range(1, num_instances):
    angle = i * angle_step
    rot_tooth = tooth.rotate((cx, cy, cz), (cx, cy, cz + 1), angle)
    combined_pattern = combined_pattern.union(rot_tooth)

model = model.union(combined_pattern)
```

## 2. Inscribing Polygons (Hexagon / Octagon)
```python
# Replace or insert regular hexagon (6 sides) of radius R
hex_profile = (
    cq.Workplane("XY")
    .transformed(offset=(cx, cy, cz))
    .polygon(nSides=6, diameter=2 * r_hex)
    .extrude(extrude_depth)
)
model = model.union(hex_profile)
```

## 3. Mirroring Across Symmetry Planes
```python
# Mirror entire solid or feature across ZX plane (symmetric along Y)
mirrored = model.mirror(mirrorPlane="ZX", basePointVector=(0, 0, 0))
model = model.union(mirrored)

# Mirror across XY plane (symmetric along Z)
mirrored = model.mirror(mirrorPlane="XY", basePointVector=(0, 0, z_mid))
model = model.union(mirrored)
```
