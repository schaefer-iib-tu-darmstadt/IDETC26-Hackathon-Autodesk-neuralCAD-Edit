# CAD Skill: Holes, Slots, Grooves & Cutouts

## 1. Simple Hole & Counterbore
```python
import cadquery as cq

# Cut a hole of diameter D on the selected face
model = model.faces(">Z").workplane().hole(diameter=1.7, depth=None) # depth=None = through-all

# Cut a hole at specific local offset (x_off, y_off)
model = model.faces(">Z").workplane().transformed(offset=(10.0, 0, 0)).hole(diameter=3.0)
```

## 2. Oblong Slots / Rounded Long Cutouts
```python
# Construct a rounded slot cutter
c1 = cq.Workplane("XY").transformed(offset=(cx, y1, z_top)).circle(r_slot).extrude(-cut_depth)
c2 = cq.Workplane("XY").transformed(offset=(cx, y2, z_top)).circle(r_slot).extrude(-cut_depth)
box = cq.Workplane("XY").transformed(offset=(cx, (y1+y2)/2.0, z_top)).rect(2 * r_slot, abs(y2 - y1)).extrude(-cut_depth)

slot_cutter = c1.union(c2).union(box)
model = model.cut(slot_cutter)
```

## 3. Annular Grooves (Concentric Retaining Rings)
```python
# Create annular groove cutter of outer radius R_out and inner radius R_in
groove_cutter = (
    cq.Workplane("XY")
    .transformed(offset=(cx, cy, cz_groove))
    .circle(r_out)
    .circle(r_in)
    .extrude(groove_width)
)
model = model.cut(groove_cutter)
```
