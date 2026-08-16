# CAD Skill: Chamfers & Fillets

## 1. Chamfer by Circular Edge Radius
Use a lambda filter to find the exact edge matching the target radius:
```python
import cadquery as cq

target_r = 6.35 # Hole radius in mm
chamfer_dist = 0.2 # Chamfer distance

model = model.edges(lambda e: e.geomType() == "CIRCLE" and abs(e.radius() - target_r) < 0.05).chamfer(chamfer_dist)
```

## 2. Chamfer/Fillet by Z-Level or Direction
```python
# Fillet only top-most circular edges
model = model.edges(">Z and %CIRCLE").fillet(1.0)

# Chamfer all vertical cylinder edges
model = model.edges("|Z and %CIRCLE").chamfer(0.5)

# Fillet all vertical rectilinear edges
model = model.edges("|Z and not %CIRCLE").fillet(2.0)
```

## 3. Removing a Fillet / Replacing Fillet with Chamfer
To remove a fillet and replace with a chamfer:
1. Cut away the filleted profile using a rectilinear box/cylinder cutter.
2. Union a sharp feature and apply `.chamfer(dist)`.
