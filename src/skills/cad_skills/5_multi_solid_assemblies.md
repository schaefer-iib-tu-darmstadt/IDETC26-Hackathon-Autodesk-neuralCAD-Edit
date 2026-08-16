# CAD Skill: Multi-Solid Assemblies

## 1. Accessing and Modifying Sub-Bodies
When a STEP file is an assembly containing multiple separate solids:
```python
import cadquery as cq

solids = model.val().Solids()
print(f"Total solids in assembly: {len(solids)}")

# Solid 0 is typically the main base/frame, Solid 1 may be the pin/lever
s0 = cq.Workplane(obj=solids[0])
s1 = cq.Workplane(obj=solids[1])

# Apply modification to Solid 0
s0_modified = s0.union(new_feature)

# Recombine all solids into final assembly compound
combined = s0_modified
for i in range(1, len(solids)):
    combined = combined.union(cq.Workplane(obj=solids[i]))

model = combined
```

## 2. Removing Clashes / Interferences Between Bodies
```python
# Remove collision where solid 1 penetrates solid 0
s0_cleared = cq.Workplane(obj=solids[0]).cut(cq.Workplane(obj=solids[1]))
model = s0_cleared.union(cq.Workplane(obj=solids[1]))
```
