"""
Cross-platform offscreen rendering and export for CadQuery/OCP shapes.

Rendering uses OCP's V3d offscreen viewer on Linux and macOS.
Requires Xvfb on headless Linux (xvfb-run python ...).
Uses Cocoa_Window on macOS.
Uses CadQuery's VTK-based vis.show on Windows.
"""

import math
import os
import shutil
import sys

import cadquery as cq
from cadquery import exporters

VIEW_PROJECTIONS = {
    'toprightiso': (1, -1, 1),
    'front': (0, 0, 1),
    'back': (0, 0, -1),
    'left': (-1, 0, 0),
    'right': (1, 0, 0),
    'top': (0, 1, 0),
    'bottom': (0, -1, 0),
}


# Fraction of the frame the part should span. Fusion's own dataset renders sit at ~0.79 of
# the image width and never touch the border; matching that keeps our views comparable to the
# ground-truth images they are shown next to.
FRAME_FILL = 0.8
# VTK's default perspective view angle. `vis.show` does not expose it, so the framing maths
# below has to assume it.
VTK_VIEW_ANGLE_DEG = 30.0
# Grey, because cadquery.vis defaults to DEFAULT_COLOR = (1, 0.8, 0) -- gold. The dataset's
# Fusion renders are grey, and a submission that mixes the two reads as two different parts.
SHAPE_COLOR = (0.72, 0.74, 0.78)


def _as_shape(occ_shape):
    """Every input form collapsed to one CadQuery Shape covering the WHOLE model.

    `Workplane.val()` returns only the first object on the stack, so a multi-solid import
    would be framed -- and rendered -- as if the other solids did not exist.
    """
    if isinstance(occ_shape, cq.Assembly):
        return occ_shape.toCompound()
    if isinstance(occ_shape, cq.Workplane):
        solids = [obj for obj in occ_shape.vals() if isinstance(obj, cq.Shape)]
        if len(solids) == 1:
            return solids[0]
        return cq.Compound.makeCompound(solids)
    if isinstance(occ_shape, cq.Shape):
        return occ_shape
    return cq.Shape.cast(occ_shape)


def _camera(shape, proj):
    """Focus, position and view-up that frame `shape` at FRAME_FILL of the image.

    A fixed distance cannot work: the dataset's parts span three orders of magnitude in size
    and are not centred on the origin. Distance is therefore derived from the extent of the
    bounding box *as projected onto the image plane*, not from its diagonal -- an elongated
    part seen down its long axis has a much smaller silhouette than its diagonal suggests,
    and framing on the diagonal leaves it a speck in the middle of the frame.
    """
    bounds = shape.BoundingBox()
    focus = (
        (bounds.xmin + bounds.xmax) / 2.0,
        (bounds.ymin + bounds.ymax) / 2.0,
        (bounds.zmin + bounds.zmax) / 2.0,
    )

    norm = math.sqrt(sum(float(c) ** 2 for c in proj)) or 1.0
    direction = tuple(float(c) / norm for c in proj)          # focus -> camera
    forward = tuple(-c for c in direction)                     # camera -> focus

    # Pick a view-up that is not parallel to the view direction, then orthonormalise.
    up_hint = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (0.0, 1.0, 0.0)
    right = _cross(forward, up_hint)
    if _length(right) < 1e-9:
        right = _cross(forward, (1.0, 0.0, 0.0))
    right = _unit(right)
    view_up = _unit(_cross(right, forward))

    half_right = half_up = 0.0
    for corner in _corners(bounds):
        offset = tuple(corner[i] - focus[i] for i in range(3))
        half_right = max(half_right, abs(_dot(offset, right)))
        half_up = max(half_up, abs(_dot(offset, view_up)))

    half_extent = max(half_right, half_up)
    tan_half_angle = math.tan(math.radians(VTK_VIEW_ANGLE_DEG) / 2.0)
    distance = half_extent / (FRAME_FILL * tan_half_angle) if half_extent else 10.0
    # Clear of the geometry even for a degenerate or near-planar shape.
    distance = max(distance, half_extent * 1.5, 1e-6)

    position = tuple(focus[i] + direction[i] * distance for i in range(3))
    return focus, position, view_up


def _corners(bounds):
    return [
        (x, y, z)
        for x in (bounds.xmin, bounds.xmax)
        for y in (bounds.ymin, bounds.ymax)
        for z in (bounds.zmin, bounds.zmax)
    ]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _length(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    length = _length(a) or 1.0
    return tuple(c / length for c in a)


def _render_to_png_vtk(occ_shape, png_path, proj=(1, -1, 1), width=1024, height=1024,
                       edges=False):
    """Render using CadQuery's VTK-based vis.show (Windows).

    `edges=True` draws the *tessellation* triangles, not the B-rep feature edges, which makes
    every cylinder read as a black band and every planar face as a wireframe. Silhouette and
    feature edges survive without it, so the default is off.
    """
    from cadquery.vis import show

    shape = _as_shape(occ_shape)
    focus, position, view_up = _camera(shape, proj)

    # An Assembly carrying an explicit Color is the only way to override vis.show's gold:
    # `show` has no color parameter, but `_to_assy` preserves the colour of an Assembly child.
    coloured = cq.Assembly(shape, color=cq.Color(*SHAPE_COLOR))

    show(
        coloured,
        screenshot=str(png_path),
        interact=False,
        width=width,
        height=height,
        position=position,
        focus=focus,
        viewup=view_up,
        roll=0,
        elevation=0,
        azimuth=0,
        edges=edges,
        trihedron=False,
        gradient=False,
        bgcolor=(1.0, 1.0, 1.0),
    )


def render_to_png(occ_shape, png_path, proj=(1, -1, 1), width=1024, height=1024):
    """Render an OCP shape to a PNG file from a given projection direction.

    Args:
        occ_shape: A CadQuery Workplane, Assembly or Shape, or a raw OCP TopoDS_Shape.
        png_path: Output PNG file path.
        proj: (Vx, Vy, Vz) camera projection direction tuple.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    # Collapse before branching: `cadquery_convert.py` hands over the whole Workplane, which the
    # OCP branch below cannot wrap (`AIS_Shape` needs a TopoDS_Shape).
    occ_shape = _as_shape(occ_shape)
    if sys.platform == "win32":
        _render_to_png_vtk(occ_shape, png_path, proj=proj, width=width, height=height)
        return

    from OCP.AIS import AIS_InteractiveContext, AIS_Shape, AIS_Shaded
    from OCP.Aspect import Aspect_DisplayConnection, Aspect_TypeOfLine
    from OCP.OpenGl import OpenGl_GraphicDriver
    from OCP.Prs3d import Prs3d_LineAspect
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
    from OCP.V3d import V3d_Viewer

    wrapped = occ_shape.wrapped if hasattr(occ_shape, "wrapped") else occ_shape

    display_connection = Aspect_DisplayConnection()
    driver = OpenGl_GraphicDriver(display_connection)
    viewer = V3d_Viewer(driver)
    viewer.SetDefaultLights()
    viewer.SetLightOn()

    context = AIS_InteractiveContext(viewer)
    view = viewer.CreateView()

    if sys.platform == "linux":
        from OCP.Xw import Xw_Window
        window = Xw_Window(display_connection, "offscreen", 0, 0, width, height)
    else:
        import AppKit
        AppKit.NSApplication.sharedApplication()
        from OCP.Cocoa import Cocoa_Window
        window = Cocoa_Window("offscreen", 0, 0, width, height)

    view.SetWindow(window)
    if not window.IsMapped():
        window.Map()

    view.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))

    from OCP.Graphic3d import Graphic3d_MaterialAspect

    ais_shape = AIS_Shape(wrapped)

    mat = Graphic3d_MaterialAspect()
    mat.SetSpecularColor(Quantity_Color(0.15, 0.15, 0.15, Quantity_TOC_RGB))
    mat.SetShininess(0.3)
    ais_shape.SetMaterial(mat)
    ais_shape.SetColor(Quantity_Color(0.6, 0.6, 0.6, Quantity_TOC_RGB))

    drawer = ais_shape.Attributes()
    edge_aspect = Prs3d_LineAspect(
        Quantity_Color(Quantity_NOC_BLACK), Aspect_TypeOfLine.Aspect_TOL_SOLID, 2.0
    )
    drawer.SetFaceBoundaryAspect(edge_aspect)
    drawer.SetFaceBoundaryDraw(True)

    context.Display(ais_shape, AIS_Shaded, 0, True)
    view.SetProj(float(proj[0]), float(proj[1]), float(proj[2]))
    view.FitAll()
    view.Dump(str(png_path))


def export_as_step(result, output_dir):
    """Export CadQuery result as STEP file."""
    try:
        output_path = os.path.join(output_dir, "tmp.step")
        if hasattr(result, 'val'):
            shape = result.val()
        elif isinstance(result, cq.Assembly):
            result.save(output_path)
            print(f"Exported assembly to: {output_path}")
            return
        else:
            shape = result

        exporters.export(shape, output_path, exportType="STEP")
        print(f"Exported shape to: {output_path}")
    except Exception as e:
        print(f"Error exporting STEP file: {e}")
        import traceback
        traceback.print_exc()


def export_as_image(result, output_dir, views=None, width=1024, height=1024):
    """Export CadQuery result as PNG image(s).

    Args:
        result: CadQuery Workplane, Shape, or Assembly.
        output_dir: Directory to write PNG files into.
        views: Optional list of view names from VIEW_PROJECTIONS.
            When None (default), renders a single ``tmp.png`` at the
            default iso projection -- fully backwards compatible.
            When a list, renders one ``tmp_{name}.png`` per view.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        List of exported PNG file paths.
    """
    try:
        if hasattr(result, 'val'):
            shape = result.val()
        elif isinstance(result, cq.Assembly):
            shape = result.toCompound()
        else:
            shape = result

        if sys.platform == "linux" and not os.environ.get("DISPLAY"):
            if not shutil.which("Xvfb") and not shutil.which("xvfb-run"):
                print("Error: Xvfb is not installed. PNG export requires Xvfb.")
                print("Install it with:")
                print("  sudo apt-get install -y xvfb    # Debian/Ubuntu")
                print("  sudo yum install -y xorg-x11-server-Xvfb  # RHEL/Amazon Linux")
                sys.exit(1)
            print("Error: No DISPLAY set. Xvfb is installed but not running.")
            print("Either run with:  xvfb-run python cadquery_script.py ...")
            print("Or start Xvfb manually:")
            print("  Xvfb :99 -screen 0 1024x768x24 &")
            print("  export DISPLAY=:99")
            sys.exit(1)

        exported = []

        if views is None:
            png_path = os.path.join(output_dir, "tmp.png")
            render_to_png(shape, png_path, width=width, height=height)
            print(f"Exported PNG image to: {png_path}")
            exported.append(png_path)
        else:
            for view_name in views:
                proj = VIEW_PROJECTIONS.get(view_name)
                if proj is None:
                    print(f"Warning: Unknown view '{view_name}', skipping.")
                    continue
                png_path = os.path.join(output_dir, f"tmp_{view_name}.png")
                render_to_png(shape, png_path, proj=proj, width=width, height=height)
                print(f"Exported PNG image to: {png_path}")
                exported.append(png_path)

        return exported

    except Exception as e:
        print(f"Error exporting image: {e}")
        import traceback
        traceback.print_exc()
        return []
