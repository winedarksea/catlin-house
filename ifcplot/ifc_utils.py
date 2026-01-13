from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import numpy.typing as npt

try:
    import ifcopenshell
    import ifcopenshell.api
    import ifcopenshell.util.placement
    import ifcopenshell.util.unit
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "ifcplot requires `ifcopenshell`. Install with `python -m pip install ifcopenshell`."
    ) from exc


@dataclass(frozen=True)
class IfcContexts:
    model: "ifcopenshell.entity_instance"
    body: "ifcopenshell.entity_instance"


def _unit(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Zero-length vector")
    return v / n


def placement_matrix(*, origin: tuple[float, float, float], x_axis: tuple[float, float, float], z_axis: tuple[float, float, float]) -> np.ndarray:
    """
    Return a 4x4 transform matrix suitable for `geometry.edit_object_placement`.

    IfcOpenShell uses column 0 as the local X axis, column 1 as Y axis, and column 2 as the local Z axis.
    The Y axis is computed as the cross product of Z and X to form a proper orthonormal basis.
    """
    m = np.eye(4, dtype=float)
    x_unit = _unit(np.array(x_axis, dtype=float))
    z_unit = _unit(np.array(z_axis, dtype=float))
    # Compute Y as cross product of Z and X for right-handed coordinate system
    y_unit = _unit(np.cross(z_unit, x_unit))
    # Recompute X to ensure orthogonality
    x_unit = _unit(np.cross(y_unit, z_unit))
    m[0:3, 0] = x_unit
    m[0:3, 1] = y_unit
    m[0:3, 2] = z_unit
    m[0:3, 3] = np.array(origin, dtype=float)
    return m


def translation_matrix(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    m = np.eye(4, dtype=float)
    m[0, 3] = float(x)
    m[1, 3] = float(y)
    m[2, 3] = float(z)
    return m


def init_ifc_project(*, name: str, schema: str = "IFC4") -> tuple["ifcopenshell.file", Any, Any, IfcContexts]:
    f = ifcopenshell.file(schema=schema)
    project = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name=name)
    ifcopenshell.api.run("unit.assign_unit", f)  # default metric units

    model_ctx = ifcopenshell.api.run("context.add_context", f, context_type="Model")
    body_ctx = ifcopenshell.api.run(
        "context.add_context",
        f,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_ctx,
    )

    site = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    ifcopenshell.api.run("aggregate.assign_object", f, products=[site], relating_object=project)
    ifcopenshell.api.run("geometry.edit_object_placement", f, product=site, matrix=translation_matrix())

    return f, project, site, IfcContexts(model=model_ctx, body=body_ctx)


def add_building(
    f: "ifcopenshell.file",
    *,
    site: Any,
    name: str,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    building = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuilding", name=name)
    ifcopenshell.api.run("aggregate.assign_object", f, products=[building], relating_object=site)
    ifcopenshell.api.run("geometry.edit_object_placement", f, product=building, matrix=translation_matrix(*origin))
    return building


def add_storey(
    f: "ifcopenshell.file",
    *,
    building: Any,
    name: str,
    elevation: float,
    global_xy: tuple[float, float] = (0.0, 0.0),
    global_z0: float = 0.0,
) -> Any:
    storey = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name=name)
    storey.Elevation = float(elevation)
    ifcopenshell.api.run("aggregate.assign_object", f, products=[storey], relating_object=building)
    ifcopenshell.api.run(
        "geometry.edit_object_placement",
        f,
        product=storey,
        matrix=translation_matrix(global_xy[0], global_xy[1], global_z0 + float(elevation)),
    )
    return storey


def ensure_pset(f: "ifcopenshell.file", *, product: Any, name: str) -> Any:
    for rel in getattr(product, "IsDefinedBy", []) or []:
        if rel.is_a("IfcRelDefinesByProperties") and rel.RelatingPropertyDefinition.Name == name:
            return rel.RelatingPropertyDefinition
    return ifcopenshell.api.run("pset.add_pset", f, product=product, name=name)


def set_pset_json(f: "ifcopenshell.file", *, product: Any, pset_name: str, prop_name: str, value: Any) -> Any:
    pset = ensure_pset(f, product=product, name=pset_name)
    ifcopenshell.api.run("pset.edit_pset", f, pset=pset, properties={prop_name: json.dumps(value, sort_keys=True)})
    return pset


def add_trade_groups(f: "ifcopenshell.file", *, trade_names: Optional[list[str]] = None) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    names = trade_names if trade_names is not None else ("Concrete", "Framing", "HVAC", "Plumbing", "Drywall", "Cladding", "Furnishings")
    for name in names:
        groups[name] = ifcopenshell.api.run("group.add_group", f, name=name)
    return groups


def assign_to_group(f: "ifcopenshell.file", *, group: Any, products: Iterable[Any]) -> None:
    products_list = [p for p in products if p is not None]
    if products_list:
        ifcopenshell.api.run("group.assign_group", f, products=products_list, group=group)


def add_wall_between_points(
    f: "ifcopenshell.file",
    *,
    context: Any,
    storey: Any,
    name: str,
    p1: tuple[float, float],
    p2: tuple[float, float],
    elevation: float,
    height: float,
    thickness: float,
    direction_sense: str = "POSITIVE",
    offset: float = 0.0,
    x_angle: float = 0.0,
    wall_type: Optional[Any] = None,
) -> Any:
    wall = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcWall", name=name)
    ifcopenshell.api.run("spatial.assign_container", f, products=[wall], relating_structure=storey)
    p1_ = np.array(p1, dtype=float)
    p2_ = np.array(p2, dtype=float)
    v = p2_ - p1_
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        raise ValueError(f"Wall `{name}` has zero length")
    v /= length

    representation = ifcopenshell.api.geometry.add_wall_representation(
        f,
        context=context,
        length=length,
        height=float(height),
        thickness=float(thickness),
        direction_sense=direction_sense,
        offset=float(offset),
        x_angle=float(x_angle),
    )

    matrix = np.array(
        [
            [v[0], -v[1], 0.0, p1_[0]],
            [v[1], v[0], 0.0, p1_[1]],
            [0.0, 0.0, 1.0, float(elevation)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    ifcopenshell.api.run("geometry.edit_object_placement", f, product=wall, matrix=matrix)
    ifcopenshell.api.run("geometry.assign_representation", f, product=wall, representation=representation)
    if wall_type is not None:
        ifcopenshell.api.run("type.assign_type", f, related_objects=[wall], relating_type=wall_type)
    return wall


def add_slab(
    f: "ifcopenshell.file",
    *,
    context: Any,
    storey: Any,
    name: str,
    polyline: list[tuple[float, float]],
    elevation: float,
    depth: float,
    slab_type: Optional[Any] = None,
    predefined_type: str = "FLOOR",
) -> Any:
    slab = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSlab", predefined_type=predefined_type, name=name)
    ifcopenshell.api.run("spatial.assign_container", f, products=[slab], relating_structure=storey)
    representation = ifcopenshell.api.run(
        "geometry.add_slab_representation",
        f,
        context=context,
        depth=float(depth),
        polyline=polyline,
    )
    ifcopenshell.api.run("geometry.assign_representation", f, product=slab, representation=representation)
    ifcopenshell.api.run(
        "geometry.edit_object_placement",
        f,
        product=slab,
        matrix=translation_matrix(0.0, 0.0, float(elevation)),
    )
    if slab_type is not None:
        ifcopenshell.api.run("type.assign_type", f, related_objects=[slab], relating_type=slab_type)
    return slab


def add_prism_from_profile(
    f: "ifcopenshell.file",
    *,
    context: Any,
    storey: Any,
    ifc_class: str,
    name: str,
    profile_points: list[tuple[float, float]],
    depth: float,
    placement_matrix: np.ndarray,
    placement_is_storey_relative: bool = True,
    element_type: Optional[Any] = None,
    predefined_type: Optional[str] = None,
) -> Any:
    element = ifcopenshell.api.run(
        "root.create_entity",
        f,
        ifc_class=ifc_class,
        predefined_type=predefined_type,
        name=name,
    )
    ifcopenshell.api.run("spatial.assign_container", f, products=[element], relating_structure=storey)

    pts = profile_points
    if pts[0] != pts[-1]:
        pts = [*pts, pts[0]]

    profile = ifcopenshell.api.run("profile.add_arbitrary_profile", f, name=name, profile=pts)

    representation = ifcopenshell.api.run(
        "geometry.add_profile_representation",
        f,
        context=context,
        profile=profile,
        depth=float(depth),
    )
    ifcopenshell.api.run("geometry.assign_representation", f, product=element, representation=representation)

    matrix = placement_matrix
    if placement_is_storey_relative and getattr(storey, "ObjectPlacement", None):
        # Convert storey placement translation into SI meters so we can combine it with our SI matrices.
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(f)
        storey_matrix = ifcopenshell.util.placement.get_local_placement(storey.ObjectPlacement).copy()
        storey_matrix[0:3, 3] *= float(unit_scale)
        matrix = storey_matrix @ placement_matrix

    ifcopenshell.api.run("geometry.edit_object_placement", f, product=element, matrix=matrix)
    if element_type is not None:
        ifcopenshell.api.run("type.assign_type", f, related_objects=[element], relating_type=element_type)
    return element


def add_prism_from_profile_with_voids(
    f: "ifcopenshell.file",
    *,
    context: Any,
    storey: Any,
    ifc_class: str,
    name: str,
    outer_profile_points: list[tuple[float, float]],
    inner_profile_points: list[list[tuple[float, float]]],
    depth: float,
    placement_matrix: np.ndarray,
    placement_is_storey_relative: bool = True,
    element_type: Optional[Any] = None,
    predefined_type: Optional[str] = None,
) -> Any:
    element = ifcopenshell.api.run(
        "root.create_entity",
        f,
        ifc_class=ifc_class,
        predefined_type=predefined_type,
        name=name,
    )
    ifcopenshell.api.run("spatial.assign_container", f, products=[element], relating_structure=storey)

    def closed(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not pts:
            raise ValueError("Profile must have at least one point")
        if pts[0] == pts[-1]:
            return pts
        return [*pts, pts[0]]

    outer = closed(outer_profile_points)
    inners = [closed(p) for p in inner_profile_points]

    profile = ifcopenshell.api.run(
        "profile.add_arbitrary_profile_with_voids",
        f,
        outer_profile=outer,
        inner_profiles=inners,
        name=name,
    )
    representation = ifcopenshell.api.run(
        "geometry.add_profile_representation",
        f,
        context=context,
        profile=profile,
        depth=float(depth),
    )
    ifcopenshell.api.run("geometry.assign_representation", f, product=element, representation=representation)

    matrix = placement_matrix
    if placement_is_storey_relative and getattr(storey, "ObjectPlacement", None):
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(f)
        storey_matrix = ifcopenshell.util.placement.get_local_placement(storey.ObjectPlacement).copy()
        storey_matrix[0:3, 3] *= float(unit_scale)
        matrix = storey_matrix @ placement_matrix

    ifcopenshell.api.run("geometry.edit_object_placement", f, product=element, matrix=matrix)
    if element_type is not None:
        ifcopenshell.api.run("type.assign_type", f, related_objects=[element], relating_type=element_type)
    return element


def add_rect_member_between_points(
    f: "ifcopenshell.file",
    *,
    context: Any,
    storey: Any,
    name: str,
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    width: float,
    depth: float,
    predefined_type: str = "JOIST",
    member_type: Optional[Any] = None,
    x_axis_hint: tuple[float, float, float] = (0.0, 1.0, 0.0),
    points_are_storey_relative: bool = True,
    ifc_class: str = "IfcBeam",
) -> Any:
    """
    Add an `IfcBeam`/`IfcMember` with a rectangular section, extruded from `p1` to `p2`.

    - `width` is the section X dimension (often the member thickness).
    - `depth` is the section Y dimension (often the member depth).
    - The extrusion direction is aligned to the member axis (local +Z).
    """
    p1v = np.array(p1, dtype=float)
    p2v = np.array(p2, dtype=float)
    if points_are_storey_relative and getattr(storey, "ObjectPlacement", None):
        # Convert storey placement translation into SI meters so we can combine it with our SI points.
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(f)
        storey_matrix = ifcopenshell.util.placement.get_local_placement(storey.ObjectPlacement).copy()
        storey_matrix[0:3, 3] *= float(unit_scale)
        p1v = (storey_matrix @ np.array([*p1v, 1.0], dtype=float))[0:3]
        p2v = (storey_matrix @ np.array([*p2v, 1.0], dtype=float))[0:3]
    axis = p2v - p1v
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        raise ValueError(f"Member `{name}` has zero length")

    z_axis = _unit(axis)
    x_hint = _unit(np.array(x_axis_hint, dtype=float))
    if abs(float(np.dot(z_axis, x_hint))) > 0.99:
        x_hint = _unit(np.array((1.0, 0.0, 0.0), dtype=float))
        if abs(float(np.dot(z_axis, x_hint))) > 0.99:
            x_hint = _unit(np.array((0.0, 0.0, 1.0), dtype=float))

    member = ifcopenshell.api.run("root.create_entity", f, ifc_class=ifc_class, predefined_type=predefined_type, name=name)
    ifcopenshell.api.run("spatial.assign_container", f, products=[member], relating_structure=storey)

    # Create profile with explicit position at origin for proper rendering
    profile_origin = f.createIfcCartesianPoint((0.0, 0.0))
    profile_position = f.createIfcAxis2Placement2D(profile_origin, None)
    profile = f.createIfcRectangleProfileDef("AREA", None, profile_position, float(width), float(depth))
    
    representation = ifcopenshell.api.run(
        "geometry.add_profile_representation",
        f,
        context=context,
        profile=profile,
        depth=length,
    )
    ifcopenshell.api.run("geometry.assign_representation", f, product=member, representation=representation)

    m = placement_matrix(origin=tuple(map(float, p1v)), x_axis=tuple(map(float, x_hint)), z_axis=tuple(map(float, z_axis)))
    ifcopenshell.api.run("geometry.edit_object_placement", f, product=member, matrix=m)

    if member_type is not None:
        ifcopenshell.api.run("type.assign_type", f, related_objects=[member], relating_type=member_type)
    return member


def create_surface_style_with_texture(
    f: "ifcopenshell.file",
    *,
    name: str,
    texture_path: str,
    repeat_s: bool = True,
    repeat_t: bool = True,
    rgb: tuple[float, float, float] | None = None,
) -> Any:
    """Creates an IfcSurfaceStyle with a texture and optional fallback color."""
    style = ifcopenshell.api.run("style.add_style", f, name=name)
    
    # Always add texture
    ifcopenshell.api.run(
        "style.add_surface_style",
        f,
        style=style,
        ifc_class="IfcSurfaceStyleWithTextures",
        attributes={
            "Textures": [
                f.create_entity(
                    "IfcImageTexture",
                    URLReference=texture_path,
                    RepeatS=repeat_s,
                    RepeatT=repeat_t,
                )
            ]
        },
    )

    # Add fallback color if provided
    if rgb:
        r, g, b = (float(c) for c in rgb)
        colour = {"Name": None, "Red": r, "Green": g, "Blue": b}
        ifcopenshell.api.run(
            "style.add_surface_style",
            f,
            style=style,
            ifc_class="IfcSurfaceStyleShading",
            attributes={"SurfaceColour": colour},
        )
        # Also add rendering style for better viewer support
        # We explicitly set DiffuseColour as many modern viewers prioritize it.
        ifcopenshell.api.run(
            "style.add_surface_style",
            f,
            style=style,
            ifc_class="IfcSurfaceStyleRendering",
            attributes={
                "SurfaceColour": colour,
                "DiffuseColour": colour,
                "Transparency": 0.0,
            },
        )
    
    return style


def create_surface_style_shading(
    f: "ifcopenshell.file",
    *,
    name: str,
    rgb: tuple[float, float, float],
    transparency: float | None = None,
) -> Any:
    """Creates a simple surface style with an RGB color (and optional transparency)."""
    r, g, b = (float(c) for c in rgb)
    style = ifcopenshell.api.run("style.add_style", f, name=name)
    colour = {"Name": None, "Red": r, "Green": g, "Blue": b}
    
    # Always add basic shading
    ifcopenshell.api.run(
        "style.add_surface_style",
        f,
        style=style,
        ifc_class="IfcSurfaceStyleShading",
        attributes={"SurfaceColour": colour},
    )

    # Also add rendering style for better viewer support (highly recommended for all materials)
    attrs: dict[str, Any] = {
        "SurfaceColour": colour,
        "DiffuseColour": colour,
        "Transparency": float(transparency) if transparency is not None else 0.0,
    }
    ifcopenshell.api.run(
        "style.add_surface_style",
        f,
        style=style,
        ifc_class="IfcSurfaceStyleRendering",
        attributes=attrs,
    )
    return style


def assign_surface_style(f: "ifcopenshell.file", *, element: Any, style: Any) -> None:
    """Assigns a surface style to an element's representation."""
    if not hasattr(element, "Representation") or not element.Representation:
        return
    if not hasattr(element.Representation, "Representations") or not element.Representation.Representations:
        return
    for rep in element.Representation.Representations:
        for item in rep.Items:
            ifcopenshell.api.run("style.assign_item_style", f, item=item, style=style)
