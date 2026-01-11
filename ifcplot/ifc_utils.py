from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.api
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "ifcplot requires `ifcopenshell`. Install with `python -m pip install ifcopenshell`."
    ) from exc


@dataclass(frozen=True)
class IfcContexts:
    model: "ifcopenshell.entity_instance"
    body: "ifcopenshell.entity_instance"


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


def add_trade_groups(f: "ifcopenshell.file") -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name in ("Concrete", "Framing", "HVAC", "Plumbing", "Drywall", "Cladding", "Furnishings"):
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
    wall_type: Optional[Any] = None,
) -> Any:
    wall = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcWall", name=name)
    ifcopenshell.api.run("spatial.assign_container", f, products=[wall], relating_structure=storey)
    representation = ifcopenshell.api.geometry.create_2pt_wall(
        f,
        element=wall,
        context=context,
        p1=p1,
        p2=p2,
        elevation=float(elevation),
        height=float(height),
        thickness=float(thickness),
        is_si=True,
    )
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

    curve = f.createIfcIndexedPolyCurve(f.createIfcCartesianPointList2D(pts))
    profile = f.createIfcArbitraryClosedProfileDef("AREA", None, curve)

    representation = ifcopenshell.api.run(
        "geometry.add_profile_representation",
        f,
        context=context,
        profile=profile,
        depth=float(depth),
    )
    ifcopenshell.api.run("geometry.assign_representation", f, product=element, representation=representation)
    ifcopenshell.api.run("geometry.edit_object_placement", f, product=element, matrix=placement_matrix)
    if element_type is not None:
        ifcopenshell.api.run("type.assign_type", f, related_objects=[element], relating_type=element_type)
    return element

