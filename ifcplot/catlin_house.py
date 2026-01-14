from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directory to path to allow imports when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import ifcopenshell

from ifcplot.detail_utils import MATERIAL_COLORS
from ifcplot.assemblies import GARAGE_ICF, GARAGE_WALL, ICFFoundationAssembly
from ifcplot.ifc_utils import (
    add_building,
    add_prism_from_profile,
    add_prism_from_profile_with_voids,
    add_rect_member_between_points,
    add_slab,
    add_storey,
    add_trade_groups,
    add_wall_between_points,
    assign_surface_style,
    assign_to_group,
    create_surface_style_shading,
    create_surface_style_with_texture,
    init_ifc_project,
    set_pset_json,
    translation_matrix,
    placement_matrix,
)
from ifcplot.units import ft, inch

# TODO: visualize the joists, studs and beams in the walls as distinct items visible in Bonsai BIM
# TODO: build the garage roof properly, adding trusses to replace the simple gable prism


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color string to RGB tuple with values 0.0-1.0."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (r / 255.0, g / 255.0, b / 255.0)


@dataclass(frozen=True)
class CatlinSitePlacement:
    house_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # If `(0,0,0)`, the garage is auto-placed relative to the house:
    # - West wall aligned with house west wall
    # - South wall 12' north of house north wall
    garage_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Plan rotation (deg) about the garage center (affects roof gable direction).
    garage_rotation_deg: float = 90.0
    # The sunken garden + porch are currently defined relative to the house.
    porch_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    breezeway_origin_m: tuple[float, float, float] = (ft(36.0), ft(6.0), 0.0)  # placeholder


@dataclass(frozen=True)
class CatlinHouseSpec:
    house_size_ft: float = 36.0  # measured at sheathing
    garage_size_ft: float = 24.0  # measured at sheathing

    basement_wall_thickness_in: float = 12.0
    basement_clear_height_ft: float = 9.0
    basement_slab_thickness_in: float = 3.5
    basement_ceiling_slab_thickness_in: float = 9.0

    main_storey_height_ft: float = 9.0  # placeholder until documented
    second_storey_height_ft: float = 9.0  # placeholder until documented
    attic_floor_elevation_ft: float = 18.0
    attic_knee_wall_height_ft: float = 5.0
    attic_ridge_height_above_floor_ft: float = 11.0
    attic_gable_extra_height_ft: float = 1.0
    attic_gable_siding_overlap_in: float = 5.0
    attic_knee_siding_extra_height_ft: float = 1.0

    roof_pitch_rise_over_run: float = 4.0 / 12.0
    roof_overhang_in: float = 0.0

    framing_spacing_in: float = 16.0
    floor_joist_width_in: float = 1.5
    floor_joist_depth_in: float = 11.875
    roof_joist_width_in: float = 1.5
    roof_joist_depth_in: float = 11.875
    centerline_wall_thickness_in: float = 5.5  # 2x6

    # Garage
    garage_icf_above_grade_in: float = 22.0
    garage_frost_depth_in: float = 42.0
    garage_wood_wall_height_ft: float = 8.0
    garage_slab_thickness_in: float = 3.5
    garage_roof_pitch_rise_over_run: float = 4.0 / 12.0
    garage_overhang_in: float = 16.0

    # Basement plan key features
    basement_grid_ft: float = 18.0  # 4 quadrants with center cross walls
    stair_opening_size_ft: tuple[float, float] = (7.0, 9.0 + 8.0 / 12.0)  # (E-W, N-S)


@dataclass(frozen=True)
class SunkenGardenSpec:
    clear_width_ft: float = 18.0  # east-west, between wall inner faces
    clear_length_ft: float = 28.0  # north-south, between wall inner faces
    porch_clear_depth_ft: float = 8.0  # north-south, between porch wall inner faces

    gap_to_house_in: float = 5.0  # gap between house south wall and sunken garden north wall (outer face)

    wall_thickness_in: float = 12.0

    footing_thickness_in: float = 12.0
    footing_toe_in: float = 36.0  # interior side
    footing_heel_in: float = 36.0  # exterior side (clamped at the house-side wall)

    aggregate_thickness_in: float = 18.0
    aggregate_extra_in: float = 6.0

    porch_joist_width_in: float = 1.5  # 2x lumber thickness
    porch_joist_depth_in: float = 7.25  # 2x8 actual
    porch_joist_spacing_in: float = 16.0
    deck_thickness_in: float = 1.5  # schematic deck/sheathing thickness

    railing_height_in: float = 36.0
    railing_thickness_in: float = 2.0

    # Arch parameters (for the two-story concrete porch box, north + south faces)
    arches_per_wall: int = 2
    arch_clear_width_ft: float = 8.0
    arch_outer_pier_ft: float = 1.0
    arch_opening_height_ft: float = 8.0


def _flip_profile_y(profile_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [tuple((p[0], -p[1])) for p in profile_points]


def _rect_polyline_xy(origin_xy: tuple[float, float], size_xy: tuple[float, float]) -> list[tuple[float, float]]:
    x0, y0 = origin_xy
    w, h = size_xy
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)]


def _arch_void_profile_points(
    *,
    x_left: float,
    x_right: float,
    opening_height: float,
    segments: int = 48,
) -> list[tuple[float, float]]:
    """
    Return a closed polyline for an arched opening (rect + semicircle).

    Local coordinates assume:
    - X is horizontal.
    - Y is vertical, with negative values "up" (matches existing gable profiles).
    - Opening base is at y=0.
    """
    w = float(x_right - x_left)
    if w <= 0:
        raise ValueError("Arch opening must have positive width")
    r = w / 2.0
    if opening_height < r:
        raise ValueError("Arch opening height must be >= half the opening width")

    spring_height = float(opening_height - r)
    spring_y = -spring_height
    cx = float((x_left + x_right) / 2.0)
    cy = float(spring_y)

    thetas = np.linspace(np.pi, 0.0, int(segments))
    pts: list[tuple[float, float]] = [(float(x_left), 0.0), (float(x_left), spring_y)]
    for t in thetas:
        pts.append((cx + r * float(np.cos(t)), cy - r * float(np.sin(t))))
    pts.extend([(float(x_right), 0.0), (float(x_left), 0.0)])
    return pts


def _arch_voids_for_wall(
    *,
    wall_width: float,
    opening_height: float,
    n_arches: int,
    arch_width: float,
    outer_pier: float,
) -> list[list[tuple[float, float]]]:
    """Compute a list of arched opening polylines for a wall face."""
    if n_arches < 1:
        return []
    if wall_width <= 0 or arch_width <= 0:
        raise ValueError("Wall width and arch width must be positive")
    if outer_pier < 0:
        raise ValueError("outer_pier must be >= 0")

    middle_length = wall_width - 2.0 * outer_pier
    arches_total = n_arches * arch_width
    interior_piers_count = max(n_arches - 1, 0)
    remaining_for_interior = middle_length - arches_total
    if remaining_for_interior < -1e-9:
        raise ValueError("Arches + outer piers exceed wall width")

    pier = (remaining_for_interior / interior_piers_count) if interior_piers_count > 0 else 0.0

    voids: list[list[tuple[float, float]]] = []
    x = float(outer_pier)
    for i in range(n_arches):
        x_left = x
        x_right = x + arch_width
        voids.append(_arch_void_profile_points(x_left=x_left, x_right=x_right, opening_height=opening_height))
        x = x_right
        if i < n_arches - 1:
            x += pier
    return voids


def _offset_segment(p0, p1, offset):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    v = p1 - p0
    if np.linalg.norm(v) == 0:
        return p0, p1
    v_perp = np.array([-v[1], v[0]])
    v_perp /= np.linalg.norm(v_perp)
    return p0 + offset * v_perp, p1 + offset * v_perp


def _pt_at_x(seg0, seg1, x):
    """Helper to find point along a segment at a given x coordinate."""
    t = (x - seg0[0]) / (seg1[0] - seg0[0] + 1e-9)
    return seg0 + t * (seg1 - seg0)


def _layer_poly(mid0, mid1, center_offset, thickness, x0, x1):
    """Create a polygon for a layer offset from a centerline segment."""
    upper0, upper1 = _offset_segment(mid0, mid1, center_offset + thickness / 2)
    lower0, lower1 = _offset_segment(mid0, mid1, center_offset - thickness / 2)
    p0u = _pt_at_x(upper0, upper1, x0)
    p1u = _pt_at_x(upper0, upper1, x1)
    p1l = _pt_at_x(lower0, lower1, x1)
    p0l = _pt_at_x(lower0, lower1, x0)
    poly_points = [p0u, p1u, p1l, p0l]
    return _flip_profile_y(poly_points)




def build_catlin_house_ifc(
    *,
    out_path: Path,
    site: CatlinSitePlacement | None = None,
    spec: CatlinHouseSpec | None = None,
    sunken_garden_spec: SunkenGardenSpec | None = None,
    include_scale_figure: bool = True,
) -> Any:
    site = site or CatlinSitePlacement()
    spec = spec or CatlinHouseSpec()
    sunken = sunken_garden_spec or SunkenGardenSpec()

    f, project, ifc_site, contexts = init_ifc_project(name="ifcPlot - Catlin House", schema="IFC4")
    groups = add_trade_groups(f)

    house_size_m = ft(spec.house_size_ft)
    house_origin_x, house_origin_y, _ = site.house_origin_m

    # Auto-place the garage relative to the house if the site placement uses the sentinel `(0,0,0)`.
    # - House footprint: x=[0..house_size], y=[0..house_size] in the current model
    # - Garage south wall is 12' north of the house north wall
    garage_origin = site.garage_origin_m
    if garage_origin == (0.0, 0.0, 0.0):
        garage_origin = (house_origin_x, house_origin_y + house_size_m + ft(12.0), 0.0)

    # ---- Materials and Styles --------------------------------------------------
    # Copy the standing-seam texture next to the exported IFC so viewers (e.g. Bonsai) can resolve it.
    texture_src = Path(__file__).parent / "textures" / "standing_seam_texture.png"
    texture_rel = Path("textures") / texture_src.name
    texture_dst = out_path.parent / texture_rel
    texture_dst.parent.mkdir(parents=True, exist_ok=True)
    if texture_src.exists():
        shutil.copy2(texture_src, texture_dst)

    concrete_style = create_surface_style_shading(f, name="Concrete", rgb=hex_to_rgb(MATERIAL_COLORS["concrete"]))
    aggregate_style = create_surface_style_shading(f, name="Aggregate", rgb=hex_to_rgb(MATERIAL_COLORS["aggregate"]))
    drywall_style = create_surface_style_shading(f, name="Drywall", rgb=hex_to_rgb(MATERIAL_COLORS["drywall"]))
    sheathing_style = create_surface_style_shading(f, name="Sheathing/OSB", rgb=hex_to_rgb(MATERIAL_COLORS["sheathing"]))
    polyiso_style = create_surface_style_shading(f, name="Polyiso", rgb=hex_to_rgb(MATERIAL_COLORS["polyiso"]))
    eps_style = create_surface_style_shading(f, name="EPS", rgb=hex_to_rgb(MATERIAL_COLORS["eps"]))
    xps_style = create_surface_style_shading(f, name="XPS", rgb=hex_to_rgb(MATERIAL_COLORS["xps"]))
    membrane_style = create_surface_style_shading(f, name="Membrane", rgb=hex_to_rgb(MATERIAL_COLORS["membrane"]))
    framing_wood_style = create_surface_style_shading(f, name="Framing Wood", rgb=hex_to_rgb(MATERIAL_COLORS["wood"]))
    metal_dark_style = create_surface_style_shading(f, name="Metal (Dark)", rgb=hex_to_rgb(MATERIAL_COLORS["metal_dark"]))
    scale_figure_style = create_surface_style_shading(
        f,
        name="Scale Figure",
        rgb=(0.2, 0.5, 0.9),
        transparency=0.65,
    )

    standing_seam_style = create_surface_style_with_texture(
        f,
        name="Standing Seam Metal",
        texture_path=texture_rel.as_posix(),
        rgb=hex_to_rgb(MATERIAL_COLORS["metal_dark"]),
    )

    # ---- Buildings and storeys -------------------------------------------------
    house_bldg = add_building(f, site=ifc_site, name="House", origin=site.house_origin_m)
    garage_bldg = add_building(f, site=ifc_site, name="Garage", origin=garage_origin)
    porch_origin = site.porch_origin_m if site.porch_origin_m != (0.0, 0.0, 0.0) else site.house_origin_m
    porch_bldg = add_building(f, site=ifc_site, name="Porch + Sunken Garden", origin=porch_origin)
    breezeway_bldg = add_building(f, site=ifc_site, name="Breezeway (placeholder)", origin=site.breezeway_origin_m)

    # Storeys (elevations are global Z)
    basement_elev_m = -ft(spec.basement_clear_height_ft)
    main_elev_m = 0.0
    second_elev_m = ft(spec.second_storey_height_ft)
    attic_elev_m = ft(spec.attic_floor_elevation_ft)

    house_basement = add_storey(
        f,
        building=house_bldg,
        name="Basement",
        elevation=basement_elev_m,
        global_xy=(site.house_origin_m[0], site.house_origin_m[1]),
        global_z0=site.house_origin_m[2],
    )
    house_main = add_storey(
        f,
        building=house_bldg,
        name="Main Floor",
        elevation=main_elev_m,
        global_xy=(site.house_origin_m[0], site.house_origin_m[1]),
        global_z0=site.house_origin_m[2],
    )
    house_second = add_storey(
        f,
        building=house_bldg,
        name="Second Floor",
        elevation=second_elev_m,
        global_xy=(site.house_origin_m[0], site.house_origin_m[1]),
        global_z0=site.house_origin_m[2],
    )
    house_attic = add_storey(
        f,
        building=house_bldg,
        name="Attic Floor",
        elevation=attic_elev_m,
        global_xy=(site.house_origin_m[0], site.house_origin_m[1]),
        global_z0=site.house_origin_m[2],
    )

    garage_level = add_storey(
        f,
        building=garage_bldg,
        name="Garage Level",
        elevation=0.0,
        global_xy=(garage_origin[0], garage_origin[1]),
        global_z0=garage_origin[2],
    )
    porch_sunken = add_storey(
        f,
        building=porch_bldg,
        name="Sunken Garden Floor",
        elevation=basement_elev_m + inch(spec.basement_slab_thickness_in),
        global_xy=(porch_origin[0], porch_origin[1]),
        global_z0=porch_origin[2],
    )
    porch_main = add_storey(
        f,
        building=porch_bldg,
        name="Porch Floor",
        elevation=main_elev_m,
        global_xy=(porch_origin[0], porch_origin[1]),
        global_z0=porch_origin[2],
    )
    porch_deck = add_storey(
        f,
        building=porch_bldg,
        name="Deck Floor",
        elevation=second_elev_m,
        global_xy=(porch_origin[0], porch_origin[1]),
        global_z0=porch_origin[2],
    )
    add_storey(
        f,
        building=breezeway_bldg,
        name="Level 0",
        elevation=0.0,
        global_xy=(site.breezeway_origin_m[0], site.breezeway_origin_m[1]),
        global_z0=site.breezeway_origin_m[2],
    )

    grid_m = ft(spec.basement_grid_ft)
    stair_w_m = ft(spec.stair_opening_size_ft[0])
    stair_d_m = ft(spec.stair_opening_size_ft[1])
    stair_opening_rect_m = {
        "x0_m": grid_m - stair_w_m,
        "y0_m": house_size_m - stair_d_m,
        "w_m": stair_w_m,
        "h_m": stair_d_m,
    }
    sauna_rect_m = {
        "x0_m": grid_m - ft(12.0),
        "y0_m": 0.0,
        "w_m": ft(12.0),
        "h_m": ft(8.0),
    }
    shower_rect_m = {
        "x0_m": grid_m - ft(4.0),
        "y0_m": 0.0,
        "w_m": ft(4.0),
        "h_m": ft(4.0),
        "recess_m": inch(4.0),
    }

    # Attach a single JSON pset on the house building for plan-driven scripts.
    plan_params = {
        "house_size_m": house_size_m,
        "basement_elevation_m": basement_elev_m,
        "basement_wall_thickness_m": inch(spec.basement_wall_thickness_in),
        "basement_grid_m": grid_m,
        "stair_opening_rect_m": stair_opening_rect_m,
        "stair_opening_note": "Located on north side, immediately west of centerline wall (per house_description.md)",
        "sauna_rect_m": sauna_rect_m,
        "shower_rect_m": shower_rect_m,
        "sauna_shower_note": "Sauna+shower located west of centerline wall, against south wall (placeholder geometry)",
    }
    set_pset_json(f, product=house_bldg, pset_name="Pset_ifcPlot_BasementPlan", prop_name="ParamsJSON", value=plan_params)

    # ---- Geometry --------------------------------------------------------------
    def hx(x: float) -> float:
        return house_origin_x + float(x)

    def hy(y: float) -> float:
        return house_origin_y + float(y)

    # Basement: perimeter concrete walls (12") + center cross walls at 18' OC.
    conc_thk_m = inch(spec.basement_wall_thickness_in)
    perim = [
        ((hx(0.0), hy(0.0)), (hx(house_size_m), hy(0.0))),  # south (eastward)
        ((hx(house_size_m), hy(0.0)), (hx(house_size_m), hy(house_size_m))),  # east (northward)
        ((hx(house_size_m), hy(house_size_m)), (hx(0.0), hy(house_size_m))),  # north (westward)
        ((hx(0.0), hy(house_size_m)), (hx(0.0), hy(0.0))),  # west (southward)
    ]
    basement_walls = [
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_basement,
            name=f"House Basement Wall {i+1}",
            p1=p1,
            p2=p2,
            elevation=basement_elev_m,
            height=ft(spec.basement_clear_height_ft),
            thickness=conc_thk_m,
        )
        for i, (p1, p2) in enumerate(perim)
    ]

    # Basement exterior insulation: 4" XPS in 2 layers, over the exterior face of perimeter concrete walls.
    # Represented as separate IfcWall elements so they can be hidden/isolated similar to other envelope layers.
    basement_xps_layer_in = 2.0
    basement_xps_layers = 2
    basement_xps_layer_m = inch(basement_xps_layer_in)
    basement_xps_walls: list[Any] = []
    for i, (p1, p2) in enumerate(perim, start=1):
        outward_offset = 0.0
        for layer in range(1, basement_xps_layers + 1):
            xps = add_wall_between_points(
                f,
                context=contexts.body,
                storey=house_basement,
                name=f"House Basement Exterior XPS L{layer} {i}",
                p1=p1,
                p2=p2,
                elevation=basement_elev_m,
                height=ft(spec.basement_clear_height_ft),
                thickness=basement_xps_layer_m,
                direction_sense="NEGATIVE",  # outward (exterior)
                offset=-outward_offset,  # stack layers outward from the concrete exterior face
            )
            basement_xps_walls.append(xps)
            outward_offset += basement_xps_layer_m
    assign_to_group(f, group=groups["Cladding"], products=basement_xps_walls)
    for xps in basement_xps_walls:
        assign_surface_style(f, element=xps, style=xps_style)

    # Center cross walls (concrete), splitting basement into four squares.
    basement_walls.append(
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_basement,
            name="House Basement Center Wall (N-S)",
            p1=(hx(grid_m), hy(0.0)),
            p2=(hx(grid_m), hy(house_size_m)),
            elevation=basement_elev_m,
            height=ft(spec.basement_clear_height_ft),
            thickness=conc_thk_m,
        )
    )
    basement_walls.append(
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_basement,
            name="House Basement Center Wall (E-W)",
            p1=(hx(0.0), hy(grid_m)),
            p2=(hx(house_size_m), hy(grid_m)),
            elevation=basement_elev_m,
            height=ft(spec.basement_clear_height_ft),
            thickness=conc_thk_m,
        )
    )

    # Stair opening is on the north side, immediately west of the centerline wall.
    # Add the basement-specific 8" concrete wall immediately west of the opening.
    stair_side_wall_thk_m = inch(8.0)
    x_stair_west_m = grid_m - stair_w_m
    y_stair_south_m = house_size_m - stair_d_m
    basement_walls.append(
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_basement,
            name='House Basement Stair Side Wall (8")',
            p1=(hx(x_stair_west_m), hy(y_stair_south_m)),
            p2=(hx(x_stair_west_m), hy(house_size_m)),
            elevation=basement_elev_m,
            height=ft(spec.basement_clear_height_ft),
            thickness=stair_side_wall_thk_m,
        )
    )
    assign_to_group(f, group=groups["Concrete"], products=basement_walls)
    for wall in basement_walls:
        assign_surface_style(f, element=wall, style=concrete_style)

    # Basement slab (+ placeholder shower recess)
    basement_slab = add_slab(
        f,
        context=contexts.body,
        storey=house_basement,
        name="House Basement Slab",
        polyline=_rect_polyline_xy((hx(0.0), hy(0.0)), (house_size_m, house_size_m)),
        elevation=basement_elev_m,
        depth=inch(spec.basement_slab_thickness_in),
        predefined_type="FLOOR",
    )
    shower_recess = add_slab(
        f,
        context=contexts.body,
        storey=house_basement,
        name="Basement Shower Recess (placeholder)",
        polyline=_rect_polyline_xy(
            (hx(grid_m - ft(4.0)), hy(0.0)),
            (ft(4.0), ft(4.0)),
        ),
        elevation=basement_elev_m - inch(4.0),
        depth=inch(spec.basement_slab_thickness_in),
        predefined_type="FLOOR",
    )
    assign_to_group(f, group=groups["Concrete"], products=[basement_slab, shower_recess])
    assign_surface_style(f, element=basement_slab, style=concrete_style)
    assign_surface_style(f, element=shower_recess, style=concrete_style)

    # Main floor concrete ceiling slab (9" thick, at elevation 0)
    main_floor_slab = add_slab(
        f,
        context=contexts.body,
        storey=house_main,
        name="House Main Floor Concrete Slab",
        polyline=_rect_polyline_xy((hx(0.0), hy(0.0)), (house_size_m, house_size_m)),
        elevation=main_elev_m - inch(spec.basement_ceiling_slab_thickness_in),
        depth=inch(spec.basement_ceiling_slab_thickness_in),
        predefined_type="FLOOR",
    )
    assign_to_group(f, group=groups["Concrete"], products=[main_floor_slab])
    assign_surface_style(f, element=main_floor_slab, style=concrete_style)

    # Upper floor coverings: drywall (5/8") + subfloor (3/4") + carpet (1/4")
    drywall_thk_m = inch(0.625)
    subfloor_thk_m = inch(0.75)
    carpet_thk_m = inch(0.25)

    # Second floor covering (below second floor joists, above main floor)
    second_floor_drywall = add_slab(
        f,
        context=contexts.body,
        storey=house_main,
        name="House Main Floor Ceiling Drywall",
        polyline=_rect_polyline_xy((hx(0.0), hy(0.0)), (house_size_m, house_size_m)),
        elevation=second_elev_m - inch(spec.floor_joist_depth_in) - drywall_thk_m,
        depth=drywall_thk_m,
        predefined_type="FLOOR",
    )
    second_floor_subfloor = add_slab(
        f,
        context=contexts.body,
        storey=house_second,
        name="House Second Floor Subfloor",
        polyline=_rect_polyline_xy((hx(0.0), hy(0.0)), (house_size_m, house_size_m)),
        elevation=second_elev_m,
        depth=subfloor_thk_m,
        predefined_type="FLOOR",
    )
    second_floor_carpet = add_slab(
        f,
        context=contexts.body,
        storey=house_second,
        name="House Second Floor Carpet",
        polyline=_rect_polyline_xy((hx(0.0), hy(0.0)), (house_size_m, house_size_m)),
        elevation=second_elev_m + subfloor_thk_m,
        depth=carpet_thk_m,
        predefined_type="FLOOR",
    )
    assign_to_group(f, group=groups["Drywall"], products=[second_floor_drywall])
    assign_to_group(f, group=groups["Framing"], products=[second_floor_subfloor, second_floor_carpet])
    assign_surface_style(f, element=second_floor_drywall, style=drywall_style)
    assign_surface_style(f, element=second_floor_subfloor, style=framing_wood_style)
    assign_surface_style(f, element=second_floor_carpet, style=framing_wood_style)

    # Attic floor covering (below attic floor joists, above second floor)
    attic_floor_drywall = add_slab(
        f,
        context=contexts.body,
        storey=house_second,
        name="House Second Floor Ceiling Drywall",
        polyline=_rect_polyline_xy((hx(0.0), hy(0.0)), (house_size_m, house_size_m)),
        elevation=attic_elev_m - inch(spec.floor_joist_depth_in) - drywall_thk_m,
        depth=drywall_thk_m,
        predefined_type="FLOOR",
    )
    attic_floor_subfloor = add_slab(
        f,
        context=contexts.body,
        storey=house_attic,
        name="House Attic Floor Subfloor",
        polyline=_rect_polyline_xy((hx(0.0), hy(0.0)), (house_size_m, house_size_m)),
        elevation=attic_elev_m,
        depth=subfloor_thk_m,
        predefined_type="FLOOR",
    )
    assign_to_group(f, group=groups["Drywall"], products=[attic_floor_drywall])
    assign_to_group(f, group=groups["Framing"], products=[attic_floor_subfloor])
    assign_surface_style(f, element=attic_floor_drywall, style=drywall_style)
    assign_surface_style(f, element=attic_floor_subfloor, style=framing_wood_style)

    # ---- Exterior Wall Layers (siding stack) ----------------------------------
    # Mirror the wall-side layers used in `catlin_house/roof_wall_eave_detail_ifc.py`.
    wall_sheathing_m = inch(0.625)
    wall_polyiso_m = inch(2.0)
    wall_eps_m = inch(2.0)
    wall_furring_m = inch(0.5)
    wall_metal_m = inch(0.5)

    # ---- Detail parameter sets ------------------------------------------------
    # Store key construction/detail parameters as JSON on the House building so matplotlib detail
    # scripts can read the same "source of truth" as the IFC geometry generator.
    basement_construction_params = {
        "foundation": {
            "wall_thickness_in": spec.basement_wall_thickness_in,
            "footing_width_in": 20.0,
            "footing_thickness_in": 8.0,
            "stone_base_thickness_in": 6.0,
            "stone_base_extra_in": 8.0,
            "french_drain_diameter_in": 4.0,
            "river_rock_depth_in": 8.0,
            "river_rock_width_in": 10.0,
        },
        "slab": {
            "slab_thickness_in": spec.basement_slab_thickness_in,
            "xps_under_in": 2.0,
            "vapor_barrier_in": 0.05,  # schematic thickness for plotting
            "stone_under_in": 4.0,
            "thermal_break_in": 1.0,
            "sealant_in": 0.5,
        },
    }
    set_pset_json(
        f,
        product=house_bldg,
        pset_name="Pset_ifcPlot_BasementConstruction",
        prop_name="ParamsJSON",
        value=basement_construction_params,
    )

    basement_to_framed_wall_detail_params = {
        "wall": {
            "drywall_in": 0.625,
            "stud_depth_in": 5.5,
            "sheathing_in": 0.625,
            "membrane_in": 0.6,  # exaggerated for visibility in 2D details
            "polyiso_in": 2.0,
            "eps_in": 2.0,
            "furring_in": 0.5,
            "cladding_in": 0.5,
        },
        "basement_exterior": {
            "xps_layer_in": 2.0,
            "xps_layers": 2,
            "xps_protect_in": 0.5,
        },
        "sill": {"gasket_in": 0.25, "plate_in": 1.5},
        "detail_view": {
            "grade_offset_in": 12.0,
            "basement_above_grade_in": 24.0,
            "basement_wall_show_height_in": 48.0,
            "slab_show_width_in": 18.0,
        },
    }
    set_pset_json(
        f,
        product=house_bldg,
        pset_name="Pset_ifcPlot_BasementToFramedWallDetail",
        prop_name="ParamsJSON",
        value=basement_to_framed_wall_detail_params,
    )

    sauna_detail_params = {
        "finish": {
            "polyiso_in": 2.0,
            "furring_in": 0.5,
            "tg_in": 1.0,
            "baseboard_height_in": 6.0,
            "membrane_in": 0.25,
            "flashing_in": 0.35,
        },
        "structure": {
            "clear_height_in": spec.basement_clear_height_ft * 12.0,
            "ceiling_slab_in": spec.basement_ceiling_slab_thickness_in,
            "drop_ceiling_depth_in": 3.5,
            "drop_ceiling_gap_in": 1.0,
        },
        "adjacent_wall": {"stud_depth_in": 3.5, "drywall_in": 0.625, "gap_to_concrete_in": 0.5},
        "benches": {"depth_in": 20.0, "thickness_in": 1.5, "lower_top_in": 18.0, "upper_top_in": 36.0},
        "heater": {"width_in": 10.0, "height_in": 18.0},
        "sauna_floor_slope": {"rise_in": 1.0, "run_ft": 8.0},  # 1/8" per ft over 8'
        "shower": {
            "recess_in": 4.0,
            "tile_in": 0.75,
            "wall_backer_in": 1.0,
            "glass_in": 0.5,
            "glass_gap_in": 1.0,
            "hrv_duct_in": 3.0,
        },
    }
    set_pset_json(
        f,
        product=house_bldg,
        pset_name="Pset_ifcPlot_SaunaShowerDetail",
        prop_name="ParamsJSON",
        value=sauna_detail_params,
    )

    def add_layered_wall_segments(
        *,
        storey: Any,
        elevation_m: float,
        height_m: float,
        stud_depth_m: float,
        segments: list[tuple[tuple[float, float], tuple[float, float]]],
        label: str,
    ) -> dict[str, list[Any]]:
        created: dict[str, list[Any]] = {"studs": [], "sheathing": [], "polyiso": [], "eps": [], "furring": [], "metal": []}
        for i, (p1, p2) in enumerate(segments, start=1):
            sheathing = add_wall_between_points(
                f,
                context=contexts.body,
                storey=storey,
                name=f"House {label} Exterior Sheathing {i}",
                p1=p1,
                p2=p2,
                elevation=elevation_m,
                height=height_m,
                thickness=wall_sheathing_m,
                direction_sense="POSITIVE",  # into house
                offset=0.0,  # starts at outer face
            )
            studs = add_wall_between_points(
                f,
                context=contexts.body,
                storey=storey,
                name=f"House {label} Stud Wall {i}",
                p1=p1,
                p2=p2,
                elevation=elevation_m,
                height=height_m,
                thickness=stud_depth_m,
                direction_sense="POSITIVE",  # into house
                offset=wall_sheathing_m,  # behind sheathing
            )
            assign_surface_style(f, element=studs, style=framing_wood_style)

            created["sheathing"].append(sheathing)
            created["studs"].append(studs)

            # Outside of sheathing: polyiso + eps + furring + standing seam (stacked outward).
            outward_offset = 0.0

            polyiso = add_wall_between_points(
                f,
                context=contexts.body,
                storey=storey,
                name=f"House {label} Exterior Polyiso {i}",
                p1=p1,
                p2=p2,
                elevation=elevation_m,
                height=height_m,
                thickness=wall_polyiso_m,
                direction_sense="NEGATIVE",  # outward
                offset=-outward_offset,
            )
            outward_offset += wall_polyiso_m

            eps = add_wall_between_points(
                f,
                context=contexts.body,
                storey=storey,
                name=f"House {label} Exterior EPS {i}",
                p1=p1,
                p2=p2,
                elevation=elevation_m,
                height=height_m,
                thickness=wall_eps_m,
                direction_sense="NEGATIVE",  # outward
                offset=-outward_offset,
            )
            outward_offset += wall_eps_m

            furring = add_wall_between_points(
                f,
                context=contexts.body,
                storey=storey,
                name=f"House {label} Exterior Furring {i}",
                p1=p1,
                p2=p2,
                elevation=elevation_m,
                height=height_m,
                thickness=wall_furring_m,
                direction_sense="NEGATIVE",  # outward
                offset=-outward_offset,
            )
            outward_offset += wall_furring_m

            metal = add_wall_between_points(
                f,
                context=contexts.body,
                storey=storey,
                name=f"House {label} Exterior Standing Seam {i}",
                p1=p1,
                p2=p2,
                elevation=elevation_m,
                height=height_m,
                thickness=wall_metal_m,
                direction_sense="NEGATIVE",  # outward
                offset=-outward_offset,
            )
            assign_surface_style(f, element=metal, style=standing_seam_style)

            created["polyiso"].append(polyiso)
            created["eps"].append(eps)
            created["furring"].append(furring)
            created["metal"].append(metal)

        assign_to_group(f, group=groups["Framing"], products=[*created["studs"], *created["sheathing"]])
        assign_to_group(f, group=groups["Cladding"], products=[*created["polyiso"], *created["eps"], *created["furring"], *created["metal"]])
        for elem in created["sheathing"]:
            assign_surface_style(f, element=elem, style=sheathing_style)
        for elem in created["polyiso"]:
            assign_surface_style(f, element=elem, style=polyiso_style)
        for elem in created["eps"]:
            assign_surface_style(f, element=elem, style=eps_style)
        for elem in created["furring"]:
            assign_surface_style(f, element=elem, style=framing_wood_style)
        return created

    perimeter_segments = [
        ((hx(0.0), hy(0.0)), (hx(house_size_m), hy(0.0))),  # south (eastward)
        ((hx(house_size_m), hy(0.0)), (hx(house_size_m), hy(house_size_m))),  # east (northward)
        ((hx(house_size_m), hy(house_size_m)), (hx(0.0), hy(house_size_m))),  # north (westward)
        ((hx(0.0), hy(house_size_m)), (hx(0.0), hy(0.0))),  # west (southward)
    ]

    add_layered_wall_segments(
        storey=house_main,
        elevation_m=main_elev_m,
        height_m=ft(spec.main_storey_height_ft),
        stud_depth_m=inch(5.5),
        segments=perimeter_segments,
        label="Main",
    )
    add_layered_wall_segments(
        storey=house_second,
        elevation_m=second_elev_m,
        height_m=ft(spec.second_storey_height_ft),
        stud_depth_m=inch(3.5),
        segments=perimeter_segments,
        label="Second",
    )

    # Attic knee walls (east/west only). Extend the siding stack upward to cover roof joists.
    attic_knee_height_m = ft(spec.attic_knee_wall_height_ft)
    attic_knee_siding_height_m = attic_knee_height_m + ft(spec.attic_knee_siding_extra_height_ft)
    attic_knee_segments = [
        ((hx(house_size_m), hy(0.0)), (hx(house_size_m), hy(house_size_m))),  # east
        ((hx(0.0), hy(house_size_m)), (hx(0.0), hy(0.0))),  # west
    ]
    attic_knee_layers: dict[str, list[Any]] = {"studs": [], "sheathing": [], "polyiso": [], "eps": [], "furring": [], "metal": []}
    for i, (p1, p2) in enumerate(attic_knee_segments, start=1):
        sheathing = add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name=f"House Attic Knee Exterior Sheathing {i}",
            p1=p1,
            p2=p2,
            elevation=attic_elev_m,
            height=attic_knee_siding_height_m,
            thickness=wall_sheathing_m,
            direction_sense="POSITIVE",  # into house
            offset=0.0,  # starts at outer face
        )
        studs = add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name=f"House Attic Knee Stud Wall {i}",
            p1=p1,
            p2=p2,
            elevation=attic_elev_m,
            height=attic_knee_height_m,
            thickness=inch(3.5),
            direction_sense="POSITIVE",  # into house
            offset=wall_sheathing_m,  # behind sheathing
        )

        outward_offset = 0.0
        polyiso = add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name=f"House Attic Knee Exterior Polyiso {i}",
            p1=p1,
            p2=p2,
            elevation=attic_elev_m,
            height=attic_knee_siding_height_m,
            thickness=wall_polyiso_m,
            direction_sense="NEGATIVE",  # outward
            offset=-outward_offset,
        )
        outward_offset += wall_polyiso_m

        eps = add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name=f"House Attic Knee Exterior EPS {i}",
            p1=p1,
            p2=p2,
            elevation=attic_elev_m,
            height=attic_knee_siding_height_m,
            thickness=wall_eps_m,
            direction_sense="NEGATIVE",  # outward
            offset=-outward_offset,
        )
        outward_offset += wall_eps_m

        furring = add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name=f"House Attic Knee Exterior Furring {i}",
            p1=p1,
            p2=p2,
            elevation=attic_elev_m,
            height=attic_knee_siding_height_m,
            thickness=wall_furring_m,
            direction_sense="NEGATIVE",  # outward
            offset=-outward_offset,
        )
        outward_offset += wall_furring_m

        metal = add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name=f"House Attic Knee Exterior Standing Seam {i}",
            p1=p1,
            p2=p2,
            elevation=attic_elev_m,
            height=attic_knee_siding_height_m,
            thickness=wall_metal_m,
            direction_sense="NEGATIVE",  # outward
            offset=-outward_offset,
        )

        attic_knee_layers["sheathing"].append(sheathing)
        attic_knee_layers["studs"].append(studs)
        attic_knee_layers["polyiso"].append(polyiso)
        attic_knee_layers["eps"].append(eps)
        attic_knee_layers["furring"].append(furring)
        attic_knee_layers["metal"].append(metal)

    assign_to_group(f, group=groups["Framing"], products=[*attic_knee_layers["studs"], *attic_knee_layers["sheathing"]])
    assign_to_group(f, group=groups["Cladding"], products=[*attic_knee_layers["polyiso"], *attic_knee_layers["eps"], *attic_knee_layers["furring"], *attic_knee_layers["metal"]])
    for elem in attic_knee_layers["studs"]:
        assign_surface_style(f, element=elem, style=framing_wood_style)
    for elem in attic_knee_layers["sheathing"]:
        assign_surface_style(f, element=elem, style=sheathing_style)
    for elem in attic_knee_layers["polyiso"]:
        assign_surface_style(f, element=elem, style=polyiso_style)
    for elem in attic_knee_layers["eps"]:
        assign_surface_style(f, element=elem, style=eps_style)
    for elem in attic_knee_layers["furring"]:
        assign_surface_style(f, element=elem, style=framing_wood_style)
    for elem in attic_knee_layers["metal"]:
        assign_surface_style(f, element=elem, style=standing_seam_style)

    # Attic gable ends: build the same layer stack as prisms so siding continues to the ridge.
    knee_wall_h_m = ft(spec.attic_knee_wall_height_ft)
    ridge_h_m = ft(spec.attic_ridge_height_above_floor_ft)
    gable_extra_h_m = ft(spec.attic_gable_extra_height_ft)
    gable_siding_overlap_m = inch(spec.attic_gable_siding_overlap_in)
    gable_knee_h_m = knee_wall_h_m + gable_extra_h_m
    gable_ridge_h_m = ridge_h_m + gable_extra_h_m

    gable_framing_profile = [
        (0.0, 0.0),
        (house_size_m, 0.0),
        (house_size_m, -gable_knee_h_m),
        (house_size_m / 2.0, -gable_ridge_h_m),
        (0.0, -gable_knee_h_m),
    ]
    gable_siding_profile = [
        (-gable_siding_overlap_m, 0.0),
        (house_size_m + gable_siding_overlap_m, 0.0),
        (house_size_m + gable_siding_overlap_m, -gable_knee_h_m),
        (house_size_m / 2.0, -gable_ridge_h_m),
        (-gable_siding_overlap_m, -gable_knee_h_m),
    ]

    def shifted_along_thickness(base_matrix: np.ndarray, dist_m: float) -> np.ndarray:
        m = base_matrix.copy()
        m[0:3, 3] = m[0:3, 3] + float(dist_m) * m[0:3, 2]
        return m

    def add_gable_layer_stack(*, base_matrix: np.ndarray, label: str) -> None:
        # Outside of sheathing (negative dist) → sheathing (0) → studs (positive dist).
        outward_layers = [
            ("Exterior Polyiso", wall_polyiso_m),
            ("Exterior EPS", wall_eps_m),
            ("Exterior Furring", wall_furring_m),
            ("Exterior Standing Seam", wall_metal_m),
        ]
        outward_cum = 0.0
        for name, thk in outward_layers:
            outward_cum += float(thk)
            el = add_prism_from_profile(
                f,
                context=contexts.body,
                storey=house_attic,
                ifc_class="IfcWall",
                name=f"House Attic {label} {name}",
                profile_points=gable_siding_profile,
                depth=float(thk),
                placement_matrix=shifted_along_thickness(base_matrix, -outward_cum),
            )
            if "Standing Seam" in name:
                assign_surface_style(f, element=el, style=standing_seam_style)
            assign_to_group(f, group=groups["Cladding"], products=[el])

        sheathing = add_prism_from_profile(
            f,
            context=contexts.body,
            storey=house_attic,
            ifc_class="IfcWall",
            name=f"House Attic {label} Exterior Sheathing",
            profile_points=gable_siding_profile,
            depth=float(wall_sheathing_m),
            placement_matrix=base_matrix,
        )
        assign_surface_style(f, element=sheathing, style=sheathing_style)
        studs = add_prism_from_profile(
            f,
            context=contexts.body,
            storey=house_attic,
            ifc_class="IfcWall",
            name=f"House Attic {label} Stud Wall",
            profile_points=gable_framing_profile,
            depth=float(inch(3.5)),
            placement_matrix=shifted_along_thickness(base_matrix, float(wall_sheathing_m)),
        )
        assign_surface_style(f, element=studs, style=framing_wood_style)
        assign_to_group(f, group=groups["Framing"], products=[sheathing, studs])

    south_gable_matrix = placement_matrix(
        origin=(0.0, 0.0, 0.0),
        x_axis=(1.0, 0.0, 0.0),
        z_axis=(0.0, 1.0, 0.0),  # into house (+Y)
    )
    north_gable_matrix = placement_matrix(
        origin=(house_size_m, house_size_m, 0.0),
        x_axis=(-1.0, 0.0, 0.0),
        z_axis=(0.0, -1.0, 0.0),  # into house (-Y)
    )
    add_gable_layer_stack(base_matrix=south_gable_matrix, label="Gable South")
    add_gable_layer_stack(base_matrix=north_gable_matrix, label="Gable North")

    # House centerline load-bearing wall (runs N-S at x=18') for upper levels.
    center_wall_thk_m = inch(spec.centerline_wall_thickness_in)
    centerline_walls = [
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_main,
            name="House Centerline Wall (Main)",
            p1=(hx(grid_m), hy(0.0)),
            p2=(hx(grid_m), hy(house_size_m)),
            elevation=main_elev_m,
            height=ft(spec.main_storey_height_ft),
            thickness=center_wall_thk_m,
        ),
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_second,
            name="House Centerline Wall (Second)",
            p1=(hx(grid_m), hy(0.0)),
            p2=(hx(grid_m), hy(house_size_m)),
            elevation=second_elev_m,
            height=ft(spec.second_storey_height_ft),
            thickness=center_wall_thk_m,
        ),
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name="House Centerline Wall (Attic, ridge support)",
            p1=(hx(grid_m), hy(0.0)),
            p2=(hx(grid_m), hy(house_size_m)),
            elevation=attic_elev_m,
            height=ft(spec.attic_ridge_height_above_floor_ft),
            thickness=center_wall_thk_m,
        ),
    ]
    assign_to_group(f, group=groups["Framing"], products=centerline_walls)
    for wall in centerline_walls:
        assign_surface_style(f, element=wall, style=framing_wood_style)

    # Floor joists (IFC framing members): 16" o.c. spanning between side walls and the centerline wall.
    spacing_m = inch(spec.framing_spacing_in)
    joist_w_m = inch(spec.floor_joist_width_in)
    joist_d_m = inch(spec.floor_joist_depth_in)

    def y_positions_m() -> list[float]:
        ys: list[float] = []
        y = 0.0
        # Include both ends for now (rim joists / gable ends handled later).
        while y <= house_size_m + 1e-9:
            ys.append(float(y))
            y += spacing_m
        if ys[-1] < house_size_m - 1e-6:
            ys.append(float(house_size_m))
        return ys

    second_floor_joists: list[Any] = []
    attic_floor_joists: list[Any] = []
    for i, y in enumerate(y_positions_m(), start=1):
        # Joists relative to storey (elevations are local)
        z_center_second = -joist_d_m / 2.0
        second_floor_joists.append(
            add_rect_member_between_points(
                f,
                context=contexts.body,
                storey=house_second,
                name=f"House Second Floor Joist W-{i:02d}",
                p1=(0.0, y, float(z_center_second)),
                p2=(grid_m, y, float(z_center_second)),
                width=joist_w_m,
                depth=joist_d_m,
                predefined_type="JOIST",
                ifc_class="IfcBeam",
            )
        )
        second_floor_joists.append(
            add_rect_member_between_points(
                f,
                context=contexts.body,
                storey=house_second,
                name=f"House Second Floor Joist E-{i:02d}",
                p1=(grid_m, y, float(z_center_second)),
                p2=(house_size_m, y, float(z_center_second)),
                width=joist_w_m,
                depth=joist_d_m,
                predefined_type="JOIST",
                ifc_class="IfcBeam",
            )
        )

        z_center_attic = -joist_d_m / 2.0
        attic_floor_joists.append(
            add_rect_member_between_points(
                f,
                context=contexts.body,
                storey=house_attic,
                name=f"House Attic Floor Joist W-{i:02d}",
                p1=(0.0, y, float(z_center_attic)),
                p2=(grid_m, y, float(z_center_attic)),
                width=joist_w_m,
                depth=joist_d_m,
                predefined_type="JOIST",
                ifc_class="IfcBeam",
            )
        )
        attic_floor_joists.append(
            add_rect_member_between_points(
                f,
                context=contexts.body,
                storey=house_attic,
                name=f"House Attic Floor Joist E-{i:02d}",
                p1=(grid_m, y, float(z_center_attic)),
                p2=(house_size_m, y, float(z_center_attic)),
                width=joist_w_m,
                depth=joist_d_m,
                predefined_type="JOIST",
                ifc_class="IfcBeam",
            )
        )

    assign_to_group(f, group=groups["Framing"], products=[*second_floor_joists, *attic_floor_joists])
    for joist in [*second_floor_joists, *attic_floor_joists]:
        assign_surface_style(f, element=joist, style=framing_wood_style)

    # Optional grouping to keep joists tidy in viewers.
    second_floor_group = ifcopenshell.api.run("group.add_group", f, name="House Second Floor Joists")
    attic_floor_group = ifcopenshell.api.run("group.add_group", f, name="House Attic Floor Joists")
    assign_to_group(f, group=second_floor_group, products=second_floor_joists)
    assign_to_group(f, group=attic_floor_group, products=attic_floor_joists)

    # ---- House Roof Assembly ---------------------------------------------------
    pitch = spec.roof_pitch_rise_over_run
    eave_z_m = attic_elev_m + ft(spec.attic_knee_wall_height_ft)
    ridge_z_m = attic_elev_m + ft(spec.attic_ridge_height_above_floor_ft)
    overhang_m = inch(spec.roof_overhang_in)
    drop_m = pitch * overhang_m
    roof_p = {
        "pitch_rise_over_run": pitch,
        "ijoist_depth_in": spec.roof_joist_depth_in,
        "joist_width_in": spec.roof_joist_width_in,
        "joist_spacing_in": spec.framing_spacing_in,
        "sheathing_in": 0.75,
        "polyiso_in": 2.0,
        "eps_in": 4.0,
        "membrane_in": 0.25,
        "furring_in": 0.75,
        "metal_roof_in": 0.5,
        "overhang_in": spec.roof_overhang_in,
    }
    sheathing_m = inch(roof_p["sheathing_in"])
    polyiso_m = inch(roof_p["polyiso_in"])
    eps_m = inch(roof_p["eps_in"])
    membrane_m = inch(roof_p["membrane_in"])
    furring_m = inch(roof_p["furring_in"])
    metal_roof_m = inch(roof_p["metal_roof_in"])

    # Reference line: top of joists
    joist_depth_m = inch(spec.roof_joist_depth_in)

    # Z-coordinates are relative to the attic storey elevation
    z_w_eave = (eave_z_m - attic_elev_m) + joist_depth_m / 2.0
    z_w_ridge = (ridge_z_m - attic_elev_m) + joist_depth_m / 2.0
    z_e_eave = z_w_eave
    z_e_ridge = z_w_ridge

    # X-coordinates for the roof centerline segments
    x_ridge = house_size_m / 2.0
    x_w_eave = 0.0
    x_e_eave = house_size_m

    # Centerline segments for the roof slopes (in local XY plane for prism profile)
    mid0_w = np.array([x_w_eave - overhang_m, z_w_eave - drop_m])
    mid1_w = np.array([x_ridge, z_w_ridge])

    mid0_e = np.array([x_ridge, z_e_ridge])
    mid1_e = np.array([x_e_eave + overhang_m, z_e_eave - drop_m])

    roof_len_m = house_size_m + 2.0 * overhang_m

    # Matrix to place the roof layers
    roof_matrix = placement_matrix(
        origin=(0.0, -overhang_m, 0.0),
        x_axis=(1.0, 0.0, 0.0),  # local X -> world X
        z_axis=(0.0, 1.0, 0.0),  # local Z (extrusion) -> world Y
    )

    house_roof_elements = []

    def create_layer(name, mid0, mid1, offset, thick, x0, x1, matrix):
        poly = _layer_poly(mid0, mid1, offset, thick, x0, x1)
        el = add_prism_from_profile(
            f, context=contexts.body, storey=house_attic,
            ifc_class="IfcRoof", name=name,
            profile_points=poly, depth=roof_len_m,
            placement_matrix=matrix,
        )
        house_roof_elements.append(el)
        return el

    # West Roof Layers
    offset = sheathing_m / 2.0
    sheath_w = create_layer("House Roof Sheathing (W)", mid0_w, mid1_w, offset, sheathing_m, mid0_w[0], mid1_w[0], roof_matrix)
    
    offset += sheathing_m / 2.0 + membrane_m / 2.0
    mem1_w = create_layer("House Roof Membrane 1 (W)", mid0_w, mid1_w, offset, membrane_m, mid0_w[0], mid1_w[0], roof_matrix)

    offset += membrane_m / 2.0 + polyiso_m / 2.0
    poly_w = create_layer("House Roof Polyiso (W)", mid0_w, mid1_w, offset, polyiso_m, mid0_w[0], mid1_w[0], roof_matrix)

    offset += polyiso_m / 2.0 + eps_m / 2.0
    eps_w = create_layer("House Roof EPS (W)", mid0_w, mid1_w, offset, eps_m, mid0_w[0], mid1_w[0], roof_matrix)
    
    offset += eps_m / 2.0 + membrane_m / 2.0
    mem2_w = create_layer("House Roof Membrane 2 (W)", mid0_w, mid1_w, offset, membrane_m, mid0_w[0], mid1_w[0], roof_matrix)

    offset += membrane_m / 2.0 + furring_m / 2.0
    furring_w = create_layer("House Roof Furring (W)", mid0_w, mid1_w, offset, furring_m, mid0_w[0], mid1_w[0], roof_matrix)
    
    offset += furring_m / 2.0 + metal_roof_m / 2.0
    metal_w = create_layer("House Roof Cladding (W)", mid0_w, mid1_w, offset, metal_roof_m, mid0_w[0], mid1_w[0], roof_matrix)
    assign_surface_style(f, element=metal_w, style=standing_seam_style)
    
    # East Roof Layers
    offset = sheathing_m / 2.0
    sheath_e = create_layer("House Roof Sheathing (E)", mid0_e, mid1_e, offset, sheathing_m, mid0_e[0], mid1_e[0], roof_matrix)

    offset += sheathing_m / 2.0 + membrane_m / 2.0
    mem1_e = create_layer("House Roof Membrane 1 (E)", mid0_e, mid1_e, offset, membrane_m, mid0_e[0], mid1_e[0], roof_matrix)
    
    offset += membrane_m / 2.0 + polyiso_m / 2.0
    poly_e = create_layer("House Roof Polyiso (E)", mid0_e, mid1_e, offset, polyiso_m, mid0_e[0], mid1_e[0], roof_matrix)
    
    offset += polyiso_m / 2.0 + eps_m / 2.0
    eps_e = create_layer("House Roof EPS (E)", mid0_e, mid1_e, offset, eps_m, mid0_e[0], mid1_e[0], roof_matrix)

    offset += eps_m / 2.0 + membrane_m / 2.0
    mem2_e = create_layer("House Roof Membrane 2 (E)", mid0_e, mid1_e, offset, membrane_m, mid0_e[0], mid1_e[0], roof_matrix)

    offset += membrane_m / 2.0 + furring_m / 2.0
    furring_e = create_layer("House Roof Furring (E)", mid0_e, mid1_e, offset, furring_m, mid0_e[0], mid1_e[0], roof_matrix)
    
    offset += furring_m / 2.0 + metal_roof_m / 2.0
    metal_e = create_layer("House Roof Cladding (E)", mid0_e, mid1_e, offset, metal_roof_m, mid0_e[0], mid1_e[0], roof_matrix)
    assign_surface_style(f, element=metal_e, style=standing_seam_style)

    assign_to_group(f, group=groups["Framing"], products=[sheath_w, sheath_e, furring_w, furring_e])
    assign_to_group(f, group=groups["Cladding"], products=[metal_w, metal_e])
    
    # Assign a primary roof element for property sets
    house_roof = sheath_w

    # Material styles for roof layers (sheathing/membranes/foam/furring).
    for elem in [sheath_w, sheath_e]:
        assign_surface_style(f, element=elem, style=sheathing_style)
    for elem in [mem1_w, mem2_w, mem1_e, mem2_e]:
        assign_surface_style(f, element=elem, style=membrane_style)
    for elem in [poly_w, poly_e]:
        assign_surface_style(f, element=elem, style=polyiso_style)
    for elem in [eps_w, eps_e]:
        assign_surface_style(f, element=elem, style=eps_style)
    for elem in [furring_w, furring_e]:
        assign_surface_style(f, element=elem, style=framing_wood_style)

    # Roof joists (IFC framing members): same spacing, 4:12 slope, spanning ridge to side walls.
    roof_joist_w_m = inch(spec.roof_joist_width_in)
    roof_joist_d_m = inch(spec.roof_joist_depth_in)
    eave_z_m = attic_elev_m + ft(spec.attic_knee_wall_height_ft)
    ridge_z_m = attic_elev_m + ft(spec.attic_ridge_height_above_floor_ft)

    roof_joists: list[Any] = []
    for i, y in enumerate(y_positions_m(), start=1):
        # Local to Attic Storey
        z_ridge_local = ridge_z_m - attic_elev_m
        z_eave_local = eave_z_m - attic_elev_m

        roof_joists.append(
            add_rect_member_between_points(
                f,
                context=contexts.body,
                storey=house_attic,
                name=f"House Roof Joist W-{i:02d}",
                p1=(grid_m, y, float(z_ridge_local)),
                p2=(0.0, y, float(z_eave_local)),
                width=roof_joist_w_m,
                depth=roof_joist_d_m,
                predefined_type="JOIST",
                ifc_class="IfcBeam",
            )
        )
        roof_joists.append(
            add_rect_member_between_points(
                f,
                context=contexts.body,
                storey=house_attic,
                name=f"House Roof Joist E-{i:02d}",
                p1=(grid_m, y, float(z_ridge_local)),
                p2=(house_size_m, y, float(z_eave_local)),
                width=roof_joist_w_m,
                depth=roof_joist_d_m,
                predefined_type="JOIST",
                ifc_class="IfcBeam",
            )
        )
    assign_to_group(f, group=groups["Framing"], products=roof_joists)
    for joist in roof_joists:
        assign_surface_style(f, element=joist, style=framing_wood_style)
    roof_joist_group = ifcopenshell.api.run("group.add_group", f, name="House Roof Joists")
    assign_to_group(f, group=roof_joist_group, products=roof_joists)

    # Detail parameters stored on the roof (for IFC-driven detail scripts).
    roof_detail_params = {
        "wall": {
            "drywall_in": 0.625,
            "stud_depth_in": 3.5,
            "sheathing_in": 0.625,
            "polyiso_in": 2.0,
            "eps_in": 2.0,
            "furring_in": 0.5,
            "cladding_in": 0.5,
        },
        "roof": {
            "pitch_rise_over_run": pitch,
            "ijoist_depth_in": 11.875,
            "joist_width_in": spec.roof_joist_width_in,
            "joist_spacing_in": spec.framing_spacing_in,
            "sheathing_in": 0.75,
            "polyiso_in": 2.0,
            "eps_in": 4.0,
            "membrane_in": 0.25,
            "furring_in": 0.75,
            "metal_roof_in": 0.5,
            "overhang_in": spec.roof_overhang_in,
        },
    }
    if house_roof:
        set_pset_json(
            f,
            product=house_roof,
            pset_name="Pset_ifcPlot_DetailParams",
            prop_name="ParamsJSON",
            value=roof_detail_params,
        )

    # House-level framing parameters for reuse in scripts.
    framing_params = {
        "spacing_in": spec.framing_spacing_in,
        "centerline_wall_thickness_in": spec.centerline_wall_thickness_in,
        "floor_joists": {"width_in": spec.floor_joist_width_in, "depth_in": spec.floor_joist_depth_in},
        "roof_joists": {"width_in": spec.roof_joist_width_in, "depth_in": spec.roof_joist_depth_in, "pitch_rise_over_run": pitch},
    }
    set_pset_json(f, product=house_bldg, pset_name="Pset_ifcPlot_HouseFraming", prop_name="ParamsJSON", value=framing_params)

    # ---- Sunken Garden + Porch/Deck (WIP) ------------------------------------
    # Model is intentionally parameterized (SunkenGardenSpec) to keep iteration easy.
    sg_clear_w_m = ft(sunken.clear_width_ft)
    sg_clear_l_m = ft(sunken.clear_length_ft)
    porch_clear_d_m = ft(sunken.porch_clear_depth_ft)
    sg_gap_m = inch(sunken.gap_to_house_in)

    sg_wall_thk_m = inch(sunken.wall_thickness_in)

    sg_footing_thk_m = inch(sunken.footing_thickness_in)
    sg_toe_m = inch(sunken.footing_toe_in)
    sg_heel_m = inch(sunken.footing_heel_in)
    # The house-side wall has only a ~5" gap, so clamp the heel there to avoid running under the house.
    sg_heel_house_side_m = min(sg_heel_m, sg_gap_m)

    sg_agg_thk_m = inch(sunken.aggregate_thickness_in)
    sg_agg_extra_m = inch(sunken.aggregate_extra_in)

    deck_thk_m = inch(sunken.deck_thickness_in)
    rail_h_m = inch(sunken.railing_height_in)
    rail_thk_m = inch(sunken.railing_thickness_in)

    # Reference elevations (global Z)
    sg_floor_top_m = basement_elev_m + inch(spec.basement_slab_thickness_in)  # top of T footing
    porch_floor_top_m = main_elev_m  # align with house main storey
    deck_floor_top_m = second_elev_m  # align with house second storey

    # Wall heights
    sg_retaining_h_m = porch_floor_top_m - sg_floor_top_m
    sg_box_h_m = deck_floor_top_m - sg_floor_top_m
    if sg_retaining_h_m <= 0:
        raise ValueError("Sunken garden retaining wall height must be positive")
    if sg_box_h_m <= 0:
        raise ValueError("Sunken garden porch box height must be positive")

    # Plan layout: centered on house south side.
    x_center_m = house_size_m / 2.0
    x_in0_m = x_center_m - sg_clear_w_m / 2.0
    x_in1_m = x_center_m + sg_clear_w_m / 2.0
    x_out0_m = x_in0_m - sg_wall_thk_m
    x_out1_m = x_in1_m + sg_wall_thk_m

    y_north_out_m = -sg_gap_m
    y_north_in_m = y_north_out_m - sg_wall_thk_m
    y_south_in_m = y_north_in_m - sg_clear_l_m
    y_south_out_m = y_south_in_m - sg_wall_thk_m

    # Porch box (two-story) occupies the first `porch_clear_d_m` of the sunken garden length.
    y_box_south_in_m = y_north_in_m - porch_clear_d_m
    y_box_south_out_m = y_box_south_in_m - sg_wall_thk_m

    # Store parameters on the porch building for future detail scripts.
    set_pset_json(
        f,
        product=porch_bldg,
        pset_name="Pset_ifcPlot_SunkenGarden",
        prop_name="ParamsJSON",
        value={
            "clear_width_m": sg_clear_w_m,
            "clear_length_m": sg_clear_l_m,
            "porch_clear_depth_m": porch_clear_d_m,
            "gap_to_house_m": sg_gap_m,
            "wall_thickness_m": sg_wall_thk_m,
            "elevations_m": {
                "t_footing_top_m": sg_floor_top_m,
                "porch_floor_m": porch_floor_top_m,
                "deck_floor_m": deck_floor_top_m,
            },
            "footing_m": {
                "thickness_m": sg_footing_thk_m,
                "toe_m": sg_toe_m,
                "heel_m": sg_heel_m,
                "heel_house_side_m": sg_heel_house_side_m,
            },
            "aggregate_m": {"thickness_m": sg_agg_thk_m, "extra_m": sg_agg_extra_m},
            "arches": {
                "arches_per_wall": sunken.arches_per_wall,
                "arch_clear_width_m": ft(sunken.arch_clear_width_ft),
                "arch_outer_pier_m": ft(sunken.arch_outer_pier_ft),
                "arch_opening_height_m": ft(sunken.arch_opening_height_ft),
            },
        },
    )

    sg_concrete: list[Any] = []
    sg_aggregate: list[Any] = []
    sg_framing: list[Any] = []

    def _add_footing_and_aggregate(*, name: str, x0_m: float, y0_m: float, w_m: float, h_m: float) -> None:
        if w_m <= 0 or h_m <= 0:
            raise ValueError(f"Invalid footing footprint for {name}")

        agg = add_prism_from_profile(
            f,
            context=contexts.body,
            storey=porch_sunken,
            ifc_class="IfcBuildingElementProxy",
            name=f"{name} Compacted Aggregate",
            profile_points=_rect_polyline_xy((0.0, 0.0), (w_m + 2.0 * sg_agg_extra_m, h_m + 2.0 * sg_agg_extra_m)),
            depth=sg_agg_thk_m,
            placement_matrix=translation_matrix(
                hx(x0_m - sg_agg_extra_m),
                hy(y0_m - sg_agg_extra_m),
                sg_floor_top_m - sg_footing_thk_m - sg_agg_thk_m,
            ),
            placement_is_storey_relative=False,
        )
        sg_aggregate.append(agg)

        footing = add_prism_from_profile(
            f,
            context=contexts.body,
            storey=porch_sunken,
            ifc_class="IfcFooting",
            name=f"{name} Footing",
            profile_points=_rect_polyline_xy((0.0, 0.0), (w_m, h_m)),
            depth=sg_footing_thk_m,
            placement_matrix=translation_matrix(
                hx(x0_m),
                hy(y0_m),
                sg_floor_top_m - sg_footing_thk_m,
            ),
            placement_is_storey_relative=False,
            predefined_type="STRIP_FOOTING",
        )
        sg_concrete.append(footing)

    # Footings (plan rectangles), derived from the retaining wall detail script.
    # North wall (house-side) footing: clamp heel to avoid running under the house.
    _add_footing_and_aggregate(
        name="Sunken Garden North Wall",
        x0_m=x_out0_m,
        y0_m=y_north_in_m - sg_toe_m,
        w_m=x_out1_m - x_out0_m,
        h_m=(y_north_out_m + sg_heel_house_side_m) - (y_north_in_m - sg_toe_m),
    )
    _add_footing_and_aggregate(
        name="Sunken Garden South Wall",
        x0_m=x_out0_m,
        y0_m=y_south_out_m - sg_heel_m,
        w_m=x_out1_m - x_out0_m,
        h_m=(y_south_in_m + sg_toe_m) - (y_south_out_m - sg_heel_m),
    )
    _add_footing_and_aggregate(
        name="Sunken Garden West Wall",
        x0_m=x_out0_m - sg_heel_m,
        y0_m=y_south_out_m,
        w_m=(x_in0_m + sg_toe_m) - (x_out0_m - sg_heel_m),
        h_m=y_north_out_m - y_south_out_m,
    )
    _add_footing_and_aggregate(
        name="Sunken Garden East Wall",
        x0_m=x_in1_m - sg_toe_m,
        y0_m=y_south_out_m,
        w_m=(x_out1_m + sg_heel_m) - (x_in1_m - sg_toe_m),
        h_m=y_north_out_m - y_south_out_m,
    )

    # ---- Retaining walls + porch box walls -----------------------------------
    def _add_wall_prism(
        *,
        name: str,
        x0_m: float,
        y0_m: float,
        w_m: float,
        h_m: float,
        z0_m: float,
        height_m: float,
    ) -> Any:
        wall = add_prism_from_profile(
            f,
            context=contexts.body,
            storey=porch_sunken,
            ifc_class="IfcWall",
            name=name,
            profile_points=_rect_polyline_xy((0.0, 0.0), (w_m, h_m)),
            depth=height_m,
            placement_matrix=translation_matrix(hx(x0_m), hy(y0_m), z0_m),
            placement_is_storey_relative=False,
        )
        sg_concrete.append(wall)
        return wall

    # Side walls: split into a tall (porch box) segment + short (open sunken) segment.
    _add_wall_prism(
        name="Sunken Garden West Wall (Open Zone)",
        x0_m=x_out0_m,
        y0_m=y_south_out_m,
        w_m=sg_wall_thk_m,
        h_m=y_box_south_out_m - y_south_out_m,
        z0_m=sg_floor_top_m,
        height_m=sg_retaining_h_m,
    )
    _add_wall_prism(
        name="Sunken Garden West Wall (Porch Box)",
        x0_m=x_out0_m,
        y0_m=y_box_south_out_m,
        w_m=sg_wall_thk_m,
        h_m=y_north_out_m - y_box_south_out_m,
        z0_m=sg_floor_top_m,
        height_m=sg_box_h_m,
    )
    _add_wall_prism(
        name="Sunken Garden East Wall (Open Zone)",
        x0_m=x_in1_m,
        y0_m=y_south_out_m,
        w_m=sg_wall_thk_m,
        h_m=y_box_south_out_m - y_south_out_m,
        z0_m=sg_floor_top_m,
        height_m=sg_retaining_h_m,
    )
    _add_wall_prism(
        name="Sunken Garden East Wall (Porch Box)",
        x0_m=x_in1_m,
        y0_m=y_box_south_out_m,
        w_m=sg_wall_thk_m,
        h_m=y_north_out_m - y_box_south_out_m,
        z0_m=sg_floor_top_m,
        height_m=sg_box_h_m,
    )

    # Far south retaining wall (end of sunken garden).
    _add_wall_prism(
        name="Sunken Garden South Wall (Retaining)",
        x0_m=x_out0_m,
        y0_m=y_south_out_m,
        w_m=x_out1_m - x_out0_m,
        h_m=sg_wall_thk_m,
        z0_m=sg_floor_top_m,
        height_m=sg_retaining_h_m,
    )

    # Porch box arch walls (north + south faces), split into a lower + upper segment.
    arch_wall_w_m = sg_clear_w_m + 2.0 * sg_wall_thk_m
    n_arches = int(sunken.arches_per_wall)
    arch_w_m = ft(sunken.arch_clear_width_ft)
    arch_outer_pier_m = ft(sunken.arch_outer_pier_ft)
    arch_open_h_m = ft(sunken.arch_opening_height_ft)

    lower_h_m = sg_retaining_h_m
    upper_h_m = deck_floor_top_m - porch_floor_top_m
    if arch_open_h_m > lower_h_m or arch_open_h_m > upper_h_m:
        raise ValueError("Arch opening height exceeds wall segment height")

    outer_lower = [(0.0, 0.0), (arch_wall_w_m, 0.0), (arch_wall_w_m, -lower_h_m), (0.0, -lower_h_m), (0.0, 0.0)]
    outer_upper = [(0.0, 0.0), (arch_wall_w_m, 0.0), (arch_wall_w_m, -upper_h_m), (0.0, -upper_h_m), (0.0, 0.0)]
    voids = _arch_voids_for_wall(
        wall_width=arch_wall_w_m,
        opening_height=arch_open_h_m,
        n_arches=n_arches,
        arch_width=arch_w_m,
        outer_pier=arch_outer_pier_m,
    )

    # North face (toward house): thickness extrudes south into the porch box.
    north_wall_base = placement_matrix(
        # Flip X so the profile's "up" direction matches the south wall (prevents inverted arches).
        origin=(hx(x_out1_m), hy(y_north_out_m), float(sg_floor_top_m)),
        x_axis=(-1.0, 0.0, 0.0),
        z_axis=(0.0, -1.0, 0.0),
    )
    south_wall_base = placement_matrix(
        origin=(hx(x_out0_m), hy(y_box_south_out_m), float(sg_floor_top_m)),
        x_axis=(1.0, 0.0, 0.0),
        z_axis=(0.0, 1.0, 0.0),
    )

    for label, base_m in [("North", north_wall_base), ("South", south_wall_base)]:
        lower = add_prism_from_profile_with_voids(
            f,
            context=contexts.body,
            storey=porch_sunken,
            ifc_class="IfcWall",
            name=f"Sunken Garden Porch {label} Arch Wall (Lower)",
            outer_profile_points=outer_lower,
            inner_profile_points=voids,
            depth=sg_wall_thk_m,
            placement_matrix=base_m,
            placement_is_storey_relative=False,
        )
        upper_base = base_m.copy()
        upper_base[2, 3] = float(porch_floor_top_m)
        upper = add_prism_from_profile_with_voids(
            f,
            context=contexts.body,
            storey=porch_main,
            ifc_class="IfcWall",
            name=f"Sunken Garden Porch {label} Arch Wall (Upper)",
            outer_profile_points=outer_upper,
            inner_profile_points=voids,
            depth=sg_wall_thk_m,
            placement_matrix=upper_base,
            placement_is_storey_relative=False,
        )
        sg_concrete.extend([lower, upper])

    # ---- Porch + deck framing -------------------------------------------------
    joist_w_m = inch(sunken.porch_joist_width_in)
    joist_d_m = inch(sunken.porch_joist_depth_in)
    joist_spacing_m = inch(sunken.porch_joist_spacing_in)

    def x_positions(start_x_m: float, end_x_m: float) -> list[float]:
        xs: list[float] = []
        x = float(start_x_m)
        while x <= end_x_m + 1e-9:
            xs.append(x)
            x += float(joist_spacing_m)
        if xs and xs[-1] < end_x_m - 1e-6:
            xs.append(float(end_x_m))
        return xs

    # Porch joists span the clear depth between the north and porch south walls.
    span_y0_m = y_north_in_m
    span_y1_m = y_box_south_in_m
    z_center_porch_joist = porch_floor_top_m - deck_thk_m - joist_d_m / 2.0
    for i, x_m in enumerate(x_positions(x_in0_m, x_in1_m), start=1):
        joist = add_rect_member_between_points(
            f,
            context=contexts.body,
            storey=porch_main,
            name=f"Porch Joist {i:02d}",
            p1=(hx(x_m), hy(span_y0_m), float(z_center_porch_joist)),
            p2=(hx(x_m), hy(span_y1_m), float(z_center_porch_joist)),
            width=float(joist_w_m),
            depth=float(joist_d_m),
            predefined_type="JOIST",
            ifc_class="IfcBeam",
            x_axis_hint=(1.0, 0.0, 0.0),
            points_are_storey_relative=False,
        )
        sg_framing.append(joist)

    # Deck joists at the second-storey deck level (placeholder framing, no slope yet).
    z_center_deck_joist = deck_floor_top_m - deck_thk_m - joist_d_m / 2.0
    for i, x_m in enumerate(x_positions(x_in0_m, x_in1_m), start=1):
        joist = add_rect_member_between_points(
            f,
            context=contexts.body,
            storey=porch_deck,
            name=f"Deck Joist {i:02d}",
            p1=(hx(x_m), hy(span_y0_m), float(z_center_deck_joist)),
            p2=(hx(x_m), hy(span_y1_m), float(z_center_deck_joist)),
            width=float(joist_w_m),
            depth=float(joist_d_m),
            predefined_type="JOIST",
            ifc_class="IfcBeam",
            x_axis_hint=(1.0, 0.0, 0.0),
            points_are_storey_relative=False,
        )
        sg_framing.append(joist)

    porch_floor_deck = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=porch_main,
        ifc_class="IfcSlab",
        name="Porch Floor Deck (placeholder)",
        profile_points=_rect_polyline_xy((0.0, 0.0), (sg_clear_w_m, porch_clear_d_m)),
        depth=deck_thk_m,
        placement_matrix=translation_matrix(hx(x_in0_m), hy(y_box_south_in_m), porch_floor_top_m - deck_thk_m),
        placement_is_storey_relative=False,
        predefined_type="FLOOR",
    )
    sg_framing.append(porch_floor_deck)

    # Deck slab (roof to porch), includes the wall thickness for a simple "cap" volume.
    deck_slab = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=porch_deck,
        ifc_class="IfcSlab",
        name="Porch Roof / Deck Slab (placeholder)",
        profile_points=_rect_polyline_xy((0.0, 0.0), (arch_wall_w_m, porch_clear_d_m + 2.0 * sg_wall_thk_m)),
        depth=deck_thk_m,
        placement_matrix=translation_matrix(hx(x_out0_m), hy(y_box_south_out_m), deck_floor_top_m - deck_thk_m),
        placement_is_storey_relative=False,
        predefined_type="ROOF",
    )
    sg_framing.append(deck_slab)

    # Railing around deck perimeter (simple solid bands for now).
    rail_elev_m = deck_floor_top_m
    rail_z0_m = float(rail_elev_m)
    rail_h_m = float(rail_h_m)
    rail_thk_m = float(rail_thk_m)

    deck_corners = {
        "x0": hx(x_out0_m),
        "x1": hx(x_out1_m),
        "y0": hy(y_box_south_out_m),
        "y1": hy(y_north_out_m),
    }
    rail_segments = [
        ("Deck Railing South", (deck_corners["x0"], deck_corners["y0"]), (deck_corners["x1"], deck_corners["y0"])),
        ("Deck Railing East", (deck_corners["x1"], deck_corners["y0"]), (deck_corners["x1"], deck_corners["y1"])),
        ("Deck Railing West", (deck_corners["x0"], deck_corners["y1"]), (deck_corners["x0"], deck_corners["y0"])),
    ]
    for name, p1, p2 in rail_segments:
        rail = add_wall_between_points(
            f,
            context=contexts.body,
            storey=porch_deck,
            name=name,
            p1=p1,
            p2=p2,
            elevation=rail_z0_m,
            height=rail_h_m,
            thickness=rail_thk_m,
        )
        sg_framing.append(rail)

    # Group + style assignments
    assign_to_group(f, group=groups["Concrete"], products=sg_concrete)
    assign_to_group(f, group=groups["Concrete"], products=sg_aggregate)
    assign_to_group(f, group=groups["Framing"], products=sg_framing)
    for el in sg_concrete:
        assign_surface_style(f, element=el, style=concrete_style)
    for el in sg_aggregate:
        assign_surface_style(f, element=el, style=aggregate_style)
    for el in sg_framing:
        assign_surface_style(f, element=el, style=framing_wood_style)

    if include_scale_figure:
        # Simple “human scale” marker for visual context (Bonsai-friendly solid).
        person_h_m = 1.75
        person_w_m = 0.55
        person_d_m = 0.25
        person_x0 = hx(x_center_m - person_w_m / 2.0) + 2.0
        person_y_mid = (y_north_in_m + y_south_in_m) / 2.0
        person_y0 = hy(person_y_mid - person_d_m / 2.0) + 0.5
        person_z0 = float(sg_floor_top_m)
        scale_person = add_prism_from_profile(
            f,
            context=contexts.body,
            storey=porch_sunken,
            ifc_class="IfcBuildingElementProxy",
            name="Scale Figure (1.75m)",
            profile_points=_rect_polyline_xy((0.0, 0.0), (person_w_m, person_d_m)),
            depth=float(person_h_m),
            placement_matrix=translation_matrix(person_x0, person_y0, person_z0),
            placement_is_storey_relative=False,
        )
        assign_to_group(f, group=groups["Furnishings"], products=[scale_person])
        assign_surface_style(f, element=scale_person, style=scale_figure_style)

    # ---- Garage shell ----------------------------------------------------------
    garage_size_m = ft(spec.garage_size_ft)
    gx0, gy0, _ = garage_origin

    rot_deg = float(site.garage_rotation_deg) % 360.0
    if rot_deg not in (0.0, 90.0, 180.0, 270.0):
        raise ValueError("CatlinSitePlacement.garage_rotation_deg must be a multiple of 90 degrees for now")
    c_m = garage_size_m / 2.0

    def _rot_xy(x: float, y: float) -> tuple[float, float]:
        # Rotate about the garage center (keeps extents stable and aligns with roof logic).
        dx = float(x) - c_m
        dy = float(y) - c_m
        if rot_deg == 0.0:
            rx, ry = dx, dy
        elif rot_deg == 90.0:
            rx, ry = -dy, dx
        elif rot_deg == 180.0:
            rx, ry = -dx, -dy
        else:  # 270
            rx, ry = dy, -dx
        return (c_m + rx, c_m + ry)

    def gxy(x: float, y: float) -> tuple[float, float]:
        xr, yr = _rot_xy(x, y)
        return (gx0 + xr, gy0 + yr)

    def _rect_polyline_garage(x0_m: float, y0_m: float, w_m: float, h_m: float) -> list[tuple[float, float]]:
        pts = _rect_polyline_xy((x0_m, y0_m), (w_m, h_m))
        return [gxy(x, y) for x, y in pts]

    garage_icf = ICFFoundationAssembly(
        core_in=GARAGE_ICF.core_in,
        eps_in=GARAGE_ICF.eps_in,
        coating_in=GARAGE_ICF.coating_in,
        above_grade_in=spec.garage_icf_above_grade_in,
        frost_depth_in=spec.garage_frost_depth_in,
    )
    icf_above_grade_m = inch(garage_icf.above_grade_in)
    icf_below_grade_m = inch(garage_icf.frost_depth_in)
    icf_total_h_m = icf_above_grade_m + icf_below_grade_m

    icf_eps_m = inch(garage_icf.eps_in)
    icf_core_m = inch(garage_icf.core_in)
    icf_total_thk_m = inch(garage_icf.total_width_in)

    wood_wall_h_m = ft(spec.garage_wood_wall_height_ft)
    garage_wall = GARAGE_WALL

    garage_perim = [
        (gxy(0.0, 0.0), gxy(garage_size_m, 0.0)),
        (gxy(garage_size_m, 0.0), gxy(garage_size_m, garage_size_m)),
        (gxy(garage_size_m, garage_size_m), gxy(0.0, garage_size_m)),
        (gxy(0.0, garage_size_m), gxy(0.0, 0.0)),
    ]
    # ICF stem wall (includes below-grade portion down to frost depth).
    icf_elev_m = -icf_below_grade_m
    garage_icf_eps_ext = []
    garage_icf_core = []
    garage_icf_eps_int = []
    for i, (p1, p2) in enumerate(garage_perim, start=1):
        eps_ext = add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage ICF EPS (Ext) {i}",
            p1=p1,
            p2=p2,
            elevation=icf_elev_m,
            height=icf_total_h_m,
            thickness=icf_eps_m,
            direction_sense="POSITIVE",  # inward from exterior face
            offset=0.0,
        )
        core = add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage ICF Concrete Core {i}",
            p1=p1,
            p2=p2,
            elevation=icf_elev_m,
            height=icf_total_h_m,
            thickness=icf_core_m,
            direction_sense="POSITIVE",
            offset=icf_eps_m,
        )
        eps_int = add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage ICF EPS (Int) {i}",
            p1=p1,
            p2=p2,
            elevation=icf_elev_m,
            height=icf_total_h_m,
            thickness=icf_eps_m,
            direction_sense="POSITIVE",
            offset=icf_eps_m + icf_core_m,
        )
        garage_icf_eps_ext.append(eps_ext)
        garage_icf_core.append(core)
        garage_icf_eps_int.append(eps_int)

    assign_to_group(f, group=groups["Concrete"], products=garage_icf_core)
    assign_to_group(f, group=groups["Cladding"], products=[*garage_icf_eps_ext, *garage_icf_eps_int])
    for wall in garage_icf_core:
        assign_surface_style(f, element=wall, style=concrete_style)
    for wall in [*garage_icf_eps_ext, *garage_icf_eps_int]:
        assign_surface_style(f, element=wall, style=eps_style)

    # Garage slab-on-grade.
    garage_slab_thk_m = inch(spec.garage_slab_thickness_in)
    slab_inner_size_m = garage_size_m - 2.0 * icf_total_thk_m
    if slab_inner_size_m <= 0:
        raise ValueError("Garage slab inner size must be positive")
    garage_slab = add_slab(
        f,
        context=contexts.body,
        storey=garage_level,
        name="Garage Floor Slab",
        polyline=_rect_polyline_garage(icf_total_thk_m, icf_total_thk_m, slab_inner_size_m, slab_inner_size_m),
        elevation=-garage_slab_thk_m,  # top of slab at grade (Z=0)
        depth=garage_slab_thk_m,
        predefined_type="FLOOR",
    )
    assign_to_group(f, group=groups["Concrete"], products=[garage_slab])
    assign_surface_style(f, element=garage_slab, style=concrete_style)

    # Above-grade framed wall layers (IFC-style: separate walls with offsets).
    drywall_m = inch(garage_wall.drywall_in)
    stud_m = inch(garage_wall.stud_depth_in)
    zip_r_m = inch(garage_wall.sheathing_in)
    rainscreen_m = inch(garage_wall.furring_in)
    metal_siding_m = inch(garage_wall.cladding_in)

    garage_wall_zip_r = []
    garage_wall_studs = []
    garage_wall_drywall = []
    garage_wall_rainscreen = []
    garage_wall_metal = []
    for i, (p1, p2) in enumerate(garage_perim, start=1):
        zip_r = add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage Zip-R Sheathing {i}",
            p1=p1,
            p2=p2,
            elevation=icf_above_grade_m,
            height=wood_wall_h_m,
            thickness=zip_r_m,
            direction_sense="POSITIVE",  # inward
            offset=0.0,
        )
        studs = add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage Stud Wall {i}",
            p1=p1,
            p2=p2,
            elevation=icf_above_grade_m,
            height=wood_wall_h_m,
            thickness=stud_m,
            direction_sense="POSITIVE",
            offset=zip_r_m,
        )
        drywall = add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage Interior Drywall {i}",
            p1=p1,
            p2=p2,
            elevation=icf_above_grade_m,
            height=wood_wall_h_m,
            thickness=drywall_m,
            direction_sense="POSITIVE",
            offset=zip_r_m + stud_m,
        )

        rainscreen = add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage Rainscreen {i}",
            p1=p1,
            p2=p2,
            elevation=icf_above_grade_m,
            height=wood_wall_h_m,
            thickness=rainscreen_m,
            direction_sense="NEGATIVE",  # outward
            offset=0.0,
        )
        metal = add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage Metal Siding {i}",
            p1=p1,
            p2=p2,
            elevation=icf_above_grade_m,
            height=wood_wall_h_m,
            thickness=metal_siding_m,
            direction_sense="NEGATIVE",
            offset=-rainscreen_m,
        )

        garage_wall_zip_r.append(zip_r)
        garage_wall_studs.append(studs)
        garage_wall_drywall.append(drywall)
        garage_wall_rainscreen.append(rainscreen)
        garage_wall_metal.append(metal)

    assign_to_group(f, group=groups["Framing"], products=[*garage_wall_zip_r, *garage_wall_studs])
    assign_to_group(f, group=groups["Drywall"], products=garage_wall_drywall)
    assign_to_group(f, group=groups["Cladding"], products=[*garage_wall_rainscreen, *garage_wall_metal])

    for wall in garage_wall_zip_r:
        assign_surface_style(f, element=wall, style=sheathing_style)
    for wall in garage_wall_studs:
        assign_surface_style(f, element=wall, style=framing_wood_style)
    for wall in garage_wall_drywall:
        assign_surface_style(f, element=wall, style=drywall_style)
    for wall in garage_wall_rainscreen:
        assign_surface_style(f, element=wall, style=membrane_style)
    for wall in garage_wall_metal:
        assign_surface_style(f, element=wall, style=metal_dark_style)

    # Garage roof prism (placeholder).
    g_overhang_m = inch(spec.garage_overhang_in)
    g_pitch = spec.garage_roof_pitch_rise_over_run
    g_eave_z_m = icf_above_grade_m + wood_wall_h_m
    g_drop_m = float(g_pitch) * g_overhang_m
    g_z0 = g_eave_z_m - g_drop_m
    g_ridge_rel_m = (g_eave_z_m + (garage_size_m / 2.0) * float(g_pitch)) - g_z0

    g_roof_profile = [
        (-g_overhang_m, 0.0),
        (garage_size_m / 2.0, g_ridge_rel_m),
        (garage_size_m + g_overhang_m, 0.0),
    ]
    g_roof_depth_m = garage_size_m + 2.0 * g_overhang_m

    g_roof_matrix = np.eye(4, dtype=float)
    # Start from the unrotated basis and apply the same plan rotation used for the garage footprint.
    def _rot_vec_xy(v: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = (float(v[0]), float(v[1]), float(v[2]))
        xr, yr = _rot_xy(x + c_m, y + c_m)  # hack: reuse _rot_xy expecting absolute-in-garage coords
        # _rot_xy returns coords rotated about center; subtract the center back out.
        return (xr - c_m, yr - c_m, z)

    g_roof_matrix[0:3, 0] = _rot_vec_xy((1.0, 0.0, 0.0))  # profile X direction
    g_roof_matrix[0:3, 1] = (0.0, 0.0, 1.0)               # profile Y direction (up)
    g_roof_matrix[0:3, 2] = _rot_vec_xy((0.0, -1.0, 0.0)) # extrusion direction

    # Rotate the original reference point (0, size+overhang) about the center so overhangs stay correct.
    x_ref, y_ref = _rot_xy(0.0, garage_size_m + g_overhang_m)
    g_roof_matrix[0:3, 3] = (gx0 + float(x_ref), gy0 + float(y_ref), float(g_z0))

    garage_roof = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=garage_level,
        ifc_class="IfcRoof",
        name="Garage Roof (placeholder prism)",
        profile_points=g_roof_profile,
        depth=g_roof_depth_m,
        placement_matrix=g_roof_matrix,
        placement_is_storey_relative=False,
        predefined_type="GABLE_ROOF",
    )
    assign_to_group(f, group=groups["Framing"], products=[garage_roof])

    wall_params = garage_wall.to_ifc_params()
    # Backward-compatible aliases for the garage detail script.
    wall_params["zip_r_in"] = wall_params["sheathing_in"]
    wall_params["rainscreen_in"] = wall_params["furring_in"]
    wall_params["metal_siding_in"] = wall_params["cladding_in"]
    wall_params["wood_wall_height_ft"] = spec.garage_wood_wall_height_ft

    garage_detail_params = {
        "icf": garage_icf.to_ifc_params(),
        "wall": wall_params,
        "roof": {
            "pitch_rise_over_run": g_pitch,
            "overhang_in": spec.garage_overhang_in,
        },
        "foundation": {
            "frost_depth_in": spec.garage_frost_depth_in,
            "footing_thick_in": 6.0,
            "footing_width_in": 12.0,
        },
        "slab": {
            "thickness_in": spec.garage_slab_thickness_in,
            "vapor_poly_in": 0.05,
            "xps_in": 2.0,
            "gravel_in": 4.0,
        },
        "framing": {
            "sill_gasket_in": 0.25,
            "sill_plate_in": 1.5,
            "top_plate_in": 1.5,
            "raised_heel_in": 6.0,
            "truss_member_in": 3.5,
        },
    }
    set_pset_json(
        f,
        product=garage_roof,
        pset_name="Pset_ifcPlot_DetailParams",
        prop_name="ParamsJSON",
        value=garage_detail_params,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(out_path))
    return f
