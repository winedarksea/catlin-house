from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directory to path to allow imports when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import ifcopenshell

from ifcplot.ifc_utils import (
    add_building,
    add_prism_from_profile,
    add_rect_member_between_points,
    add_slab,
    add_storey,
    add_trade_groups,
    add_wall_between_points,
    assign_surface_style,
    assign_to_group,
    create_surface_style_with_texture,
    init_ifc_project,
    set_pset_json,
    translation_matrix,
    placement_matrix,
)
from ifcplot.units import ft, inch


@dataclass(frozen=True)
class CatlinSitePlacement:
    house_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    garage_origin_m: tuple[float, float, float] = (ft(48.0), ft(0.0), 0.0)  # placeholder
    porch_origin_m: tuple[float, float, float] = (ft(-40.0), ft(-40.0), 0.0)  # placeholder
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

    roof_pitch_rise_over_run: float = 4.0 / 12.0
    roof_overhang_in: float = 16.0

    framing_spacing_in: float = 16.0
    floor_joist_width_in: float = 1.5
    floor_joist_depth_in: float = 11.875
    roof_joist_width_in: float = 1.5
    roof_joist_depth_in: float = 11.875
    centerline_wall_thickness_in: float = 5.5  # 2x6

    # Garage
    garage_icf_above_grade_in: float = 22.0
    garage_wood_wall_height_ft: float = 8.0
    garage_roof_pitch_rise_over_run: float = 4.0 / 12.0
    garage_overhang_in: float = 16.0

    # Basement plan key features
    basement_grid_ft: float = 18.0  # 4 quadrants with center cross walls
    stair_opening_size_ft: tuple[float, float] = (7.0, 9.0 + 8.0 / 12.0)  # (E-W, N-S)


def _rect_polyline_xy(origin_xy: tuple[float, float], size_xy: tuple[float, float]) -> list[tuple[float, float]]:
    x0, y0 = origin_xy
    w, h = size_xy
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)]


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
    return [tuple(p) for p in [p0u, p1u, p1l, p0l]]




def build_catlin_house_ifc(*, out_path: Path, site: CatlinSitePlacement | None = None, spec: CatlinHouseSpec | None = None) -> Any:
    site = site or CatlinSitePlacement()
    spec = spec or CatlinHouseSpec()

    f, project, ifc_site, contexts = init_ifc_project(name="ifcPlot - Catlin House", schema="IFC4")
    groups = add_trade_groups(f)

    # ---- Materials and Styles --------------------------------------------------
    standing_seam_style = create_surface_style_with_texture(
        f,
        name="Standing Seam Metal",
        texture_path="ifcplot/textures/standing_seam_texture.png",
    )

    # ---- Buildings and storeys -------------------------------------------------
    house_bldg = add_building(f, site=ifc_site, name="House", origin=site.house_origin_m)
    garage_bldg = add_building(f, site=ifc_site, name="Garage", origin=site.garage_origin_m)
    porch_bldg = add_building(f, site=ifc_site, name="Porch + Sunken Garden (placeholder)", origin=site.porch_origin_m)
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
        global_xy=(site.garage_origin_m[0], site.garage_origin_m[1]),
        global_z0=site.garage_origin_m[2],
    )
    add_storey(
        f,
        building=porch_bldg,
        name="Level 0",
        elevation=0.0,
        global_xy=(site.porch_origin_m[0], site.porch_origin_m[1]),
        global_z0=site.porch_origin_m[2],
    )
    add_storey(
        f,
        building=breezeway_bldg,
        name="Level 0",
        elevation=0.0,
        global_xy=(site.breezeway_origin_m[0], site.breezeway_origin_m[1]),
        global_z0=site.breezeway_origin_m[2],
    )

    house_size_m = ft(spec.house_size_ft)
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
    house_origin_x, house_origin_y, _ = site.house_origin_m

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

    # House above-grade shell walls: simple sheathing-envelope walls.
    wall_main_thk_m = inch(5.5 + 0.625)  # 2x6 + 5/8" sheathing
    wall_upper_thk_m = inch(3.5 + 0.625)  # 2x4 + 5/8" sheathing

    def add_house_storey_shell(storey: Any, *, elev_m: float, height_m: float, thickness_m: float, label: str) -> list[Any]:
        segments = [
            ((hx(0.0), hy(0.0)), (hx(house_size_m), hy(0.0))),
            ((hx(house_size_m), hy(0.0)), (hx(house_size_m), hy(house_size_m))),
            ((hx(house_size_m), hy(house_size_m)), (hx(0.0), hy(house_size_m))),
            ((hx(0.0), hy(house_size_m)), (hx(0.0), hy(0.0))),
        ]
        return [
            add_wall_between_points(
                f,
                context=contexts.body,
                storey=storey,
                name=f"House {label} Exterior Wall {i+1}",
                p1=p1,
                p2=p2,
                elevation=elev_m,
                height=height_m,
                thickness=thickness_m,
            )
            for i, (p1, p2) in enumerate(segments)
        ]

    main_shell_walls = add_house_storey_shell(
        house_main, elev_m=main_elev_m, height_m=ft(spec.main_storey_height_ft), thickness_m=wall_main_thk_m, label="Main"
    )
    second_shell_walls = add_house_storey_shell(
        house_second,
        elev_m=second_elev_m,
        height_m=ft(spec.second_storey_height_ft),
        thickness_m=wall_upper_thk_m,
        label="Second",
    )
    # Attic walls: east/west knee walls at 5', north/south gable ends with triangular top.
    # Gable ends: rectangular base (5' high) + triangular peak (6' additional at center)
    knee_wall_h_m = ft(spec.attic_knee_wall_height_ft)
    ridge_h_m = ft(spec.attic_ridge_height_above_floor_ft)
    gable_triangle_h_m = ridge_h_m - knee_wall_h_m  # Height of the triangular portion

    # South gable wall: base rectangle (5' high) + triangular peak
    # Profile for gable: rectangle with triangle on top
    # Points: bottom-left, bottom-right, top-right (at knee height), peak (center at ridge), top-left (at knee height)
    gable_profile_south = [
        (0.0, 0.0),  # bottom-left
        (house_size_m, 0.0),  # bottom-right
        (house_size_m, knee_wall_h_m),  # top-right at knee wall
        (house_size_m / 2.0, ridge_h_m),  # peak at center
        (0.0, knee_wall_h_m),  # top-left at knee wall
    ]

    # South gable wall matrix: profile in XY plane, extrude in +Z direction (local)
    # We need: local X → world X, local Y → world Z (up), local Z → world Y (into house)
    south_gable_matrix = placement_matrix(
        origin=(0.0, wall_upper_thk_m, 0.0),
        x_axis=(1.0, 0.0, 0.0),  # local X -> world +X (along wall)
        y_axis=(0.0, 0.0, 1.0),  # local Y -> world +Z (up)
    )

    south_gable_wall = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=house_attic,
        ifc_class="IfcWall",
        name="House Attic Exterior Wall South (gable)",
        profile_points=gable_profile_south,
        depth=wall_upper_thk_m,
        placement_matrix=south_gable_matrix,
    )

    # North gable wall matrix: profile in XY plane, extrude in local +Z direction
    # We need: local X → world -X (mirror), local Y → world +Z (up), local Z → world -Y (into house)
    north_gable_matrix = placement_matrix(
        origin=(house_size_m, house_size_m - wall_upper_thk_m, 0.0),
        x_axis=(-1.0, 0.0, 0.0),  # local X -> world -X (mirrored along wall)
        y_axis=(0.0, 0.0, 1.0),  # local Y -> world +Z (up)
    )

    north_gable_wall = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=house_attic,
        ifc_class="IfcWall",
        name="House Attic Exterior Wall North (gable)",
        profile_points=gable_profile_south,  # Same profile, different orientation
        depth=wall_upper_thk_m,
        placement_matrix=north_gable_matrix,
    )

    attic_shell_walls = [
        south_gable_wall,
        # East wall (knee wall)
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name="House Attic Exterior Wall East (knee)",
            p1=(hx(house_size_m), hy(0.0)),
            p2=(hx(house_size_m), hy(house_size_m)),
            elevation=attic_elev_m,
            height=ft(spec.attic_knee_wall_height_ft),
            thickness=wall_upper_thk_m,
        ),
        north_gable_wall,
        # West wall (knee wall)
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=house_attic,
            name="House Attic Exterior Wall West (knee)",
            p1=(hx(0.0), hy(house_size_m)),
            p2=(hx(0.0), hy(0.0)),
            elevation=attic_elev_m,
            height=ft(spec.attic_knee_wall_height_ft),
            thickness=wall_upper_thk_m,
        ),
    ]
    assign_to_group(f, group=groups["Framing"], products=[*main_shell_walls, *second_shell_walls, *attic_shell_walls])

    # House Cladding
    cladding_thk_m = inch(0.5)

    def add_house_storey_cladding(storey: Any, *, elev_m: float, height_m: float, thickness_m: float, label: str) -> list[Any]:
        offset = thickness_m / 2.0
        segments = [
            ((hx(-offset), hy(-offset)), (hx(house_size_m + offset), hy(-offset))),
            ((hx(house_size_m + offset), hy(-offset)), (hx(house_size_m + offset), hy(house_size_m + offset))),
            ((hx(house_size_m + offset), hy(house_size_m + offset)), (hx(-offset), hy(house_size_m + offset))),
            ((hx(-offset), hy(house_size_m + offset)), (hx(-offset), hy(-offset))),
        ]
        walls = [
            add_wall_between_points(
                f,
                context=contexts.body,
                storey=storey,
                name=f"House {label} Exterior Cladding {i+1}",
                p1=p1,
                p2=p2,
                elevation=elev_m,
                height=height_m,
                thickness=cladding_thk_m,
            )
            for i, (p1, p2) in enumerate(segments)
        ]
        for wall in walls:
            assign_surface_style(f, element=wall, style=standing_seam_style)
        return walls

    main_cladding_walls = add_house_storey_cladding(
        house_main, elev_m=main_elev_m, height_m=ft(spec.main_storey_height_ft), thickness_m=wall_main_thk_m, label="Main"
    )
    second_cladding_walls = add_house_storey_cladding(
        house_second,
        elev_m=second_elev_m,
        height_m=ft(spec.second_storey_height_ft),
        thickness_m=wall_upper_thk_m,
        label="Second",
    )
    assign_to_group(f, group=groups["Cladding"], products=[*main_cladding_walls, *second_cladding_walls])

    # Cladding for attic gable walls
    cladding_thk_m = inch(0.5)
    offset = wall_upper_thk_m / 2.0

    cladding_south_gable_matrix = south_gable_matrix.copy()
    cladding_south_gable_matrix[0:3, 3] += offset * south_gable_matrix[0:3, 2]
    south_gable_cladding = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=house_attic,
        ifc_class="IfcWall",
        name="House Attic Exterior Cladding South (gable)",
        profile_points=gable_profile_south,
        depth=cladding_thk_m,
        placement_matrix=cladding_south_gable_matrix,
    )
    assign_surface_style(f, element=south_gable_cladding, style=standing_seam_style)

    cladding_north_gable_matrix = north_gable_matrix.copy()
    cladding_north_gable_matrix[0:3, 3] += offset * north_gable_matrix[0:3, 2]
    north_gable_cladding = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=house_attic,
        ifc_class="IfcWall",
        name="House Attic Exterior Cladding North (gable)",
        profile_points=gable_profile_south,  # Same profile, different orientation
        depth=cladding_thk_m,
        placement_matrix=cladding_north_gable_matrix,
    )
    assign_surface_style(f, element=north_gable_cladding, style=standing_seam_style)

    assign_to_group(f, group=groups["Cladding"], products=[south_gable_cladding, north_gable_cladding])

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
        origin=(0.0, house_size_m + overhang_m, 0.0),
        x_axis=(1.0, 0.0, 0.0),
        y_axis=(0.0, 0.0, 1.0),
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

    # ---- Garage shell ----------------------------------------------------------
    garage_size_m = ft(spec.garage_size_ft)
    gx0, gy0, _ = site.garage_origin_m

    def gx(x: float) -> float:
        return gx0 + float(x)

    def gy(y: float) -> float:
        return gy0 + float(y)

    icf_h_m = inch(spec.garage_icf_above_grade_in)
    icf_thk_m = inch(13.0)  # 8" core + ~2.5" EPS each side
    wood_wall_h_m = ft(spec.garage_wood_wall_height_ft)
    wood_wall_thk_m = inch(5.5 + 1.5)  # 2x6 + Zip-R 1.5" (used as sheathing envelope)

    garage_perim = [
        ((gx(0.0), gy(0.0)), (gx(garage_size_m), gy(0.0))),
        ((gx(garage_size_m), gy(0.0)), (gx(garage_size_m), gy(garage_size_m))),
        ((gx(garage_size_m), gy(garage_size_m)), (gx(0.0), gy(garage_size_m))),
        ((gx(0.0), gy(garage_size_m)), (gx(0.0), gy(0.0))),
    ]
    garage_icf_walls = [
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage ICF Stem Wall {i+1}",
            p1=p1,
            p2=p2,
            elevation=0.0,
            height=icf_h_m,
            thickness=icf_thk_m,
        )
        for i, (p1, p2) in enumerate(garage_perim)
    ]
    garage_wood_walls = [
        add_wall_between_points(
            f,
            context=contexts.body,
            storey=garage_level,
            name=f"Garage Wood Wall {i+1}",
            p1=p1,
            p2=p2,
            elevation=icf_h_m,
            height=wood_wall_h_m,
            thickness=wood_wall_thk_m,
        )
        for i, (p1, p2) in enumerate(garage_perim)
    ]
    assign_to_group(f, group=groups["Concrete"], products=garage_icf_walls)
    assign_to_group(f, group=groups["Framing"], products=garage_wood_walls)

    # Garage roof prism (placeholder).
    g_overhang_m = inch(spec.garage_overhang_in)
    g_pitch = spec.garage_roof_pitch_rise_over_run
    g_eave_z_m = icf_h_m + wood_wall_h_m
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
    g_roof_matrix[0:3, 0] = (1.0, 0.0, 0.0)   # local X -> world +X (east-west)
    g_roof_matrix[0:3, 1] = (0.0, 0.0, 1.0)   # local Y -> world +Z (up)
    g_roof_matrix[0:3, 2] = (0.0, -1.0, 0.0)  # local Z (extrusion) -> world -Y (south)
    # Relative to Garage Level (Elev 0)
    g_roof_matrix[0:3, 3] = (0.0, garage_size_m + g_overhang_m, float(g_z0))

    garage_roof = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=garage_level,
        ifc_class="IfcRoof",
        name="Garage Roof (placeholder prism)",
        profile_points=g_roof_profile,
        depth=g_roof_depth_m,
        placement_matrix=g_roof_matrix,
        predefined_type="GABLE_ROOF",
    )
    assign_to_group(f, group=groups["Framing"], products=[garage_roof])

    garage_detail_params = {
        "icf": {
            "core_in": 8.0,
            "eps_in": 2.5,
            "total_in": 13.0,
            "above_grade_in": spec.garage_icf_above_grade_in,
        },
        "wall": {
            "drywall_in": 0.625,
            "stud_depth_in": 5.5,
            "zip_r_in": 1.5,
            "rainscreen_in": 0.375,
            "metal_siding_in": 0.5,
            "wood_wall_height_ft": spec.garage_wood_wall_height_ft,
        },
        "roof": {
            "pitch_rise_over_run": g_pitch,
            "overhang_in": spec.garage_overhang_in,
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
