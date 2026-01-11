from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ifc_utils import (
    add_building,
    add_prism_from_profile,
    add_slab,
    add_storey,
    add_trade_groups,
    add_wall_between_points,
    assign_to_group,
    init_ifc_project,
    set_pset_json,
    translation_matrix,
)
from .units import ft, inch


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


def build_catlin_house_ifc(*, out_path: Path, site: CatlinSitePlacement | None = None, spec: CatlinHouseSpec | None = None) -> Any:
    site = site or CatlinSitePlacement()
    spec = spec or CatlinHouseSpec()

    f, project, ifc_site, contexts = init_ifc_project(name="ifcPlot - Catlin House", schema="IFC4")
    groups = add_trade_groups(f)

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
    attic_shell_walls = add_house_storey_shell(
        house_attic,
        elev_m=attic_elev_m,
        height_m=ft(spec.attic_knee_wall_height_ft),
        thickness_m=wall_upper_thk_m,
        label="Attic",
    )
    assign_to_group(f, group=groups["Framing"], products=[*main_shell_walls, *second_shell_walls, *attic_shell_walls])

    # Roof: gable prism (placeholder) aligned to north-south ridge at x=18'.
    overhang_m = inch(spec.roof_overhang_in)
    pitch = spec.roof_pitch_rise_over_run

    eave_z_m = attic_elev_m + ft(spec.attic_knee_wall_height_ft)
    ridge_z_m = attic_elev_m + ft(spec.attic_ridge_height_above_floor_ft)
    drop_m = float(pitch) * overhang_m
    z0 = eave_z_m - drop_m
    ridge_rel_m = ridge_z_m - z0

    roof_profile = [
        (-overhang_m, 0.0),
        (house_size_m / 2.0, ridge_rel_m),
        (house_size_m + overhang_m, 0.0),
    ]
    roof_depth_m = house_size_m + 2.0 * overhang_m

    roof_matrix = np.eye(4, dtype=float)
    roof_matrix[0:3, 0] = (1.0, 0.0, 0.0)  # local X -> world +X (east)
    roof_matrix[0:3, 2] = (0.0, -1.0, 0.0)  # local Z -> world -Y (south)
    roof_matrix[0:3, 3] = (hx(0.0), hy(house_size_m + overhang_m), float(z0))

    house_roof = add_prism_from_profile(
        f,
        context=contexts.body,
        storey=house_attic,
        ifc_class="IfcRoof",
        name="House Roof (placeholder prism)",
        profile_points=roof_profile,
        depth=roof_depth_m,
        placement_matrix=roof_matrix,
        predefined_type="GABLE_ROOF",
    )
    assign_to_group(f, group=groups["Framing"], products=[house_roof])

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
            "sheathing_in": 0.75,
            "polyiso_in": 2.0,
            "eps_in": 4.0,
            "membrane_in": 0.25,
            "furring_in": 0.75,
            "metal_roof_in": 0.5,
            "overhang_in": spec.roof_overhang_in,
        },
    }
    set_pset_json(
        f,
        product=house_roof,
        pset_name="Pset_ifcPlot_DetailParams",
        prop_name="ParamsJSON",
        value=roof_detail_params,
    )

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
    g_roof_matrix[0:3, 0] = (1.0, 0.0, 0.0)
    g_roof_matrix[0:3, 2] = (0.0, -1.0, 0.0)
    g_roof_matrix[0:3, 3] = (gx(0.0), gy(garage_size_m + g_overhang_m), float(g_z0))

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
