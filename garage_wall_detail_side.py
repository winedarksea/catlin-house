from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon

from ifcplot.detail_utils import (
    COLORS,
    HATCHES,
    _dim_h,
    _dim_v,
    _flashing,
    _leader,
    _lumber,
    _offset_segment,
    _path_from_steps,
    _quad_from_segment,
    _rect,
    load_markdown_notes,
    _wrap_notes,
)


def _unit(v):
    return f'{v:g}"'


def main():
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.1], wspace=0.01)
    ax = fig.add_subplot(gs[0, 0])
    ax_notes = fig.add_subplot(gs[0, 1])

    # Units: inches (schematic)
    # -----------------------
    grade_y = 0.0

    # Foundation: ICF stem wall on compacted stone footing
    frost_depth = 42.0
    icf_above_grade = 6.0
    footing_thick = 6.0
    footing_width = 12.0

    # ICF (8" core, 13" total including EPS foam)
    icf_core = 8.0
    icf_eps = 2.5
    icf_coating = 0.5  # protective coating thickness (schematic)
    icf_total = icf_core + 2 * icf_eps  # 13"

    # Above-grade wall (2x6 + Zip-R 1.5")
    drywall = 0.625
    stud_depth = 5.5
    zip_r = 1.5  # show 1.5" R-6.6 version
    rainscreen = 0.375  # optional
    metal_siding = 0.5

    # Slab assembly (optional layers are shown, labeled optional)
    slab_thick = 3.5
    vapor_poly = 0.05  # schematic
    xps = 2.0
    gravel = 4.0

    # Layout: x=0 is exterior face of ICF protective coating (exposed portion).
    x_exterior = 0.0
    x_icf_eps_ext0 = x_exterior - icf_coating - icf_eps
    x_icf_eps_ext1 = x_exterior - icf_coating
    x_icf_core0 = x_icf_eps_ext0 - icf_core
    x_icf_core1 = x_icf_eps_ext0
    x_icf_eps_int0 = x_icf_core0 - icf_eps
    x_icf_eps_int1 = x_icf_core0

    y_footing_top = grade_y - frost_depth
    y_footing_bot = y_footing_top - footing_thick
    y_icf_bot = y_footing_top
    y_icf_top = grade_y + icf_above_grade

    # Place stud wall so sill/bottom plate bears over concrete core (schematic).
    x_stud_ext = x_icf_core1  # align to exterior face of concrete core
    x_stud_int = x_stud_ext - stud_depth
    x_dry_int = x_stud_int - drywall
    x_zip_ext = x_stud_ext + zip_r
    x_rain_ext = x_zip_ext + rainscreen
    x_siding_ext = x_rain_ext + metal_siding

    sill_gasket = 0.25
    sill_plate = 1.5  # 2x material thickness
    y_sill0 = y_icf_top
    y_sill1 = y_sill0 + sill_gasket + sill_plate

    wall_height = 90.0  # 7.5' framed wall height (above sill)
    raised_heel = 6.0  # 6" raised heel for insulation
    y_topplate_top = y_sill1 + wall_height
    top_plate = 1.5
    y_topplate0 = y_topplate_top - top_plate
    y_raised_heel_top = y_topplate_top + raised_heel  # total 8' wall + heel

    # Colors (shared palette)
    COL = COLORS
    COL_CONC = COL.concrete
    COL_EPS = COL.eps
    COL_WOOD = COL.wood
    COL_DRY = COL.drywall
    COL_GRAVEL = COL.aggregate
    COL_SOIL = COL.soil
    COL_METAL = COL.metal_dark
    COL_FLASH = COL.flashing
    COL_POLY = COL.membrane
    COL_XPS = COL.xps
    COL_SHEATH = COL.sheathing
    COL_UNDERLAY = COL.underlayment
    COL_RIVER_ROCK = COL.river_rock
    COL_SOFFIT = COL.soffit
    COL_INSUL = COL.insulation

    # -----------------------
    # Exterior grade / soil
    # -----------------------
    grade_line = np.array(
        [
            [x_exterior + 0.2, grade_y],
            [x_exterior + 30.0, grade_y],
        ]
    )
    soil_poly = np.vstack(
        [
            grade_line,
            [x_exterior + 30.0, y_footing_bot - 7.0],
            [x_exterior + 0.2, y_footing_bot - 7.0],
        ]
    )
    ax.add_patch(
        Polygon(soil_poly, closed=True, facecolor=COL_SOIL, edgecolor="none", alpha=0.55, zorder=0)
    )
    ax.plot(grade_line[:, 0], grade_line[:, 1], color="black", linewidth=1.2, zorder=5)
    ax.text(x_exterior + 18.0, grade_y + 0.2, "Grade (typ.)", fontsize=9, ha="center", va="bottom")

    # River rock trench beneath overhang edge (for drainage)
    # Position: beneath the eave overhang (calculated later, but we'll place it roughly)
    river_rock_depth = 6.0
    river_rock_width = 12.0
    # Approximate position: at exterior wall + overhang (16")
    river_rock_x = x_exterior + 16.0
    river_rock_x0 = river_rock_x - river_rock_width / 2
    river_rock_x1 = river_rock_x + river_rock_width / 2
    _rect(
        ax,
        river_rock_x0,
        grade_y - river_rock_depth,
        river_rock_width,
        river_rock_depth,
        fc=COL_RIVER_ROCK,
        ec="black",
        lw=1.0,
        hatch=HATCHES.gravel,
        z=2,
    )
    _leader(
        ax,
        (river_rock_x, grade_y - river_rock_depth / 2),
        (x_exterior + 30.0, grade_y - 12.0),
        '12"×6" river rock trench\n(drainage below overhang), geotextile-lined',
        ha="left",
        va="center",
    )

    # -----------------------
    # Footing (compacted stone)
    # -----------------------
    footing_center = (x_exterior - icf_total / 2) - icf_coating / 2
    x_foot0 = footing_center - footing_width / 2
    x_foot1 = footing_center + footing_width / 2
    _rect(
        ax,
        x_foot0,
        y_footing_bot,
        x_foot1 - x_foot0,
        footing_thick,
        fc=COL_GRAVEL,
        ec="black",
        lw=1.2,
        hatch=HATCHES.compacted,
        z=2,
    )
    _leader(
        ax,
        (footing_center, (y_footing_top + y_footing_bot) / 2),
        (x_exterior + 30.0, y_footing_bot - 2.0),
        '12"×6" compacted stone footing\n(IRC R403.1, R403.4.1)',
        ha="left",
        va="top",
    )

    # -----------------------
    # ICF stem wall (EPS + concrete + EPS)
    # -----------------------
    # Exterior EPS
    _rect(ax, x_icf_eps_ext0, y_icf_bot, x_icf_eps_ext1 - x_icf_eps_ext0, y_icf_top - y_icf_bot, fc=COL_EPS, z=2)
    # Concrete core
    _rect(ax, x_icf_core0, y_icf_bot, x_icf_core1 - x_icf_core0, y_icf_top - y_icf_bot, fc=COL_CONC, z=2)
    # Interior EPS
    _rect(ax, x_icf_eps_int0, y_icf_bot, x_icf_eps_int1 - x_icf_eps_int0, y_icf_top - y_icf_bot, fc=COL_EPS, z=2)

    # Protective coating (only where EPS is exposed above grade)
    # Exterior coating
    _rect(
        ax,
        x_icf_eps_ext1,
        grade_y,
        icf_coating,
        y_icf_top - grade_y,
        fc=COL_METAL,
        ec="black",
        lw=1.2,
        z=5,
    )
    # Interior coating (15-min thermal barrier per IRC R316.4)
    _rect(
        ax,
        x_icf_eps_int0,
        grade_y,
        icf_coating,
        y_icf_top - grade_y,
        fc=COL_DRY,
        ec="black",
        lw=1.2,
        z=5,
    )
    _leader(
        ax,
        (x_icf_eps_ext1 + icf_coating / 2, (grade_y + y_icf_top) / 2),
        (x_exterior + 30.0, grade_y + 2.0),
        "Protective coating over exposed\nICF EPS (above grade, both sides)\nInterior: 15-min thermal barrier\n(IRC R316.4)",
        ha="left",
        va="bottom",
    )

    _leader(
        ax,
        ((x_icf_core0 + x_icf_core1) / 2, y_icf_bot + 10.0),
        (x_exterior + 30.0, y_footing_top + 10.0),
        'ICF stem wall\n8" concrete core,\n13" total width incl. EPS foam',
        ha="left",
    )

    # Dimensions for ICF depth / above-grade exposure
    _dim_v(ax, y_footing_top, grade_y, x_exterior + 2.0, '42" to frost (typ.)')
    # _dim_v(ax, grade_y, y_icf_top, x_exterior + 42.0, '≥6" ICF above grade')

    # -----------------------
    # Sill, gasket, anchors (schematic)
    # -----------------------
    _rect(ax, x_stud_int, y_sill0, stud_depth, sill_gasket, fc="#FFFFFF", ec="black", lw=1.0, z=4)
    _lumber(ax, x_stud_int, y_sill0 + sill_gasket, stud_depth, sill_plate, fc=COL_WOOD, ec="black", lw=1.2, z=4)
    _leader(
        ax,
        (x_stud_int + stud_depth / 2, y_sill0 + sill_gasket + sill_plate / 2),
        (x_dry_int - 10.0, y_sill1 + 6.0),
        "PT sill board + sill gasket",
        ha="right",
        va="bottom",
    )

    # -----------------------
    # Above-grade framed wall
    # -----------------------
    # Stud (one cut shown)
    _rect(ax, x_stud_int, y_sill1, stud_depth, y_topplate0 - y_sill1, fc=COL_WOOD, ec="black", lw=1.2, z=3)
    # Top plate
    _lumber(ax, x_stud_int, y_topplate0, stud_depth, top_plate, fc=COL_WOOD, ec="black", lw=1.2, z=5)
    _leader(
        ax,
        (x_stud_int + stud_depth / 2, (y_sill1 + y_topplate_top) / 2),
        (x_dry_int - 10.0, y_topplate_top - 18.0),
        '2×6 framed wall @ 16" o.c.\n(1 stud shown in section)',
        ha="right",
    )

    # Interior drywall
    _rect(ax, x_dry_int, y_sill0, drywall, y_topplate_top - y_sill0, fc=COL_DRY, ec="black", lw=1.0, z=4)
    _leader(
        ax,
        (x_dry_int + drywall / 2, y_sill1 + 24.0),
        (x_dry_int - 10.0, y_sill1 + 24.0),
        '5/8" drywall (interior)',
        ha="right",
    )

    # Zip-R sheathing (extends down over sill plate onto flashing, and up over raised heel)
    zip_r_bottom = y_sill0 - 0.5  # extends down over sill plate onto flashing
    _rect(ax, x_stud_ext, zip_r_bottom, zip_r, y_raised_heel_top - zip_r_bottom, fc=COL_SHEATH, ec="black", lw=1.0, z=4)
    _leader(
        ax,
        (x_stud_ext + zip_r, y_sill1 + 62.0),
        (x_exterior + 30.0, y_sill1 + 62.0),
        'Zip-R (or similar) sheathing\n1.5" (R-6.6), taped\n(extends over sill & flashing,\nup over raised heel)',
    )

    # Optional rainscreen mesh (extends over raised heel)
    _rect(
        ax,
        x_zip_ext,
        y_sill1,
        rainscreen,
        y_raised_heel_top - y_sill1,
        fc="none",
        ec="black",
        lw=1.0,
        ls="--",
        z=4,
    )

    # Metal siding (extends down over sill plate onto flashing, and up over raised heel)
    metal_siding_bottom = y_sill0 - 0.5  # extends down over sill plate onto flashing
    _rect(ax, x_rain_ext, metal_siding_bottom, metal_siding, y_raised_heel_top - metal_siding_bottom, fc=COL_METAL, ec="black", lw=1.0, z=4)
    _leader(
        ax,
        (x_rain_ext + metal_siding / 2, y_sill1 + 34.0),
        (x_exterior + 30.0, y_sill1 + 34.0),
        "Metal siding over optional rainscreen",
    )

    # Z-flashing / drip edge at base of wall cladding (schematic)
    z_y = y_sill0 - 0.2
    z_thk = 0.2
    # Exterior Z-flashing centerline: down at wall face, out to cladding face, then short down leg.
    z_ext_start = (x_stud_ext + z_thk, z_y + 0.7)
    z_ext_steps = [
        (0.0, -0.8),  # down along wall face
        (x_exterior + 0.2 - z_ext_start[0], 0.0),  # out toward cladding edge
        (0.0, -0.2),  # short down leg (drip)
    ]
    _flashing(ax, _path_from_steps(z_ext_start, z_ext_steps), z_thk, fc=COL_FLASH, ec="black", lw=0.9, z=6, alpha=0.95)
    _leader(
        ax,
        (x_exterior + 0.1, z_y + 0.2),
        (x_exterior + 30.0, grade_y + 24.0),
        "Z-flashing / drip edge (exterior)\n(liquid flashing recommended)\nDirects water outward",
        va="bottom",
    )
    
    # Interior Z-flashing (mirror image)
    z_int_start = (x_stud_int - z_thk, z_y + 0.7)
    z_int_steps = [
        (0.0, -0.8),
        (x_dry_int - 0.2 - z_int_start[0], 0.0),
        (0.0, -0.2),
    ]
    _flashing(ax, _path_from_steps(z_int_start, z_int_steps), z_thk, fc=COL_FLASH, ec="black", lw=0.9, z=6, alpha=0.95)
    _leader(
        ax,
        (x_dry_int - 6.0, z_y + 0.1),
        (x_dry_int - 16.0, grade_y + 3.0),
        "Z-flashing / drip edge (interior)\nDirects water inward to sloped slab",
        ha="right",
        va="bottom",
    )

    # -----------------------
    # Slab (interior, schematic section)
    # -----------------------
    x_slab0 = x_icf_eps_int0 - 28.0
    x_slab1 = x_icf_eps_int0

    _rect(ax, x_slab0, grade_y - slab_thick, x_slab1 - x_slab0, slab_thick, fc=COL_CONC, ec="black", lw=1.2, z=2)
    _leader(
        ax,
        ((x_slab0 + x_slab1) / 2, grade_y - slab_thick / 2),
        (x_slab0 - 10.0, grade_y - 8.0),
        'Concrete slab (min. 3.5")\n≥3,500 psi (IRC R506.1)\nSlope to drain/door (into page)',
        ha="right",
        va="top",
    )

    # Slope note (since slope may be into/out of section plane)
    ax.annotate(
        "Slope (typ.)",
        xy=(x_slab0 + 6.0, grade_y - 0.2),
        xytext=(x_slab0 + 18.0, grade_y + 5.0),
        ha="left",
        va="bottom",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
    )

    # Optional vapor retarder, XPS, gravel
    _rect(ax, x_slab0, grade_y - slab_thick - vapor_poly, x_slab1 - x_slab0, vapor_poly, fc=COL_POLY, ec="black", lw=0.8, ls="--", z=3)
    _rect(ax, x_slab0, grade_y - slab_thick - vapor_poly - xps, x_slab1 - x_slab0, xps, fc=COL_XPS, ec="black", lw=0.8, ls="--", z=2)
    _rect(ax, x_slab0, grade_y - slab_thick - vapor_poly - xps - gravel, x_slab1 - x_slab0, gravel, fc=COL_GRAVEL, ec="black", lw=0.8, ls="--", hatch=HATCHES.compacted, z=1)

    _leader(
        ax,
        (x_slab1 - 2.0, grade_y - slab_thick - vapor_poly - xps / 2),
        (x_slab0 - 8.0, grade_y - 22.0),
        "Below slab:\n- Poly vapor retarder\n- 2\" XPS (≥25 psi)\n- 4\" gravel base",
        ha="center",
        va="top",
    )

    # -----------------------
    # Roof / truss at eave (schematic) - showing ~4' of proper triangular truss
    # -----------------------
    # Truss geometry: 2x4 members (3.5" actual depth)
    truss_member = 3.5
    roof_pitch = 4 / 12  # 4:12 pitch

    # Bottom chord: horizontal, rests on top of top plate
    # Bottom surface of bottom chord sits on top plate
    bottom_chord_bottom = y_topplate_top
    bottom_chord_y = bottom_chord_bottom + truss_member / 2  # centerline
    bottom_chord_int = np.array([x_stud_int - 48.0, bottom_chord_y])  # ~4' into interior
    bottom_chord_ext = np.array([x_stud_ext, bottom_chord_y])  # ends at exterior wall face

    # Top chord: angled, from interior rising to heel at wall, then extending to eave
    # Calculate where top chord is at interior end (same x as bottom chord interior)
    top_chord_int_x = bottom_chord_int[0]
    # Height rises from heel going toward ridge
    # Heel: top chord sits above bottom chord with 6" raised heel clearance
    heel_y = bottom_chord_y + raised_heel  # 6" raised heel
    heel_x = x_stud_ext
    top_chord_int_y = heel_y + (heel_x - top_chord_int_x) * roof_pitch
    top_chord_int = np.array([top_chord_int_x, top_chord_int_y])
    heel = np.array([heel_x, heel_y])

    # Eave: top chord extends beyond wall for overhang
    overhang_length = 16.0  # 16" overhang
    eave_x = heel_x + overhang_length
    eave_y = heel_y - overhang_length * roof_pitch
    eave = np.array([eave_x, eave_y])

    # Draw bottom chord as 2x4
    bottom_chord_poly = _quad_from_segment(bottom_chord_int, bottom_chord_ext, truss_member)
    ax.add_patch(Polygon(bottom_chord_poly, closed=True, facecolor=COL_WOOD, edgecolor="black", linewidth=1.2, zorder=5))

    # Draw top chord as 2x4 (from interior through heel to eave)
    top_chord_poly = _quad_from_segment(top_chord_int, eave, truss_member)
    ax.add_patch(Polygon(top_chord_poly, closed=True, facecolor=COL_WOOD, edgecolor="black", linewidth=1.2, zorder=5))

    # Raised heel support at wall (vertical member from top plate to top chord) - 2x6
    heel_support_x = x_stud_ext - 3.0  # moved left to avoid overlapping Zip-R sheathing
    heel_support_bottom = bottom_chord_y + truss_member / 2
    heel_support_top_y = heel_y + (heel_x - heel_support_x) * roof_pitch - truss_member / 2 + 0.5  # extended slightly taller
    heel_support_poly = _quad_from_segment(np.array([heel_support_x, heel_support_bottom]), np.array([heel_support_x, heel_support_top_y]), stud_depth)  # 2x6 (5.5")
    ax.add_patch(Polygon(heel_support_poly, closed=True, facecolor=COL_WOOD, edgecolor="black", linewidth=1.0, zorder=4))

    # Diagonal web member (from interior bottom chord to mid top chord)
    diag_bottom = np.array([bottom_chord_int[0] + 6.0, bottom_chord_y + truss_member / 2])
    diag_top_x = (bottom_chord_int[0] + heel_x) / 2
    diag_top_y = heel_y + (heel_x - diag_top_x) * roof_pitch - truss_member / 2
    diag_top = np.array([diag_top_x, diag_top_y])
    diag_poly = _quad_from_segment(diag_bottom, diag_top, 1.5)  # thinner for schematic
    ax.add_patch(Polygon(diag_poly, closed=True, facecolor=COL_WOOD, edgecolor="black", linewidth=0.8, zorder=4))

    _leader(ax, ((bottom_chord_int[0] + heel_x) / 2, (bottom_chord_y + top_chord_int_y) / 2), (x_dry_int - 10.0, y_topplate_top + 10.0), 'Gable roof truss @ 16" o.c.\n(~4\' shown, not to full scale)', ha="right", va="bottom")

    # Roof sheathing + underlayment + metal roofing on top of top chord
    # Offset outward from top chord top surface
    roof_seg0, roof_seg1 = top_chord_int, eave
    osb_offset = truss_member / 2 + 0.3
    osb_seg0, osb_seg1 = _offset_segment(roof_seg0, roof_seg1, osb_offset)
    osb_poly = _quad_from_segment(osb_seg0, osb_seg1, 0.5)
    # Underlayment layer
    underlayment_seg0, underlayment_seg1 = _offset_segment(roof_seg0, roof_seg1, osb_offset + 0.5)
    underlayment_poly = _quad_from_segment(underlayment_seg0, underlayment_seg1, 0.1)
    # Metal roofing
    metal_seg0, metal_seg1 = _offset_segment(roof_seg0, roof_seg1, osb_offset + 0.6)
    metal_poly = _quad_from_segment(metal_seg0, metal_seg1, 0.5)
    ax.add_patch(Polygon(osb_poly, closed=True, facecolor=COL_SHEATH, edgecolor="black", linewidth=0.8, zorder=6))
    ax.add_patch(Polygon(underlayment_poly, closed=True, facecolor=COL_UNDERLAY, edgecolor="black", linewidth=0.6, zorder=7))
    ax.add_patch(Polygon(metal_poly, closed=True, facecolor=COL_METAL, edgecolor="black", linewidth=0.8, zorder=8))
    _leader(ax, ((osb_seg0[0] + osb_seg1[0]) / 2, (osb_seg0[1] + osb_seg1[1]) / 2 + 1.5), (x_exterior + 30.0, y_topplate_top + 14.0), "Roof: OSB + underlayment + (optional)\nrainscreen mesh + metal roofing")

    # Fascia board at end of top chord / eave - positioned closer to truss end
    fascia_w = 1.5  # wider fascia board (1.5")
    fascia_x = eave[0] + truss_member / 2 - 2.5  # positioned closer to truss end
    # Calculate fascia height: from soffit to just below the OSB sheathing at the eave
    soffit_y = eave[1] - truss_member / 2 - 0.5
    # Top of fascia should align with bottom of roof sheathing
    fascia_top_y = metal_seg1[1] - 0.2  # just below metal roofing at eave
    fascia_bot_y = soffit_y
    fascia_h = fascia_top_y - fascia_bot_y
    _rect(ax, fascia_x, fascia_bot_y, fascia_w, fascia_h, fc=COL_WOOD, ec="black", lw=1.2, z=8)

    # Drip edge flashing: over roof edge and down fascia face
    # Simple L-shaped flashing with visible thickness
    drip_thickness = 0.15
    roof_dir = (metal_seg1 - metal_seg0) / (np.linalg.norm(metal_seg1 - metal_seg0) + 1e-9)
    drip_roof_in = metal_seg1 - roof_dir * 0.6
    drip_roof_out = metal_seg1 + roof_dir * 0.8
    fascia_face_x = fascia_x + fascia_w
    
    # Create clean L-shape: over roof, then down fascia (centerline path)
    drip_centerline = [
        drip_roof_in,
        drip_roof_out,
        [fascia_face_x + drip_thickness / 2, drip_roof_out[1]],
        [fascia_face_x + drip_thickness / 2, fascia_bot_y - 0.3],
    ]
    _flashing(ax, drip_centerline, drip_thickness, fc=COL_FLASH, ec="black", lw=1.0, z=9, alpha=0.95)
    _leader(ax, (fascia_face_x + drip_thickness/2, (drip_roof_out[1] + fascia_bot_y) / 2), (x_exterior + 30.0, fascia_bot_y - 0.0), "Drip edge flashing wraps over\nroof edge and down fascia face")

    # Vented soffit: horizontal panel under overhang, from wall face to fascia
    soffit_x0 = x_siding_ext  # start at outer face of wall cladding
    soffit_x1 = fascia_x
    _rect(ax, soffit_x0, soffit_y, soffit_x1 - soffit_x0, 0.5, fc=COL_SOFFIT, ec="black", lw=1.0, z=7)
    ax.plot([soffit_x0 + 1.0, soffit_x1 - 1.0], [soffit_y + 0.25, soffit_y + 0.25], color="black", linewidth=0.8, zorder=8, linestyle=":")
    _leader(ax, ((soffit_x0 + soffit_x1) / 2, soffit_y + 0.25), (x_exterior + 30.0, soffit_y - 12.0), "Vented soffit (aluminum recommended)")

    # Ceiling drywall: attaches to underside of bottom chord (extends to meet wall drywall)
    ceiling_drywall_y = bottom_chord_y - truss_member / 2 - 0.625
    _rect(ax, x_slab0, ceiling_drywall_y, x_dry_int - x_slab0, 0.625, fc=COL_DRY, ec="black", lw=0.9, z=4)

    # Blown insulation above ceiling, filling truss cavity
    insulation_bottom_y = bottom_chord_y + truss_member / 2
    original_height = (top_chord_int_y - truss_member / 2) - insulation_bottom_y
    insulation_top_y = insulation_bottom_y + original_height / 3.0  # Reduced by 2/3
    ax.add_patch(
        Polygon(
            np.array(
                [
                    [x_slab0, insulation_bottom_y],
                    [x_stud_int - 4.0, insulation_bottom_y],
                    [x_stud_int - 4.0, insulation_top_y],
                    [x_slab0, insulation_top_y],
                ]
            ),
            closed=True,
            facecolor=COL_INSUL,
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES.compacted,
            zorder=1,
        )
    )
    _leader(ax, (x_stud_int - 14.0, insulation_bottom_y + 8.0), (x_dry_int - 10.0, insulation_bottom_y + 18.0), 'Ceiling: airtight 5/8" drywall\nAttic insulation above', ha="right")

    # -----------------------
    # Labels / title / formatting
    # -----------------------
    ax.set_title(
        "Detached Garage Wall Detail (Side View) — Insulated, Unheated (Builder Reference)",
        fontsize=13,
        fontweight="bold",
        pad=12,
        loc='left',
    )

    # Author label at right side, aligned with legend area.
    ax.text(x_exterior - 0.0, y_footing_bot - 50.0, "Colin Catlin, 2026", ha="center", va="bottom", fontsize=7)

    ax.set_xlim(x_dry_int - 18.0, x_exterior + 45.0)
    # Leave a little blank space at the bottom for the legend (inside the axes).
    ax.set_ylim(y_footing_bot - 44.0, y_topplate_top + 24.0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        Patch(facecolor=COL_CONC, edgecolor="black", label="Concrete"),
        Patch(facecolor=COL_EPS, edgecolor="black", label="EPS foam"),
        Patch(facecolor=COL_XPS, edgecolor="black", label="XPS foam"),
        Patch(facecolor=COL_WOOD, edgecolor="black", label="Wood framing / plates"),
        Patch(facecolor=COL_DRY, edgecolor="black", label="Drywall"),
        Patch(facecolor=COL_INSUL, edgecolor="black", label="Attic Insulation"),
        Patch(facecolor=COL_METAL, edgecolor="black", label="Metal cladding / roofing"),
        Patch(facecolor=COL_GRAVEL, edgecolor="black", label="Stone / gravel"),
    ]

    # Legend under the main drawing, but *inside* the main axes so it doesn't inflate the saved bbox.
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        bbox_transform=ax.transAxes,
        frameon=True,
        fontsize=8.5,
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.2,
    )
    # -----------------------
    # Notes panel
    # -----------------------
    ax_notes.set_xlim(0, 100)
    ax_notes.set_ylim(0, 145)
    ax_notes.axis("off")

    raw_notes = load_markdown_notes(Path(__file__).resolve().parent / "notes/garage_wall_detail_side.md")
    ax_notes.text(5, 142, _wrap_notes(raw_notes, width=54), fontsize=9, va="top", ha="left", family="monospace")

    out_path = "./garage_wall_detail_side.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out_path


if __name__ == "__main__":
    print(main())
