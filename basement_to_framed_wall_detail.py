"""
Schematic exterior wall detail: basement concrete wall transitioning to wood-framed wall above.

Run:
  python3 catlin-house/basement_to_framed_wall_detail.py
Output:
  catlin-house/basement_to_framed_wall_detail.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon

from ifcplot.detail_utils import (
    COLORS,
    HATCHES,
    _dim_h,
    _flashing,
    _french_drain,
    _leader,
    _lumber,
    BasementConcreteWallAssembly,
    ExteriorWoodWallAssembly,
    _path_from_steps,
    _rect,
    _slab_assembly,
    _wrap_notes,
)


def main():
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.85, 1.15], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_notes = fig.add_subplot(gs[0, 1])

    # Units: inches (schematic)
    grade_y = 12.0  # raised 1' to reduce figure height
    basement_above_grade = 24.0  # top 2' shown above grade
    basement_wall_height = 48.0  # 4' shown

    y_conc_top = grade_y + basement_above_grade
    y_conc_bot = y_conc_top - basement_wall_height

    # Wall assemblies (x: interior negative, exterior positive)
    drywall = 0.625
    stud_depth = 5.5  # 2x6 (LSL recommended)
    sheathing = 0.625
    membrane_thk = 0.6  # exaggerated for visibility

    # Continuous insulation target: 4" total above + below (schematic)
    polyiso_wall = 2.0
    eps_wall = 2.0
    xps_layer = 2.0  # basement: two 2" layers

    furring = 0.5
    cladding = 0.5
    conc_wall_thk = 10.0

    wall_layers = ExteriorWoodWallAssembly(
        drywall=drywall,
        stud_depth=stud_depth,
        sheathing=sheathing,
        membrane=membrane_thk,
        polyiso=polyiso_wall,
        eps=eps_wall,
        furring=furring,
        cladding=cladding,
        x_clad1=1.0,
    ).coords()
    x_dry0, x_dry1 = wall_layers["drywall"]
    x_stud0, x_stud1 = wall_layers["stud"]
    x_sheath0, x_sheath1 = wall_layers["sheathing"]
    x_mem0, x_mem1 = wall_layers["membrane"]
    x_poly0, x_poly1 = wall_layers["polyiso"]
    x_eps0, x_eps1 = wall_layers["eps"]
    x_fur0, x_fur1 = wall_layers["furring"]
    x_clad0, x_clad1 = wall_layers["cladding"]
    x_foam_inner = x_poly0
    x_foam_outer = x_eps1

    conc_layers = BasementConcreteWallAssembly(
        conc_thk=conc_wall_thk,
        membrane=membrane_thk,
        polyiso=polyiso_wall,
        eps=eps_wall,
        furring=furring,
        cladding=cladding,
        x_clad1=x_clad1,
    ).coords()
    x_conc_int, x_conc_ext = conc_layers["concrete"]

    # Interior slab assembly (at grade level)
    slab_thk = 4.0
    xps_under_thk = 2.0  # R-10 XPS
    poly_sheet_thk = 0.05  # schematic
    stone_under_slab = 4.0
    thermal_break_thk = 1.0
    sealant_thk = 0.5
    slab_width = 18.0  # show 18" of interior slab

    # Sill assembly (schematic)
    sill_gasket = 0.25
    sill_plate = 1.5
    y_sill0 = y_conc_top
    y_gasket1 = y_sill0 + sill_gasket
    y_sill1 = y_gasket1 + sill_plate

    # How much of the framed wall to show above the sill
    wall_show_h = 46.0
    y_wall_top = y_sill1 + wall_show_h

    # Colors
    COL = COLORS
    COL_WOOD = COL.wood
    COL_CONC = COL.concrete
    COL_SHEATH = COL.sheathing
    COL_MEM = COL.membrane
    COL_XPS = COL.xps
    COL_EPS = COL.eps
    COL_POLYISO = COL.polyiso
    COL_SOIL = COL.soil
    COL_METAL = COL.metal_dark
    COL_FLASH = COL.metal
    COL_SPRAY = COL.spray_foam
    COL_INSUL = COL.insulation
    COL_RIVER_ROCK = COL.river_rock

    # -----------------------
    # Exterior grade / soil
    # -----------------------
    grade_line = np.array([[x_clad1 + 0.2, grade_y], [x_clad1 + 32.0, grade_y]])
    soil_poly = np.vstack(
        [
            grade_line,
            [x_clad1 + 32.0, y_conc_bot - 18.0],
            [x_clad1 + 0.2, y_conc_bot - 18.0],
        ]
    )
    ax.add_patch(Polygon(soil_poly, closed=True, facecolor=COL_SOIL, edgecolor="none", alpha=0.55, zorder=0))
    ax.plot(grade_line[:, 0], grade_line[:, 1], color="black", linewidth=1.2, zorder=5)
    ax.text(x_clad1 + 18.0, grade_y + 0.2, "Grade (typ.)", fontsize=9, ha="center", va="bottom")
    
    # Slope annotation (soil slopes away from foundation at 6" per 10')
    ax.text(x_clad1 + 18.0, grade_y - 1.8, "Slope: 6\" per 10' away from foundation", fontsize=9, ha="center", va="top", style="italic")

    # -----------------------
    # Interior slab assembly (basement floor)
    # -----------------------
    # Slab at basement floor level, extending into basement (negative X direction)
    basement_floor_y = y_conc_bot + 2.0
    slab_x0 = x_conc_int - slab_width
    slab_x1 = x_conc_int
    
    # Calculate footing position first (needs to connect to bottom of concrete wall)
    footing_w = 20.0
    footing_d = 8.0
    stone_base_thk = 6.0
    
    # Footing starts at bottom of concrete wall
    y_foot0 = y_conc_bot
    y_foot1 = y_foot0 - footing_d
    y_stone0 = y_foot1
    y_stone1 = y_stone0 - stone_base_thk
    
    # Center footing under the concrete wall
    x_foot0 = (x_conc_int + x_conc_ext) / 2 - footing_w / 2
    x_foot1 = x_foot0 + footing_w
    
    # Calculate slab position relative to footing (slab XPS bottom is 2" above footing top, matching sauna)
    slab_xps_bot = y_foot0 + 2.0
    basement_floor_y = slab_xps_bot + xps_under_thk + poly_sheet_thk + slab_thk
    
    slab_layers = _slab_assembly(
        ax,
        slab_x0,
        slab_x1,
        basement_floor_y,
        slab_thk=slab_thk,
        xps_thk=xps_under_thk,
        vapor_thk=poly_sheet_thk,
        stone_thk=stone_under_slab,
        thermal_break_thk=thermal_break_thk,
        sealant_thk=sealant_thk,
        thermal_break_side='right',
        thermal_break_x=x_conc_int - thermal_break_thk,
        show_stone=True,
    )
    
    # Single consistent aggregate base (wider than footing for french drain)
    stone_base_extra = 8.0
    drain_diam = 4.0
    x_stone0 = x_foot0 - stone_base_extra
    x_stone1 = x_foot1 + stone_base_extra

    _rect(ax, x_conc_int, y_conc_bot, conc_wall_thk, y_conc_top - y_conc_bot, fc=COL_CONC, lw=1.2, z=2)
    _rect(ax, x_foot0, y_foot1, footing_w, footing_d, fc=COL_CONC, lw=1.2, hatch=HATCHES.compacted, z=2)
    _rect(ax, x_stone0, y_stone1, x_stone1 - x_stone0, stone_base_thk, fc=COL.aggregate, lw=1.0, hatch=HATCHES.gravel, z=1)
    
    # French drain (in wider stone area, not under footing)
    drain_x = x_stone0 + drain_diam * 0.8
    drain_y = y_stone1 + stone_base_thk / 2
    _french_drain(ax, drain_x, drain_y, drain_diam, fc=COL_METAL, z=3)
    _french_drain(ax, drain_x, drain_y, drain_diam, fc=COL_METAL, z=3)

    # -----------------------
    # Air/water barrier continuity (liquid membrane) + insulation
    # -----------------------
    # Continuous membrane from sheathing down to foundation wall.
    _rect(ax, x_mem0, y_conc_bot, membrane_thk, y_wall_top - y_conc_bot, fc=COL_MEM, ec="black", lw=0.9, z=6)
    # XPS: 4" total, in two 2" layers (outer layer taped seams)
    x_xps_in0 = x_mem1
    x_xps_in1 = x_xps_in0 + xps_layer
    x_xps_out0 = x_xps_in1
    x_xps_out1 = x_xps_out0 + xps_layer
    _rect(ax, x_xps_in0, y_conc_bot, xps_layer, y_conc_top - y_conc_bot, fc=COL_XPS, lw=1.0, z=4)
    _rect(ax, x_xps_out0, y_conc_bot, xps_layer, y_conc_top - y_conc_bot, fc=COL_XPS, lw=1.0, z=4)

    # Above-grade XPS protection (coating or rigid trim), only on the exposed portion
    protect_thk = 0.5
    _rect(ax, x_xps_out1, grade_y, protect_thk, y_conc_top - grade_y, fc=COL_METAL, lw=1.0, z=7, alpha=0.85)

    # River rock trench (top 8" of soil, against foundation for drainage)
    river_rock_depth = 8.0
    river_rock_width = 10.0
    river_rock_x0 = x_xps_out1 + protect_thk
    river_rock_x1 = river_rock_x0 + river_rock_width
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

    # -----------------------
    # Sill gasket + sill plate (mudsill)
    # -----------------------
    _rect(ax, x_stud0, y_sill0, stud_depth, sill_gasket, fc=COL.rubber, ec="black", lw=0.9, z=7)
    _lumber(ax, x_stud0, y_gasket1, stud_depth, sill_plate, fc=COL_WOOD, lw=1.2, z=10)

    # -----------------------
    # Above-grade wood wall (portion)
    # -----------------------
    _rect(ax, x_dry0, y_sill1, drywall, y_wall_top - y_sill1, fc=COL.drywall, lw=1.0, z=3)
    _rect(ax, x_stud0, y_sill1, stud_depth, y_wall_top - y_sill1, fc=COL_INSUL, lw=0.9, hatch=HATCHES.compacted, z=1)
    _rect(ax, x_stud0, y_sill1, stud_depth, y_wall_top - y_sill1, fc="none", lw=1.2, z=5)
    _rect(ax, x_sheath0, y_sill1, sheathing, y_wall_top - y_sill1, fc=COL_SHEATH, lw=1.1, z=4)
    _rect(ax, x_poly0, y_sill0, polyiso_wall, y_wall_top - y_sill0, fc=COL_POLYISO, lw=1.0, z=3)
    _rect(ax, x_eps0, y_sill0, eps_wall, y_wall_top - y_sill0, fc=COL_EPS, lw=1.0, z=3)

    # Furring + cladding start at the basement/first-floor transition (y_conc_top)
    _rect(ax, x_fur0, y_conc_top, furring, y_wall_top - y_conc_top, fc=COL_WOOD, lw=1.0, hatch=HATCHES.diagonal, z=6)
    _rect(ax, x_clad0, y_conc_top, cladding, y_wall_top - y_conc_top, fc=COL_METAL, lw=1.1, z=7)

    # -----------------------
    # Flashings + insect screen at the transition
    # -----------------------
    # L-flashing at sheathing bottom, runs down and across the top of basement foam,
    # but terminates short of the exterior face to reduce thermal bridging.
    l_flash_thk = 0.45
    l_flash_x = x_mem1 + 0.15
    l_flash_y = y_conc_top + 0.12
    l_flash_pts = np.array([[l_flash_x, y_sill1 + 0.15], [l_flash_x, l_flash_y], [x_foam_outer - 0.7, l_flash_y]])
    _flashing(ax, l_flash_pts, l_flash_thk, fc=COL_FLASH, lw=0.9, z=12, alpha=0.9)

    # Spray foam at the "buried" flashing termination / joint
    spray_h = 0.9
    _rect(ax, x_foam_outer - 0.75, y_conc_top, 0.75, spray_h, fc=COL_SPRAY, ec="black", lw=0.8, z=11)

    # Z-flashing w/ drip edge at bottom of furring strips; add insect screen above.
    z_thk = 0.5
    z_start = (x_fur0 + 0.06, y_conc_top + 2.4)
    z_steps = [
        (0.0, -(2.15)),  # down behind furring (foam side)
        (x_fur1 - x_fur0 + 0.18, 0.0),  # under bottom of furring
        (0.55, 0.0),  # out toward the drip
        (0.0, -4.2),  # down leg
        (0.7, -0.25),  # hem / drip
    ]
    _flashing(ax, _path_from_steps(z_start, z_steps), z_thk, fc=COL_FLASH, lw=0.9, z=13, alpha=0.9)

    # Insect screen (Cor-A-Vent strip or SS mesh), just above flashing.
    screen_h = 0.5
    _rect(ax, x_fur0 + 0.02, y_conc_top + 2.65, x_clad1 - x_fur0 - 0.04, screen_h, fc="white", ec="black", lw=0.8, hatch=HATCHES.cross, z=14, alpha=0.9)

    # -----------------------
    # Callouts / dimensions
    # -----------------------
    _dim_h(ax, x_xps_in0, x_xps_out1, y_conc_bot + 24.0, '4" XPS (2 layers)')
    _dim_h(ax, x_poly0, x_eps1, y_conc_top + 28.0, '4" CI above (2" + 2")')

    _leader(
        ax,
        xy=(x_conc_ext + 0.35, (y_conc_top + y_conc_bot) / 2),
        text_xy=(x_clad1 + 5.0, y_conc_bot + 28.0),
        text="Liquid-applied waterproofing / air barrier\n(thickness exaggerated; continuous)",
        ha="left",
    )
    _leader(
        ax,
        xy=(x_xps_out0 + 0.9, y_conc_top - 14.0),
        text_xy=(x_clad1 + 5.0, y_conc_top - 10.0),
        text='4" XPS (2 layers, staggered seams;\nouter layer taped; low water absorption)',
        ha="left",
    )
    _leader(
        ax,
        xy=(x_xps_out1 + protect_thk / 2, grade_y + 10.0),
        text_xy=(x_clad1 + 5.0, grade_y + 8.0),
        text="Exposed XPS protection above grade:\ncoating or rigid metal/PVC trim",
        ha="left",
    )
    _leader(
        ax,
        xy=(river_rock_x0 + river_rock_width / 2, grade_y - river_rock_depth / 2),
        text_xy=(x_clad1 + 5.0, grade_y - 6.0),
        text='10"×8" river rock trench (geotextile-lined)\n(top 8" of soil, drainage against foundation)',
        ha="left",
    )
    _leader(
        ax,
        xy=(drain_x, drain_y),
        text_xy=(x_clad1 + 5.0, y_foot1 - 6.0),
        text='4" perforated french drain\nin washed stone (wider area,\nnot under footing)',
        ha="left",
    )
    _leader(
        ax,
        xy=((x_foot0 + x_foot1) / 2, (y_foot0 + y_foot1) / 2),
        text_xy=(x_dry0 - 12.0, y_foot1 + 2.0),
        text='20"×8" footing (5000 psi)\non 6" compacted aggregate\n(geotextile-lined)',
        ha="right",
        va="center",
    )
    _leader(
        ax,
        xy=((slab_x0 + slab_x1) / 2, basement_floor_y - slab_thk / 2),
        text_xy=(x_dry0 - 12.0, y_conc_bot - 12.0),
        text='Interior basement slab:\n4" concrete over R-10 XPS,\nvapor barrier, 4" gravel base\n1" thermal break at foundation wall',
        ha="right",
        va="center",
    )
    _leader(
        ax,
        xy=(x_stud0 + 0.6, y_gasket1 + sill_plate / 2),
        text_xy=(x_dry0 - 12.0, y_conc_top + 18.0),
        text="Sill gasket + treated mudsill\n(air seal at sill; anchors such as MASAP not shown)",
        ha="right",
    )
    _leader(
        ax,
        xy=(x_fur0 + 0.3, y_conc_top + 1.2),
        text_xy=(x_clad1 + 5.0, y_conc_top + 5.5),
        text="Stainless / thick aluminum Z-flashing\nw/ drip edge at bottom of rainscreen furring",
        ha="left",
    )
    _leader(
        ax,
        xy=(x_fur0 + 0.2, y_conc_top + 2.9),
        text_xy=(x_clad1 + 5.0, y_conc_top + 12.0),
        text="Insect screen (Cor-A-Vent or SS mesh)\njust above flashing",
        ha="left",
    )
    _leader(
        ax,
        xy=(x_foam_outer - 0.55, y_conc_top + 0.4),
        text_xy=(x_dry0 - 12.0, y_conc_top + 2.0),
        text="L-flashing from sheathing down onto\nbasement foam; terminate within foam\nand seal outer end with spray foam",
        ha="right",
    )

    ax.set_title("Basement to Wood-Framed Wall Transition Detail", fontsize=13, fontweight="bold", loc="center", pad=12)
    ax.text(x_conc_int + 30.0, y_stone1 - 8.0, "Colin Catlin, 2026", ha="center", va="top", fontsize=7)

    ax.set_xlim(x_conc_int - 14.0, x_clad1 + 38.0)
    ax.set_ylim(y_stone1 - 6.0, y_wall_top + 12.0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        Patch(facecolor=COL_CONC, edgecolor="black", label="Concrete"),
        Patch(facecolor=COL_WOOD, edgecolor="black", label="Wood framing / plates / furring"),
        Patch(facecolor=COL_SHEATH, edgecolor="black", label='Sheathing (5/8")'),
        Patch(facecolor=COL_MEM, edgecolor="black", label="Liquid-applied membrane"),
        Patch(facecolor=COL_XPS, edgecolor="black", label='XPS (4")'),
        Patch(facecolor=COL_POLYISO, edgecolor="black", label='Polyiso (2")'),
        Patch(facecolor=COL_EPS, edgecolor="black", label='EPS (2")'),
        Patch(facecolor=COL.aggregate, edgecolor="black", label="Aggregate / washed stone"),
        Patch(facecolor=COL_RIVER_ROCK, edgecolor="black", label="River rock (drainage)"),
        Patch(facecolor=COL_FLASH, edgecolor="black", label="Metal flashing"),
        Patch(facecolor=COL_SPRAY, edgecolor="black", label="Spray foam (seal gaps)"),
    ]
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
        columnspacing=1.0,
    )

    # -----------------------
    # Notes panel
    # -----------------------
    ax_notes.set_xlim(0, 100)
    ax_notes.set_ylim(0, 150)
    ax_notes.axis("off")

    raw_notes = [
        "NOTES:",
        "",
        "• Detail intent: schematic section showing basement exterior wall and transition to wood-framed exterior wall above.",
        "",
        "• Above-grade wall: 2x6 framing (LSL recommended) with R-19 cavity insulation and continuous exterior CI totaling 4\". Above-grade CI shown as 2\" polyiso + 2\" EPS; seams staggered, outer layer taped.",
        "",
        "• Foundation waterproofing / air barrier: liquid-applied membrane on sheathing and on concrete foundation wall; maintain continuity at the sill/rim transition with overlap, sealant, and/or additional liquid membrane as needed.",
        "",
        "• Basement CI: 4\" XPS in two layers with staggered seams; tape seams on the outer layer. Use lower water-absorption XPS types (most are) and confirm compressive strength / below-grade suitability.",
        "",
        "• Exposed XPS (top 2' above grade): protect with appropriate elastomeric coating or rigid metal/PVC trim per manufacturer.",
        "",
        "• Sill: include sill gasket and treated mudsill. Prioritize air sealing at sill plate (sealant + spray foam at gaps). Use mudsill anchors (e.g., MASAP) as required (not shown).",
        "",
        "• Flashings: provide stainless (preferred) or thick aluminum Z-flashing with drip edge at bottom of rainscreen furring. Install insect barrier mesh/strip (Cor-A-Vent or SS screen) just above flashing. Mesh can be stapled to furring strips, run behind the flashing, and into the layer between basement and wall foam.",
        "",
        "• Interface flashing: provide L-flashing from bottom of sheathing down onto the top of basement foam. Terminate within the insulation plane and seal the outer end with spray foam (Pestblock) to avoid an exterior thermal bridge. This is meant as a foam layer insect barrier.",
        "",
        "• Drainage: 4\" perforated french drain in geotextile-lined washed stone (wider area, not under footing). Additional compacted aggregate in front of footing (equal to footing height, geotextile-lined). 10\"×8\" river rock trench (geotextile-lined) against foundation for top 8\" of soil.",
        "",
        "• Interior slab: 4\" concrete slab (min. 3,500 psi, IRC R506.1) over R-10 XPS insulation (≥25 psi), 10 mil (min) polyethylene vapor barrier, and 4\" compacted gravel base. Provide 1\" XPS thermal break with 1/2\" polyurethane sealant at foundation wall perimeter.",
        "",
        "• Grading: soil must slope away from foundation at minimum 6\" per 10' for first 10' (IRC R401.3).",
        "",
        "• All gaps: fill voids and transitions with low-expansion spray foam as needed for air sealing and continuity.",
    ]
    ax_notes.text(3, 146, _wrap_notes(raw_notes), fontsize=9, va="top", ha="left", family="monospace")

    out_path = Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return str(out_path)


if __name__ == "__main__":
    print(main())
