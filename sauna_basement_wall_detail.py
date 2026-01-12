import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch, Polygon

from ifcplot.detail_utils import (
    COLORS,
    HATCHES,
    BasementConcreteWallAssembly,
    _dim_h,
    _dim_v,
    _french_drain,
    _leader,
    _lumber,
    _rect,
    _slab_assembly,
    _wrap_notes,
)


def main():
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.15], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_notes = fig.add_subplot(gs[0, 1])

    # Colors
    COL = COLORS
    COL_CONC = COL.concrete
    COL_STONE = COL.aggregate
    COL_WOOD = COL.wood
    COL_POLYISO = COL.polyiso
    COL_XPS = COL.xps
    COL_MEM = COL.membrane
    COL_METAL = COL.metal
    COL_FC = COL.fiber_cement
    COL_RUBBER = COL.rubber
    COL_INSUL = COL.insulation
    COL_SEALANT = COL.sealant
    COL_POLY = COL.membrane
    COL_EQUIP = COL.equipment
    COL_DRY = COL.drywall

    # -----------------------
    # Geometry (inches, schematic)
    # -----------------------
    sauna_clear_w = 96.0  # 8'
    total_h = 108.0  # 9' to underside of joists (schematic)

    conc_wall_thk = 10.0
    polyiso_thk = 2.0
    furring_thk = 0.5  # 1/2" plywood furring
    tg_thk = 1.0  # 5/4 T&G (1" actual)
    baseboard_h = 6.0
    baseboard_thk = furring_thk + tg_thk  # replaces both

    slab_thk = 4.0
    xps_under_thk = 2.0  # R-10 XPS (typically 2")
    vapor_sheet_thk = 0.05  # schematic only
    stone_under_slab = 4.0

    thermal_break_thk = 1.0  # 1" XPS at slab perimeter
    sealant_thk = 0.5  # 1/2" polyurethane sealant (schematic)

    stud_depth = 3.5  # 2x4
    drywall = 0.625  # adjacent side (schematic)

    footing_w = 20.0
    footing_d = 8.0
    footing_top_y = -48.0
    stone_base_thk = 6.0
    stone_base_extra = 8.0  # wider than footing on each side
    drain_diam = 4.0

    joist_depth = 9.25  # 2x10 (schematic)
    drop_depth = 3.5  # 2x4 dropped ceiling framing

    # Define footing position first (independent reference point)
    footing_center_x = 0.0  # footing and aggregate are correctly positioned here
    
    # Calculate concrete wall position (centered over footing)
    x_conc_center = footing_center_x
    x_conc_ext = x_conc_center - conc_wall_thk / 2
    x_conc_int = x_conc_ext + conc_wall_thk

    # Left wall framing against concrete (supports drop ceiling)
    gap_to_conc = 0.5
    x_studL0 = x_conc_int + gap_to_conc
    x_studL1 = x_studL0 + stud_depth

    # Left wall layers (toward sauna interior, +x)
    x_polyL0 = x_studL1
    x_polyL1 = x_polyL0 + polyiso_thk
    x_furL0 = x_polyL1
    x_furL1 = x_furL0 + furring_thk
    x_tgL0 = x_furL1
    x_tgL1 = x_tgL0 + tg_thk
    x_sauna_left = x_tgL1
    x_sauna_right = x_sauna_left + sauna_clear_w

    # Right wall layers (away from sauna interior, +x)
    x_tgR0 = x_sauna_right
    x_tgR1 = x_tgR0 + tg_thk
    x_furR0 = x_tgR1
    x_furR1 = x_furR0 + furring_thk
    x_polyR0 = x_furR1
    x_polyR1 = x_polyR0 + polyiso_thk
    x_stud0 = x_polyR1
    x_stud1 = x_stud0 + stud_depth
    x_dry0 = x_stud1
    x_dry1 = x_dry0 + drywall

    # Ceiling build-up (inside sauna)
    y_slab_top = 0.0
    y_joist_bot = total_h  # primary structure (joists or concrete deck)
    y_joist_top = y_joist_bot + joist_depth
    y_drop_top = y_joist_bot - 1.0
    y_drop_bot = y_drop_top - drop_depth
    y_ceil_poly0 = y_drop_bot - polyiso_thk
    y_ceil_fur0 = y_ceil_poly0 - furring_thk
    y_ceil_tg0 = y_ceil_fur0 - tg_thk  # interior ceiling face (underside of T&G)

    # -----------------------
    # Sauna slab assembly (interior)
    # -----------------------
    y_slab_bot = y_slab_top - slab_thk
    # Typical radon/vapor barrier location is directly under the slab.
    y_vapor_bot = y_slab_bot - vapor_sheet_thk
    y_xps_bot = y_vapor_bot - xps_under_thk
    y_stone_under_bot = y_xps_bot - stone_under_slab

    # Foundation / footing layout (x positions) - based on footing center position
    x_foot0 = footing_center_x - footing_w / 2
    x_foot1 = x_foot0 + footing_w
    x_stone0 = x_foot0 - stone_base_extra
    x_stone1 = x_foot1 + stone_base_extra
    y_foot0 = y_xps_bot - 2.0
    y_foot1 = y_foot0 - footing_d
    y_stone0 = y_foot1
    y_stone1 = y_stone0 - stone_base_thk

    # -----------------------
    # Concrete wall + footing
    # -----------------------
    _rect(ax, x_conc_ext, y_foot0, conc_wall_thk, (y_joist_top - y_foot0), fc=COL_CONC, lw=1.2, z=2)
    _rect(ax, x_foot0, y_foot1, footing_w, footing_d, fc=COL_CONC, lw=1.2, hatch=HATCHES.compacted, z=2)
    _rect(ax, x_stone0, y_stone1, (x_stone1 - x_stone0), stone_base_thk, fc=COL_STONE, lw=1.0, hatch=HATCHES.gravel, z=1)

    # French drain in wider stone (not under footing)
    drain_x = x_stone0 + drain_diam * 0.8
    drain_y = y_stone1 + stone_base_thk / 2
    _french_drain(ax, drain_x, drain_y, drain_diam, fc=COL_METAL, z=3)

    # Sauna slab assembly with thermal breaks on both sides
    # First draw left thermal break
    _rect(ax, x_conc_int, y_slab_bot, thermal_break_thk, slab_thk, fc=COL_XPS, lw=1.0, z=3)
    _rect(ax, x_conc_int, y_slab_top, thermal_break_thk, sealant_thk, fc=COL_SEALANT, ec="black", lw=0.9, z=5)

    slab_x0 = x_conc_int + thermal_break_thk
    # Sauna slab extends to include thermal breaks on both sides.
    slab_x1 = x_polyR0
    
    # Use helper function for main slab layers
    slab_layers = _slab_assembly(
        ax,
        slab_x0,
        slab_x1,
        y_slab_top,
        slab_thk=slab_thk,
        xps_thk=xps_under_thk,
        vapor_thk=vapor_sheet_thk,
        stone_thk=stone_under_slab,
        thermal_break_side='none',  # We handle thermal breaks separately for sauna
        show_stone=True,
    )
    
    # Right side thermal break (sauna slab to adjacent slab / room, schematic)
    slab_break_x0 = slab_x1
    slab_break_x1 = slab_break_x0 + thermal_break_thk
    _rect(ax, slab_break_x0, y_slab_bot, thermal_break_thk, slab_thk, fc=COL_XPS, lw=1.0, z=3)
    _rect(ax, slab_break_x0, y_slab_top, thermal_break_thk, sealant_thk, fc=COL_SEALANT, ec="black", lw=0.9, z=5)
    _rect(ax, slab_break_x1, y_slab_bot, 18.0, slab_thk, fc=COL_CONC, lw=1.0, ls="--", z=1)

    # -----------------------
    # Right wall framing (2x4 with optional R-13)
    # -----------------------
    plate_thk = 1.5
    _lumber(ax, x_stud0, y_slab_top, stud_depth, plate_thk, fc=COL_WOOD, lw=1.2, z=6)
    _rect(ax, x_stud0, y_slab_top + plate_thk, stud_depth, (y_joist_bot - (y_slab_top + plate_thk)), fc=COL_INSUL, lw=1.0, hatch=HATCHES.compacted, z=1)
    _rect(ax, x_dry0, y_slab_top, drywall, (y_joist_bot - y_slab_top), fc=COL_DRY, lw=1.0, z=0)

    # Left wall framing against concrete (supports drop ceiling)
    _lumber(ax, x_studL0, y_slab_top, stud_depth, plate_thk, fc=COL_WOOD, lw=1.2, z=6)
    _rect(ax, x_studL0, y_slab_top + plate_thk, stud_depth, (y_drop_bot - (y_slab_top + plate_thk)), fc=COL_INSUL, lw=1.0, hatch=HATCHES.compacted, z=1)

    # -----------------------
    # Sauna interior wall & ceiling finishes
    # -----------------------
    # Left wall polyiso (continuous up to dropped ceiling)
    _rect(ax, x_polyL0, y_slab_top, polyiso_thk, (y_drop_bot - y_slab_top), fc=COL_POLYISO, lw=1.0, z=4)

    # Right wall polyiso (continuous up to dropped ceiling)
    _rect(ax, x_polyR0, y_slab_top, polyiso_thk, (y_drop_bot - y_slab_top), fc=COL_POLYISO, lw=1.0, z=4)

    # Ceiling polyiso (between wall polyiso faces to avoid overlap)
    _rect(ax, x_polyL1, y_ceil_poly0, (x_polyR0 - x_polyL1), polyiso_thk, fc=COL_POLYISO, lw=1.0, z=4)

    # Furring + T&G walls (stop at ceiling interior face)
    _rect(ax, x_furL0, baseboard_h, furring_thk, (y_ceil_tg0 - baseboard_h), fc=COL_WOOD, lw=1.0, hatch=HATCHES.diagonal, z=6)
    _rect(ax, x_tgL0, baseboard_h, tg_thk, (y_ceil_tg0 - baseboard_h), fc=COL_WOOD, lw=1.1, z=7)

    _rect(ax, x_furR0, baseboard_h, furring_thk, (y_ceil_tg0 - baseboard_h), fc=COL_WOOD, lw=1.0, hatch=HATCHES.diagonal, z=6)
    _rect(ax, x_tgR0, baseboard_h, tg_thk, (y_ceil_tg0 - baseboard_h), fc=COL_WOOD, lw=1.1, z=7)

    # Ceiling furring + T&G (span between inside faces)
    _rect(ax, x_furL1, y_ceil_fur0, (x_furR0 - x_furL1), furring_thk, fc=COL_WOOD, lw=1.0, hatch=HATCHES.diagonal, z=6)
    _rect(ax, x_sauna_left, y_ceil_tg0, (x_sauna_right - x_sauna_left), tg_thk, fc=COL_WOOD, lw=1.1, z=7)

    # Fiber cement baseboard (replaces furring + T&G at bottom 6")
    _rect(ax, x_furL0, y_slab_top, baseboard_thk, baseboard_h, fc=COL_FC, lw=1.1, z=8)
    _rect(ax, x_sauna_right, y_slab_top, baseboard_thk, baseboard_h, fc=COL_FC, lw=1.1, z=8)

    # Flashing from polyiso over baseboard (both sides)
    flash_thk = 0.35
    flash_drop = 0.6
    left_flash = np.array(
        [
            [x_polyL1, baseboard_h + flash_thk],
            [x_sauna_left, baseboard_h + flash_thk],
            [x_sauna_left, baseboard_h],
            [x_polyL1 + 0.15, baseboard_h],
            [x_polyL1 + 0.15, baseboard_h + 3.0],
            [x_polyL1, baseboard_h + 3.0],
        ]
    )
    ax.add_patch(Polygon(left_flash, closed=True, facecolor=COL_METAL, edgecolor="black", linewidth=0.9, zorder=9))

    right_flash = np.array(
        [
            [x_polyR0, baseboard_h + flash_thk],
            [x_sauna_right, baseboard_h + flash_thk],
            [x_sauna_right, baseboard_h],
            [x_polyR0 - 0.15, baseboard_h],
            [x_polyR0 - 0.15, baseboard_h + 3.0],
            [x_polyR0, baseboard_h + 3.0],
        ]
    )
    ax.add_patch(Polygon(right_flash, closed=True, facecolor=COL_METAL, edgecolor="black", linewidth=0.9, zorder=9))

    # Liquid floor membrane + up baseboard
    mem_thk = 0.25
    _rect(ax, x_sauna_left, y_slab_top, (x_sauna_right - x_sauna_left), mem_thk, fc=COL_MEM, lw=0.8, z=10)
    _rect(ax, x_sauna_left, y_slab_top, mem_thk, baseboard_h, fc=COL_MEM, lw=0.8, z=10)
    _rect(ax, x_sauna_right - mem_thk, y_slab_top, mem_thk, baseboard_h, fc=COL_MEM, lw=0.8, z=10)

    # Dropped ceiling framing (2x4s below primary structure)
    _rect(ax, x_studL0, y_drop_bot, (x_stud0 + stud_depth - x_studL0), drop_depth, fc=COL_WOOD, lw=1.1, hatch=HATCHES.cross, z=5)

    # Duckboards on rubber feet (schematic)
    duck_thk = 1.0
    foot_h = 0.5
    duck_y0 = y_slab_top + mem_thk + foot_h
    duck_x0 = x_sauna_left + 4.0
    duck_x1 = x_sauna_right - 4.0
    _rect(ax, duck_x0, duck_y0, duck_x1 - duck_x0, duck_thk, fc=COL_WOOD, lw=1.0, hatch=HATCHES.compacted, z=11)
    for fx in [duck_x0 + 6.0, duck_x1 - 6.0]:
        _rect(ax, fx, y_slab_top + mem_thk, 1.2, foot_h, fc=COL_RUBBER, ec="black", lw=0.8, z=12)

    # Benches and heater (schematic, Law of Löyly: heater low, benches stepped)
    bench_depth = 20.0
    bench_thk = 1.5
    lower_bench_top = 18.0
    upper_bench_top = 36.0
    bench_x0 = x_sauna_right - bench_depth - 2.0
    bench_x1 = x_sauna_right - 2.0
    lower_bench_x0 = x_sauna_right - bench_depth * 2 - 2.0
    _rect(ax, lower_bench_x0, y_slab_top + lower_bench_top - bench_thk, bench_depth * 2, bench_thk, fc=COL_WOOD, lw=1.0, z=12)
    _rect(ax, bench_x0, y_slab_top + upper_bench_top - bench_thk, bench_depth, bench_thk, fc=COL_WOOD, lw=1.0, z=12)
    # Heater
    heater_w = 10.0
    heater_h = 18.0
    heater_x0 = x_sauna_left + 6.0
    heater_y0 = y_slab_top + mem_thk
    _rect(ax, heater_x0, heater_y0, heater_w, heater_h, fc=COL_EQUIP, ec="black", lw=1.0, hatch=HATCHES.dense_plus, z=12)

    # Joists above (schematic)
    _rect(ax, x_conc_ext - 6.0, y_joist_bot, (x_dry1 - (x_conc_ext - 6.0)) + 8.0, joist_depth, fc=COL_WOOD, lw=1.2, hatch=HATCHES.joist, z=3)

    # -----------------------
    # Labels / leaders
    # -----------------------
    _leader(
        ax,
        (x_conc_ext + conc_wall_thk / 2, 54.0),
        (x_conc_ext - 40.0, 82.0),
        'Concrete wall\n(basement foundation)',
        ha="right",
    )
    _leader(
        ax,
        ((x_foot0 + x_foot1) / 2, (y_foot0 + y_foot1) / 2),
        (x_conc_ext - 26.0, -26.0),
        '20"×8" footing\n(5000 psi concrete)',
        ha="right",
    )
    _leader(
        ax,
        (drain_x, drain_y),
        (x_conc_ext - 26.0, -54.0),
        'French drain in washed stone\n(wider area, not under footing)',
        ha="right",
    )
    _leader(
        ax,
        (x_polyL1 - 0.6, 62.0),
        (x_sauna_left + 18.0, 68.0),
        '2" foil-faced polyiso (taped)\n(walls + ceiling)',
    )
    _leader(
        ax,
        ((x_furL0 + x_furL1) / 2, 76.0),
        (x_conc_ext - 26.0, 92.0),
        '1/2" plywood furring strips\n(fasteners per IRC Table R703.15.2)',
        ha="right",
    )
    _leader(
        ax,
        ((x_tgL0 + x_tgL1) / 2, 86.0),
        (x_sauna_left + 18.0, 92.0),
        '5/4 T&G boards (1" actual)\n(low-k species: basswood/poplar/aspen)',
    )
    _leader(
        ax,
        (x_studL0 + (x_stud0 + stud_depth - x_studL0) / 2, y_drop_bot + drop_depth / 2),
        (x_sauna_left + 34.0, y_drop_top + 10.0),
        '2x4 dropped ceiling framing\n below primary structure',
    )
    _leader(
        ax,
        (x_sauna_left + 0.4, 3.0),
        (x_sauna_left - 80.0, 18.0),
        '6" fiber cement baseboard\n(replaces furring + T&G)',
    )
    _leader(
        ax,
        ((x_sauna_left + x_sauna_right) / 2, y_slab_top + mem_thk / 2),
        (x_sauna_left + 20.0, 12.0),
        "Liquid membrane on slab\nextends up baseboard",
    )
    _leader(
        ax,
        ((duck_x0 + duck_x1) / 2, duck_y0 + duck_thk / 2),
        (x_sauna_left + 55.0, 24.0),
        "Duckboards on rubber feet\n(SS fasteners)",
    )
    _leader(
        ax,
        ((x_polyR1 + x_stud0) / 2, 62.0),
        (x_dry1 + 16.0, 80.0),
        "2×4 framing\nR-13 batt optional",
    )
    _leader(
        ax,
        (slab_x0 + 30.0, y_xps_bot + xps_under_thk / 2),
        (x_dry1 + 16.0, -24.0),
        'R-10 XPS under slab\n(≥25 psi)',
    )
    _leader(
        ax,
        (slab_x0 + 30.0, y_vapor_bot + vapor_sheet_thk / 2),
        (x_dry1 + 16.0, -44.0),
        "10 mil (min) polyethylene\nradon / vapor barrier",
    )
    _leader(
        ax,
        (x_conc_int + thermal_break_thk / 2, y_slab_bot + slab_thk / 2),
        (x_sauna_left + 4.0, -14.0),
        'Perimeter thermal break:\n1" XPS + 1/2" polyurethane sealant',
    )
    _leader(
        ax,
        (bench_x0 + bench_depth / 2, y_slab_top + upper_bench_top),
        (x_sauna_left + 30.0, 50.0),
        "Two-tier benches (Law of Löyly):\nupper ≈36\", lower ≈18\"",
    )
    _leader(
        ax,
        (heater_x0 + heater_w / 2, heater_y0 + heater_h / 2),
        (x_sauna_left + 15.0, 32.0),
        "Electric sauna heater (shown)\nplace near airflow path",
    )

    # Key dimensions
    # _dim_h(ax, x_sauna_left, x_sauna_right, 40.0, '8\'-0" clear width (sauna)')
    # _dim_v(ax, y_slab_top, y_joist_bot, x_dry1 + 8.0, "9'-0\" total height\n(to underside of joists)")

    ax.set_title("Sauna Detail", fontsize=13, fontweight="bold", loc="center", pad=12)
    ax.text(x_conc_ext + 54.0, y_stone1 - 20.0, "Colin Catlin, 2026", ha="center", va="top", fontsize=7)

    ax.set_xlim(x_conc_ext - 34.0, x_dry1 + 30.0)
    ax.set_ylim(y_stone1 - 18.0, y_joist_top + 24.0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        Patch(facecolor=COL_CONC, edgecolor="black", label="Concrete"),
        Patch(facecolor=COL_WOOD, edgecolor="black", label="Wood framing / T&G / furring"),
        Patch(facecolor=COL_POLYISO, edgecolor="black", label='Foil-faced polyiso (2")'),
        Patch(facecolor=COL_XPS, edgecolor="black", label='XPS (under slab / thermal break)'),
        Patch(facecolor=COL_MEM, edgecolor="black", label="Liquid membrane"),
        Patch(facecolor=COL_FC, edgecolor="black", label='Fiber cement baseboard (6")'),
        Patch(facecolor=COL_INSUL, edgecolor="black", label="Cavity insulation (optional)"),
        Patch(facecolor=COL_STONE, edgecolor="black", label="Washed stone / aggregate"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        bbox_to_anchor=(0.8, 0.01),
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
        "• Detail intent: schematic section showing sauna interior finishes, slab, and foundation bearing. Confirm all dimensions with the project plan set.",
        "",
        "• Interior sauna liner (walls + ceiling): 2\" foil-faced polyiso (taped seams). Polyiso is held in place with 1/2\" plywood furring strips; fasten per IRC Table R703.15.2 (table is for exterior cladding attachment—verify applicability and embedment for interior concrete/framing substrates).",
        "",
        "• Interior finish over furring: 5/4 tongue-and-groove boards (1\" actual). Use low thermal conductivity species such as American basswood, Canadian poplar, or aspen.",
        "",
        "• Wall/ceiling junction: detail as continuous layers (insulation meets insulation, wood meets wood). Tape/flash seams as required for vapor control and durability.",
        "",
        "• Support framing: 2x4 wall framed against concrete supports the dropped 2x4 ceiling. Primary structure above may be joists or concrete deck; hang drop framing accordingly.",
        "",
        "• Benches + heater (Law of Löyly): show two-tier bench (≈18\" + ≈36\" heights). Heater low and near airflow path; maintain clearances per manufacturer.",
        "",
        "• Base: 6\" fiber cement baseboard (or tile backer) at bottom of walls replaces T&G and furring. Provide flashing from polyiso over baseboard at top edge. Liquid floor membrane extends up the fiber cement baseboard.",
        "",
        "• Floor: 4\" concrete slab over R-10 XPS (≥25 psi) and 10 mil (min) polyethylene sheet (radon/vapor barrier). Top of slab: liquid membrane plus removable duckboards on rubber feet; stainless fasteners.",
        "",
        "• Thermal break / isolation joint: 1\" XPS with 1/2\" polyurethane sealant around sauna slab perimeter (shown schematically).",
        "",
        "• Electrical: supply 240V, 50A GFCI breaker and wiring to sauna heater (max 10.5 kW). For gas/wood appliances, reference MPC Section 615 and the appliance listing.",
        "",
        "• Lighting: IP65-rated LED strips concealed under lower bench lips + one waterproof wall sconce; keep drivers/transformers outside hot zone.",
        "",
        "• Ventilation: include HRV/ERV connections with adjustable cedar vent registers; intake low and away from heater, exhaust high above/near heater. Keep plastic vent pipe behind insulation.",
        "",
        "• Indicators: provide an exterior “in use” light, tied to heater control or via current-sensing relay.",
        "",
        "• Foundation: 10\" concrete wall bearing on 20\"×8\" footing per IRC Table R403.1 (confirm local requirements). Footing concrete 5000 psi. Footing bears on 6\" compacted washed stone aggregate (wider than footing) with French drain located in the wider area, not under the footing.",
    ]
    ax_notes.text(2, 146, _wrap_notes(raw_notes), fontsize=9, va="top", ha="left", family="monospace")

    out_path = "./sauna_basement_wall_detail.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out_path


if __name__ == "__main__":
    print(main())
