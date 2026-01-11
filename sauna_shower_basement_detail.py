"""
Schematic basement detail for a combined sauna + shower.

Run:
  python3 catlin-house/sauna_shower_basement_detail.py
Output:
  catlin-house/sauna_shower_basement_detail.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch, Polygon

from ifcplot.detail_utils import COLORS, HATCHES, _dim_h, _dim_v, _leader, _poly, _rect, _wrap_notes


def _sloped_layer(x0, x1, y0, y1, thickness):
    return np.array(
        [
            [x0, y0],
            [x1, y1],
            [x1, y1 - thickness],
            [x0, y0 - thickness],
        ]
    )


def main():
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.9, 1.1], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_notes = fig.add_subplot(gs[0, 1])

    # Colors
    COL = COLORS
    COL_CONC = COL.concrete
    COL_STONE = COL.stone
    COL_WOOD = COL.wood
    COL_POLYISO = COL.polyiso
    COL_XPS = COL.xps
    COL_MEM = COL.membrane
    COL_METAL = COL.metal
    COL_FC = COL.fiber_cement
    COL_DRY = COL.drywall
    COL_RUBBER = COL.rubber
    COL_INSUL = COL.insulation
    COL_GLASS = COL.glass
    COL_TILE = COL.tile
    COL_SEALANT = COL.sealant
    COL_POLY = COL.poly
    COL_EQUIP = COL.equipment

    # -----------------------
    # Geometry (inches, schematic)
    # -----------------------
    sauna_L = 96.0  # 8'
    shower_L = 48.0  # 4'
    total_L = sauna_L + shower_L  # 12'

    total_h = 108.0  # 9' to underside of joists (schematic)
    joist_depth = 9.25  # 2x10 (schematic)

    slab_thk = 4.0
    underfoam_thk = 2.0
    poly_sheet_thk = 0.05  # schematic only
    stone_under_thk = 4.0

    thermal_break_thk = 1.0
    sealant_thk = 0.5

    # Sauna floor slopes 1/8" per foot over 8' => 1" total
    sauna_rise_far_end = 1.0

    # Shower recess (top of shower slab is 4" below main basement floor level)
    shower_recess = 4.0

    # Shower finish build-up (varies in reality; shown here to bring finish flush at partition)
    shower_buildup_nom = 4.0
    tile_thk = 0.75  # tile + thinset (schematic)

    # Wall + finish thicknesses (schematic)
    stud_depth = 3.5
    wall_thk = 6.0
    polyiso_thk = 2.0
    furring_thk = 0.5
    tg_thk = 1.0
    baseboard_h = 6.0
    baseboard_thk = furring_thk + tg_thk
    drywall_thk = 0.625
    shower_wall_backer_thk = 1.0

    # Partition glass wall/door details
    glass_thk = 0.5
    glass_gap_bot = 1.0
    handle_standoff = 1.0

    # Reference levels
    y_main_ff = 0.0
    y_joist_bot = total_h
    y_joist_top = y_joist_bot + joist_depth

    drop_depth = 3.5
    y_drop_top = y_joist_bot - 1.0
    y_drop_bot = y_drop_top - drop_depth
    glass_h = y_drop_bot - (y_main_ff + glass_gap_bot)

    y_ceil_poly0 = y_drop_bot - polyiso_thk
    y_ceil_fur0 = y_ceil_poly0 - furring_thk
    y_ceil_tg0 = y_ceil_fur0 - tg_thk

    # x reference
    x0 = 0.0
    x_part = sauna_L
    x1 = total_L

    # Sauna slab top is sloped: +1" at far end -> 0" at partition
    y_sauna_top_0 = y_main_ff + sauna_rise_far_end
    y_sauna_top_1 = y_main_ff

    # Shower slab (recessed)
    y_shower_slab_top = y_main_ff - shower_recess

    # Under-slab elevations
    y_sauna_slab_bot_0 = y_sauna_top_0 - slab_thk
    y_sauna_slab_bot_1 = y_sauna_top_1 - slab_thk
    y_shower_slab_bot = y_shower_slab_top - slab_thk

    y_sauna_poly_bot_0 = y_sauna_slab_bot_0 - poly_sheet_thk
    y_sauna_poly_bot_1 = y_sauna_slab_bot_1 - poly_sheet_thk
    y_shower_poly_bot = y_shower_slab_bot - poly_sheet_thk

    y_sauna_foam_bot_0 = y_sauna_poly_bot_0 - underfoam_thk
    y_sauna_foam_bot_1 = y_sauna_poly_bot_1 - underfoam_thk
    y_shower_foam_bot = y_shower_poly_bot - underfoam_thk

    y_sauna_stone_bot_0 = y_sauna_foam_bot_0 - stone_under_thk
    y_sauna_stone_bot_1 = y_sauna_foam_bot_1 - stone_under_thk
    y_shower_stone_bot = y_shower_foam_bot - stone_under_thk

    # -----------------------
    # Slab + under-slab assemblies
    # -----------------------
    _poly(ax, _sloped_layer(x0, x_part, y_sauna_top_0, y_sauna_top_1, slab_thk), fc=COL_CONC, z=4)
    _poly(ax, _sloped_layer(x0, x_part, y_sauna_slab_bot_0, y_sauna_slab_bot_1, poly_sheet_thk), fc=COL_GLASS, lw=0.7, z=3)
    _poly(ax, _sloped_layer(x0, x_part, y_sauna_poly_bot_0, y_sauna_poly_bot_1, underfoam_thk), fc=COL_XPS, lw=1.0, z=2)
    _poly(ax, _sloped_layer(x0, x_part, y_sauna_foam_bot_0, y_sauna_foam_bot_1, stone_under_thk), fc=COL_STONE, lw=0.9, hatch=HATCHES.gravel, z=1)

    _rect(ax, x_part, y_shower_slab_top - slab_thk, shower_L, slab_thk, fc=COL_CONC, z=4)
    _rect(ax, x_part, y_shower_slab_bot - poly_sheet_thk, shower_L, poly_sheet_thk, fc=COL_GLASS, lw=0.7, z=3)
    _rect(ax, x_part, y_shower_poly_bot - underfoam_thk, shower_L, underfoam_thk, fc=COL_XPS, lw=1.0, z=2)
    _rect(ax, x_part, y_shower_foam_bot - stone_under_thk, shower_L, stone_under_thk, fc=COL_STONE, lw=0.9, hatch=HATCHES.gravel, z=1)

    # Thermal break / isolation joint around perimeter (schematic; shown at both ends)
    _rect(ax, x0 - thermal_break_thk, y_sauna_slab_bot_0, thermal_break_thk, slab_thk, fc=COL_XPS, lw=1.0, z=6)
    _rect(ax, x0 - thermal_break_thk, y_sauna_top_0, thermal_break_thk, sealant_thk, fc=COL_SEALANT, lw=0.9, z=7)

    _rect(ax, x1, y_shower_slab_top - slab_thk, thermal_break_thk, slab_thk, fc=COL_XPS, lw=1.0, z=6)
    _rect(ax, x1, y_main_ff, thermal_break_thk, sealant_thk, fc=COL_SEALANT, lw=0.9, z=7)

    # Adjacent main basement slab (outside shower/sauna room)
    main_slab_w = 24.0
    main_x0 = x1 + thermal_break_thk
    y_main_slab_bot = y_main_ff - slab_thk
    y_main_poly_bot = y_main_slab_bot - poly_sheet_thk
    y_main_foam_bot = y_main_poly_bot - underfoam_thk
    y_main_stone_bot = y_main_foam_bot - stone_under_thk
    _rect(ax, main_x0, y_main_slab_bot, main_slab_w, slab_thk, fc=COL_CONC, lw=1.1, z=4)
    _rect(ax, main_x0, y_main_poly_bot, main_slab_w, poly_sheet_thk, fc=COL_GLASS, lw=0.7, z=3)
    _rect(ax, main_x0, y_main_foam_bot, main_slab_w, underfoam_thk, fc=COL_XPS, lw=1.0, z=2)
    _rect(ax, main_x0, y_main_stone_bot, main_slab_w, stone_under_thk, fc=COL_STONE, lw=0.9, hatch=HATCHES.gravel, z=1)

    # Concrete step face at sauna/shower transition (recess)
    _rect(ax, x_part - 2.0, y_shower_slab_top, 2.0, (y_sauna_top_1 - y_shower_slab_top), fc=COL_CONC, lw=1.2, z=5)
    _rect(ax, x_part - 2.0, y_shower_slab_bot, 2.0, (y_shower_slab_top - y_shower_slab_bot), fc=COL_CONC, lw=1.2, z=5)

    # -----------------------
    # Sauna finishes (schematic, like reference)
    # -----------------------
    mem_thk = 0.25
    _poly(ax, _sloped_layer(x0 + 2.0, x_part - 2.0, y_sauna_top_0, y_sauna_top_1, mem_thk), fc=COL_MEM, lw=0.8, z=10)

    duck_thk = 1.0
    foot_h = 0.5
    duck_x0 = x0 + 4.0
    duck_x1 = x_part - 4.0
    duck_y0_0 = y_sauna_top_0 + mem_thk + foot_h
    duck_y0_1 = y_sauna_top_1 + mem_thk + foot_h
    _poly(ax, _sloped_layer(duck_x0, duck_x1, duck_y0_0, duck_y0_1, duck_thk), fc=COL_WOOD, lw=1.0, hatch=HATCHES.compacted, z=11)
    for fx in [duck_x0 + 6.0, duck_x1 - 6.0]:
        fy0 = y_sauna_top_0 + (fx - x0) * (y_sauna_top_1 - y_sauna_top_0) / (x_part - x0) + mem_thk
        _rect(ax, fx, fy0, 1.2, foot_h, fc=COL_RUBBER, ec="black", lw=0.8, z=12)

    # -----------------------
    # Sauna benches (schematic, two-tier along length)
    # -----------------------
    bench_depth = 20.0
    bench_thk = 1.5
    lower_bench_top = 18.0
    upper_bench_top = 36.0
    bench_setback = 2.0
    
    # Benches extend along the sauna length (x0 to x_part)
    bench_x0 = x0 + bench_setback
    bench_x1 = x_part - bench_setback - 4.0  # leave room near partition
    bench_length = bench_x1 - bench_x0
    
    # Lower bench (wider, extends full depth)
    lower_bench_y = y_main_ff + lower_bench_top - bench_thk
    _rect(ax, bench_x0, lower_bench_y, bench_length, bench_thk, fc=COL_WOOD, lw=1.0, z=12)
    
    # Upper bench (narrower, set back from lower bench)
    upper_bench_y = y_main_ff + upper_bench_top - bench_thk
    upper_bench_length = bench_length * 0.7  # shorter upper bench
    _rect(ax, bench_x0, upper_bench_y, upper_bench_length, bench_thk, fc=COL_WOOD, lw=1.0, z=12)
    
    # -----------------------
    # Sauna heater (schematic)
    # -----------------------
    heater_w = 10.0
    heater_h = 18.0
    heater_x0 = x0 + 6.0
    heater_y0 = y_main_ff + mem_thk
    _rect(ax, heater_x0, heater_y0, heater_w, heater_h, fc=COL_EQUIP, ec="black", lw=1.0, hatch=HATCHES.dense_plus, z=12)

    # HRV exhaust above heater (pipe in wall)
    hrv_exhaust_diam = 3.0
    hrv_exhaust_x = x0 - 0.5  # recessed into wall
    hrv_exhaust_y = heater_y0 + heater_h + 4.0  # just above heater
    _rect(ax, hrv_exhaust_x, hrv_exhaust_y, hrv_exhaust_diam, hrv_exhaust_diam, fc="white", ec="black", lw=1.0, z=13)
    ax.add_patch(Circle((hrv_exhaust_x + hrv_exhaust_diam / 2, hrv_exhaust_y + hrv_exhaust_diam / 2), radius=hrv_exhaust_diam / 2 - 0.2, facecolor=COL_METAL, edgecolor="none", alpha=0.6, zorder=13))

    # -----------------------
    # Shower finishes (shown as flush at partition; slope to drain shown in inset)
    # -----------------------
    # GoBoard on left side of recess (vertical wall piece - bottom 4" only)
    _rect(ax, x_part, y_shower_slab_top, shower_wall_backer_thk, shower_recess, fc=COL_POLYISO, lw=1.2, z=15)
    _rect(ax, x_part - tile_thk, y_shower_slab_top, tile_thk, shower_recess, fc=COL_TILE, lw=1.2, z=15)
    
    shower_fill_thk = max(shower_buildup_nom - shower_wall_backer_thk - tile_thk, 0)
    _rect(ax, x_part, y_shower_slab_top, shower_L, shower_wall_backer_thk, fc=COL_POLYISO, lw=1.0, hatch=HATCHES.compacted, z=9)
    if shower_fill_thk > 0:
        _rect(ax, x_part, y_shower_slab_top + shower_wall_backer_thk, shower_L, shower_fill_thk, fc=COL_POLYISO, lw=1.0, hatch=HATCHES.compacted, z=9)
    _rect(ax, x_part, y_shower_slab_top + shower_buildup_nom - tile_thk, shower_L, tile_thk, fc=COL_TILE, lw=1.0, z=10)

    # -----------------------
    # Shower entry door (shown in section, dotted)
    # -----------------------
    door_width = 36.0  # 3' standard
    door_height = 80.0  # 6'8" standard
    door_x0 = x_part + (shower_L - door_width) / 2  # centered in shower section
    door_y0 = y_main_ff
    _rect(ax, door_x0, door_y0, door_width, door_height, fc="none", ec="black", lw=1.0, ls=":", z=15)

    # -----------------------
    # Glass partition (enclosure wall + door; schematic in section)
    # -----------------------
    glass_x0 = x_part - glass_thk / 2
    glass_y0 = y_main_ff + glass_gap_bot
    _rect(ax, glass_x0, glass_y0, glass_thk, glass_h, fc=COL_GLASS, lw=1.1, z=20, alpha=0.55)

    # Door handle (spigots typically clamp at base; not shown in section)
    handle_w = 0.65
    handle_h = 12.0
    handle_y0 = y_main_ff + 40.0
    _rect(ax, x_part + handle_standoff, handle_y0, handle_w, handle_h, fc=COL_METAL, lw=0.9, z=21)
    ax.plot(
        [x_part + glass_thk / 2, x_part + handle_standoff],
        [handle_y0 + handle_h / 2, handle_y0 + handle_h / 2],
        color="black",
        lw=0.9,
        zorder=21,
    )

    # -----------------------
    # Suggestive end walls (schematic)
    # -----------------------
    wall_y0 = y_main_ff
    wall_y1 = y_joist_bot

    # HRV intake on shower wall (halfway up)
    hrv_intake_diam = 3.0
    hrv_intake_x = x1 - shower_wall_backer_thk - tile_thk + 0.5  # recessed into wall
    hrv_intake_y = y_main_ff + (y_joist_bot - y_main_ff) / 2 - hrv_intake_diam / 2
    _rect(ax, hrv_intake_x, hrv_intake_y, hrv_intake_diam, hrv_intake_diam, fc="white", ec="black", lw=1.0, z=13)
    ax.add_patch(Circle((hrv_intake_x + hrv_intake_diam / 2, hrv_intake_y + hrv_intake_diam / 2), radius=hrv_intake_diam / 2 - 0.2, facecolor=COL_METAL, edgecolor="none", alpha=0.6, zorder=13))

    # Sauna end wall (left)
    x_sauna_tg0 = x0 - tg_thk
    x_sauna_fur0 = x_sauna_tg0 - furring_thk
    x_sauna_poly0 = x_sauna_fur0 - polyiso_thk
    _rect(ax, x0 - wall_thk, wall_y0, (x_sauna_poly0 - (x0 - wall_thk)), wall_y1 - wall_y0, fc=COL_INSUL, lw=1.0, hatch=HATCHES.compacted, z=0)
    _rect(ax, x_sauna_poly0, wall_y0, polyiso_thk, y_drop_bot - wall_y0, fc=COL_POLYISO, lw=1.0, z=2)
    _rect(ax, x_sauna_fur0, baseboard_h, furring_thk, y_ceil_tg0 - baseboard_h, fc=COL_WOOD, lw=1.0, hatch=HATCHES.diagonal, z=3)
    _rect(ax, x_sauna_tg0, baseboard_h, tg_thk, y_ceil_tg0 - baseboard_h, fc=COL_WOOD, lw=1.1, z=4)
    _rect(ax, x_sauna_fur0, wall_y0, baseboard_thk, baseboard_h, fc=COL_FC, lw=1.1, z=5)

    # Shower end wall (right) with foam tile backer to ceiling
    _rect(ax, x1, wall_y0 - shower_recess, stud_depth, wall_y1 - (wall_y0 - shower_recess), fc=COL_INSUL, lw=1.0, hatch=HATCHES.compacted, z=0)
    _rect(ax, x1, wall_y0 - shower_recess, shower_wall_backer_thk, wall_y1 - (wall_y0 - shower_recess), fc=COL_POLYISO, lw=1.0, z=2)
    _rect(ax, x1 - shower_wall_backer_thk, wall_y0 - shower_recess, tile_thk, wall_y1 - (wall_y0 - shower_recess), fc=COL_TILE, lw=1.0, z=3)
    _rect(ax, x1 + stud_depth, wall_y0 - shower_recess, drywall_thk, wall_y1 - (wall_y0 - shower_recess), fc=COL_DRY, lw=1.0, z=1)

    # Joists above (schematic)
    joist_band_x0 = x0 - 10.0
    joist_band_w = (x1 - x0) + 20.0
    _rect(ax, joist_band_x0, y_joist_bot, joist_band_w, joist_depth, fc=COL_WOOD, lw=1.2, hatch=HATCHES.joist, z=2)
    joist_oc = 16.0
    for jx in np.arange(x0, x1 + 1e-6, joist_oc):
        ax.plot([jx, jx], [y_joist_bot, y_joist_top], color="black", lw=0.7, alpha=0.65, zorder=3)

    # Dropped ceiling framing below joists (schematic)
    # Framing extends over side walls
    drop_frame_x0 = x0 - wall_thk
    drop_frame_w = (x1 - x0) + wall_thk + stud_depth + drywall_thk
    _rect(ax, drop_frame_x0, y_drop_bot, drop_frame_w, drop_depth, fc=COL_WOOD, lw=1.1, hatch=HATCHES.cross, z=4, alpha=0.9)
    # Polyiso on left extends 2" past x0, but not over right wall
    poly_extend_left = 2.0
    drop_poly_x0 = x0 - poly_extend_left
    drop_poly_w = (x1 - x0) + poly_extend_left
    _rect(ax, drop_poly_x0, y_ceil_poly0, drop_poly_w, polyiso_thk, fc=COL_POLYISO, lw=1.0, z=5)
    # Furring and T&G stay at original extent
    drop_x0 = x0
    drop_w = (x1 - x0)
    _rect(ax, drop_x0, y_ceil_fur0, drop_w, furring_thk, fc=COL_WOOD, lw=1.0, hatch=HATCHES.diagonal, z=6)
    _rect(ax, drop_x0, y_ceil_tg0, drop_w, tg_thk, fc=COL_WOOD, lw=1.1, z=7)

    # -----------------------
    # Inset: shower transverse slope to drain (8' width)
    # -----------------------
    ax_in = ax.inset_axes([0.03, -0.06, 0.24, 0.18])
    ax_in.axis("off")
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(0.97)
    for spine in ax_in.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor(COL_METAL)

    sh_w = 96.0  # 8' width (door wall -> far wall)
    y_ff = 0.0
    y_slab_top = -shower_recess
    y_slab_bot = y_slab_top - slab_thk

    slope = 0.25 / 12.0  # 1/4" per foot
    drop = slope * sh_w  # 2" total over 8'
    y_tile_at_door = y_ff
    y_tile_at_drain = y_ff - drop

    # slab + under slab (flat in this inset)
    _rect(ax_in, 0.0, y_slab_bot, sh_w, slab_thk, fc=COL_CONC, lw=1.0, z=2)
    _rect(ax_in, 0.0, y_slab_bot - poly_sheet_thk, sh_w, poly_sheet_thk, fc=COL_GLASS, lw=0.6, z=1)
    _rect(ax_in, 0.0, y_slab_bot - poly_sheet_thk - underfoam_thk, sh_w, underfoam_thk, fc=COL_XPS, lw=0.8, z=1)

    # GoBoard/wedge build-up (schematic taper)
    buildup0 = y_tile_at_door - y_slab_top - tile_thk
    buildup1 = y_tile_at_drain - y_slab_top - tile_thk
    base_backer_thk = shower_wall_backer_thk
    _rect(ax_in, 0.0, y_slab_top, sh_w, base_backer_thk, fc=COL_POLYISO, lw=0.9, hatch=HATCHES.compacted, z=2.5)
    fill0 = max(buildup0 - base_backer_thk, 0)
    fill1 = max(buildup1 - base_backer_thk, 0)
    _poly(
        ax_in,
        [
            [0.0, y_slab_top + base_backer_thk],
            [sh_w, y_slab_top + base_backer_thk],
            [sh_w, y_slab_top + base_backer_thk + fill1],
            [0.0, y_slab_top + base_backer_thk + fill0],
        ],
        fc=COL_POLYISO,
        ec="black",
        lw=0.9,
        hatch=HATCHES.compacted,
        z=3,
    )
    _poly(
        ax_in,
        [
            [0.0, y_tile_at_door],
            [sh_w, y_tile_at_drain],
            [sh_w, y_tile_at_drain - tile_thk],
            [0.0, y_tile_at_door - tile_thk],
        ],
        fc=COL_TILE,
        ec="black",
        lw=0.9,
        z=4,
    )

    # Drain at far wall: centered in 4' length, shown schematically at far wall
    drain_r = 2.2
    ax_in.add_patch(Circle((sh_w - 2.0, y_tile_at_drain - tile_thk / 2), radius=drain_r, facecolor="white", edgecolor="black", lw=1.0, zorder=6))
    ax_in.add_patch(Circle((sh_w - 2.0, y_tile_at_drain - tile_thk / 2), radius=drain_r - 0.3, facecolor=COL_METAL, edgecolor="none", alpha=0.75, zorder=6))

    ax_in.set_xlim(-6.0, sh_w + 10.0)
    ax_in.set_ylim(-12.0, 12.0)
    
    # Add title below the inset
    ax_in.text(sh_w / 2, -11.5, "Shower floor slope (transverse section)", fontsize=9, ha="center", va="top")

    # -----------------------
    # Drain + pipe (schematic, in main section)
    # -----------------------
    drain_x = x_part + shower_L / 2
    pipe_diam = 2.0
    pipe_x0 = drain_x - pipe_diam / 2
    pipe_y_top = y_main_ff - tile_thk
    pipe_y_bot = y_shower_stone_bot + 0.5
    _rect(ax, pipe_x0, pipe_y_bot, pipe_diam, pipe_y_top - pipe_y_bot, fc=COL_METAL, lw=0.9, z=25, alpha=0.85)

    # -----------------------
    # Leaders / callouts
    # -----------------------
    _leader(
        ax,
        (x_part + 18.0, y_main_ff - 1.0),
        (x_part + 18.0, 18.0),
        'Shower area (4\'-0") recessed slab\n(-4" to top of concrete)',
        ha="left",
    )
    _leader(
        ax,
        (x0 + 18.0, (y_sauna_top_0 + y_sauna_top_1) / 2),
        (x0 + 10.0, 42.0),
        'Sauna floor slopes 1/8" per ft\n(1" over 8\') toward shower',
        ha="left",
    )
    _leader(
        ax,
        (x_part, glass_y0 + 12.0),
        (x_part - 36.0, 84.0),
        'Glass partition wall + 36" door\n1/2" tempered glass\nhandle shown; base spigots not shown',
        ha="right",
    )
    _leader(
        ax,
        (x_part, y_main_ff + glass_gap_bot / 2),
        (x_part - 34.0, 14.0),
        '1" gap under partition wall\n(air + water flow; maintain cleanable edge)',
        ha="right",
    )
    _leader(
        ax,
        (x1 - 1.0, 66.0),
        (x1 + 28.0, 80.0),
        "Shower walls: foam tile backer\n(GoBoard/polyiso preferred)\nup to ceiling",
        ha="left",
    )
    _leader(
        ax,
        (drain_x, pipe_y_top - 0.2),
        (x1 + 26.0, 44.0),
        "Shower drain pipe (schematic)\nroute to trap/vent per plumbing",
        ha="left",
    )
    _leader(
        ax,
        ((bench_x0 + bench_x1) / 2, upper_bench_y + bench_thk / 2),
        (x0 + 48.0, 52.0),
        "Two-tier benches (Law of Löyly):\nupper ≈36\", lower ≈18\"",
        ha="left",
    )
    _leader(
        ax,
        (heater_x0 + heater_w / 2, heater_y0 + heater_h / 2),
        (x0 + 32.0, 32.0),
        "Sauna heater (electric)\nplace near airflow path",
        ha="left",
    )
    _leader(
        ax,
        (hrv_exhaust_x + hrv_exhaust_diam / 2, hrv_exhaust_y + hrv_exhaust_diam / 2),
        (x0 + 32.0, 42.0),
        "HRV exhaust\n(low, above heater)",
        ha="left",
    )
    _leader(
        ax,
        (hrv_intake_x + hrv_intake_diam / 2, hrv_intake_y + hrv_intake_diam / 2),
        (x1 + 26.0, 60.0),
        "HRV intake\n(mid-height)",
        ha="left",
    )
    _leader(
        ax,
        (x0 - thermal_break_thk / 2, y_sauna_top_0 + 0.2),
        (x0 + 22.0, 96.0),
        'Thermal break: 1" XPS + sealant\naround shower/sauna perimeter',
        ha="left",
    )

    # Key dimensions
    _dim_h(ax, x0, x_part, -28.0, '8\'-0" sauna length')
    _dim_h(ax, x_part, x1, -28.0, '4\'-0" shower length')
    _dim_h(ax, x0, x1, -38.0, '12\'-0" overall length')
    _dim_v(ax, y_main_ff - shower_recess, y_main_ff, x1 + 10.0, '4" shower recess')

    ax.set_title("Sauna + Shower Basement Detail (schematic section)", fontsize=13, fontweight="bold", loc="center", pad=12)
    ax.text(x0 + total_L / 2, y_shower_stone_bot - 10.0, "Colin Catlin, 2026", ha="center", va="top", fontsize=7)

    ax.set_xlim(x0 - 28.0, x1 + 44.0)
    ax.set_ylim(y_shower_stone_bot - 16.0, y_joist_top + 22.0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        Patch(facecolor=COL_CONC, edgecolor="black", label="Concrete"),
        Patch(facecolor=COL_XPS, edgecolor="black", label='Foam under slab / XPS (2")'),
        Patch(facecolor=COL_GLASS, edgecolor="black", label="Vapor barrier (schematic)"),
        Patch(facecolor=COL_STONE, edgecolor="black", label="Washed stone / aggregate"),
        Patch(facecolor=COL_MEM, edgecolor="black", label="Liquid membrane"),
        Patch(facecolor=COL_POLYISO, edgecolor="black", label="Foam board / wedges / backer"),
        Patch(facecolor=COL_TILE, edgecolor="black", label="Tile / finish layer"),
        Patch(facecolor=COL_GLASS, edgecolor="black", label="Glass (partition)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        bbox_to_anchor=(0.92, 0.01),
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
    ax_notes.set_ylim(0, 165)
    ax_notes.axis("off")

    raw_notes = [
        "NOTES:",
        "",
        "• Intent: schematic section for a combined sauna + shower room in a basement on slab. Confirm all dimensions, waterproofing, and structural requirements with the project drawings and manufacturers.",
        "",
        '• Room size: 12\'×8\'. Shower is the 4\' end (4\'×8\'). Primary entry door is on the shower from the long (12\') side wall (not shown in section).',
        "",
        '• Slabs: both sauna and shower are slab-on-grade over vapor barrier and foam, with a thermal break around the perimeter of the combined room. Shower slab is recessed 4" below the main basement floor level (entire assembly shifted down).',
        "",
        '• Sauna floor: slopes 1/8" per foot over 8\' (1" total) down toward the shower end; detail shown as sloped slab schematically.',
        "",
        '• Shower floor: finish is built up using foam tile backer (GoBoard or other polyiso-based board preferred) and wedges; slope to drain is transverse from the entry wall to the far wall (see inset). Coordinate with drain height and waterproofing membrane.',
        "",
        '• Drain: centered in the 4\' shower length and set against the far wall from the entry door (plan location). Provide appropriate trap/vent and waterproofing transitions per the drain system.',
        "",
        '• Shower walls: foam tile backer board to ceiling (polyiso backer preferred for heat tolerance); finish with tile system and compatible waterproofing.',
        "",
        '• Partition: glass shower enclosure wall + 36" glass divider door between shower and sauna. Partition is elevated leaving a 1" gap at bottom for air + water flow. Use 1/2" tempered glass with 2–3 spigots (sealant anchor method) and ≥1" standoff; maintain cleanable, durable edges at the floor.',
        "",
        "• Electrical: supply 240V, 50A GFCI breaker and wiring to sauna heater (max 10.5 kW). For gas/wood appliances, reference MPC Section 615 and the appliance listing.",
        "",
        "• Lighting: IP65-rated LED strips concealed under lower bench lips + one waterproof wall sconce; keep drivers/transformers outside hot zone.",
        "",
        "• Ventilation: include HRV/ERV connections with adjustable cedar vent registers; intake low and away from heater, exhaust high above/near heater. Keep plastic vent pipe behind insulation.",
        "",
        "• Coordinate: thresholds (flush vs. step), any curbs, slip resistance, and transitions between sauna membrane/duckboards and shower tile.",
    ]
    ax_notes.text(2, 162, _wrap_notes(raw_notes), fontsize=9, va="top", ha="left", family="monospace")

    out_path = Path(__file__).resolve().with_suffix(".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return str(out_path)


if __name__ == "__main__":
    print(main())
