"""
Schematic retaining wall detail for a "sunken garden" (double-sided section).

Run:
  python3 catlin-house/sunken_garden_retaining_wall_detail.py
Output:
  catlin-house/sunken_garden_retaining_wall_detail.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch

from detail_utils import COLORS, _dim_h, _dim_v, _leader, _poly, _rect, _wrap_notes


def _pipe(ax, x, y, d, *, fc=COLORS.metal, ec="black", lw=1.1, z=6):
    ax.add_patch(Circle((x, y), radius=d / 2, facecolor="white", edgecolor=ec, lw=lw, zorder=z))
    ax.add_patch(Circle((x, y), radius=d / 2 - 0.25, facecolor=fc, edgecolor="none", alpha=0.85, zorder=z + 1))


def _dotted_pipe(ax, p0, p1, d, *, z=15):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color="black",
        linewidth=1.6,
        linestyle=(0, (1.0, 2.2)),
        zorder=z,
    )
    for p in (p0, p1):
        ax.add_patch(Circle((p[0], p[1]), radius=d / 2, facecolor="white", edgecolor="black", lw=1.0, zorder=z + 1))
        ax.add_patch(
            Circle((p[0], p[1]), radius=d / 2 - 0.25, facecolor=COLORS.metal, edgecolor="none", alpha=0.85, zorder=z + 2)
        )


def _rebar_L(ax, x_cover_from_face, x_face, y_base_bot, y_base_top, y_wall_top, *, hook=14.0, z=20):
    """
    Draw a schematic L-bar: vertical along wall, hooks into base slab.

    x_face: soil-facing outer face of stem.
    x_cover_from_face: positive distance into concrete from x_face.
    """

    x_vert = x_face + x_cover_from_face
    y_hook = y_base_bot + 3.0
    x_hook = x_vert + np.sign(x_cover_from_face) * hook

    ax.plot([x_vert, x_vert, x_hook], [y_wall_top - 6.0, y_hook, y_hook], color="black", linewidth=1.6, zorder=z)
    ax.add_patch(Circle((x_vert, y_base_top + 24.0), radius=0.85, facecolor="black", edgecolor="none", zorder=z + 1))


def _rebar_h(ax, x0, x1, y, *, z=20):
    ax.plot([x0, x1], [y, y], color="black", linewidth=1.4, zorder=z)
    for x in (x0 + 10.0, x1 - 10.0):
        ax.add_patch(Circle((x, y), radius=0.75, facecolor="black", edgecolor="none", zorder=z + 1))


def main():
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.9, 1.1], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    ax_notes = fig.add_subplot(gs[0, 1])

    # Colors
    COL = COLORS
    COL_CONC = COL.concrete
    COL_GRAVEL = COL.stone
    COL_AGG = COL.aggregate
    COL_SOIL = COL.soil
    COL_XPS = COL.xps
    COL_BLOCK = COL.block
    COL_MEM = COL.membrane

    # Units: inches (schematic)
    inner_clear = 216.0  # 18'
    wall_h = 120.0  # 10'
    backfill_h = 84.0  # 7'
    raised_bed_h = 36.0  # 3'
    raised_bed_w = 36.0  # 3'

    stem_thk = 12.0
    base_thk = 12.0
    toe = 36.0
    heel = 36.0

    frost_depth = 42.0
    toe_gravel_depth = frost_depth

    xps_thk = 1.0  # under outer heel arm

    drain_diam = 4.0
    drain_stone_w = 24.0
    drain_backfill_h = backfill_h

    compacted_agg_thk = 18.0
    compacted_agg_extra = 6.0

    # Coordinate system: sunken garden "interior" grade is y=0.
    y_int_grade = 0.0
    y_base_top = y_int_grade
    y_base_bot = y_base_top - base_thk
    y_agg_top = y_base_bot
    y_agg_bot = y_agg_top - compacted_agg_thk
    y_wall_top = y_base_top + wall_h

    # Two walls: inner faces at x=0 and x=inner_clear
    xL_in = 0.0
    xL_out = xL_in - stem_thk
    xR_in = inner_clear
    xR_out = xR_in + stem_thk

    # Base extents (toe is interior side, heel is soil side)
    xL_base0 = xL_out - heel
    xL_base1 = xL_in + toe
    xR_base0 = xR_in - toe
    xR_base1 = xR_out + heel

    # Raised garden (top 3' high) sits on soil side of each wall and extends outward 3'
    block_thk = 12.0  # retaining wall block depth (schematic)
    xL_bed_inner = xL_out
    xL_bed_outer = xL_bed_inner - raised_bed_w
    xL_block0 = xL_bed_outer - block_thk
    xL_block1 = xL_bed_outer

    xR_bed_inner = xR_out
    xR_bed_outer = xR_bed_inner + raised_bed_w
    xR_block0 = xR_bed_outer
    xR_block1 = xR_bed_outer + block_thk

    # -----------------------
    # Concrete: stems + bases
    # -----------------------
    # Compacted aggregate base (extends under stem and beyond footing edges)
    _rect(
        ax,
        xL_base0 - compacted_agg_extra,
        y_agg_bot,
        (xL_base1 - xL_base0) + 2 * compacted_agg_extra,
        compacted_agg_thk,
        fc=COL_AGG,
        ec="black",
        lw=1.0,
        hatch="..",
        z=2,
    )
    _rect(
        ax,
        xR_base0 - compacted_agg_extra,
        y_agg_bot,
        (xR_base1 - xR_base0) + 2 * compacted_agg_extra,
        compacted_agg_thk,
        fc=COL_AGG,
        ec="black",
        lw=1.0,
        hatch="..",
        z=2,
    )

    # Left wall stem and base
    _rect(ax, xL_out, y_base_top, stem_thk, wall_h, fc=COL_CONC, z=5)
    _rect(ax, xL_base0, y_base_bot, xL_base1 - xL_base0, base_thk, fc=COL_CONC, z=4)

    # Right wall stem and base
    _rect(ax, xR_in, y_base_top, stem_thk, wall_h, fc=COL_CONC, z=5)
    _rect(ax, xR_base0, y_base_bot, xR_base1 - xR_base0, base_thk, fc=COL_CONC, z=4)

    # Interior grade line (sunken garden floor)
    ax.plot([xL_in, xR_in], [y_int_grade, y_int_grade], color="black", linewidth=1.2, zorder=30)
    ax.text((xL_in + xR_in) / 2, y_int_grade + 1.5, "Sunken garden floor / grade (typ.)", fontsize=9, ha="center", va="bottom")

    # -----------------------
    # Toe frost-protected gravel trench (geotextile-wrapped)
    # -----------------------
    # Left toe: under toe area inside span.
    xL_toe0 = xL_in
    xL_toe1 = xL_in + toe
    _rect(ax, xL_toe0, y_base_bot - toe_gravel_depth, xL_toe1 - xL_toe0, toe_gravel_depth, fc=COL_GRAVEL, hatch="o", lw=1.0, z=1)
    _rect(
        ax,
        xL_toe0 - 1.5,
        y_base_bot - toe_gravel_depth - 1.5,
        (xL_toe1 - xL_toe0) + 3.0,
        toe_gravel_depth + 3.0,
        fc="none",
        ec="black",
        lw=1.0,
        ls="--",
        z=2,
    )

    # Right toe
    xR_toe0 = xR_in - toe
    xR_toe1 = xR_in
    _rect(ax, xR_toe0, y_base_bot - toe_gravel_depth, xR_toe1 - xR_toe0, toe_gravel_depth, fc=COL_GRAVEL, hatch="o", lw=1.0, z=1)
    _rect(
        ax,
        xR_toe0 - 1.5,
        y_base_bot - toe_gravel_depth - 1.5,
        (xR_toe1 - xR_toe0) + 3.0,
        toe_gravel_depth + 3.0,
        fc="none",
        ec="black",
        lw=1.0,
        ls="--",
        z=2,
    )

    # -----------------------
    # XPS under outer heel arms
    # -----------------------
    # Left heel zone under base
    _rect(ax, xL_base0, y_base_bot - xps_thk, (xL_out - xL_base0), xps_thk, fc=COL_XPS, ec="black", lw=0.9, z=3)
    # Right heel zone under base
    _rect(ax, xR_out, y_base_bot - xps_thk, (xR_base1 - xR_out), xps_thk, fc=COL_XPS, ec="black", lw=0.9, z=3)

    # -----------------------
    # Soil + raised garden soil
    # -----------------------
    # Outside (soil side) grade is 7' above interior grade.
    # Left side: backfill soil to 7', then raised garden soil to 10' within a 3' wide bed.
    xL_soil_far = xL_block0 - 36.0
    left_backfill = [
        [xL_out, y_base_top],
        [xL_soil_far, y_base_top],
        [xL_soil_far, backfill_h],
        [xL_out, backfill_h],
    ]
    _poly(ax, left_backfill, fc=COL_SOIL, ec="none", alpha=0.55, z=0)
    ax.plot([xL_out, xL_soil_far], [backfill_h, backfill_h], color="black", linewidth=1.1, zorder=10)

    left_bed_soil = [
        [xL_bed_inner, backfill_h],
        [xL_bed_outer, backfill_h],
        [xL_bed_outer, y_wall_top],
        [xL_bed_inner, y_wall_top],
    ]
    _poly(ax, left_bed_soil, fc=COL_SOIL, ec="none", alpha=0.55, z=0)

    # Right side
    xR_soil_far = xR_block1 + 36.0
    right_backfill = [
        [xR_out, y_base_top],
        [xR_soil_far, y_base_top],
        [xR_soil_far, backfill_h],
        [xR_out, backfill_h],
    ]
    _poly(ax, right_backfill, fc=COL_SOIL, ec="none", alpha=0.55, z=0)
    ax.plot([xR_out, xR_soil_far], [backfill_h, backfill_h], color="black", linewidth=1.1, zorder=10)

    right_bed_soil = [
        [xR_bed_inner, backfill_h],
        [xR_bed_outer, backfill_h],
        [xR_bed_outer, y_wall_top],
        [xR_bed_inner, y_wall_top],
    ]
    _poly(ax, right_bed_soil, fc=COL_SOIL, ec="none", alpha=0.55, z=0)

    # -----------------------
    # Raised garden outer block walls (retaining wall blocks)
    # -----------------------
    _rect(ax, xL_block0, backfill_h, xL_block1 - xL_block0, raised_bed_h, fc=COL_BLOCK, hatch="++", lw=1.1, z=6)
    _rect(ax, xR_block0, backfill_h, xR_block1 - xR_block0, raised_bed_h, fc=COL_BLOCK, hatch="++", lw=1.1, z=6)

    # -----------------------
    # Granular backfill + drain tile (geotextile wrapped) on soil side (extends above footing)
    # -----------------------
    # Left drain stone/backfill zone
    xL_drain0 = xL_out - drain_stone_w
    xL_drain1 = xL_out
    y_drain0 = y_base_top
    y_drain1 = drain_backfill_h
    _rect(ax, xL_drain0, y_drain0, xL_drain1 - xL_drain0, y_drain1 - y_drain0, fc=COL_GRAVEL, hatch="o", lw=0.9, z=2)
    _rect(ax, xL_drain0 - 1.2, y_drain0 - 1.2, (xL_drain1 - xL_drain0) + 2.4, (y_drain1 - y_drain0) + 2.4, fc="none", ec="black", ls="--", lw=1.0, z=3)
    _pipe(ax, xL_out - drain_stone_w * 0.55, y_base_top + drain_diam * 1.0, drain_diam)

    # Right drain stone/backfill zone
    xR_drain0 = xR_out
    xR_drain1 = xR_out + drain_stone_w
    _rect(ax, xR_drain0, y_drain0, xR_drain1 - xR_drain0, y_drain1 - y_drain0, fc=COL_GRAVEL, hatch="o", lw=0.9, z=2)
    _rect(ax, xR_drain0 - 1.2, y_drain0 - 1.2, (xR_drain1 - xR_drain0) + 2.4, (y_drain1 - y_drain0) + 2.4, fc="none", ec="black", ls="--", lw=1.0, z=3)
    _pipe(ax, xR_out + drain_stone_w * 0.55, y_base_top + drain_diam * 1.0, drain_diam)

    # -----------------------
    # Waterproofing / dampproofing on soil-facing side of wall
    # -----------------------
    mem_thk = 0.35
    _rect(ax, xL_out - mem_thk, y_base_top, mem_thk, wall_h, fc=COL_MEM, ec="black", lw=0.8, z=7)
    _rect(ax, xR_out, y_base_top, mem_thk, wall_h, fc=COL_MEM, ec="black", lw=0.8, z=7)

    # -----------------------
    # Weep holes: sloped dotted pipes through full wall depth
    # -----------------------
    weep_y = y_base_top + 18.0
    weep_pipe_d = 2.0
    weep_rise = 2.0
    _dotted_pipe(ax, (xL_in, weep_y), (xL_out, weep_y + weep_rise), weep_pipe_d, z=12)
    _dotted_pipe(ax, (xR_in, weep_y), (xR_out, weep_y + weep_rise), weep_pipe_d, z=12)

    # -----------------------
    # Rebar: L-bar from base into wall; vertical along soil face with 3" cover
    # -----------------------
    cover = 3.0
    # Left wall soil face is xL_out
    _rebar_L(ax, +cover, xL_out, y_base_bot, y_base_top, y_wall_top)
    # Right wall soil face is xR_out (outer face to the right)
    _rebar_L(ax, -cover, xR_out, y_base_bot, y_base_top, y_wall_top)
    # Footing steel (schematic, continuous)
    _rebar_h(ax, xL_base0 + 6.0, xL_base1 - 6.0, y_base_bot + 4.0)
    _rebar_h(ax, xR_base0 + 6.0, xR_base1 - 6.0, y_base_bot + 4.0)

    # -----------------------
    # Dimensions
    # -----------------------
    _dim_h(ax, xL_in, xR_in, y_int_grade + 36.0, '18\'-0" clear between walls')
    _dim_v(ax, y_base_top, y_wall_top, xR_block1 + 22.0, '10\'-0" retaining wall height')
    _dim_v(ax, y_base_top, backfill_h, xR_block1 + 12.0, '7\'-0" soil backfill (typ.)')
    _dim_v(ax, backfill_h, y_wall_top, xR_block1 + 12.0, '3\'-0" raised garden soil')
    dim_foot_y = (y_base_bot - toe_gravel_depth) - 10.0
    _dim_h(ax, xL_base0, xL_out, dim_foot_y, '3\'-0" heel')
    _dim_h(ax, xL_in, xL_base1, dim_foot_y, '3\'-0" toe')

    # -----------------------
    # Leaders / callouts
    # -----------------------
    _leader(
        ax,
        (xL_out + stem_thk / 2, y_base_top + 70.0),
        (xL_block0 - 80.0, y_wall_top - 6.0),
        '12" concrete stem\n(soil side is outside)',
        ha="right",
    )
    _leader(
        ax,
        (xL_base0 + 14.0, y_base_bot + base_thk / 2),
        (xL_block0 - 80.0, y_base_bot - 16.0),
        '12" thick footing/base slab\nextends 3\'-0" beyond each wall face (typ.)',
        ha="right",
        va="top",
    )
    _leader(
        ax,
        (xL_base0 + 10.0, y_agg_bot + compacted_agg_thk / 2),
        (xL_out - 86.0, y_agg_bot - 10.0),
        "Compacted crushed stone base\nextends under stem + footing (typ.)",
        ha="right",
        va="top",
    )
    _leader(
        ax,
        (xL_out - mem_thk / 2, y_base_top + 40.0),
        (xL_block0 - 80.0, y_base_top + 28.0),
        "Dampproofing / waterproofing\nrequired (soil-facing)",
        ha="right",
    )
    _leader(
        ax,
        (xL_out + cover, y_base_top + 60.0),
        (xL_in + 56.0, y_wall_top - 18.0),
        'Primary vertical stem steel\nat soil face (3" cover)\nL-dowels into footing (typ.)',
    )
    _leader(
        ax,
        (xL_out - drain_stone_w * 0.55, y_base_top + drain_diam * 1.0),
        (xL_block0 - 80.0, y_base_top + 6.0),
        '4" perf drain tile (typ.)\nplace at base of drain stone; slope to daylight / sump',
        ha="right",
    )
    _leader(
        ax,
        (xL_out - drain_stone_w / 2, backfill_h * 0.55),
        (xL_block0 - 80.0, y_base_top + 26.0),
        "Granular backfill / drain stone\nextends up wall; geotextile-wrapped",
        ha="right",
    )
    _leader(
        ax,
        ((xL_toe0 + xL_toe1) / 2, y_base_bot - toe_gravel_depth / 2),
        (xL_in + 28.0, (y_base_bot - toe_gravel_depth) - 12.0),
        '42" washed angular gravel\n(geotextile-wrapped) under toe',
    )
    _leader(
        ax,
        (xL_base0 + 12.0, y_base_bot - xps_thk / 2),
        (xL_out - 86.0, y_base_bot - 6.0),
        '1" XPS (taped seams)\nunder outer heel arm',
        ha="right",
        va="top",
    )
    _leader(
        ax,
        (xL_in + 2.0, weep_y + 0.1),
        (xL_in + 34.0, weep_y + 14.0),
        "Weep holes @ 8' o.c. (typ.)\nsloped weep pipes through wall",
    )
    _leader(
        ax,
        ((xL_block0 + xL_block1) / 2, backfill_h + raised_bed_h / 2),
        (xL_block0 - 80.0, backfill_h + 28.0),
        'Retaining wall blocks (typ.)\n8"H × 18"L × 12"D',
        ha="right",
    )

    ax.set_title("Sunken Garden — Double Cantilever Retaining Wall Section (Schematic)", fontsize=13, fontweight="bold", pad=12)
    ax.text((xL_in + xR_in) / 2, (y_base_bot - toe_gravel_depth) - 18.0, "Colin Catlin, 2026", ha="center", va="top", fontsize=7)

    # Framing / plot formatting
    ax.set_xlim(xL_block0 - 120.0, xR_block1 + 120.0)
    y_bottom = min(y_agg_bot, y_base_bot - toe_gravel_depth) - 38.0
    ax.set_ylim(y_bottom, y_wall_top + 22.0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        Patch(facecolor=COL_CONC, edgecolor="black", label="Concrete"),
        Patch(facecolor=COL_SOIL, edgecolor="black", label="Soil / backfill"),
        Patch(facecolor=COL_GRAVEL, edgecolor="black", label="Washed stone / gravel"),
        Patch(facecolor=COL_AGG, edgecolor="black", label="Compacted aggregate base"),
        Patch(facecolor=COL_XPS, edgecolor="black", label='XPS (1", taped)'),
        Patch(facecolor=COL_BLOCK, edgecolor="black", label="Retaining wall blocks"),
        Patch(facecolor=COL_MEM, edgecolor="black", label="Dampproofing / waterproofing"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        bbox_transform=ax.transAxes,
        frameon=True,
        fontsize=8.5,
        ncol=3,
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
        "• Detail intent: schematic section showing a double-sided “sunken garden” with two identical cantilever retaining walls; confirm design and reinforcing with structural engineer.",
        "",
        "• Retaining walls: 10' tall, 12\" stem thickness. Base slab is 12\" thick with 3'-0\" heel (soil side) and 3'-0\" toe (sunken garden side) (schematic).",
        "",
        "• Soil/backfill: soil backfilled to 7' above sunken garden grade. Above that: 3'-0\" wide × 3'-0\" tall raised garden; inner wall is the retaining wall, outer wall is retaining-wall blocks (8\"H × 18\"L × 12\"D typ.).",
        "",
        "• Drainage: provide 4\" perforated drain tile at base of wall on soil side, sloped to daylight/sump. Provide granular drainage stone / backfill behind wall extending up from the footing; wrap drainage stone in geotextile and protect waterproofing.",
        "",
        "• Base: provide compacted crushed stone under footing and stem bearing; confirm thickness, compaction, and bearing requirements.",
        "",
        "• Frost: under the toe (sunken garden side) provide ≥42\" depth of washed angular crushed stone (geotextile-wrapped) per schematic; confirm frost protection strategy and bearing requirements.",
        "",
        "• Insulation: provide 1\" XPS (taped seams) under outer heel arms (soil side) as shown. Near house foundation footings, provide 3\" XPS under grade extending 48\" out from the house foundation perimeter to reduce frost impact on footings (coordinate with foundation details).",
        "",
        "• Waterproofing: dampproofing or waterproofing required along the soil-facing side of retaining wall. Protect membrane and provide drainage path to drain tile.",
        "",
        "• Weeps: provide sloped weep pipes through full wall thickness @ 8' o.c. (typ.); coordinate with waterproofing, drain stone, and interior water management.",
        "",
        "• Reinforcing (schematic): provide L-shaped dowels from footing into stem; provide continuous footing steel (shown schematically). Place primary vertical stem steel on the soil-facing side only with 3\" cover; coordinate development length into footing, footing steel, and temperature/shrinkage reinforcement with the structural design.",
    ]
    ax_notes.text(2, 146, _wrap_notes(raw_notes), fontsize=9, va="top", ha="left", family="monospace")

    out_path = Path(__file__).resolve().with_suffix(".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return str(out_path)


if __name__ == "__main__":
    print(main())
