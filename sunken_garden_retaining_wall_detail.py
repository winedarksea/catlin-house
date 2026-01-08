"""
Schematic retaining wall detail for a "sunken garden" (double-sided section).

Run:
  python3 catlin-house/sunken_garden_retaining_wall_detail.py
Output:
  catlin-house/sunken_garden_retaining_wall_detail.png
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch, Polygon, Rectangle


def _wrap_notes(lines, width=58):
    wrapped = []
    for line in lines:
        if line.strip().startswith("•"):
            wrapped.extend(textwrap.wrap(line, width=width))
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


def _rect(ax, x, y, w, h, *, fc="white", ec="black", lw=1.2, ls="-", hatch=None, z=1, alpha=1.0):
    r = Rectangle(
        (x, y),
        w,
        h,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        hatch=hatch,
        zorder=z,
        alpha=alpha,
    )
    ax.add_patch(r)
    return r


def _poly(ax, pts, *, fc="white", ec="black", lw=1.2, ls="-", hatch=None, z=1, alpha=1.0):
    p = Polygon(
        np.asarray(pts, dtype=float),
        closed=True,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        hatch=hatch,
        zorder=z,
        alpha=alpha,
    )
    ax.add_patch(p)
    return p


def _leader(ax, xy, text_xy, text, *, ha="left", va="center", zorder=100):
    ax.annotate(
        text,
        xy=xy,
        xytext=text_xy,
        textcoords="data",
        ha=ha,
        va=va,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
        arrowprops=dict(arrowstyle="-", linewidth=1.0, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )


def _dim_h(ax, x0, x1, y, text, *, text_dy=0.9):
    ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->", linewidth=1.1))
    ax.text((x0 + x1) / 2, y + text_dy, text, ha="center", va="bottom", fontsize=9)


def _dim_v(ax, y0, y1, x, text, *, text_dx=0.9):
    ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=dict(arrowstyle="<->", linewidth=1.1))
    ax.text(x + text_dx, (y0 + y1) / 2, text, ha="left", va="center", fontsize=9, rotation=90)


def _pipe(ax, x, y, d, *, fc="#A7B5C6", ec="black", lw=1.1, z=6):
    ax.add_patch(Circle((x, y), radius=d / 2, facecolor="white", edgecolor=ec, lw=lw, zorder=z))
    ax.add_patch(Circle((x, y), radius=d / 2 - 0.25, facecolor=fc, edgecolor="none", alpha=0.85, zorder=z + 1))


def _rebar_L(ax, x_cover_from_face, x_face, y_base_bot, y_base_top, y_wall_top, *, hook=14.0, z=20):
    """
    Draw a schematic L-bar: vertical along wall, hooks into base slab.

    x_face: soil-facing outer face of stem.
    x_cover_from_face: positive distance into concrete from x_face.
    """

    x_vert = x_face + x_cover_from_face
    y_hook = y_base_bot + 4.0

    ax.plot([x_vert, x_vert], [y_base_top, y_wall_top - 6.0], color="black", linewidth=1.2, zorder=z)
    ax.plot([x_vert, x_vert + np.sign(x_cover_from_face) * hook], [y_hook, y_hook], color="black", linewidth=1.2, zorder=z)
    ax.add_patch(Circle((x_vert, y_base_top + 24.0), radius=0.7, facecolor="black", edgecolor="none", zorder=z + 1))


def main():
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.9, 1.1], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    ax_notes = fig.add_subplot(gs[0, 1])

    # Colors
    COL_CONC = "#BFBFBF"
    COL_GRAVEL = "#9C9C9C"
    COL_SOIL = "#D2B48C"
    COL_XPS = "#A7D7C5"
    COL_BLOCK = "#A0A0A0"
    COL_MEM = "#1E3A5F"

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
    drain_stone_h = 18.0
    drain_stone_w = 24.0

    # Coordinate system: sunken garden "interior" grade is y=0.
    y_int_grade = 0.0
    y_base_top = y_int_grade
    y_base_bot = y_base_top - base_thk
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
    # Left wall stem and base
    _rect(ax, xL_out, y_base_top, stem_thk, wall_h, fc=COL_CONC, z=5)
    _rect(ax, xL_base0, y_base_bot, xL_base1 - xL_base0, base_thk, fc=COL_CONC, hatch="..", z=4)

    # Right wall stem and base
    _rect(ax, xR_in, y_base_top, stem_thk, wall_h, fc=COL_CONC, z=5)
    _rect(ax, xR_base0, y_base_bot, xR_base1 - xR_base0, base_thk, fc=COL_CONC, hatch="..", z=4)

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
    # Drain tile + granular backfill (geotextile wrapped) on soil side at bottom
    # -----------------------
    # Left drain stone envelope
    xL_drain0 = xL_out - drain_stone_w
    xL_drain1 = xL_out
    y_drain0 = y_base_bot
    y_drain1 = y_base_bot + drain_stone_h
    _rect(ax, xL_drain0, y_drain0, xL_drain1 - xL_drain0, y_drain1 - y_drain0, fc=COL_GRAVEL, hatch="o", lw=0.9, z=2)
    _rect(ax, xL_drain0 - 1.2, y_drain0 - 1.2, (xL_drain1 - xL_drain0) + 2.4, (y_drain1 - y_drain0) + 2.4, fc="none", ec="black", ls="--", lw=1.0, z=3)
    _pipe(ax, xL_out - drain_stone_w * 0.55, y_base_bot + drain_diam * 0.75, drain_diam)

    # Right drain stone envelope
    xR_drain0 = xR_out
    xR_drain1 = xR_out + drain_stone_w
    _rect(ax, xR_drain0, y_drain0, xR_drain1 - xR_drain0, y_drain1 - y_drain0, fc=COL_GRAVEL, hatch="o", lw=0.9, z=2)
    _rect(ax, xR_drain0 - 1.2, y_drain0 - 1.2, (xR_drain1 - xR_drain0) + 2.4, (y_drain1 - y_drain0) + 2.4, fc="none", ec="black", ls="--", lw=1.0, z=3)
    _pipe(ax, xR_out + drain_stone_w * 0.55, y_base_bot + drain_diam * 0.75, drain_diam)

    # -----------------------
    # Waterproofing / dampproofing on soil-facing side of wall
    # -----------------------
    mem_thk = 0.35
    _rect(ax, xL_out - mem_thk, y_base_top, mem_thk, wall_h, fc=COL_MEM, ec="black", lw=0.8, z=7)
    _rect(ax, xR_out, y_base_top, mem_thk, wall_h, fc=COL_MEM, ec="black", lw=0.8, z=7)

    # -----------------------
    # Weep holes (schematic) through wall near base
    # -----------------------
    weep_y = y_base_top + 18.0
    weep_h = 2.0
    weep_w = 4.0
    _rect(ax, xL_out + stem_thk - weep_w, weep_y, weep_w, weep_h, fc="white", ec="black", lw=1.0, z=12)
    _rect(ax, xR_in, weep_y, weep_w, weep_h, fc="white", ec="black", lw=1.0, z=12)

    # -----------------------
    # Rebar: L-bar from base into wall; vertical along soil face with 3" cover
    # -----------------------
    cover = 3.0
    # Left wall soil face is xL_out
    _rebar_L(ax, +cover, xL_out, y_base_bot, y_base_top, y_wall_top)
    # Right wall soil face is xR_out (outer face to the right)
    _rebar_L(ax, -cover, xR_out, y_base_bot, y_base_top, y_wall_top)

    # -----------------------
    # Dimensions
    # -----------------------
    _dim_h(ax, xL_in, xR_in, y_int_grade + 36.0, '18\'-0" clear between walls')
    _dim_v(ax, y_base_top, y_wall_top, xR_block1 + 22.0, '10\'-0" retaining wall height')
    _dim_v(ax, y_base_top, backfill_h, xR_block1 + 12.0, '7\'-0" soil backfill (typ.)')
    _dim_v(ax, backfill_h, y_wall_top, xR_block1 + 12.0, '3\'-0" raised garden soil')

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
        '12" thick base slab\n(3\' heel + 12" stem + 3\' toe)',
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
        (xL_out - drain_stone_w * 0.55, y_base_bot + drain_diam * 0.75),
        (xL_block0 - 80.0, y_base_bot + 6.0),
        '4" perf drain tile (typ.)\nsloped to daylight / sump',
        ha="right",
    )
    _leader(
        ax,
        (xL_out - drain_stone_w / 2, y_base_bot + drain_stone_h / 2),
        (xL_block0 - 80.0, y_base_bot + 22.0),
        "Granular backfill / drain stone\ngeotextile-wrapped (typ.)",
        ha="right",
    )
    _leader(
        ax,
        ((xL_toe0 + xL_toe1) / 2, y_base_bot - toe_gravel_depth / 2),
        (xL_in + 28.0, y_base_bot - 34.0),
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
        (xL_out + stem_thk - weep_w / 2, weep_y + weep_h / 2),
        (xL_in + 34.0, weep_y + 14.0),
        "Weep holes @ 8' o.c. (typ.)",
    )
    _leader(
        ax,
        ((xL_block0 + xL_block1) / 2, backfill_h + raised_bed_h / 2),
        (xL_block0 - 80.0, backfill_h + 28.0),
        'Retaining wall blocks (typ.)\n8"H × 18"L × 12"D',
        ha="right",
    )

    ax.set_title("Sunken Garden — Double Cantilever Retaining Wall Section (Schematic)", fontsize=13, fontweight="bold", pad=12)
    ax.text((xL_in + xR_in) / 2, y_base_bot - toe_gravel_depth - 16.0, "Colin Catlin, 2026", ha="center", va="top", fontsize=7)

    # Framing / plot formatting
    ax.set_xlim(xL_block0 - 120.0, xR_block1 + 120.0)
    ax.set_ylim(y_base_bot - toe_gravel_depth - 28.0, y_wall_top + 22.0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        Patch(facecolor=COL_CONC, edgecolor="black", label="Concrete"),
        Patch(facecolor=COL_SOIL, edgecolor="black", label="Soil / backfill"),
        Patch(facecolor=COL_GRAVEL, edgecolor="black", label="Washed stone / gravel"),
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
        "• Drainage: provide 4\" perforated drain tile at base of wall on soil side, sloped to daylight/sump. Use granular drainage stone / backfill at the bottom behind wall, wrapped in geotextile.",
        "",
        "• Frost: under the toe (sunken garden side) provide ≥42\" depth of washed angular crushed stone (geotextile-wrapped) per schematic; confirm frost protection strategy and bearing requirements.",
        "",
        "• Insulation: provide 1\" XPS (taped seams) under outer heel arms (soil side) as shown. Near house foundation footings, provide 3\" XPS under grade extending 48\" out from the house foundation perimeter to reduce frost impact on footings (coordinate with foundation details).",
        "",
        "• Waterproofing: dampproofing or waterproofing required along the soil-facing side of retaining wall. Protect membrane and provide drainage path to drain tile.",
        "",
        "• Weeps: provide weep holes through wall @ 8' o.c. (typ.); coordinate location with waterproofing, drain stone, and interior water management.",
        "",
        "• Reinforcing (schematic): L-shaped bars from base into wall; vertical wall steel placed on soil face only with 3\" concrete cover (verify sizing/spacing, development, and temperature/shrinkage reinforcement).",
    ]
    ax_notes.text(2, 146, _wrap_notes(raw_notes), fontsize=9, va="top", ha="left", family="monospace")

    out_path = Path(__file__).resolve().with_suffix(".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return str(out_path)


if __name__ == "__main__":
    print(main())

