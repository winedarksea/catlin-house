"""
Shared helper functions for plotting construction detail schematics.
"""

import textwrap
from types import SimpleNamespace

import numpy as np
from matplotlib.patches import Polygon, Rectangle

# Standard material colors for consistent drawings.
MATERIAL_COLORS = {
    "aggregate": "#7F7F7F",
    "block": "#A0A0A0",
    "concrete": "#BFBFBF",
    "drywall": "#E6E6E6",
    "drywall_alt": "#D8D8D8",
    "eps": "#C8E0F8",
    "equipment": "#7F7F7F",
    "fiber_cement": "#E6E6E6",
    "flashing": "#7A0C0C",
    "glass": "#BEE3F8",
    "insulation": "#DDECC8",
    "membrane": "#1E3A5F",
    "metal": "#A7B5C6",
    "metal_dark": "#2F2F2F",
    "poly": "#BEE3F8",
    "polyiso": "#F4E6B1",
    "river_rock": "#A9A9A9",
    "rubber": "#3A3A3A",
    "sealant": "#6E4F2A",
    "sheathing": "#D9C8A0",
    "soil": "#D2B48C",
    "soffit": "#E0E0E0",
    "spray_foam": "#FFD966",
    "stone": "#9C9C9C",
    "tile": "#E9ECEF",
    "underlayment": "#4A4A4A",
    "wood": "#C8A26A",
    "xps": "#A7D7C5",
}

COLORS = SimpleNamespace(**MATERIAL_COLORS)


def _wrap_notes(lines, width=58):
    wrapped = []
    for line in lines:
        if line.strip().startswith("•"):
            wrapped.extend(textwrap.wrap(line, width=width))
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


def _rect(ax, x, y, w, h, *, fc="white", ec="black", lw=1.2, ls="-", hatch=None, z=1, alpha=1.0):
    rect = Rectangle(
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
    ax.add_patch(rect)
    return rect


def _poly(ax, pts, *, fc="white", ec="black", lw=1.2, ls="-", hatch=None, z=1, alpha=1.0):
    poly = Polygon(
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
    ax.add_patch(poly)
    return poly


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


def _offset_segment(p0, p1, offset):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    normal = np.array([-delta[1], delta[0]], dtype=float)
    normal /= np.linalg.norm(normal) + 1e-9
    return p0 + offset * normal, p1 + offset * normal


def _quad_from_segment(p0, p1, thickness):
    a0, a1 = _offset_segment(p0, p1, thickness / 2)
    b0, b1 = _offset_segment(p0, p1, -thickness / 2)
    return np.vstack([a0, a1, b1, b0])
