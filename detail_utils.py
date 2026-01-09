"""
Shared helper functions for plotting construction detail schematics.
"""

import textwrap
from types import SimpleNamespace

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

# Standard material colors for consistent drawings.
MATERIAL_COLORS = {
    "aggregate": "#7F7F7F",
    "block": "#A0A0A0",
    "concrete": "#BFBFBF",
    "drywall": "#E6E6E6",
    "eps": "#C8E0F8",
    "equipment": "#7F7F7F",
    "fiber_cement": "#E6E6E6",
    "gutter": "#8B8B8B",
    "flashing": "#7A0C0C",
    "glass": "#BEE3F8",
    "insulation": "#DDECC8",
    "membrane": "#1E3A5F",
    "metal": "#A7B5C6",
    "metal_dark": "#2F2F2F",
    "mineral_wool": "#A8A8A8",
    "osb": "#D9C8A0",
    "polyiso": "#F4E6B1",
    "river_rock": "#A9A9A9",
    "rubber": "#3A3A3A",
    "sealant": "#6E4F2A",
    "sheathing": "#D9C8A0",
    "soil": "#D2B48C",
    "soffit": "#E0E0E0",
    "spray_foam": "#FFD966",
    "tile": "#E9ECEF",
    "underlayment": "#4A4A4A",
    "wood": "#C8A26A",
    "xps": "#A7D7C5",
}

COLORS = SimpleNamespace(**MATERIAL_COLORS)
HATCHES = SimpleNamespace(
    gravel="o",
    compacted="..",
    diagonal="//",
    cross="xx",
    joist="\\\\",
    dense_plus="++",
)


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


def _round_rect(ax, x, y, w, h, *, fc="white", ec="black", lw=1.0, ls="-", radius=0.5, z=1, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def _pipe(ax, x, y, d, *, fc=COLORS.metal, ec="black", lw=1.1, z=6, fill_alpha=0.85):
    ax.add_patch(Circle((x, y), radius=d / 2, facecolor="white", edgecolor=ec, lw=lw, zorder=z))
    ax.add_patch(Circle((x, y), radius=d / 2 - 0.25, facecolor=fc, edgecolor="none", alpha=fill_alpha, zorder=z + 1))


def _dotted_pipe(ax, p0, p1, d, *, fc=COLORS.metal, ec="black", lw=1.6, dash=(1.0, 2.2), z=15):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ec, linewidth=lw, linestyle=(0, dash), zorder=z)
    for p in (p0, p1):
        ax.add_patch(Circle((p[0], p[1]), radius=d / 2, facecolor="white", edgecolor=ec, lw=1.0, zorder=z + 1))
        ax.add_patch(Circle((p[0], p[1]), radius=d / 2 - 0.25, facecolor=fc, edgecolor="none", alpha=0.85, zorder=z + 2))


def _rebar_L(ax, x_cover_from_face, x_face, y_base_bot, y_base_top, y_wall_top, *, hook=14.0, color="black", lw=1.6, z=20):
    """
    Draw a schematic L-bar: vertical along wall, hooks into base slab.

    x_face: soil-facing outer face of stem.
    x_cover_from_face: positive distance into concrete from x_face.
    """

    x_vert = x_face + x_cover_from_face
    y_hook = y_base_bot + 3.0
    x_hook = x_vert + np.sign(x_cover_from_face) * hook

    ax.plot([x_vert, x_vert, x_hook], [y_wall_top - 6.0, y_hook, y_hook], color=color, linewidth=lw, zorder=z)
    ax.add_patch(Circle((x_vert, y_base_top + 24.0), radius=0.85, facecolor=color, edgecolor="none", zorder=z + 1))


def _rebar_h(ax, x0, x1, y, *, color="black", lw=1.4, z=20, end_hooks=True):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, zorder=z)
    if end_hooks:
        for x in (x0 + 10.0, x1 - 10.0):
            ax.add_patch(Circle((x, y), radius=0.75, facecolor=color, edgecolor="none", zorder=z + 1))


def _screw(ax, x_start, x_end, y_center, diam, *, color=COLORS.flashing, z=6):
    lw = max(1.2, diam * 6)
    ax.plot([x_start, x_end], [y_center, y_center], color=color, linewidth=lw, solid_capstyle="round", zorder=z)
    head_r = diam / 2
    tip_r = diam / 2 * 0.8
    ax.add_patch(Circle((x_start, y_center), radius=head_r, facecolor=color, edgecolor="black", linewidth=0.6, zorder=z + 1))
    ax.add_patch(Circle((x_end, y_center), radius=tip_r, facecolor=color, edgecolor="black", linewidth=0.6, zorder=z + 1))


def _batt_insulation(ax, x, y, w, h, *, fc=COLORS.insulation, ec="black", lw=0.8, hatch=HATCHES.compacted, radius=0.6, z=1):
    patch = _round_rect(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, ls="-", radius=radius, z=z, alpha=1.0)
    patch.set_hatch(hatch)
    return patch


def _stud_pattern(ax, positions, *, axis="y", fixed=0.0, depth, thickness, fc=COLORS.wood, ec="black", lw=1.2, z=3, hatch=None):
    """
    Draw repeated studs either along y (top view) or along x (section view).
    """

    patches = []
    if axis == "y":
        for pos in positions:
            patches.append(_rect(ax, fixed, pos, depth, thickness, fc=fc, ec=ec, lw=lw, hatch=hatch, z=z))
    elif axis == "x":
        for pos in positions:
            patches.append(_rect(ax, pos, fixed, thickness, depth, fc=fc, ec=ec, lw=lw, hatch=hatch, z=z))
    else:
        raise ValueError("axis must be 'x' or 'y'")
    return patches


def _path_from_steps(start_xy, steps):
    """
    Build a polyline from a start point and a list of (dx, dy) steps.

    Useful for sheet-metal flashing profiles (Z-flashing, drip edges, etc.) when combined
    with `_thick_polyline`.
    """

    x, y = (float(start_xy[0]), float(start_xy[1]))
    pts = [(x, y)]
    for dx, dy in steps:
        x += float(dx)
        y += float(dy)
        pts.append((x, y))
    return np.asarray(pts, dtype=float)


def _thick_polyline(points, thickness, *, miter_min_dot=0.25):
    """
    Convert a polyline centerline into a closed polygon with constant thickness.

    `miter_min_dot` prevents miter spikes at sharp corners; lower values allow longer miters.
    """

    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("need at least 2 points")

    seg = pts[1:] - pts[:-1]
    norms = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-9

    vnorms = np.zeros_like(pts)
    vnorms[0] = norms[0]
    vnorms[-1] = norms[-1]

    for i in range(1, len(pts) - 1):
        n0 = norms[i - 1]
        n1 = norms[i]
        m = n0 + n1
        m_norm = np.linalg.norm(m)
        if m_norm < 1e-6:
            vnorms[i] = n1
            continue
        m /= m_norm
        denom = float(np.dot(m, n1))
        denom = max(denom, float(miter_min_dot))
        vnorms[i] = m / denom

    half = float(thickness) / 2.0
    outer = pts + half * vnorms
    inner = pts - half * vnorms
    return np.vstack([outer, inner[::-1]])


def _flashing(ax, centerline_points, thickness, *, fc=COLORS.metal, ec="black", lw=1.0, z=8, alpha=0.9):
    """
    Draw sheet-metal flashing as a thickened polyline.

    `centerline_points` is an Nx2 polyline; `thickness` is schematic/visual thickness.
    """

    poly = _thick_polyline(centerline_points, thickness)
    patch = Polygon(poly, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha)
    ax.add_patch(patch)
    return patch


def _lumber(ax, x, y, w, h, *, fc=COLORS.wood, ec="black", lw=1.1, ls="-", z=5, alpha=1.0, draw_x=True):
    """
    Draw cross-section lumber with standard X marking (for sill plates, top plates, headers, etc.).
    
    The X is drawn as actual lines that scale with the rectangle size, not as a matplotlib hatch pattern.
    Set draw_x=False to omit the X marking if needed.
    """
    rect = _rect(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, ls=ls, hatch=None, z=z, alpha=alpha)
    
    if draw_x:
        # Draw diagonal X lines across the lumber cross-section
        x_lw = 0.8  # line weight for X
        ax.plot([x, x + w], [y, y + h], color="black", linewidth=x_lw, zorder=z + 1)
        ax.plot([x + w, x], [y, y + h], color="black", linewidth=x_lw, zorder=z + 1)
    
    return rect
