import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon, Rectangle


def _wrap_notes(lines, width=58):
    wrapped = []
    for line in lines:
        if line.strip().startswith("•"):
            wrapped.extend(textwrap.wrap(line, width=width))
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


def _rect(ax, x, y, w, h, *, fc="white", ec="black", lw=1.2, ls="-", hatch=None, z=1):
    r = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, hatch=hatch, zorder=z)
    ax.add_patch(r)
    return r


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
    d = p1 - p0
    n = np.array([-d[1], d[0]], dtype=float)
    n /= np.linalg.norm(n) + 1e-9
    return p0 + offset * n, p1 + offset * n


def _quad_from_segment(p0, p1, thickness):
    a0, a1 = _offset_segment(p0, p1, thickness / 2)
    b0, b1 = _offset_segment(p0, p1, -thickness / 2)
    return np.vstack([a0, a1, b1, b0])


def _thick_polyline(points, thickness):
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
        denom = max(denom, 0.25)  # keep miters from exploding at sharp corners
        vnorms[i] = m / denom

    half = thickness / 2
    outer = pts + half * vnorms
    inner = pts - half * vnorms
    return np.vstack([outer, inner[::-1]])


def _pt_at_x(seg0, seg1, x):
    t = (x - seg0[0]) / (seg1[0] - seg0[0] + 1e-9)
    return seg0 + t * (seg1 - seg0)


def _layer_poly(mid0, mid1, center_offset, thickness, x0, x1):
    upper0, upper1 = _offset_segment(mid0, mid1, center_offset + thickness / 2)
    lower0, lower1 = _offset_segment(mid0, mid1, center_offset - thickness / 2)
    p0u = _pt_at_x(upper0, upper1, x0)
    p1u = _pt_at_x(upper0, upper1, x1)
    p1l = _pt_at_x(lower0, lower1, x1)
    p0l = _pt_at_x(lower0, lower1, x0)
    return np.vstack([p0u, p1u, p1l, p0l])


def _ijoist_with_birdsmouth(mid0, mid1, depth, *, x0, x_end, seat_x0, seat_y):
    top0, top1 = _offset_segment(mid0, mid1, depth / 2)
    bot0, bot1 = _offset_segment(mid0, mid1, -depth / 2)

    p_top_start = _pt_at_x(top0, top1, x0)
    p_top_end = _pt_at_x(top0, top1, x_end)

    p_bot_end = np.array([x_end, seat_y], dtype=float)
    p_seat0 = np.array([seat_x0, seat_y], dtype=float)
    p_bot_seat0_uncut = _pt_at_x(bot0, bot1, seat_x0)
    p_bot_start = _pt_at_x(bot0, bot1, x0)

    return np.vstack(
        [
            p_top_start,
            p_top_end,
            p_bot_end,
            p_seat0,
            p_bot_seat0_uncut,
            p_bot_start,
        ]
    )


def main():
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.15], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_notes = fig.add_subplot(gs[0, 1])

    # Colors
    COL_WOOD = "#C8A26A"
    COL_DRY = "#E6E6E6"
    COL_SHEATH = "#D9C8A0"
    COL_MEM = "#1E3A5F"
    COL_METAL = "#2F2F2F"
    COL_FLASH = "#7A0C0C"
    COL_EPS = "#C8E0F8"
    COL_POLYISO = "#F4E6B1"
    COL_SPRAY = "#FFD966"
    COL_INSUL = "#DDECC8"
    COL_STEEL = "#A7B5C6"

    # Wall geometry (schematic, inches)
    drywall = 0.625
    stud_depth = 3.5  # 2x4
    stud_width = 1.5
    sheathing = 0.625  # 5/8" Struct 1
    polyiso_wall = 2.0
    eps_wall = 2.0
    furring_wall = 0.5  # 1/2" plywood, 3.5" wide
    cladding = 0.5
    plate_thick = 1.5  # each top plate

    wall_show_height = 32.0  # portion shown below top plates
    y_top_plate_top = 0.0
    y_top_plate_mid = y_top_plate_top - plate_thick
    y_top_plate_bot = y_top_plate_mid - plate_thick
    y_wall_bot = y_top_plate_bot - wall_show_height

    # X positions (interior negative, exterior positive)
    x_dry0 = -8.75
    x_dry1 = x_dry0 + drywall
    x_stud0 = x_dry1
    x_stud1 = x_stud0 + stud_depth
    x_sheath0 = x_stud1
    x_sheath1 = x_sheath0 + sheathing
    x_poly0 = x_sheath1
    x_poly1 = x_poly0 + polyiso_wall
    x_eps0 = x_poly1
    x_eps1 = x_eps0 + eps_wall
    x_fur0 = x_eps1
    x_fur1 = x_fur0 + furring_wall
    x_clad0 = x_fur1
    x_clad1 = x_clad0 + cladding

    # Roof geometry (precompute here so the wall extension can align cleanly)
    roof_pitch = -4 / 12  # slope down to exterior
    depth_ijoist = 11.875
    flange_thk = 1.375  # I-joist flange thickness (top & bottom)
    sheathing_thk = 0.75  # Struct 1 roof deck
    polyiso_roof = 2.0
    eps_roof = 4.0
    roof_mem = 0.25  # roofing membrane over EPS
    furring_roof = 0.75  # 3/4" plywood, 3.5" wide
    metal_roof = 0.5  # standing seam schematic thickness

    x_bearing = x_stud1  # wall line / outer face of top plate
    eave_x = x_bearing  # exterior face of top plate
    overhang = 16.0  # foam/furring/roofing beyond wall line (schematic)
    eave_outer_x = eave_x + overhang

    birdsmouth_depth = 1.17  # depth of birdsmouth seat cut
    y_joist_bot = y_top_plate_top - birdsmouth_depth
    y_joist_center = y_joist_bot + depth_ijoist / 2

    mid0 = np.array([-34.0, y_joist_center + roof_pitch * (-34.0 - x_bearing)])
    mid1 = np.array([eave_x, y_joist_center + roof_pitch * (eave_x - x_bearing)])
    mid1_ext = np.array([eave_outer_x, y_joist_center + roof_pitch * (eave_outer_x - x_bearing)])

    # Wall foam/furring extend up close to underside of roof foam (leave a small spray-foam gap).
    # Underside of roof foam = top of roof sheathing membrane plane (extended).
    membrane_thk = 0.25
    foam_base_seg0, foam_base_seg1 = _offset_segment(mid0, mid1_ext, depth_ijoist / 2 + sheathing_thk + membrane_thk)
    y_roof_foam_under_at_wall = _pt_at_x(foam_base_seg0, foam_base_seg1, x_eps1)[1]
    y_roof_foam_under_at_sheath = _pt_at_x(foam_base_seg0, foam_base_seg1, x_sheath1)[1]
    y_wall_ext = min(y_roof_foam_under_at_wall, y_roof_foam_under_at_sheath) - 0.4
    y_wall_ext = min(y_wall_ext, y_top_plate_top + depth_ijoist - 0.25)

    # Wall components (main portion below top plate)
    _rect(ax, x_dry0, y_wall_bot, drywall, y_top_plate_top - y_wall_bot, fc=COL_DRY, lw=1.1, z=4)
    
    # Ceiling drywall under I-joist (sloped to follow bottom of joist)
    ceiling_dry_x0 = -34.0  # match start of I-joist
    ceiling_dry_x1 = x_stud1  # extend to bearing point
    # Use bottom flange offset minus drywall thickness to position ceiling drywall just below I-joist
    ceiling_offset = -depth_ijoist / 2 - drywall / 2
    ceiling_seg0, ceiling_seg1 = _offset_segment(mid0, mid1, ceiling_offset)
    ceiling_dry_poly = _layer_poly(mid0, mid1, ceiling_offset, drywall, ceiling_dry_x0, ceiling_dry_x1)
    ax.add_patch(Polygon(ceiling_dry_poly, closed=True, facecolor=COL_DRY, edgecolor="black", linewidth=1.1, zorder=2))
    
    _rect(ax, x_stud0, y_wall_bot, stud_depth, y_top_plate_bot - y_wall_bot, fc=COL_INSUL, lw=0.9, hatch="..", z=1)
    _rect(ax, x_stud0, y_top_plate_bot, stud_depth, plate_thick, fc=COL_WOOD, lw=1.2, z=10)
    _rect(ax, x_stud0, y_top_plate_mid, stud_depth, plate_thick, fc=COL_WOOD, lw=1.2, z=10)
    _rect(ax, x_stud0, y_wall_bot, stud_depth, y_top_plate_bot - y_wall_bot, fc="none", lw=1.2, z=5)
    # Wall sheathing/membrane continue up to align with roof layers.
    _rect(ax, x_sheath0, y_wall_bot, sheathing, (y_top_plate_top + depth_ijoist) - y_wall_bot, fc=COL_SHEATH, lw=1.1, z=4)
    _rect(ax, x_sheath1, y_wall_bot, 0.25, (y_top_plate_top + depth_ijoist) - y_wall_bot, fc=COL_MEM, ec="black", lw=0.8, z=7)
    # Wall foam layers extend up past top plate to meet roof
    _rect(ax, x_poly0, y_wall_bot, polyiso_wall, y_wall_ext - y_wall_bot, fc=COL_POLYISO, lw=1.0, z=4)
    _rect(ax, x_eps0, y_wall_bot, eps_wall, y_wall_ext - y_wall_bot, fc=COL_EPS, lw=1.0, z=4)
    # Wall furring extends up behind gutter to meet roof furring
    offset_furring_bot = depth_ijoist / 2 + sheathing_thk + membrane_thk + polyiso_roof + eps_roof + roof_mem
    fur_bot_seg0, fur_bot_seg1 = _offset_segment(mid0, mid1_ext, offset_furring_bot)
    wall_fur_top_y = _pt_at_x(fur_bot_seg0, fur_bot_seg1, x_fur1)[1]
    _rect(ax, x_fur0, y_wall_bot, furring_wall, wall_fur_top_y - y_wall_bot, fc=COL_WOOD, lw=1.0, hatch="//", z=5)
    _rect(ax, x_clad0, y_wall_bot, cladding, y_wall_ext - y_wall_bot, fc=COL_METAL, lw=1.1, z=6)

    # Roof layers along slope
    # I-joist shown with birdsmouth seat cut bearing across full top plate (no cantilever).
    seat_x0 = x_stud0
    seat_y = y_top_plate_top - 0.1  # show cut down into seat for clarity
    joist_poly = _ijoist_with_birdsmouth(
        mid0,
        mid1,
        depth_ijoist,
        x0=-34.0,
        x_end=eave_x,
        seat_x0=seat_x0,
        seat_y=seat_y,
    )
    ax.add_patch(Polygon(joist_poly, closed=True, facecolor=COL_WOOD, edgecolor="black", linewidth=1.2, zorder=3))

    # Dotted flange lines (schematic) to show top/bottom flanges without obscuring web
    top_flange_seg0, top_flange_seg1 = _offset_segment(mid0, mid1, depth_ijoist / 2 - flange_thk)
    bot_flange_seg0, bot_flange_seg1 = _offset_segment(mid0, mid1, -(depth_ijoist / 2 - flange_thk))
    top_flange_p0 = _pt_at_x(top_flange_seg0, top_flange_seg1, -34.0)
    top_flange_p1 = _pt_at_x(top_flange_seg0, top_flange_seg1, eave_x)
    bot_flange_p0 = _pt_at_x(bot_flange_seg0, bot_flange_seg1, -34.0)
    bot_flange_p1 = _pt_at_x(bot_flange_seg0, bot_flange_seg1, seat_x0)
    ax.plot(
        [top_flange_p0[0], top_flange_p1[0]],
        [top_flange_p0[1], top_flange_p1[1]],
        color="black",
        linewidth=0.9,
        linestyle=(0, (2.5, 2.5)),
        alpha=0.8,
    )
    ax.plot(
        [bot_flange_p0[0], bot_flange_p1[0]],
        [bot_flange_p0[1], bot_flange_p1[1]],
        color="black",
        linewidth=0.9,
        linestyle=(0, (2.5, 2.5)),
        alpha=0.8,
    )

    # Beveled bearing stiffener at eave (4:12 bevel to match slope), inside I-joist between flanges
    stiff_w = stud_depth
    stiff_x0 = eave_x - stiff_w
    # Bottom flange connection points
    stiff_bot0 = _pt_at_x(bot_flange_seg0, bot_flange_seg1, stiff_x0)
    stiff_bot1 = _pt_at_x(bot_flange_seg0, bot_flange_seg1, eave_x)
    # Top flange connection points
    stiff_top0 = _pt_at_x(top_flange_seg0, top_flange_seg1, stiff_x0)
    stiff_top1 = _pt_at_x(top_flange_seg0, top_flange_seg1, eave_x)
    stiff_poly = np.array(
        [
            stiff_bot0,
            stiff_bot1,
            stiff_top1,
            stiff_top0,
        ]
    )
    ax.add_patch(Polygon(stiff_poly, closed=True, facecolor=COL_WOOD, edgecolor="black", linewidth=1.0, hatch="xx", zorder=4))

    _leader(
        ax,
        (eave_x - 0.6, y_top_plate_top + 0.15),
        (x_dry0 - 18.0, y_joist_center - 6.0),
        "I-joist bearing on top plate",
    )

    # Roof overhang: foam/membrane/furring/roofing extend beyond wall line to create eave.

    # Roof sheathing (air barrier: liquid membrane over Struct 1 plywood)
    offset_sheath = depth_ijoist / 2 + sheathing_thk / 2
    sheath_seg0, sheath_seg1 = _offset_segment(mid0, mid1_ext, offset_sheath)
    # Extend sheathing by wall sheathing thickness to overlap wall sheathing
    sheath_poly = _layer_poly(mid0, mid1_ext, offset_sheath, sheathing_thk, -34.0, eave_x + sheathing)
    ax.add_patch(Polygon(sheath_poly, closed=True, facecolor=COL_SHEATH, edgecolor="black", linewidth=1.0, zorder=5))
    
    # Liquid membrane over roof sheathing (0.25" thickness for visibility)
    membrane_thk = 0.25
    offset_roof_sheath_mem = depth_ijoist / 2 + sheathing_thk + membrane_thk / 2
    roof_mem_seg0, roof_mem_seg1 = _offset_segment(mid0, mid1_ext, offset_roof_sheath_mem)
    roof_sheath_mem_poly = _layer_poly(mid0, mid1_ext, offset_roof_sheath_mem, membrane_thk, -34.0, eave_x + sheathing)
    ax.add_patch(Polygon(roof_sheath_mem_poly, closed=True, facecolor=COL_MEM, edgecolor="black", linewidth=0.8, zorder=7))
    
    _leader(
        ax,
        _pt_at_x(sheath_seg0, sheath_seg1, -6.0),
        (x_clad1 + 14.0, y_joist_center + 14.0),
        '3/4" Struct 1 roof sheathing\nLiquid membrane = roof air barrier\nSheathing extends to overlap wall sheathing',
    )

    # Inner roof foam (polyiso) then outer EPS
    offset_polyiso = depth_ijoist / 2 + sheathing_thk + membrane_thk + polyiso_roof / 2
    poly_seg0, poly_seg1 = _offset_segment(mid0, mid1_ext, offset_polyiso)
    poly_poly = _layer_poly(mid0, mid1_ext, offset_polyiso, polyiso_roof, -34.0, x_fur0)
    ax.add_patch(Polygon(poly_poly, closed=True, facecolor=COL_POLYISO, edgecolor="black", linewidth=1.0, zorder=5))

    offset_eps = depth_ijoist / 2 + sheathing_thk + membrane_thk + polyiso_roof + eps_roof / 2
    eps_seg0, eps_seg1 = _offset_segment(mid0, mid1_ext, offset_eps)
    eps_poly = _layer_poly(mid0, mid1_ext, offset_eps, eps_roof, -34.0, x_fur0)
    ax.add_patch(Polygon(eps_poly, closed=True, facecolor=COL_EPS, edgecolor="black", linewidth=1.0, zorder=5))
    _leader(
        ax,
        _pt_at_x(eps_seg0, eps_seg1, -2.0),
        (x_clad1 + 14.0, y_joist_center + 22.0),
        'Roof CI: 2" polyiso (inner, seams staggered)\n+ 4" EPS (outer, seams taped)',
    )

    # Roofing membrane over EPS, under roof furring; turns down to lap wall EPS
    offset_roof_mem = depth_ijoist / 2 + sheathing_thk + membrane_thk + polyiso_roof + eps_roof + roof_mem / 2
    mem_seg0, mem_seg1 = _offset_segment(mid0, mid1_ext, offset_roof_mem)
    mem_poly = _layer_poly(mid0, mid1_ext, offset_roof_mem, roof_mem, -34.0, x_fur0)
    ax.add_patch(Polygon(mem_poly, closed=True, facecolor=COL_MEM, edgecolor="black", linewidth=0.8, zorder=6))

    # Lap/downturn of roofing membrane onto wall EPS, behind wall furring (avoid geometry overlaps)
    mem_turn_pt = _pt_at_x(mem_seg0, mem_seg1, x_fur0 + 0.02)
    mem_down_x = x_fur0 + 0.02
    mem_down_w = 0.25
    mem_down_y_bot = y_top_plate_top + 8.0
    mem_down = np.array(
        [
            mem_turn_pt,
            mem_turn_pt + np.array([mem_down_w, 0.0]),
            [mem_down_x + mem_down_w, mem_down_y_bot],
            [mem_down_x, mem_down_y_bot],
        ]
    )
    ax.add_patch(Polygon(mem_down, closed=True, facecolor=COL_MEM, edgecolor="black", linewidth=0.8, zorder=7))
    _leader(
        ax,
        (mem_down_x + mem_down_w / 2, (mem_turn_pt[1] + mem_down_y_bot) / 2),
        (x_clad1 + 15.0, y_top_plate_top + 4.0),
        "Roofing membrane laps down\nonto outer wall EPS under furring",
    )

    # Roof furring strips (vent cavity to ridge)
    offset_furring_center = depth_ijoist / 2 + sheathing_thk + membrane_thk + polyiso_roof + eps_roof + roof_mem + furring_roof / 2
    fur_seg0, fur_seg1 = _offset_segment(mid0, mid1_ext, offset_furring_center)
    x_fur_end = x_fur1  # Align roof furring with wall furring strip
    fur_poly = _layer_poly(mid0, mid1_ext, offset_furring_center, furring_roof, -34.0, x_fur_end)
    ax.add_patch(Polygon(fur_poly, closed=True, facecolor=COL_WOOD, edgecolor="black", linewidth=1.0, hatch="//", zorder=7))
    _leader(
        ax,
        _pt_at_x(fur_seg0, fur_seg1, -12.0),
        (-22.0, y_joist_center + 22.0),
        '3/4" plywood roof furring, 3.5" wide\nVent channel continuous wall→eave→ridge',
    )

    # Standing seam roofing
    offset_roofing_center = offset_furring_center + furring_roof / 2 + metal_roof / 2
    roof_seg0, roof_seg1 = _offset_segment(mid0, mid1_ext, offset_roofing_center)
    roof_poly = _layer_poly(mid0, mid1_ext, offset_roofing_center, metal_roof, -34.0, x_fur1 + 0.6)
    ax.add_patch(Polygon(roof_poly, closed=True, facecolor=COL_METAL, edgecolor="black", linewidth=1.0, zorder=8))

    # Spray foam wedge where roof foam meets wall foam
    # The roof foam (polyiso inner) sits on top of roof sheathing membrane
    # Wall foam extends up and needs to meet the underside of roof foam
    # The gap between them gets filled with spray foam
    foam_base_seg0, foam_base_seg1 = _offset_segment(mid0, mid1_ext, depth_ijoist / 2 + sheathing_thk + membrane_thk)
    # Find where the underside of roof foam crosses the wall interface.
    roof_foam_bot_at_sheath = _pt_at_x(foam_base_seg0, foam_base_seg1, x_sheath1)
    roof_foam_bot_at_eps = _pt_at_x(foam_base_seg0, foam_base_seg1, x_eps1)
    spray_poly = np.array(
        [
            (x_eps1, y_wall_ext),
            roof_foam_bot_at_eps,
            roof_foam_bot_at_sheath,
            (x_sheath1, y_wall_ext),
        ]
    )
    ax.add_patch(Polygon(spray_poly, closed=True, facecolor=COL_SPRAY, edgecolor="black", linewidth=0.8, zorder=6))
    _leader(
        ax,
        ((x_sheath1 + x_eps1) / 2, stiff_top1[1] + 1.0),
        (x_clad1 + 10.0, y_top_plate_top - 1.0),
        "Fill gap between roof & wall foam\nwith closed-cell spray foam",
    )

    gutter_h = 5.0
    gutter_depth = 5.0
    gutter_back_x = x_fur_end + furring_wall - 0.2
    gutter_front_x = gutter_back_x + gutter_depth

    roof_fur_top_seg0, roof_fur_top_seg1 = _offset_segment(mid0, mid1_ext, offset_furring_center + furring_roof / 2)
    roof_fur_top_at_eave = _pt_at_x(roof_fur_top_seg0, roof_fur_top_seg1, x_fur_end)
    roof_fur_bot_at_eave = _pt_at_x(fur_bot_seg0, fur_bot_seg1, x_fur_end)
    gutter_top_y = roof_fur_bot_at_eave[1] - 1.2

    # Drip edge flashing (thickened for visibility). Starts at the lower end of the roof furring strip,
    # runs upslope a few inches, then continues out over the gutter and turns down to drip into it.
    drip_thk = 0.5  # intentionally thick (schematic visibility)
    drip_back_leg = 3.5
    drip_face_x = gutter_back_x + 0.15 * gutter_depth
    drip_turn_x = drip_face_x - 0.9
    drip_down_y = gutter_top_y - gutter_h + 4.0
    drip_hem_dx = min(1.2, (gutter_front_x - 0.25) - drip_face_x)

    drip_back_pt = _pt_at_x(roof_fur_top_seg0, roof_fur_top_seg1, x_fur_end - drip_back_leg)
    drip_turn_pt = _pt_at_x(roof_fur_top_seg0, roof_fur_top_seg1, drip_turn_x)
    drip_eave_pt = roof_fur_top_at_eave
    drip_pts = [
        drip_back_pt,
        drip_eave_pt,
        drip_turn_pt,
        [drip_face_x, drip_turn_pt[1] - 0.2],
        [drip_face_x, drip_down_y],
        [drip_face_x + drip_hem_dx, drip_down_y - 0.15],
        # [drip_face_x + drip_hem_dx, drip_down_y + 0.9],
    ]
    drip_poly = _thick_polyline(drip_pts, drip_thk)
    ax.add_patch(Polygon(drip_poly, closed=True, facecolor=COL_STEEL, edgecolor="black", linewidth=1.0, zorder=9))
    _leader(
        ax,
        (drip_face_x + 0.05, (gutter_top_y + drip_down_y) / 2),
        (x_clad1 + 16.0, gutter_top_y - gutter_h + 0.5),
        "Drip edge fastened to roof furring;\nturns into gutter",
    )

    # Stainless Z-flashing behind gutter
    # Top of flashing aligns with bottom edge of roof furring strip
    flash_top_pt = _pt_at_x(fur_bot_seg0, fur_bot_seg1, x_fur1 + furring_wall)
    flash_top_y = flash_top_pt[1]
    flash_bot_y = flash_top_y - gutter_depth - 2.0
    z_flash_thick = 0.4  # visual thickness for schematic
    z_flash_out = 1.5  # horizontal extension 
    z_flash_down = 2.0  # downward leg length
    z_flash_mid_y = flash_bot_y + 1.0  # where horizontal leg starts
    z_flash = np.array([
        # Back vertical leg (behind wall furring, running up behind gutter)
        [x_fur1, flash_bot_y],
        [x_fur1, flash_top_y],
        [x_fur1 + z_flash_thick, flash_top_y],
        [x_fur1 + z_flash_thick, z_flash_mid_y],
        # Horizontal leg (extending outward at bottom, near where siding starts)
        [x_fur1 + z_flash_thick + z_flash_out, z_flash_mid_y],
        [x_fur1 + z_flash_thick + z_flash_out, z_flash_mid_y - z_flash_thick],
        # Front downward leg (in front of siding)
        [x_fur1 + z_flash_out + z_flash_thick, z_flash_mid_y - z_flash_thick],
        [x_fur1 + z_flash_out + z_flash_thick, z_flash_mid_y - z_flash_thick - z_flash_down],
        [x_fur1 + z_flash_out, z_flash_mid_y - z_flash_thick - z_flash_down],
        [x_fur1 + z_flash_out, z_flash_mid_y - z_flash_thick],
        # Back along horizontal
        [x_fur1, z_flash_mid_y - z_flash_thick],
    ])
    ax.add_patch(Polygon(z_flash, closed=True, facecolor=COL_STEEL, edgecolor="black", linewidth=0.9, zorder=8))

    # Gutter (6" box, fascia style) - positioned at eave
    gutter = np.array(
        [
            [gutter_back_x, gutter_top_y],
            [gutter_back_x, gutter_top_y - gutter_h],
            [gutter_front_x, gutter_top_y - gutter_h],
            [gutter_front_x + 1.0, gutter_top_y - gutter_h + 2.5],  # angled fascia face
            [gutter_front_x + 1.0, gutter_top_y + 0.5],
            [gutter_back_x + 0.5, gutter_top_y + 0.5],
        ]
    )
    ax.add_patch(Polygon(gutter, closed=True, facecolor="#8B8B8B", edgecolor="black", linewidth=1.0, zorder=7))
    _leader(
        ax,
        (gutter_back_x + gutter_depth / 2, gutter_top_y - gutter_h / 2),
        (x_clad1 + 16.0, y_top_plate_top + 8.0),
        '6" box gutter (fascia style)',
    )

    # Ventilation arrows (roof and wall furring cavities)
    ax.annotate(
        "",
        xy=_pt_at_x(fur_seg0, fur_seg1, x_fur_end - 2.0),
        xytext=_pt_at_x(fur_seg0, fur_seg1, -8.0),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
    )
    ax.annotate(
        "",
        xy=(x_fur1 + 0.05, y_top_plate_top - 2.0),
        xytext=(x_fur1 + 0.05, y_wall_bot + 6.0),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
    )
    ax.text(x_fur1 + 0.9, y_top_plate_top - 1.0, "Vent path\nroof→wall", fontsize=8, ha="left", va="top")

    # Wall labels (drawn after geometry to appear on top)
    _leader(
        ax,
        (x_stud0 + stud_depth / 2, y_top_plate_bot - 12),
        (x_dry0 - 1.5, y_top_plate_bot - 9.0),
        "2x4 wall w/ R-13 batt fill",
        ha="right",
    )
    _leader(
        ax,
        (x_sheath1 + 0.04, y_top_plate_bot - 4.0),
        (x_sheath1 + 10.0, y_top_plate_bot - 8.0),
        '5/8" Struct 1 plywood, taped\nLiquid-applied membrane = wall air barrier',
    )
    _leader(
        ax,
        (x_eps0 + eps_wall / 2, y_top_plate_bot - 12.0),
        (x_poly1 + 12.0, y_top_plate_bot - 16.0),
        '2" polyiso (inner) + 2" EPS (outer)\nStagger seams, tape outer layer',
    )
    _leader(
        ax,
        (x_fur0 + furring_wall / 2, y_top_plate_top - 4.0),
        (x_clad1 + 10.0, y_top_plate_top - 6.0),
        '1/2" plywood furring, 3.5" wide\nFasten per IRC Table R703.15.2',
    )

    # Labels & dimensions
    # ax.text(x_dry0, y_wall_bot - 2.2, "INTERIOR", ha="left", va="top", fontsize=10, fontweight="bold")
    # ax.text(x_clad1, y_wall_bot - 2.2, "EXTERIOR", ha="right", va="top", fontsize=10, fontweight="bold")
    # _dim_h(ax, x_poly0, x_eps1, y_top_plate_bot - 14.0, '2" polyiso + 2" EPS (wall)')
    # _dim_h(ax, x_eps1, x_clad1, y_top_plate_bot - 11.0, '1/2" furring + cladding')
    # _dim_h(ax, eave_x, eave_outer_x, y_joist_center + 8.0, f'{overhang:.0f}" eave overhang')
    # _dim_v(ax, y_top_plate_bot, y_top_plate_top, x_stud0 - 1.0, 'Double top plate\n(3")')

    ax.set_title(
        "Roof–Wall Eave Detail with Exterior CI, No Overhang Roof Foam, and Box Gutter (4:12 Slope)",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.text(x_dry0 - 1.0, y_wall_bot - 6.5, "Colin Catlin, 2026", ha="left", va="top", fontsize=7)

    ax.set_xlim(x_dry0 - 4.0, gutter_front_x + 12.0)
    ax.set_ylim(y_wall_bot - 8.0, y_joist_center + 24.0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        Patch(facecolor=COL_WOOD, edgecolor="black", label="Wood framing / plates / furring"),
        Patch(facecolor=COL_DRY, edgecolor="black", label="Drywall (5/8\")"),
        Patch(facecolor=COL_SHEATH, edgecolor="black", label="Struct 1 plywood"),
        Patch(facecolor=COL_MEM, edgecolor="black", label="Liquid / roofing membrane"),
        Patch(facecolor=COL_POLYISO, edgecolor="black", label="Polyiso foam"),
        Patch(facecolor=COL_EPS, edgecolor="black", label="EPS foam"),
        Patch(facecolor=COL_SPRAY, edgecolor="black", label="Spray foam infill"),
        Patch(facecolor=COL_METAL, edgecolor="black", label="Metal roofing / siding"),
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

    # Notes panel
    ax_notes.set_xlim(0, 100)
    ax_notes.set_ylim(0, 150)
    ax_notes.axis("off")

    raw_notes = [
        "NOTES:",
        "",
        "• Roof framing: I-joists bear on double top plate with birdsmouth cut + beveled bearing stiffeners (APA D710 10h) OR beveled plate (APA D710 10q, requires additional fasteners for uplift). Ridge framing per APA D710 10c. Roof slope 4:12.",
        "",
        "• Roof sheathing: 3/4\" Struct 1 plywood with liquid-applied membrane as PRIMARY AIR BARRIER. Roof sheathing extends to overlap wall sheathing/membrane for continuity.",
        "",
        "• Roof CI: inner layer 2\" polyiso (seams staggered but not taped) + outer 4\" EPS (seams taped). Roofing membrane over EPS, under roof furring, turns down to lap top of wall EPS behind furring/drip flashing.",
        "",
        "• Roof nailer cavity: 3/4\" plywood furring strips (3.5\" wide) over membrane. Maintains vented path from ridge vent (not shown) down to eave; air can also enter under drip edge. Connect junction of wall furring and roof furring with nails or adhesive.",
        "",
        "• Wall: 2x4 studs with R-13 batt insulation. 5/8\" Struct 1 plywood sheathing, taped, with liquid membrane air barrier. Continuous CI = 2\" polyiso + 2\" EPS (outer layer taped).",
        "",
        "• Wall furring: 1/2\" Struct 1 plywood, 3.5\" wide, over CI; fasten per IRC Table R703.15.2 (Note C: wood sheathing counts toward embedment). Class III interior paint on drywall.",
        "",
        "• Foam interface: leave angled mismatch between roof foam and wall foam; fill gaps with closed-cell spray polyurethane foam (avoid high-expansion spray foams).",
        "",
        "• Roofing membrane and drip edge: membrane under furring; drip edge nailed to roof furring and empties into gutter.",
        "",
        "• Gutter: 6\" box gutter with angled fascia face. Stainless steel flashing behind gutter over wall furring and under drip edge; connect downspout with conduit pipe clamps to steady (not primary support).",
        "",
        "• Maintain continuous ventilation path within roof nailer cavity and wall rainscreen; keep outlets clear at ridge and eave.",
    ]
    ax_notes.text(4, 146, _wrap_notes(raw_notes), fontsize=9, va="top", ha="left", family="monospace")

    out_path = "./roof_wall_eave_detail.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out_path


if __name__ == "__main__":
    print(main())
