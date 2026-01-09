import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, PathPatch
from matplotlib.path import Path

# ============= CONFIGURATION =============
# House dimensions
HOUSE_WIDTH = 36.0
VISIBLE_BASEMENT = 3.0
FLOOR_HEIGHT = 10.0

# Roof dimensions
GABLE_EAVE_HEIGHT = 5.0    # Height at sides for gable roof
GABLE_RIDGE_HEIGHT = 11.0  # Height at center peak for gable roof
MONO_LEFT_HEIGHT = 1.0     # Height at left for mono-slope roof
MONO_RIGHT_HEIGHT = 10.0   # Height at right for mono-slope roof

# Porch dimensions
PORCH_WIDTH = 20.0
PORCH_RAILING_HEIGHT = 4.0
PORCH_ARCHES = 2
PORCH_ARCH_WIDTH = 8.0
PORCH_OUTER_PIER = 1.0
PORCH_OPENING_HEIGHT = 8.0

show_floor_lines = False
# =========================================

def arch_opening_path(x_left, x_right, y_base, opening_height):
    w = x_right - x_left
    r = w / 2.0
    spring_y = y_base + (opening_height - r)
    cx = (x_left + x_right) / 2.0
    cy = spring_y

    thetas = np.linspace(np.pi, 0, 240)
    xs = cx + r * np.cos(thetas)
    ys = cy + r * np.sin(thetas)

    verts = [(x_left, y_base), (x_left, spring_y)]
    codes = [Path.MOVETO, Path.LINETO]
    for x, y in zip(xs, ys):
        verts.append((x, y))
        codes.append(Path.LINETO)
    verts += [(x_right, y_base), (x_left, y_base)]
    codes += [Path.LINETO, Path.CLOSEPOLY]
    return Path(verts, codes), spring_y, (cx, cy), r

def draw_arch_wall(ax, x0, y0, wall_width=20.0, wall_height=9.0,
                   n_arches=2, arch_width=8.0, outer_pier=1.0,
                   opening_height=8.0,
                   face="0.65", edge="0.15"):
    middle_length = wall_width - 2 * outer_pier
    arches_total = n_arches * arch_width
    interior_piers_count = max(n_arches - 1, 0)
    remaining_for_interior = middle_length - arches_total
    pier = remaining_for_interior / interior_piers_count if interior_piers_count > 0 else 0.0

    # Draw rectangle without edges to avoid horizontal lines between stacked walls
    ax.add_patch(Rectangle((x0, y0), wall_width, wall_height,
                           facecolor=face, edgecolor="none"))
    # Draw left and right edges manually
    ax.plot([x0, x0], [y0, y0 + wall_height], color=edge, linewidth=2)
    ax.plot([x0 + wall_width, x0 + wall_width], [y0, y0 + wall_height], color=edge, linewidth=2)

    x = x0 + outer_pier
    for i in range(n_arches):
        x_left = x
        x_right = x + arch_width

        path, spring_y, (cx, cy), r = arch_opening_path(x_left, x_right, y0, opening_height)
        ax.add_patch(PathPatch(path, facecolor="0.88", edgecolor="0.2", linewidth=1.5))

        ax.plot([x_left, x_left], [y0, spring_y], color="0.2", linewidth=1.2)
        ax.plot([x_right, x_right], [y0, spring_y], color="0.2", linewidth=1.2)
        thetas = np.linspace(np.pi, 0, 320)
        ax.plot(cx + r*np.cos(thetas), cy + r*np.sin(thetas), color="0.2", linewidth=1.2)

        x = x_right
        if i < n_arches - 1:
            x += pier

def roof_height_at_x(x, roof_type, y3, house_w=HOUSE_WIDTH):
    if roof_type == "gable":
        eave_h, ridge_h = GABLE_EAVE_HEIGHT, GABLE_RIDGE_HEIGHT
        ridge_x = house_w / 2.0
        if x <= ridge_x:
            return y3 + eave_h + (ridge_h - eave_h) * (x / ridge_x)
        else:
            return y3 + ridge_h + (eave_h - ridge_h) * ((x - ridge_x) / (house_w - ridge_x))
    else:
        left_h, right_h = MONO_LEFT_HEIGHT, MONO_RIGHT_HEIGHT
        return y3 + left_h + (right_h - left_h) * (x / house_w)

def draw_columns_and_juliet(ax, y2, columns_xs, juliet_span, pole_x,
                            roof_type, y3):
    col_w = 1.33  # 16 inches
    column_h = 9.0
    rail_h = 4.0

    for xc in columns_xs:
        ax.add_patch(Rectangle((xc - col_w/2, y2), col_w, column_h,
                               facecolor="0.65", edgecolor="none"))
        # Draw column side lines only above the lower railing (y2 + rail_h)
        ax.plot([xc - col_w/2, xc - col_w/2], [y2 + rail_h, y2 + column_h], 
                color="0.15", linewidth=2)
        ax.plot([xc + col_w/2, xc + col_w/2], [y2 + rail_h, y2 + column_h], 
                color="0.15", linewidth=2)

    # Draw Juliet balcony only if juliet_span is provided
    if juliet_span is not None:
        rail_y0 = y2 + column_h
        x_left, x_right = juliet_span
        ax.add_patch(Rectangle((x_left, rail_y0), x_right - x_left, rail_h,
                               facecolor="0.97", edgecolor="0.2", linewidth=2))
        for x in np.linspace(x_left + 0.5, x_right - 0.5, max(6, int((x_right-x_left)*0.8))):
            ax.plot([x, x], [rail_y0, rail_y0 + rail_h], color="0.55", linewidth=1)
        rail_y0_for_pole = rail_y0
    else:
        rail_y0_for_pole = y2 + column_h

    pole_d = 0.5  # 6"
    pole_y0 = rail_y0_for_pole
    roof_y = roof_height_at_x(pole_x, roof_type, y3, house_w=HOUSE_WIDTH)
    pole_y1 = roof_y + 2.0
    ax.add_patch(Rectangle((pole_x - pole_d/2, pole_y0), pole_d, pole_y1 - pole_y0,
                           facecolor="0.92", edgecolor="0.15", linewidth=1.5))

def draw_house(ax, porch_x0, title, roof_type, version):
    y_grade = 0.0
    visible_basement_main = VISIBLE_BASEMENT
    h_floor = FLOOR_HEIGHT
    y1 = y_grade + visible_basement_main
    y2 = y1 + h_floor
    y3 = y2 + h_floor
    house_w = HOUSE_WIDTH

    # Background fills
    ax.add_patch(Rectangle((-2, y_grade), 42, 37, facecolor="#f5f9ff", edgecolor="none"))  # Sky
    ax.add_patch(Rectangle((-2, -7), 42, 7, facecolor="#faf8f0", edgecolor="none"))  # Ground

    ax.add_patch(Rectangle((0, y_grade), house_w, visible_basement_main + 2*h_floor,
                           facecolor="white", edgecolor="none"))
    # Draw only the vertical wall edges to avoid a seam at the roof line
    ax.plot([0, 0], [y_grade, y3], color="0.1", linewidth=2)
    ax.plot([house_w, house_w], [y_grade, y3], color="0.1", linewidth=2)
    
    # Add faint vertical lines for standing seam siding (16" = 1.33' spacing)
    seam_spacing = 1.33
    for x_seam in np.arange(seam_spacing, house_w, seam_spacing):
        roof_top = roof_height_at_x(x_seam, roof_type, y3, house_w=house_w)
        ax.plot([x_seam, x_seam], [y_grade, roof_top], color="0.7", linewidth=0.5, alpha=0.6)

    if roof_type == "gable":
        eave_h, ridge_h = GABLE_EAVE_HEIGHT, GABLE_RIDGE_HEIGHT
        roof_poly = Polygon([(0, y3), (0, y3+eave_h), (house_w/2, y3+ridge_h),
                     (house_w, y3+eave_h), (house_w, y3)],
                    closed=True, facecolor="white", edgecolor="none")
        ax.add_patch(roof_poly)
        # Draw vertical gable walls
        ax.plot([0, 0], [y3, y3+eave_h], color="0.1", linewidth=2)
        ax.plot([house_w, house_w], [y3, y3+eave_h], color="0.1", linewidth=2)
        # Draw sloped roof lines
        ax.plot([0, house_w/2], [y3+eave_h, y3+ridge_h], color="0.1", linewidth=2)
        ax.plot([house_w/2, house_w], [y3+ridge_h, y3+eave_h], color="0.1", linewidth=2)
    else:
        left_h, right_h = MONO_LEFT_HEIGHT, MONO_RIGHT_HEIGHT
        roof_poly = Polygon([(0, y3), (0, y3+left_h), (house_w, y3+right_h), (house_w, y3)],
                    closed=True, facecolor="white", edgecolor="none")
        ax.add_patch(roof_poly)
        # Draw vertical side walls
        ax.plot([0, 0], [y3, y3+left_h], color="0.1", linewidth=2)
        ax.plot([house_w, house_w], [y3, y3+right_h], color="0.1", linewidth=2)
        # Draw sloped roof line
        ax.plot([0, house_w], [y3+left_h, y3+right_h], color="0.1", linewidth=2)

    # For version 1: add 32" (2.67') charcoal gray panels on each side extending to roof peak
    # Draw AFTER roof so they overlay on top
    if version == 1:
        color_area_width = 16.0  # recommended it be equal to multiples of 16", seam OC spacing
        gray_width = color_area_width / 12.0  # 32 inches = 2.67 feet
        # Left side gray panel - extends to top of roof
        left_top_y = roof_height_at_x(0, roof_type, y3, house_w=house_w)
        left_roof_y = roof_height_at_x(gray_width, roof_type, y3, house_w=house_w)
        left_verts = [(0, y_grade), (0, left_top_y),
                      (gray_width, left_roof_y), (gray_width, y_grade)]
        ax.add_patch(Polygon(left_verts, facecolor="#3a3a3a", edgecolor="none", alpha=0.9))
        # Right side gray panel - extends to top of roof
        right_start_x = house_w - gray_width
        right_roof_y = roof_height_at_x(right_start_x, roof_type, y3, house_w=house_w)
        right_top_y = roof_height_at_x(house_w, roof_type, y3, house_w=house_w)
        right_verts = [(right_start_x, y_grade), (right_start_x, right_roof_y),
                       (house_w, right_top_y), (house_w, y_grade)]
        ax.add_patch(Polygon(right_verts, facecolor="#3a3a3a", edgecolor="none", alpha=0.9))
        # Redraw seam lines on gray sections extending to roof top
        for x_seam in np.arange(seam_spacing, gray_width, seam_spacing):
            roof_top = roof_height_at_x(x_seam, roof_type, y3, house_w=house_w)
            ax.plot([x_seam, x_seam], [y_grade, roof_top], color="0.5", linewidth=0.5, alpha=0.8)
        for x_seam in np.arange(right_start_x + seam_spacing, house_w, seam_spacing):
            roof_top = roof_height_at_x(x_seam, roof_type, y3, house_w=house_w)
            ax.plot([x_seam, x_seam], [y_grade, roof_top], color="0.5", linewidth=0.5, alpha=0.8)

    porch_w = PORCH_WIDTH
    railing_h = PORCH_RAILING_HEIGHT
    porch_basement_bottom = y1 - 9.0

    draw_arch_wall(ax, porch_x0, porch_basement_bottom, wall_width=porch_w, wall_height=9.0,
                   n_arches=PORCH_ARCHES, arch_width=PORCH_ARCH_WIDTH, outer_pier=PORCH_OUTER_PIER, opening_height=PORCH_OPENING_HEIGHT)
    draw_arch_wall(ax, porch_x0, y1, wall_width=porch_w, wall_height=10.0,
                   n_arches=PORCH_ARCHES, arch_width=PORCH_ARCH_WIDTH, outer_pier=PORCH_OUTER_PIER, opening_height=PORCH_OPENING_HEIGHT)
    
    # Draw bottom edge for the outer boundary of stacked concrete walls
    ax.plot([porch_x0, porch_x0 + porch_w], [porch_basement_bottom, porch_basement_bottom], color="0.15", linewidth=2)

    # Draw railing without bottom edge to avoid line between arches and railing
    ax.add_patch(Rectangle((porch_x0, y2), porch_w, railing_h,
                           facecolor="0.65", edgecolor="none"))
    # Manually draw top, left, and right edges (but not bottom)
    ax.plot([porch_x0, porch_x0], [y2, y2 + railing_h], color="0.2", linewidth=2)  # left
    ax.plot([porch_x0 + porch_w, porch_x0 + porch_w], [y2, y2 + railing_h], color="0.2", linewidth=2)  # right
    ax.plot([porch_x0, porch_x0 + porch_w], [y2 + railing_h, y2 + railing_h], color="0.2", linewidth=2)  # top

    if version == 1:
        # Add 3 slits in the railing (0.5' wide, 2' tall, 1' below top)
        slit_width = 0.5
        slit_height = 2.0
        slit_y_bottom = y2 + railing_h - 1.0 - slit_height  # 1 foot below top, then down by slit height
        num_slits = 3
        # Evenly space the slits across the porch width
        total_spacing = porch_w
        gap = (total_spacing - num_slits * slit_width) / (num_slits + 1)
        for i in range(num_slits):
            slit_x = porch_x0 + gap * (i + 1) + slit_width * i
            ax.add_patch(Rectangle((slit_x, slit_y_bottom), slit_width, slit_height,
                                   facecolor="0.35", edgecolor="0.15", linewidth=1))
        cols = (porch_x0 + porch_w/2,)  # Only center column
        juliet_span = None  # No Juliet balcony
        pole_x = porch_x0 + porch_w/2
        draw_columns_and_juliet(ax, y2, cols, juliet_span, pole_x, roof_type, y3)
    else:
        cols = (porch_x0 + porch_w/2, porch_x0 + porch_w - 0.5)
        juliet_span = (porch_x0 + porch_w/2, porch_x0 + porch_w)
        pole_x = porch_x0 + porch_w - 0.5
        draw_columns_and_juliet(ax, y2, cols, juliet_span, pole_x, roof_type, y3)

    # Draw ground level line in segments to avoid crossing arches
    ax.plot([ax.get_xlim()[0], 0], [y_grade, y_grade], linewidth=2, color="0.0")  # Left extension
    if porch_x0 > 0:
        ax.plot([0, porch_x0], [y_grade, y_grade], linewidth=2, color="0.0")
    ax.plot([porch_x0 + porch_w, house_w], [y_grade, y_grade], linewidth=2, color="0.0")
    ax.plot([house_w, ax.get_xlim()[1]], [y_grade, y_grade], linewidth=2, color="0.0")  # Right extension

    ax.set_title(title, fontsize=12, weight="bold")
    ax.text(0, y3 + 11.2, f"{HOUSE_WIDTH}' main width", fontsize=10, color="0.25", va="bottom")

    if show_floor_lines:
        ax.plot([0, house_w], [y1, y1], color="0.70", linewidth=1)
        ax.plot([0, house_w], [y2, y2], color="0.70", linewidth=1)

        ax.text(-0.2, y1, "top of visible\nbasement (3')", ha="right", va="center", fontsize=9, color="0.25")
        ax.text(-0.2, y2, "2nd floor line", ha="right", va="center", fontsize=9, color="0.25")

        ax.text(porch_x0 + porch_w/2, y2 + railing_h + 0.2, "20' porch/balcony (18' clear + 1' walls)",
                ha="center", fontsize=9.5, color="0.25")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

porch_x_centered = (HOUSE_WIDTH - PORCH_WIDTH) / 2.0
draw_house(axes[0], porch_x_centered,
           "Version 1: porch centered + gable-like roof\n(+ 1 center column with pole, no Juliet balcony)",
           roof_type="gable", version=1)

porch_x_right = HOUSE_WIDTH - PORCH_WIDTH
draw_house(axes[1], porch_x_right,
           "Version 2: porch right-aligned + roof rises L→R\n(+ 2 columns, half Juliet balcony, right pole)",
           roof_type="mono", version=2)

for ax in axes:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2, 40)
    ax.set_ylim(-7, 37)
    ax.set_xlabel("Feet")
    ax.set_ylabel("Feet")
    ax.grid(False)

out_path = "./house_side_sketches_v2.png"
plt.savefig(out_path, dpi=220)
out_path
