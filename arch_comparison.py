import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

def arch_opening_path(x_left, x_right, y_base, opening_height):
    """Path for an opening with vertical sides + semicircular top spanning [x_left, x_right].
    opening_height is total opening height (here 8'). The semicircle radius is w/2.
    """
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

def draw_option(ax, y0, n_arches, arch_width, middle_length=18.0, outer_pier=1.0,
                wall_height=9.0, opening_height=8.0):
    # Geometry: total wall length includes two fixed outer piers (not counted in middle_length)
    total_length = middle_length + 2 * outer_pier

    arches_total = n_arches * arch_width
    interior_piers_count = max(n_arches - 1, 0)
    remaining_for_interior = middle_length - arches_total
    if interior_piers_count > 0:
        pier = remaining_for_interior / interior_piers_count
    else:
        pier = 0.0

    # Wall body
    ax.add_patch(Rectangle((0, y0), total_length, wall_height,
                           facecolor="0.85", edgecolor="0.1", linewidth=2))

    # Draw openings within the middle 18' region: [outer_pier, outer_pier + middle_length]
    x = outer_pier
    for i in range(n_arches):
        x_left = x
        x_right = x + arch_width
        path, spring_y, (cx, cy), r = arch_opening_path(x_left, x_right, y0, opening_height)
        ax.add_patch(PathPatch(path, facecolor="1.0", edgecolor="0.2", linewidth=1.5))

        # Outline
        ax.plot([x_left, x_left], [y0, spring_y], color="0.2", linewidth=1.5)
        ax.plot([x_right, x_right], [y0, spring_y], color="0.2", linewidth=1.5)
        thetas = np.linspace(np.pi, 0, 320)
        ax.plot(cx + r*np.cos(thetas), cy + r*np.sin(thetas), color="0.2", linewidth=1.5)

        # advance
        x = x_right
        if i < n_arches - 1:
            x += pier

    # Middle region boundary (the 18' that contains arches + interior piers)
    ax.plot([outer_pier, outer_pier], [y0, y0 + wall_height], color="0.35", linewidth=1, linestyle=":")
    ax.plot([outer_pier + middle_length, outer_pier + middle_length], [y0, y0 + wall_height], color="0.35", linewidth=1, linestyle=":")
    ax.text(outer_pier, y0 + wall_height + 0.05, "start of 18'", ha="left", va="bottom", fontsize=9, color="0.25")
    ax.text(outer_pier + middle_length, y0 + wall_height + 0.05, "end of 18'", ha="right", va="bottom", fontsize=9, color="0.25")

    # Arch apex line
    ax.plot([0, total_length], [y0 + opening_height, y0 + opening_height],
            linestyle="--", linewidth=1, color="0.4")
    ax.text(total_length + 0.2, y0 + opening_height, "8' arch apex", va="center", fontsize=10, color="0.25")

    # Labels
    label = f"{n_arches} arches × {arch_width:g}' wide"
    ax.text(0, y0 + wall_height + 0.28, label, fontsize=12, weight="bold", va="bottom")

    ax.text(0, y0 + wall_height + 0.05,
            f"Total wall = {total_length:.1f}' (includes 1' outer piers each side). Middle = 18'. "
            f"Arches = {arches_total:.2f}'. Interior piers: {interior_piers_count} × {pier:.2f}'",
            fontsize=10, va="bottom", color="0.25")

# Plot
fig, ax = plt.subplots(figsize=(12, 8))

gap = 2.0
base = 0.0
draw_option(ax, base, 4, 4.0)
draw_option(ax, base + 9 + gap, 3, 5.5)
draw_option(ax, base + 2*(9 + gap), 2, 8.0)

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-0.5, 22.0)
ax.set_ylim(-0.5, base + 3*(9 + gap) + 1.7)
ax.set_xlabel("Feet")
ax.set_ylabel("Feet")
ax.set_title("Concrete Wall Arch Layout Comparison\n(18' middle length for arches + interior piers; fixed 1' outer piers; 9' wall height; 8' openings)")

# 1' cap annotation
ax.annotate("1' concrete above arch apex", xy=(20.0, 8.0), xytext=(20.4, 9.0),
            arrowprops=dict(arrowstyle="->", lw=1), fontsize=10, color="0.25")

ax.grid(False)
plt.tight_layout()


out_path = "./arch_comparison.png"
plt.savefig(out_path, dpi=200)
out_path
