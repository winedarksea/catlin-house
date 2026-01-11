import matplotlib.pyplot as plt
import numpy as np
import textwrap
from matplotlib.patches import Patch

from ifcplot.detail_utils import COLORS, HATCHES, _batt_insulation, _rect, _screw, _stud_pattern

# ----------------------------
# Figure layout with notes panel
# ----------------------------
fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.2], wspace=0.15)
ax = fig.add_subplot(gs[0, 0])
ax_notes = fig.add_subplot(gs[0, 1])

# ----------------------------
# Parameters (inches) - schematic top view
# ----------------------------
plate_depth = 7.25        # 2x8 actual depth
stud_thick = 1.5          # 2x4 thickness
stud_depth = 3.5          # 2x4 depth

drywall = 0.625
osb = 0.5
rigid_mw = 2.0            # 2" rigid mineral wool
rigid_eps = 2.0           # 2" EPS
furring_thick = 0.75      # 1x actual thickness
siding = 0.75             # schematic thickness

# Screw spec
screw_shaft_diam = 0.275

# Vertical extent (length along wall shown)
H = 24.0
y0 = 0.0

# X positions (interior -> exterior)
x0 = 0.0
x_dry0, x_dry1 = x0, x0 + drywall
x_plate0, x_plate1 = x_dry1, x_dry1 + plate_depth

# Interior studs: 3.5" perpendicular to wall
x_in_stud0, x_in_stud1 = x_plate0, x_plate0 + stud_depth

# OSB flush to interior studs
x_osb0, x_osb1 = x_in_stud1, x_in_stud1 + osb

# Outer studs rotated "flat"
x_out_stud1, x_out_stud0 = x_plate1, x_plate1 - stud_thick

# ccSPF wraps around outer studs
x_cc0, x_cc1 = x_osb1, x_plate1

# Exterior CI (split: mineral wool + EPS), furring, siding
x_mw0, x_mw1 = x_plate1, x_plate1 + rigid_mw
x_eps0, x_eps1 = x_mw1, x_mw1 + rigid_eps
x_fur0, x_fur1 = x_eps1, x_eps1 + furring_thick
x_sid0, x_sid1 = x_fur1, x_fur1 + siding

# ----------------------------
# Colors
# ----------------------------
COL = COLORS
COL_DRY = COL.drywall
COL_STUD = COL.wood
COL_OSB = COL.osb
COL_CCS = COL.spray_foam
COL_MW = COL.mineral_wool
COL_EPS = COL.eps
COL_FUR = COL.wood
COL_SID = COL.metal_dark
COL_BATT = COL.insulation
COL_PLATE_EDGE = "black"
screw_color = COL.flashing

# ----------------------------
# Base layers
# ----------------------------
# Plate outline (dotted)
_rect(ax, x_plate0, y0, plate_depth, H, fc="none", ec=COL_PLATE_EDGE, lw=1.2, ls=':')

# Drywall
_rect(ax, x_dry0, y0, drywall, H, fc=COL_DRY, ec="black", lw=1.2)

# Interior studs @24" o.c. (show 2)
in_stud_positions = [5, 17]
_stud_pattern(ax, in_stud_positions, fixed=x_in_stud0, depth=stud_depth, thickness=stud_thick, fc=COL_STUD, ec="black", lw=1.2, z=3)

# Optional batts between interior studs (rounded)
batt_x0 = x_plate0 + 0.2
batt_x1 = x_osb0 - 0.2
batt_w = batt_x1 - batt_x0
# Place batts in cavities between shown studs
batt_positions = [(in_stud_positions[0] + stud_thick + 0.5, in_stud_positions[1] - 0.5)]
for yb0, yb1 in batt_positions:
    _batt_insulation(ax, batt_x0, yb0, batt_w, yb1 - yb0, fc=COL_BATT, ec="black", lw=0.8, radius=0.6, hatch=HATCHES.compacted, z=1)

# OSB
_rect(ax, x_osb0, y0, osb, H, fc=COL_OSB, ec="black", lw=1.2)

# ccSPF (wraps around outer studs)
_rect(ax, x_cc0, y0, x_cc1 - x_cc0, H, fc=COL_CCS, ec="black", lw=1.0)

# Outer studs @18" o.c. (draw on top)
out_stud_positions = [2, 10, 18]
_stud_pattern(ax, out_stud_positions, fixed=x_out_stud0, depth=stud_thick, thickness=4, fc=COL_STUD, ec="black", lw=1.2, z=4)

# Rigid mineral wool (interior CI layer)
_rect(ax, x_mw0, y0, rigid_mw, H, fc=COL_MW, ec="black", lw=1.2)

# EPS (exterior CI layer)
_rect(ax, x_eps0, y0, rigid_eps, H, fc=COL_EPS, ec="black", lw=1.2)

# Furring strips @18" o.c.
fur_positions = [2, 10, 18]
_stud_pattern(ax, fur_positions, fixed=x_fur0, depth=furring_thick, thickness=4, fc=COL_FUR, ec="black", lw=1.2, z=5)

# Standing seam siding
_rect(ax, x_sid0, y0, siding, H, fc=COL_SID, ec="black", lw=1.2)

# ----------------------------
# Screws: from furring outer edge to stud inner edge
# ----------------------------
for yy in fur_positions:
    y_center = yy + 2.0
    _screw(ax, x_fur1, x_out_stud0, y_center, screw_shaft_diam, color=screw_color, z=6)

# ----------------------------
# Legend
# ----------------------------
legend_patches = [
    Patch(facecolor=COL_DRY, edgecolor="black", label='Drywall (5/8" + Class III paint)'),
    Patch(facecolor=COL_STUD, edgecolor="black", label='Wood framing (2x4s)'),
    Patch(facecolor=COL_OSB, edgecolor="black", label='OSB (1/2", taped seams)'),
    Patch(facecolor=COL_CCS, edgecolor="black", label='Closed-cell spray foam'),
    Patch(facecolor=COL_MW, edgecolor="black", label='Rigid mineral wool (2")'),
    Patch(facecolor=COL_EPS, edgecolor="black", label='EPS rigid foam (2")'),
    Patch(facecolor=COL_BATT, edgecolor="black", label='Cavity insulation (≤R-20)'),
    Patch(facecolor=COL_FUR, edgecolor="black", label='Furring (1x4)'),
    Patch(facecolor=COL_SID, edgecolor="black", label='Standing seam siding'),
    Patch(facecolor=screw_color, edgecolor="black", label='Structural screw (Ø0.275")'),
]
ax.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0.0, -0.25),
          frameon=True, fontsize=9, ncol=2)

# ----------------------------
# Dimension arrows
# ----------------------------
def dim(xa, xb, y, txt):
    ax.annotate('', xy=(xa, y), xytext=(xb, y),
                arrowprops=dict(arrowstyle='<->', linewidth=1.1))
    ax.text((xa+xb)/2, y+0.5, txt, ha='center', va='bottom', fontsize=9)

# Top horizontal dims (spaced to avoid overlap)
dim(x_plate0, x_plate1, H+1.0, '2x8 plate (7.25")')
dim(x_mw0, x_mw1, H+3.4, '2" MW')
dim(x_eps0, x_eps1, H+5.8, '2" EPS')

# Vertical spacing dims on right side
def vdim(y0, y1, x, txt):
    ax.annotate('', xy=(x, y0), xytext=(x, y1),
                arrowprops=dict(arrowstyle='<->', linewidth=1.1))
    ax.text(x+0.4, (y0+y1)/2, txt, ha='left', va='center', fontsize=9, rotation=90)

# 18" o.c. for exterior studs / furring
vdim(out_stud_positions[0]+2, out_stud_positions[1]+2, x_sid1+1.2, '18" o.c.')

# 24" o.c. for interior studs (representative)
vdim(in_stud_positions[0]+stud_thick/2, in_stud_positions[1]+stud_thick/2,
     x_sid1+3.4, '24" o.c.')

# ----------------------------
# Labels
# ----------------------------
ax.text(x_dry0, -1.6, "INTERIOR", ha='left', va='top', fontsize=10, fontweight='bold')
ax.text(x_sid1, -1.6, "EXTERIOR", ha='right', va='top', fontsize=10, fontweight='bold')

ax.set_title("Wall Section — Standing Seam over Rainscreen over Continuous Insulation",
             ha='center', fontsize=12, fontweight='bold', pad=14)
ax.text((x0 + x_sid1)/2, -4.2, "Colin Catlin, 2025", ha='center', va='top', fontsize=7)

# ----------------------------
# Formatting
# ----------------------------
ax.set_xlim(x0-1.0, x_sid1+5.5)
ax.set_ylim(-5.0, H+10.0)
ax.set_aspect('equal')
ax.axis('off')

# ----------------------------
# Notes Panel
# ----------------------------
ax_notes.set_xlim(0, 100)
ax_notes.set_ylim(0, 130)
ax_notes.axis('off')

raw_notes = [
    "NOTES:",
    "",
    "• Both interior and exterior stud lines are mounted to the same 2×8 top plates and sill plates (7.25\" actual depth), creating a unified structural base for the double-stud wall system.",
    "",
    "• Apply liquid membrane flashing on the exterior side of sills, top plates, and rim joists, extending to the edge of the taped OSB sheathing.",
    "",
    "• Additional reinforcement against warping of exterior stud line is strongly recommended. Use LSL studs or install horizontal 2×4 blocking or 6\" wide strips of 3/4\" plywood as strongbacks at quarter-point and midpoint heights up the wall.",
    "",
    "• Interior cavity insulation (optional) should not exceed R-20 to maintain adequate sheathing temperature and prevent condensation.",
    "",
    "• Closed-cell spray foam may be replaced with mineral wool batts in the outer stud cavity. If using mineral wool, apply a full liquid membrane coating to the exterior face of OSB sheathing.",
    "",
    "• Exterior continuous insulation consists of 2\" rigid mineral wool (interior layer) plus 2\" unfaced EPS (exterior layer, class III vapor barrier). Stagger seams between layers.",
    "",
    "• Taped OSB sheathing is the primary air barrier. Detail continuity at floor lines, corners, and roof connections.",
    "",
    "• Outer studs (2×4 flat) and furring strips aligned at 18\" o.c. for structural screw attachment.",
    "",
    "• Use engineered long structural screws (min. #12) through furring, CI, and into outer studs per engineer. Stainless steel recommended.",
    "",
    "• Interior finish with Class III vapor retarder paint. Do NOT use polyethylene or Class I/II vapor barriers on interior.",
    "",
    "• Provide rainscreen gap via furring strips; ensure continuous drainage and ventilation path.",
]

wrapped_lines = []
for line in raw_notes:
    if line.strip().startswith("•"):
        wrapped = textwrap.wrap(line, width=48)
        wrapped_lines.extend(wrapped)
    else:
        wrapped_lines.append(line)

notes_text = "\n".join(wrapped_lines)

# ax_notes.add_patch(Rectangle((2, -10), 96, 137, facecolor='white', edgecolor='black', linewidth=1.4))
ax_notes.text(5, 127, notes_text, fontsize=9, va='top', ha='left', family='monospace')

fig.suptitle("High-Performance Wall Assembly Detail (Top View) — Builder Reference", fontsize=14, fontweight='bold')

out_path = "./wall_detail_catlin.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
out_path
