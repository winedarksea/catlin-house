import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import textwrap

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.4], wspace=0.25)

ax_main = fig.add_subplot(gs[0, 0])
ax_notes = fig.add_subplot(gs[0, 1])

# ---------------- Main Section ----------------
ax_main.set_xlim(0, 120)
ax_main.set_ylim(0, 80)
ax_main.axis('off')

x0, w = 12, 90

layers = [
    ("Standing seam metal roofing", 4, '////'),
    ("1x4 furring strips (vented/drained cavity)\n(vertical orientation for eave→ridge airflow)", 7, '..'),
    ("High-temp roofing underlayment\n(HT synthetic or HT peel & stick)", 3, 'xx'),
    ("4.5\" rigid foam insulation (polyiso, eps)\n(stagger/tape seams; tape-compatible facer)", 18, '//'),
    ("1\" mineral wool board\n(continuous; protects/levels deck)", 4, '\\\\'),
    ("Taped OSB roof deck (PRIMARY AIR BARRIER)\n(tape seams; seal penetrations)", 5, '--'),
    ("I-joist / cavity insulation (typ.)\nR-20 mineral wool batts or dense-pack cellulose\nNo interior Class I/II vapor barrier", 22, ''),
    ("Interior finish (e.g., gypsum board)\nSeal penetrations for airtightness", 6, 'oo'),
]

y = 70
rects = []
for label, h, hatch in layers:
    y1 = y - h
    rect = Rectangle((x0, y1), w, h, facecolor='white', edgecolor='black', hatch=hatch, linewidth=1.4)
    ax_main.add_patch(rect)
    rects.append((rect, label))
    y = y1 - 1.2

callout_x = 108
callout_y = 70
dy = 7.5

for rect, label in rects:
    rx, ry = rect.get_xy()
    cx = rx + w
    cy = ry + rect.get_height()/2
    ax_main.plot([cx, callout_x-2], [cy, callout_y], color='black', linewidth=1.0)
    wrapped = "\n".join(textwrap.wrap(label, width=30))
    ax_main.text(callout_x, callout_y, wrapped, fontsize=9, va='center', ha='left')
    callout_y -= dy

def dim(ax, x, y1, y2, txt):
    arrow = FancyArrowPatch((x, y1), (x, y2), arrowstyle='<->', mutation_scale=12, linewidth=1.2)
    ax.add_patch(arrow)
    ax.text(x-1.5, (y1+y2)/2, txt, rotation=90, fontsize=9, va='center', ha='right')

def y_bounds(idx):
    rect, _ = rects[idx]
    rx, ry = rect.get_xy()
    return ry, ry + rect.get_height()

dim(ax_main, 7, *y_bounds(0), "Metal")
dim(ax_main, 7, *y_bounds(1), "Vent +\n1x4")
dim(ax_main, 7, *y_bounds(2), "Under-\nlayment")
dim(ax_main, 7, *y_bounds(3), "Polyiso\n4.5\"")
dim(ax_main, 7, *y_bounds(4), "MW\n1\"")
dim(ax_main, 7, *y_bounds(5), "OSB")
dim(ax_main, 7, *y_bounds(6), "Joist\ncavity")

ax_main.set_title("Roof Assembly Section — Standing Seam over Vented Furring over Continuous Insulation",
                  fontsize=14, fontweight='bold')

# ---------------- Notes ----------------
ax_notes.set_xlim(0, 100)
ax_notes.set_ylim(0, 100)
ax_notes.axis('off')

raw_notes = [
    "NOTES:",
    "",
    '• Joists, furring strips, and roofing all on aligned 18" o.c. spacing.',
    "",
    '• #12 stainless screws @ 24" o.c. along each 1×4 furring strip. #12 @ 12" o.c within 4\'-0" of eaves, rakes, and ridge. Embedded ≥1.5" into joist',
    "",
    "• Underlayment is installed ABOVE rigid insulation (below furring/metal) as secondary water barrier.",
    "",
    "• Provide continuous ventilation path in furring cavity.",
    "",
    "• Use engineered long structural screws through furring + foam into framing per manufacturer/engineer.",
    "",
    "• Stainless steel fasteners are recommended to reduce thermal bridging and improve durability.",
    "",
    "• Stagger insulation seams; tape polyiso seams; shingle underlayment laps for drainage.",
    "",
    "• Taped OSB deck is primary air barrier—detail continuity to wall air barrier at eaves, rakes, and ridge.",
    "",
    "• Joists to be supported with adjustable / slopeable metal hangers suitable for roof pitch (by engineer).",
    "",
    "• Provide web stiffeners at bearings and as required when using I-joists.",
    "",
    "• Either standard I-joists or trim joists may be used; select depth and series appropriate for span and loads.",
    "",
    "• Do NOT install interior polyethylene or other Class I/II vapor barriers in this warm-roof assembly."
]

wrapped_lines = []
for line in raw_notes:
    if line.strip().startswith("•"):
        wrapped = textwrap.wrap(line, width=45)
        wrapped_lines.extend(wrapped)
    else:
        wrapped_lines.append(line)

notes_text = "\n".join(wrapped_lines)

ax_notes.add_patch(Rectangle((3, 3), 94, 94, facecolor='white', edgecolor='black', linewidth=1.4))
ax_notes.text(6, 96, notes_text, fontsize=10, va='top', ha='left')

fig.suptitle("High-Performance Warm Roof Detail (CZ6) — Builder Reference", fontsize=15, fontweight='bold')

outpath = "./roof_details.png"
plt.savefig(outpath, dpi=300, bbox_inches='tight')
outpath
