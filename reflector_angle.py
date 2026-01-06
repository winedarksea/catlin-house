"""
The idea is that a fixed reflector can increase light selectively (winter and early morninge/late evening)
while also reducing required window area (windows being poor insulators).
See also: window awnings for reduced summer heat gain

This is mostly for south facing windows (in the northern hemisphere)
"""
import numpy as np, math
import matplotlib.pyplot as plt

# Geometry (I'm using inches, but math is same if all is in mm)
d = 8.0  # distance top of mirror from bottom of window, horizontally
window_h = 36.0  # window height, top to bottom
mirror_len = 18.0  # hypotenuse of mirror, mirror size

P_mirror_top = np.array([d, 0.0])
P_window_top = np.array([0.0, window_h])

# --- Solve mirror orientation ---
# I chose sun angle at 4 pm on August 31st, trying to set my "too hot above" boundary
# angles found here: https://gml.noaa.gov/grad/solcalc/azel.html
sun_ref_deg = 30.0
a = np.array([math.cos(math.radians(sun_ref_deg)),
              math.sin(math.radians(sun_ref_deg))])
b = P_window_top - P_mirror_top
b = b / np.linalg.norm(b)

n = a + b
n = n / np.linalg.norm(n)

t = np.array([-n[1], n[0]])
if t[0] < 0:
    t = -t

P_mirror_bottom = P_mirror_top + mirror_len * t
mirror_drop = abs(P_mirror_bottom[1] - P_mirror_top[1])

def reflected_ray(theta_deg):
    s_in = np.array([-math.cos(math.radians(theta_deg)),
                     -math.sin(math.radians(theta_deg))])
    s_in = s_in / np.linalg.norm(s_in)
    r_out = s_in - 2*np.dot(s_in, n)*n
    return s_in, r_out

def y_at_window(theta_deg, P):
    s_in, r_out = reflected_ray(theta_deg)
    tau = -P[0] / r_out[0]
    return P[1] + tau * r_out[1]

# Find bottom-edge threshold angle
thetas = np.linspace(0.1, 89.0, 3000)
vals = [y_at_window(th, P_mirror_bottom) - window_h for th in thetas]
idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
lo, hi = thetas[idx[0]], thetas[idx[0]+1]
for _ in range(80):
    mid = 0.5*(lo+hi)
    if (y_at_window(lo, P_mirror_bottom)-window_h)*(y_at_window(mid, P_mirror_bottom)-window_h) <= 0:
        hi = mid
    else:
        lo = mid
theta_bottom_hit = 0.5*(lo+hi)

# Plot
fig, ax = plt.subplots(figsize=(7.6,5.6))

# Window & mirror
ax.plot([0,0],[0,window_h], color="tab:blue", linewidth=3, label="window")
ax.plot([P_mirror_top[0], P_mirror_bottom[0]],
        [P_mirror_top[1], P_mirror_bottom[1]],
        color="gray", linewidth=4, label="mirror")

# Angles and colors
angles = [10, 30, 45]
colors = {10:"tab:green", 30:"tab:orange", 45:"tab:purple"}

for ang in angles:
    s_in, r_out = reflected_ray(ang)
    P_in_start = P_mirror_top - 45*s_in
    P_out_end = P_mirror_top + 70*r_out

    if ang == 30:
        ls, lw, aalpha = "-", 2.5, 1.0
    else:
        ls, lw, aalpha = "dotted", 1.8, 0.5

    ax.plot([P_in_start[0], P_mirror_top[0]],
            [P_in_start[1], P_mirror_top[1]],
            color=colors[ang], linewidth=lw, linestyle=ls, alpha=aalpha,
            label=f"incoming {ang}°")
    ax.plot([P_mirror_top[0], P_out_end[0]],
            [P_mirror_top[1], P_out_end[1]],
            color=colors[ang], linewidth=lw, linestyle=ls, alpha=aalpha,
            label=f"reflected {ang}°")

# Bottom-edge threshold rays
s_in, r_out = reflected_ray(theta_bottom_hit)
P_in_start = P_mirror_bottom - 55*s_in
tau = -P_mirror_bottom[0]/r_out[0]
P_hit = P_mirror_bottom + tau*r_out

ax.plot([P_in_start[0], P_mirror_bottom[0]],
        [P_in_start[1], P_mirror_bottom[1]],
        color="tab:red", linestyle="--", linewidth=2.2,
        label=f"incoming {theta_bottom_hit:.1f}° (mirror bottom)")
ax.plot([P_mirror_bottom[0], P_hit[0]],
        [P_mirror_bottom[1], P_hit[1]],
        color="tab:red", linestyle="--", linewidth=2.2,
        label=f"reflected {theta_bottom_hit:.1f}° (hits window top)")

# Annotations
tilt_down = -math.degrees(math.atan2(t[1], t[0]))
ax.text(0.02, 0.97,
        f"Mirror tilt: {tilt_down:.1f}° down away from window",
        transform=ax.transAxes, va="top", fontsize=9)

mid_mirror = 0.5*(P_mirror_top + P_mirror_bottom)
ax.text(mid_mirror[0]-6, mid_mirror[1]-3,
        f"{mirror_len:.1f}\"",
        fontsize=9, color="black")

ax.text(P_mirror_bottom[0]+1, P_mirror_bottom[1]-3,
        f"{mirror_drop:.1f}\" drop",
        fontsize=9)

ax.set_aspect('equal', 'box')
ax.set_xlabel("x (in), outside is +x")
ax.set_ylabel("y (in), up")
ax.set_title("Fixed Solar Reflector Geometry")
ax.grid(True)
ax.legend(loc="upper right", fontsize=8)

# plt.show()
plt.savefig("solar_reflector.png", dpi=300, bbox_inches="tight")

print(f"Mirror tilt = {tilt_down:.3f}°")
print(f"Mirror bottom drop = {mirror_drop:.3f} in")
print(f"Bottom-edge → window-top solar elevation = {theta_bottom_hit:.3f}°")
