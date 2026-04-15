"""
Environment figure for drone communication paper.
Shows user access links and backhaul links in an MST topology.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

# ── colours ───────────────────────────────────────────────────────────────────
C_ACCESS = "#2166ac"
C_BH     = "#d6604d"
C_GROUND = "#bdbdbd"
C_DRONE  = "#333333"
C_USER   = "#4dac26"

# ── layout constants ──────────────────────────────────────────────────────────
GROUND_Y   = 0.0
DRONE1_POS = np.array([3.2, 3.4])
DRONE2_POS = np.array([6.8, 5.4])
DRONE_RADIUS = 0.62   # outer extent of drone (arm tip + rotor), data units

fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(-0.8, 6.2)
ax.set_aspect("equal")
ax.axis("off")


# ── ground ────────────────────────────────────────────────────────────────────
ax.axhline(GROUND_Y, color=C_GROUND, lw=2.5, zorder=1)
ax.fill_between([0, 10], [GROUND_Y, GROUND_Y], [-0.75, -0.75],
                color=C_GROUND, alpha=0.35, zorder=1)
ax.text(9.8, -0.45, "Ground level", fontsize=9, color="gray",
        ha="right", va="center", style="italic")


# ── helpers ───────────────────────────────────────────────────────────────────
def draw_drone(ax, pos, label, label_dx=0.65, label_dy=0.0):
    """Simple quadrotor: body + 4 arms + rotor discs."""
    x, y = pos
    arm = 0.42
    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        ex, ey = x + arm * np.cos(rad), y + arm * np.sin(rad)
        ax.plot([x, ex], [y, ey], color=C_DRONE, lw=1.8, zorder=8,
                solid_capstyle="round")
        rotor = plt.Circle((ex, ey), 0.18, color="white", ec=C_DRONE,
                           lw=1.5, zorder=9)
        ax.add_patch(rotor)
    body = plt.Circle((x, y), 0.13, color=C_DRONE, zorder=10)
    ax.add_patch(body)
    ax.text(x + label_dx, y + label_dy, label, fontsize=11,
            fontweight="bold", va="center", zorder=11)


def draw_stick_figure(ax, x, y_ground=0.0, height=0.55, color=C_USER):
    """Stick figure standing at (x, y_ground)."""
    hr  = height * 0.17          # head radius
    bh  = height * 0.33          # body (torso) height
    lh  = height * 0.33          # leg height
    aw  = height * 0.26          # arm half-width
    lw  = 1.6

    foot_y  = y_ground
    hip_y   = foot_y + lh
    neck_y  = hip_y + bh
    head_cy = neck_y + hr

    # head
    head = plt.Circle((x, head_cy), hr, color=color, zorder=6)
    ax.add_patch(head)
    # torso
    ax.plot([x, x], [hip_y, neck_y], color=color, lw=lw, zorder=6)
    # arms
    arm_y = hip_y + bh * 0.65
    ax.plot([x - aw, x + aw], [arm_y, arm_y], color=color, lw=lw, zorder=6)
    # legs
    ax.plot([x, x - height * 0.14], [hip_y, foot_y], color=color, lw=lw, zorder=6)
    ax.plot([x, x + height * 0.14], [hip_y, foot_y], color=color, lw=lw, zorder=6)

    # return top of head for link anchoring
    return head_cy + hr


def shrunk_arrow(ax, src, dst, color, shrink_a=DRONE_RADIUS,
                 shrink_b=DRONE_RADIUS, lw=2.0, ls="-", head_w=0.22, head_l=0.18):
    """Arrow in data coordinates with explicit shrink (data units)."""
    vec  = np.array(dst) - np.array(src)
    dist = np.linalg.norm(vec)
    unit = vec / dist
    p0 = np.array(src) + unit * shrink_a
    p1 = np.array(dst) - unit * shrink_b
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(
                    arrowstyle=f"->,head_width={head_w},head_length={head_l}",
                    color=color, lw=lw, linestyle=ls,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=7)


def access_link(ax, drone_pos, user_top_x, user_top_y, color=C_ACCESS,
                shrink_drone=DRONE_RADIUS, shrink_user=0.04):
    """Thin line + small arrowhead from drone to user."""
    src = np.array(drone_pos)
    dst = np.array([user_top_x, user_top_y])
    vec  = dst - src
    dist = np.linalg.norm(vec)
    unit = vec / dist
    p0 = src + unit * shrink_drone
    p1 = dst - unit * shrink_user
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(
                    arrowstyle="->,head_width=0.12,head_length=0.10",
                    color=color, lw=1.0, alpha=0.7,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=3)


# ── user cluster for Drone 1 ──────────────────────────────────────────────────
# Increased number of users and tightened their spacing.
d1_user_xs = [2.35, 2.65, 2.95, 3.25, 3.55, 3.85, 4.15]
d1_user_tops = []
for x in d1_user_xs:
    top = draw_stick_figure(ax, x, GROUND_Y, height=0.52, color=C_USER)
    d1_user_tops.append((x, top))

ax.text(np.mean(d1_user_xs), -0.52, "Users", fontsize=10, color=C_USER,
        ha="center", va="top", fontweight="bold")

# ── user cluster for Drone 2 ──────────────────────────────────────────────────
d2_user_xs = [5.6, 6.2, 6.8, 7.4, 8.0]
d2_user_tops = []
for x in d2_user_xs:
    top = draw_stick_figure(ax, x, GROUND_Y, height=0.52, color=C_USER)
    d2_user_tops.append((x, top))

ax.text(np.mean(d2_user_xs), -0.52, "Users", fontsize=10, color=C_USER,
        ha="center", va="top", fontweight="bold")


# ── user access links ─────────────────────────────────────────────────────────
for ux, uy in d1_user_tops:
    access_link(ax, DRONE1_POS, ux, uy)

for ux, uy in d2_user_tops:
    access_link(ax, DRONE2_POS, ux, uy)


# ── backhaul link: Drone 1 → Drone 2 ─────────────────────────────────────────
shrunk_arrow(ax, DRONE1_POS, DRONE2_POS, color=C_BH, lw=2.2)


# ── backhaul link: Drone 2 → gateway (off-screen) ────────────────────────────
# Shifted gateway down to reduce figure height.
gateway_dir = np.array([9.7, 4.8])
shrunk_arrow(ax, DRONE2_POS, gateway_dir, color=C_BH, lw=2.2,
             ls=(0, (6, 3)), shrink_b=0.0)

ax.text(9.75, 4.9, "…", fontsize=20, color=C_BH,
        ha="left", va="center", alpha=0.8)
ax.text(9.75, 5.15, "Gateway\n(MST root)", fontsize=9, color="gray",
        ha="left", va="center", style="italic")


# ── draw drones (on top of links) ────────────────────────────────────────────
draw_drone(ax, DRONE1_POS, "Drone 1", label_dx=0.65)
draw_drone(ax, DRONE2_POS, "Drone 2", label_dx=0.65)


# ── link-type labels ──────────────────────────────────────────────────────────
# Shifted right and up to avoid intersecting the altitude annotation.
ax.text(1.75, 2.15, "User access\nlink", fontsize=10, color=C_ACCESS,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_ACCESS,
                  lw=0.8, alpha=0.9))

# Backhaul link label (above midpoint of Drone 1 → Drone 2)
mid_bh = (DRONE1_POS + DRONE2_POS) / 2
ax.text(mid_bh[0] - 0.3, mid_bh[1] + 0.55, "Backhaul link",
        fontsize=10, color=C_BH, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_BH,
                  lw=0.8, alpha=0.9))


# ── legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=C_ACCESS, lw=2.0, label="User access link"),
    Line2D([0], [0], color=C_BH,     lw=2.0, label="Backhaul link (MST)"),
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.96),
    fontsize=10,
    framealpha=0.9,
    edgecolor="lightgray",
    borderpad=0.25,
    labelspacing=0.3,
    handlelength=1.6,
    handletextpad=0.5,
    borderaxespad=0.2,
)

# ── altitude annotation ───────────────────────────────────────────────────────
ax.annotate("", xy=(0.45, GROUND_Y), xytext=(0.45, DRONE1_POS[1]),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
ax.text(0.62, DRONE1_POS[1] / 2, "Altitude", fontsize=10, color="gray",
        ha="left", va="center", rotation=90)


plt.tight_layout()
plt.savefig("results/environment_figure.pdf", bbox_inches="tight")
plt.savefig("results/environment_figure.png", dpi=180, bbox_inches="tight")
print("Saved to results/environment_figure.{pdf,png}")
plt.show()