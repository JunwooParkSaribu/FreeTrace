# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
"""Generate help.png for the FreeTrace GUI Help tab.

Draws a trajectory diagram (left) with annotated variables,
and two columns of variable reference (right).
Run: python generate_help_image.py
Output: icon/help.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import os

fig = plt.figure(figsize=(20, 12), facecolor='#1e1e1e')

# Colors
c_point = '#66ccff'
c_consec = '#44dd88'
c_gap = '#ff6666'
c_angle = '#ffaa33'
c_text = '#cccccc'
c_dim = '#888888'
c_1d = '#ff99cc'

# ============================================================
# LEFT: Trajectory diagram with annotated variables
# ============================================================
ax = fig.add_axes([0.02, 0.06, 0.44, 0.88])  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
ax.set_facecolor('#1a1a1a')
ax.set_aspect('equal')

# Sample trajectory points (with a frame gap between frame 4 and 6)
frames = [1, 2, 3, 4, 6, 7, 8]
xs = [0.5, 1.2, 2.4, 3.0, 4.5, 5.0, 5.8]
ys = [1.0, 2.0, 2.3, 3.5, 4.0, 3.2, 3.8]

# Draw trajectory path
for i in range(len(xs) - 1):
    if frames[i+1] - frames[i] == 1:
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], '-', color=c_consec, linewidth=2, alpha=0.7)
    else:
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], '--', color=c_gap, linewidth=2, alpha=0.7)

# Draw points
for i, (x, y, f) in enumerate(zip(xs, ys, frames)):
    ax.plot(x, y, 'o', color=c_point, markersize=10, zorder=5)
    ax.annotate(f'f={f}', (x, y), textcoords='offset points',
                xytext=(8, 8), fontsize=9, color=c_text, fontweight='bold')

# --- Jump Distance (frame 2→3) ---
i = 1
mid_x = (xs[i] + xs[i+1]) / 2
mid_y = (ys[i] + ys[i+1]) / 2
dx = xs[i+1] - xs[i]
dy = ys[i+1] - ys[i]
jd = np.sqrt(dx**2 + dy**2)
ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle='->', color=c_consec, lw=2.5))
perp_x, perp_y = -dy / jd * 0.25, dx / jd * 0.25
ax.text(mid_x + perp_x, mid_y + perp_y, 'Jump dist.\n= √(Δx²+Δy²)',
        fontsize=9, color=c_consec, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor=c_consec, alpha=0.9))

# --- 1D displacement (frame 1→2) ---
i = 0
ax.annotate('', xy=(xs[i+1], ys[i]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle='->', color=c_1d, lw=2))
ax.text((xs[i] + xs[i+1]) / 2, ys[i] - 0.25, 'Δx (1D disp.)',
        fontsize=9, color=c_1d, ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a1a', edgecolor=c_1d, alpha=0.9))
ax.plot([xs[i+1], xs[i+1]], [ys[i], ys[i+1]], ':', color=c_1d, linewidth=1.5, alpha=0.5)
ax.text(xs[i+1] + 0.15, (ys[i] + ys[i+1]) / 2, 'Δy', fontsize=8, color=c_1d, ha='left', alpha=0.7)

# --- Angle θ (frame 2→3→4) ---
v1 = np.array([xs[2] - xs[1], ys[2] - ys[1]])
v2 = np.array([xs[3] - xs[2], ys[3] - ys[2]])
cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle_deg = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
a1 = np.degrees(np.arctan2(v1[1], v1[0]))
a2 = np.degrees(np.arctan2(v2[1], v2[0]))
arc = Arc((xs[2], ys[2]), 0.8, 0.8, angle=0, theta1=min(a1, a2), theta2=max(a1, a2),
          color=c_angle, linewidth=2)
ax.add_patch(arc)
ax.text(xs[2] + 0.5, ys[2] + 0.25, f'θ = {angle_deg:.0f}°',
        fontsize=10, color=c_angle, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor=c_angle, alpha=0.9))

# --- Frame Gap (frame 4→6) ---
gap_mid_x = (xs[3] + xs[4]) / 2
gap_mid_y = (ys[3] + ys[4]) / 2
ax.text(gap_mid_x, gap_mid_y + 0.35, 'Frame gap\nΔt = 2\n(excluded)',
        fontsize=9, color=c_gap, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor=c_gap, alpha=0.9))

# --- Duration ---
ax.annotate('', xy=(xs[-1], 0.4), xytext=(xs[0], 0.4),
            arrowprops=dict(arrowstyle='<->', color='#bb88ff', lw=2))
ax.text((xs[0] + xs[-1]) / 2, 0.1,
        'Duration = Σ(frame diffs) × framerate  (includes gaps)',
        fontsize=8, color='#bb88ff', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor='#bb88ff', alpha=0.9))

# Legend
ax.plot([], [], '-', color=c_consec, linewidth=2, label='Consecutive (Δt = 1)')
ax.plot([], [], '--', color=c_gap, linewidth=2, label='Frame gap (Δt > 1, excluded)')
ax.plot([], [], 'o', color=c_point, markersize=8, label='Detection')
ax.legend(loc='upper left', fontsize=8, facecolor='#2a2a2a', edgecolor='#555555',
          labelcolor=c_text)

ax.set_xlim(-0.3, 6.8)
ax.set_ylim(-0.5, 5.0)
ax.set_xlabel('x (μm)', color=c_text, fontsize=11)
ax.set_ylabel('y (μm)', color=c_text, fontsize=11)
ax.set_title('Trajectory Variables', color=c_text, fontsize=13, fontweight='bold', pad=10)
ax.tick_params(colors=c_dim)
for spine in ax.spines.values():
    spine.set_color('#555555')

# ============================================================
# RIGHT: Two-column variable reference
# ============================================================

# Column 1 (Basic Stats variables)
col1_entries = [
    ('H (Hurst exponent)', '#66ccff',
     'Characterizes diffusion type:\n'
     '  H < 0.5 → Subdiffusion\n'
     '  H = 0.5 → Brownian motion\n'
     '  H > 0.5 → Superdiffusion'),
    ('K (Diffusion coefficient)', '#66ccff',  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
     'Generalized diffusion coefficient.\n'
     'Higher K → faster diffusion.\n'
     'Computed in pixel & frame scale,\n'
     'not converted to μm & s.\n'
     'Log-scale (geometrically spaced bins).'),
    ('Jump Distance', c_consec,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
     'Euclidean distance between consecutive\n'
     'detections: √(Δx² + Δy²).\n'
     'Only consecutive frames (Δt = 1).\n'
     'Assumes isotropic motion.'),
    ('Mean Jump Distance', c_consec,
     'Average jump distance per trajectory\n'
     '(one value per trajectory).'),
    ('Duration', '#bb88ff',
     'Total observation time:\n'
     'Σ(frame diffs) × framerate.\n'
     'Includes frame gaps.'),
    ('EA-SD', '#ff9966',
     'Ensemble-Averaged Squared Displacement.\n'
     'Average SD over all trajectories at\n'
     'each time point. Uses absolute\n'
     'positions; handles gaps naturally.'),
    ('Angle (0°–180°)', c_angle,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
     'Deflection angle between two\n'
     'consecutive jump vectors (dot product).\n'
     '0° = straight, 180° = reversal.\n'
     'Both steps must be Δt = 1.\n'
     'Uniform if isotropic & Brownian.'),
    ('Polar Angle (0°–360°)', c_angle,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
     'Signed turning angle (atan2).\n'
     '0°/360° = straight ahead (right),\n'
     '180° = reversal (left).\n'
     'Uniform if isotropic & Brownian.'),
]

# Column 2 (Advanced Stats)
col2_entries = [
    ('TA-EA-SD', '#ff9966',
     'Time-Averaged, Ensemble-Averaged SD.\n'
     'For each trajectory, average SD over\n'
     'all time windows of lag τ, then\n'
     'ensemble-average across trajectories.\n'
     'More robust than EA-SD for short trajectories.'),
    ('1D Displacement (Δx, Δy)', c_1d,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
     'Projection of a step onto one axis.\n'
     'Gaussian for Brownian or fBm molecules\n'
     'in a homogeneous population.\n'
     'Non-Gaussian → heterogeneous population\n'
     '(mixed diffusion states). Only Δt = 1.\n'
     'Assumes isotropic motion (Δx ≡ Δy).'),
    ('1D Displacement Ratio', c_1d,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
     'Ratio of consecutive 1D displacements:\n'
     'Δx(t+1) / Δx(t). For fBm (any H),\n'
     'the ratio follows a Cauchy-like dist.\n'
     'Deviation from Cauchy → the motion\n'
     'is not purely fractional Brownian.\n'
     'Assumes homogeneous & isotropic.'),
]


def _draw_column(ax_col, title, entries, title_color='#66ccff'):
    ax_col.axis('off')
    y = 0.97
    ax_col.text(0.0, y, title, color=title_color, fontsize=12,
                fontweight='bold', transform=ax_col.transAxes, va='top')
    y -= 0.05
    for name, color, desc in entries:
        ax_col.text(0.0, y, f'■ {name}', color=color, fontsize=9.5,
                    fontweight='bold', transform=ax_col.transAxes, va='top')
        y -= 0.025
        ax_col.text(0.02, y, desc, color='#aaaaaa', fontsize=8,
                    transform=ax_col.transAxes, va='top', family='monospace',
                    linespacing=1.3)
        n_lines = desc.count('\n') + 1
        y -= 0.028 * n_lines + 0.015


ax_c1 = fig.add_axes([0.48, 0.03, 0.26, 0.94])  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
_draw_column(ax_c1, 'Basic Stats', col1_entries)

ax_c2 = fig.add_axes([0.72, 0.03, 0.26, 0.94])  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
_draw_column(ax_c2, 'Advanced Stats', col2_entries)

# Save
out_path = os.path.join(os.path.dirname(__file__), 'icon', 'help.png')
fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
print(f'Saved: {out_path}')
plt.close(fig)
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
