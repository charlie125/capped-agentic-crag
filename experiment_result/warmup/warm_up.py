import json
from pathlib import Path
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR.parent / "plot"
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Load the JSON record and split it into x and y series
with open(SCRIPT_DIR / "warmup_run_times.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

runs = [data["run"] for data in raw_data]
latency = [data["execution_time_seconds"] for data in raw_data]

# Figure and size
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Main line and markers
ax.plot(runs, latency, marker='o', color='#1f77b4', linewidth=2.5, markersize=7, label='Latency (s)')

# 1. Shade the steady-state region from Run 3 onwards (light green)
ax.axvspan(3, 15, color='#e8f5e9', alpha=0.6, label='Stable Region (Run 3–15)')

# 2. Vertical convergence boundary at Run 3
ax.axvline(x=3, color='#2ca02c', linestyle='--', linewidth=1.5, label='Convergence Point (Run 3)')

# 3. Annotation arrow and caption
ax.annotate('Stable Latency Range\n(~4.58s - 4.63s)',
            xy=(3, 4.60), xytext=(6.5, 8.0),
            arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=8),
            fontsize=10, fontweight='bold', color='#1b5e20',
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#2ca02c", lw=1.5))

# 4. Value labels on the key points (Runs 1-3)
ax.text(1, 10.84 + 0.3, '10.84s', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d62728')
ax.text(2, 5.63 + 0.3, '5.63s', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#ff7f0e')
ax.text(3, 4.60 - 0.5, '4.60s', ha='center', va='top', fontsize=9, fontweight='bold', color='#2ca02c')

# Style: the y-axis starts at 0 so the cold-start drop is not visually exaggerated
ax.set_title('Warm-up Latency Convergence (Run 1–15)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Run Number', fontsize=11, fontweight='bold')
ax.set_ylabel('Latency (s)', fontsize=11, fontweight='bold')
ax.set_xticks(runs)
ax.set_ylim(0, 12)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

# 5. Inset: detail of the Run 3-15 steady-state region
axins = inset_axes(ax, width="42%", height="35%", loc='center right',
                    bbox_to_anchor=(0.02, -0.05, 1, 1), bbox_transform=ax.transAxes)
axins.plot(runs[2:], latency[2:], marker='o', color='#2ca02c', linewidth=2, markersize=5)
axins.set_xlim(2.5, 15.5)
axins.set_ylim(4.5, 4.7)
axins.set_title('Zoom: Run 3–15', fontsize=8, fontweight='bold')
axins.tick_params(labelsize=7)
axins.grid(True, linestyle=':', alpha=0.5)

# Dashed rectangle marking the inset region on the main axes
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--', linewidth=0.8)
PLOT_DIR.mkdir(exist_ok=True)
plt.savefig(PLOT_DIR / 'Warm-up_Latency_Convergence.png', dpi=400, bbox_inches='tight')
plt.show()