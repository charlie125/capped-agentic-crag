import json
from pathlib import Path
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR.parent / "plot"                  # experiment_result/plot/
OUT_DIRS = [PLOT_DIR]

# When this repository sits inside the dissertation working tree, refresh the
# manuscript's copy as well, so the two cannot drift apart. Skipped in a bare
# clone, where the directory does not exist.
PAPER_DIR = SCRIPT_DIR.parents[2] / "paper" / "experimental result"
if PAPER_DIR.is_dir():
    OUT_DIRS.append(PAPER_DIR)

capped_latency = []
with open(SCRIPT_DIR / "capped_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        capped_latency.append(round(s["wall_clock_seconds"], 2))

uncapped_latency = []
with open(SCRIPT_DIR / "uncapped_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        uncapped_latency.append(round(s["wall_clock_seconds"], 2))

naive_latency = []
with open(SCRIPT_DIR / "naive_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        naive_latency.append(round(s["wall_clock_seconds"], 2))
# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(12, 7), dpi=150)

plt.plot(naive_latency, marker='o', linestyle='-', color='#e377c2', label='Naive')

# The remaining two series
plt.plot(uncapped_latency, marker='o', linestyle='-', color='#1f77b4', label='Uncapped')
plt.plot(capped_latency, marker='o', linestyle='-', color='#ff7f0e', label='Capped')

# Title and axis labels
plt.title('Latency: Naive vs Uncapped vs Capped', fontsize=14)
plt.xlabel('Test Case ID', fontsize=11)
plt.ylabel('Elapsed Time (seconds)', fontsize=11)

# Background grid
plt.grid(True, linestyle='-', color='lightgray', alpha=0.6)

# Category dividers: the dataset switches question type at items 12 and 17
split_1 = 12
plt.axvline(x=split_1, color='gray', linestyle='--', alpha=0.7)

split_2 = 17
plt.axvline(x=split_2, color='gray', linestyle='--', alpha=0.7)

# Category band labels
# Anchor the label height to the current maximum so it never overlaps a line
max_val = max(uncapped_latency) if uncapped_latency else 50
total_len = len(uncapped_latency)

# 1. First band: Single-hop (midpoint of 0 .. split_1)
plt.text(split_1 / 2, max_val * 0.9, 'Single-hop',
         color='gray', fontsize=12, ha='center')

# 2. Second band: Multi-hop (midpoint of split_1 .. split_2)
plt.text(split_1 + (split_2 - split_1) / 2, max_val * 0.9, 'Multi-hop',
         color='gray', fontsize=12, ha='center')

# 3. Third band: Unanswerable (midpoint of split_2 .. end)
plt.text(split_2 + (total_len - split_2) / 2, max_val * 0.9, 'Unanswerable',
         color='gray', fontsize=12, ha='center')

# Legend
plt.legend(loc='upper left', fontsize=11)

# Tight layout and render
plt.tight_layout()
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)
    plt.savefig(d / "latency_comparison_plot.png", dpi=400)
    print("wrote", d / "latency_comparison_plot.png")
plt.show()