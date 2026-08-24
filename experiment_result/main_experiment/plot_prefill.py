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

# 1. Load the Capped report (prompt_tokens is the Prefill field)
capped_tokens = []
with open(SCRIPT_DIR / "capped_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        capped_tokens.append(s["prompt_tokens"])

uncapped_tokens = []
with open(SCRIPT_DIR / "uncapped_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        uncapped_tokens.append(s["prompt_tokens"])

naive_tokens = []
with open(SCRIPT_DIR / "naive_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        naive_tokens.append(s["prompt_tokens"])

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(12, 7), dpi=150)
x_axis = range(len(capped_tokens))  # X-axis index used by fill_between

# If the Naive report loaded, draw its line and filled area (pink)
plt.plot(naive_tokens, marker='s', linestyle='-', color='#e377c2', label='Naive')

# Uncapped line and filled area (blue)
plt.plot(uncapped_tokens, marker='s', linestyle='-', color='#1f77b4', label='Uncapped')

# Capped line and filled area (orange)
plt.plot(capped_tokens, marker='s', linestyle='-', color='#ff7f0e', label='Capped')

# ==========================================
# Title and y-axis label (Prefill)
# ==========================================
plt.title('Prefill Token Usage: Naive vs Uncapped vs Capped', fontsize=14)
plt.xlabel('Test Case ID', fontsize=11)
plt.ylabel('Prefill Tokens Consumed', fontsize=11)

# Dashed background grid, kept light so the filled areas stay dominant
plt.grid(True, linestyle=':', color='gray', alpha=0.5)

# Category dividers: the dataset switches question type at items 12 and 17
split_1 = 12
plt.axvline(x=split_1, color='gray', linestyle='--', alpha=0.7)

split_2 = 17
plt.axvline(x=split_2, color='gray', linestyle='--', alpha=0.7)

# Category band labels
# Anchor the label height to the current maximum so it never overlaps a line
max_val = max(uncapped_tokens) if uncapped_tokens else 500  # Prefill counts run high, so the fallback default is raised accordingly
total_len = len(uncapped_tokens)

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

# Commented-out save path renamed to prefill_plot.png
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)
    plt.savefig(d / "prefill_plot.png", dpi=400)
    print("wrote", d / "prefill_plot.png")
plt.show()