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

# 1. Load llm_calls for all three architectures
capped_calls = []
with open(SCRIPT_DIR / "capped_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        capped_calls.append(s["llm_calls"])

uncapped_calls = []
with open(SCRIPT_DIR / "uncapped_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        uncapped_calls.append(s["llm_calls"])

naive_calls = []
with open(SCRIPT_DIR / "naive_resource_report.json", "r") as f:
    data = json.load(f)
    for s in data:
        naive_calls.append(s["llm_calls"])

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(12, 7), dpi=150)

plt.plot(naive_calls, marker='o', linestyle='-', color='#e377c2', label='Naive')
plt.plot(uncapped_calls, marker='o', linestyle='-', color='#1f77b4', label='Uncapped')
plt.plot(capped_calls, marker='o', linestyle='-', color='#ff7f0e', label='Capped')

plt.title('LLM Calls: Naive vs Uncapped vs Capped', fontsize=14)
plt.xlabel('Test Case ID', fontsize=11)
plt.ylabel('LLM Calls', fontsize=11)

plt.grid(True, linestyle=':', color='gray', alpha=0.5)

# Category dividers: the dataset switches question type at items 12 and 17
split_1 = 12
plt.axvline(x=split_1, color='gray', linestyle='--', alpha=0.7)

split_2 = 17
plt.axvline(x=split_2, color='gray', linestyle='--', alpha=0.7)

total_len = len(uncapped_calls)
plt.ylim(0, 27)

# Category band labels
label_y = 25.2
plt.text(split_1 / 2, label_y, 'Single-hop',
         color='gray', fontsize=12, ha='center')
plt.text(split_1 + (split_2 - split_1) / 2, label_y, 'Multi-hop',
         color='gray', fontsize=12, ha='center')
plt.text(split_2 + (total_len - split_2) / 2, label_y, 'Unanswerable',
         color='gray', fontsize=12, ha='center')

# Two cap reference lines: the harness K=6 safety valve, and this study's K=2
for y, text in [(22, 'harness ceiling, K=6  (4 + 3×6)'),
                (10, 'K=2 ceiling  (4 + 3×2)')]:
    plt.axhline(y=y, color='gray', linestyle='-.', alpha=0.6)
    plt.text(0.2, y + 0.4, text, color='gray', fontsize=10, ha='left')

# On answerable items Uncapped and Capped coincide exactly at 4; annotated so the
# overlap is not misread as a missing series
plt.annotate('Uncapped and Capped coincide at 4 —\nthe rewrite loop never starts',
             xy=(8, 4), xytext=(8, 6.6), color='black', fontsize=10.5, ha='center',
             arrowprops=dict(arrowstyle='-', color='gray', linewidth=1))

# ids 17 and 20 reach the safety valve: terminated by the harness rather than converged
plt.annotate('ids 17 and 20 hit the ceiling — halted, not converged',
             xy=(17, 22.2), xytext=(13.5, 24.0), color='black', fontsize=10.5, ha='center',
             arrowprops=dict(arrowstyle='-', color='gray', linewidth=1))
plt.annotate('', xy=(20, 22.2), xytext=(16.2, 23.8),
             arrowprops=dict(arrowstyle='-', color='gray', linewidth=1))

plt.legend(loc='center left', fontsize=11)

plt.tight_layout()

for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)
    plt.savefig(d / "llm_calls_plot.png", dpi=400)
    print("wrote", d / "llm_calls_plot.png")
plt.show()
