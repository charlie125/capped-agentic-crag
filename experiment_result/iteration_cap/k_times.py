from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR.parent / "plot"
import numpy as np

# RAGAS metrics data from log
data = {
    'Iteration Cap (K)': [1, 2, 3, 4, 5, 6],
    'Faithfulness': [0.5000, 0.5556, 0.3889, 0.3889, 0.5139, 0.7639],
    'Answer Relevancy': [0.4434, 0.4828, 0.4384, 0.4461, 0.6235, 0.7092],
    'Context Precision': [0.5833, 0.5833, 0.5556, 0.5278, 0.6667, 0.7222],
    'Context Recall': [0.9167, 0.9167, 0.7778, 0.9167, 0.9167, 0.8981]
}

df = pd.DataFrame(data)

# Academic styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Slightly wider figure
fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(df['Iteration Cap (K)']))
width = 0.18

rects1 = ax.bar(x - 1.5*width, df['Faithfulness'], width, label='Faithfulness', color='#1f77b4')
rects2 = ax.bar(x - 0.5*width, df['Answer Relevancy'], width, label='Answer Relevancy', color='#ff7f0e')
rects3 = ax.bar(x + 0.5*width, df['Context Precision'], width, label='Context Precision', color='#2ca02c')
rects4 = ax.bar(x + 1.5*width, df['Context Recall'], width, label='Context Recall', color='#d62728')

# --- Value labels ---
# fmt '%.2f' keeps two decimal places
label_kwargs = {'padding': 3, 'fmt': '%.2f', 'fontsize': 8, 'rotation': 0}
ax.bar_label(rects1, **label_kwargs)
ax.bar_label(rects2, **label_kwargs)
ax.bar_label(rects3, **label_kwargs)
ax.bar_label(rects4, **label_kwargs)
# ------------------------

# No bold weight
ax.set_xlabel('Iteration Cap (K)', fontsize=12)
ax.set_ylabel('RAGAS Score', fontsize=12)
ax.set_title('RAGAS Quality Metrics Across Iteration Caps (K)', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in df['Iteration Cap (K)']])

# Higher y-limit leaves headroom for the labels
ax.set_ylim(0, 1.15)

ax.legend(loc='lower right', frameon=True, edgecolor='none')

# Add subtle highlight background for K=2
ax.axvspan(0.5, 1.5, color='gray', alpha=0.15, label='Optimal Cap (K=2)')

plt.tight_layout()
PLOT_DIR.mkdir(exist_ok=True)
plt.savefig(PLOT_DIR / 'ragas_metrics_bar_k_times.png', dpi=300)
plt.show()