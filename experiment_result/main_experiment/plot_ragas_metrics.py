from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd

# Resolve paths from the script's own location so it runs from any working directory
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent / "figures"             # experiment_result/figures/
OUT_DIRS = [FIGURE_DIR]

# When this repository sits inside the dissertation working tree, refresh the
# manuscript's copy as well, so the two cannot drift apart. Skipped in a bare
# clone, where the directory does not exist.
PAPER_DIR = SCRIPT_DIR.parents[2] / "paper" / "experimental result"
if PAPER_DIR.is_dir():
    OUT_DIRS.append(PAPER_DIR)

RAGAS_DIR = SCRIPT_DIR                                  # the CSVs sit beside this script

capped_csv = pd.read_csv(RAGAS_DIR / "ragas_results_capped.csv")
naive_csv = pd.read_csv(RAGAS_DIR / "ragas_results_naive.csv")
uncapped_csv = pd.read_csv(RAGAS_DIR / "ragas_results_uncapped.csv")


def extract_index(test_name, m: str):
    target = ""
    if test_name == "capped":
        target = capped_csv
    elif test_name == "uncapped":
        target = uncapped_csv
    elif test_name == "naive":
        target = naive_csv
    return round(mean([each for each in target[f"{m}"]]), 4)


uncapped_result = {'faithfulness': extract_index("uncapped", "faithfulness"), 'answer_relevancy': extract_index("uncapped", "answer_relevancy"),
                   'context_precision': extract_index("uncapped", "context_precision"), 'context_recall': extract_index("uncapped", "context_recall")}

naïve_result = {'faithfulness': extract_index("naive", "faithfulness"), 'answer_relevancy': extract_index("naive", "answer_relevancy"),
                'context_precision': extract_index("naive", "context_precision"), 'context_recall': extract_index("naive", "context_recall")}

capped_result = {'faithfulness': extract_index("capped", "faithfulness"), 'answer_relevancy': extract_index("capped", "answer_relevancy"),
                 'context_precision': extract_index("capped", "context_precision"), 'context_recall': extract_index("capped", "context_recall")}

labels = [
    "Naïve",
    "Uncapped",
    "Capped"
]

conditions = [
    naïve_result,
    uncapped_result,
    capped_result,
]

metrics = ["faithfulness", "answer_relevancy",
           "context_precision", "context_recall"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
axs = axs.flatten()

for ax, metric in zip(axs, metrics):
    # --- Dashed grid ---
    # Horizontal dashed gridlines on the y-axis
    ax.grid(axis='y', linestyle='--', color='lightgray', alpha=0.8)
    # ------------------------

    values = [c[metric] for c in conditions]
    bars = ax.bar(labels, values, color=colors, width=0.7)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_ylabel("Score")
    ax.tick_params(axis='x', rotation=0, labelsize=8)

fig.suptitle("RAGAS Metrics Comparison Across Three Conditions", fontsize=14)
plt.tight_layout()
for _d in OUT_DIRS:
    _d.mkdir(parents=True, exist_ok=True)
    plt.savefig(_d / "ragas_metrics_all_items.png", dpi=400)
    print("wrote", _d / "ragas_metrics_all_items.png")
plt.show()
