"""Resource cost of the three architectures, split by question type.

Answerable items (single-hop + multi-hop, n=17) and unanswerable items (n=8) have
opposite target behaviour, so the 25-item mean hides the effect this figure exists
to show: the three architectures behave almost identically while retrieval succeeds,
and separate by roughly an order of magnitude once it cannot.

CPU time is computed per item (avg_cpu_pct x wall_clock) and then averaged, not as
the product of the two means -- avg_cpu_pct is a rate diluted by the sampling window,
so the two differ. Mean CPU utilisation is deliberately absent: on the unanswerable
subset it ranks the uncapped baseline as the most economical, which reverses once
expenditure is measured as CPU time.

Single run, n=25, Apple M2 Pro. No error bars are available; the overlaid dots show
the per-item spread instead.
"""
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent / "figures"             # experiment_result/figures/
OUT_DIRS = [FIGURE_DIR]

# When this repository sits inside the dissertation working tree, refresh the
# manuscript's copy as well, so the two cannot drift apart. Skipped in a bare
# clone, where the directory does not exist.
PAPER_DIR = SCRIPT_DIR.parents[2] / "paper" / "experimental result"
if PAPER_DIR.is_dir():
    OUT_DIRS.append(PAPER_DIR)

# Architecture colours match the other figures in this repository: identity is
# carried by the same hue throughout the dissertation.
ARCHITECTURES = [
    ("naive", "Naive RAG", "#e377c2"),
    ("uncapped", "Uncapped Agentic CRAG", "#1f77b4"),
    ("capped", "Capped Agentic CRAG", "#ff7f0e"),
]

# Each panel: (title, y-axis label, per-item accessor, value format)
PANELS = [
    ("End-to-end latency", "seconds",
     lambda r: r["wall_clock_seconds"], "{:.2f}"),
    ("CPU time", "seconds",
     lambda r: r["avg_cpu_pct"] / 100 * r["wall_clock_seconds"], "{:.2f}"),
    ("Prefill tokens", "tokens per question",
     lambda r: r["prompt_tokens"], "{:,.0f}"),
    ("Decode tokens", "tokens per question",
     lambda r: r["completion_tokens"], "{:,.0f}"),
]

INK = "#3f3f3f"
MUTED = "#6b6b6b"


def load(name):
    """Split one architecture's report into answerable and unanswerable items."""
    rows = json.load(open(SCRIPT_DIR / f"{name}_resource_report.json"))
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    return answerable, unanswerable


data = {key: load(key) for key, _, _ in ARCHITECTURES}
n_answerable = len(data["naive"][0])
n_unanswerable = len(data["naive"][1])

GROUPS = [f"Answerable\n(n = {n_answerable})", f"Unanswerable\n(n = {n_unanswerable})"]
group_x = np.arange(len(GROUPS))
bar_w = 0.24
offsets = np.array([-bar_w, 0.0, bar_w])

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
rng = np.random.default_rng(0)          # fixed seed: the jitter is reproducible

for ax, (title, ylabel, value_of, fmt) in zip(axes.flat, PANELS):
    panel_max = 0.0

    for (key, label, colour), dx in zip(ARCHITECTURES, offsets):
        means, xs = [], []
        for gi, subset in enumerate(data[key]):
            values = [value_of(r) for r in subset]
            x = group_x[gi] + dx
            means.append(mean(values))
            xs.append(x)
            panel_max = max(panel_max, max(values))

            # Per-item points, jittered inside the bar. The main experiment is a
            # single run, so this spread is the only dispersion the data carries.
            jitter = rng.uniform(-bar_w * 0.28, bar_w * 0.28, size=len(values))
            ax.scatter(np.full(len(values), x) + jitter, values,
                       s=16, color=INK, alpha=0.45, linewidths=0.5,
                       edgecolors="white", zorder=3)

        bars = ax.bar(xs, means, bar_w * 0.92, label=label,
                      color=colour, zorder=2)

        # Value labels are not decoration here: two of the three hues fall below
        # 3:1 contrast on a white surface, and the worst tritan pair sits in the
        # 6-8 separation band, so the figure needs a non-colour channel.
        for rect, value in zip(bars, means):
            ax.annotate(fmt.format(value),
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=8.5, color=INK)

    ax.set_title(title, fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9.5, color=MUTED)
    ax.set_xticks(group_x)
    ax.set_xticklabels(GROUPS, fontsize=10, color=INK)
    ax.set_ylim(0, panel_max * 1.16)
    ax.grid(axis="y", linestyle="--", color="lightgray", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("lightgray")
    ax.tick_params(colors=MUTED, length=0)

fig.tight_layout(rect=[0, 0.030, 1, 0.905])

fig.suptitle("Resource cost by question type", fontsize=15, color=INK, y=0.985)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center", ncol=3, frameon=False,
           fontsize=10.5, bbox_to_anchor=(0.5, 0.935))

fig.text(0.5, 0.012,
         "Bars are means; dots are individual questions. Single run, n = 25, Apple M2 Pro. "
         "CPU time is computed per question as avg_cpu_pct x wall_clock, then averaged.",
         ha="center", fontsize=8.5, color=MUTED)

for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "question_type_comparison.png", dpi=400, bbox_inches="tight")
    print("wrote", d / "question_type_comparison.png")
