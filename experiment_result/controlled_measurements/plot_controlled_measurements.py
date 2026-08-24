"""Resource comparison figures for the two controlled measurements.

Produces two four-panel figures, each written to both experiment_result/figures/ and
paper/experimental result/:

  prompt_refinement_resources.png   prompt refinement (§4.2.2 / Figure 4.5)
  iteration_depth_resources.png                iteration depth (§5.5.1)

The metrics reported are Wall Clock and CPU time (avg_cpu_pct x wall_clock)
rather than Peak CPU: the latter differs by roughly one percentage point
between the two conditions and so carries no information (see §3.4 on how the
CPU measure is defined here).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent / "figures"             # experiment_result/figures/
OUT_DIRS = [FIGURE_DIR]

# When this repository sits inside the dissertation working tree, refresh the
# manuscript's copy as well, so the two cannot drift apart. Skipped in a bare
# clone, where the directory does not exist.
PAPER_DIR = SCRIPT_DIR.parents[2] / "paper" / "experimental result"
if PAPER_DIR.is_dir():
    OUT_DIRS.append(PAPER_DIR)


def load_and_average(filename):
    """Load one condition's JSON, compute CPU time per run, then average."""
    with open(SCRIPT_DIR / filename, "r") as f:
        data = json.load(f)
    out = {k: sum(v) / len(v) for k, v in data.items() if isinstance(v, list)}
    cpu_time = [c / 100 * w for c, w in
                zip(data["avg_cpu_pct"], data["wall_clock_seconds"])]
    out["cpu_time_s"] = sum(cpu_time) / len(cpu_time)
    return out


# Large-effect metrics on the top row, memory on the bottom
METRICS = [
    ("wall_clock_seconds", "Wall Clock (s)", "Wall Clock Time", "%.2f"),
    ("cpu_time_s", "CPU Time (s)", "CPU Time", "%.2f"),
    ("avg_ram_mb", "Avg RAM (MB)", "Avg RAM Usage", "%.1f"),
    ("peak_ram_mb", "Peak RAM (MB)", "Peak RAM Usage", "%.1f"),
]

FIGURES = [
    ("orig_2.json", "refined_2.json", ["Original", "Refined"],
     "Original vs Refined Prompt - 2 iterations",
     "prompt_refinement_resources.png"),
    ("2_iters.json", "6_iters.json", ["K = 2", "K = 6"],
     "Iteration Cap: K = 2 vs K = 6",
     "iteration_depth_resources.png"),
]

colors = ["#4C72B0", "#DD8452"]
x_positions = [0.32, 0.68]
bar_width = 0.28

for file_a, file_b, labels, suptitle, outname in FIGURES:
    data_a = load_and_average(file_a)
    data_b = load_and_average(file_b)

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.5), layout="constrained")
    axs = axs.flatten()

    for ax, (key, ylabel, title, fmt) in zip(axs, METRICS):
        vals = [data_a[key], data_b[key]]
        bars = ax.bar(x_positions, vals, color=colors, width=bar_width)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlim(0, 1)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

        ax.set_ylim(0, max(vals) * 1.15)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=9.5)

        # The secondary label carries the relative change so small differences stay legible
        delta = (vals[1] - vals[0]) / vals[0] * 100
        ax.set_xlabel("%+.1f%%" % delta, fontsize=9.5, labelpad=2)

        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        plt.savefig(d / outname, dpi=400, bbox_inches="tight")
        print("wrote", d / outname)
    plt.close(fig)
