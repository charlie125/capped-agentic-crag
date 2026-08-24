"""Build a grouped summary JSON from the three ragas_results_*.csv files and
print it in readable form.

Groups: ALL 0-24, ANS 0-16 (answerable), UNANS 17-24 (unanswerable).
"""
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_JSON = SCRIPT_DIR / "ragas_metrics_summary.json"

SYSTEMS = ["naive", "uncapped", "capped"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
GROUPS = {"ALL": (0, 25), "ANS": (0, 17), "UNANS": (17, 25)}


def load(system):
    with open(SCRIPT_DIR / f"ragas_results_{system}.csv", newline="", encoding="utf-8") as f:
        return [{m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)]


def group_mean(rows, lo, hi):
    sub = rows[lo:hi]
    return {m: round(sum(r[m] for r in sub) / len(sub), 4) for m in METRICS}


summary = {"metrics": METRICS,
           "groups": {g: {"start": lo, "end": hi - 1, "n": hi - lo} for g, (lo, hi) in GROUPS.items()},
           "systems": {}}

for system in SYSTEMS:
    rows = load(system)
    summary["systems"][system] = {
        "n": len(rows),
        "group_means": {g: group_mean(rows, lo, hi) for g, (lo, hi) in GROUPS.items()},
        "per_question": {f"id{i}": {m: round(r[m], 4) for m in METRICS}
                         for i, r in enumerate(rows)},
    }

OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

for system in SYSTEMS:
    s = summary["systems"][system]
    print(f"===== {system} {s['n']}")
    for g, (lo, hi) in GROUPS.items():
        tag = f"{g} {lo}-{hi - 1}" if g != "ALL" else f"{g} {hi - lo}"
        vals = " ".join(f"{m}={s['group_means'][g][m]:.4f}" for m in METRICS)
        print(f"  {tag:<12} {vals}")
    for i in range(*GROUPS["UNANS"]):
        print(f"   id{i} {[str(v) for v in s['per_question'][f'id{i}'].values()]}")
print(f"\nwritten: {OUT_JSON}")
