"""
§5.6 Correlation analysis.

Pairs the Stage 1 hardware measurements (resource_report_*.json) with the
Stage 2 RAGAS scores (ragas_results_*.csv) item by item, and computes:

  §5.6.1  Token-level latency decomposition -- Pearson r between latency and
          the Prefill / Decode token counts separately.
  §5.6.2  Hardware load x generation quality -- 4 hardware metrics x 4 RAGAS
          metrics, reporting Pearson r alongside Spearman rho, at two levels:
          within each architecture, and pooled across all three.

Usage:  python experiment_result/correlation/correlation_analysis.py
        (runs from any working directory)
Output: three CSV files written beside this script.
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Resolve paths from the script's own location so it runs from any working directory
SCRIPT_DIR = Path(__file__).resolve().parent          # experiment_result/correlation/
RESULT_ROOT = SCRIPT_DIR.parent                       # experiment_result/
# Both inputs live beside one another in main_experiment/: the Stage 1 hardware
# reports and the Stage 2 RAGAS scores. Reading them from inside the repository
# keeps this analysis reproducible from a bare clone.
RAGAS_DIR = RESULT_ROOT / "main_experiment"

ARCHITECTURES = {"naive": "Naive", "uncapped": "Uncapped", "capped": "Capped"}

HARDWARE_METRICS = ["wall_clock_seconds",
                    "avg_cpu_pct", "peak_ram_mb", "total_tokens"]
QUALITY_METRICS = ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]
TOKEN_PHASES = {"prompt_tokens": "Prefill", "completion_tokens": "Decode"}

OUTPUT_DIR = SCRIPT_DIR


def load_paired_data():
    """Load the hardware reports and RAGAS scores for all three architectures,
    pairing them item by item into a single DataFrame.

    The RAGAS CSVs carry no id column, so rows are paired by position; the
    alignment is then verified by comparing user_input verbatim before the id
    is taken from the JSON. This is what makes the one-to-one correspondence
    by question id described in §5.6.2 hold.
    """
    frames = []
    for key, label in ARCHITECTURES.items():
        resource = pd.DataFrame(
            json.load(open(RAGAS_DIR / f"{key}_resource_report.json")))
        ragas = pd.read_csv(
            RAGAS_DIR / f"ragas_results_{key}.csv", encoding="utf-8-sig"
        )

        if len(resource) != len(ragas):
            raise ValueError(
                f"{label}: row count mismatch ({len(resource)} vs {len(ragas)})")

        mismatched = (
            resource["user_input"].str.strip(
            ).values != ragas["user_input"].str.strip().values
        ).sum()
        if mismatched:
            raise ValueError(f"{label}: user_input failed to align on {mismatched} item(s); pairing aborted")

        # RAGAS returns 0.9999999999 where the computation lands on 1: a
        # floating-point artefact, not a score difference. Left raw it gives
        # distinct Spearman ranks to scores that are in fact tied. Six places
        # is well beyond anything an 8B judge resolves, and matches the
        # treatment in experiment_result/main_experiment/stats_tests.py.
        scores = ragas[QUALITY_METRICS].reset_index(drop=True).round(6)

        columns = ["id", "category"] + HARDWARE_METRICS + list(TOKEN_PHASES)
        paired = pd.concat(
            [resource[columns].reset_index(drop=True), scores],
            axis=1,
        )
        paired.insert(0, "architecture", label)
        frames.append(paired)

    return pd.concat(frames, ignore_index=True)


def latency_decomposition(df):
    """§5.6.1: whether latency is driven by Prefill context accumulation or by Decode output length."""
    rows = []
    for label in ARCHITECTURES.values():
        subset = df[df.architecture == label]
        for column, phase in TOKEN_PHASES.items():
            r, p = pearsonr(subset[column], subset["wall_clock_seconds"])
            rows.append({
                "architecture": label,
                "phase": phase,
                "token_field": column,
                "pearson_r": round(r, 4),
                "p_value": round(p, 4),
                "n": len(subset),
            })
    return pd.DataFrame(rows)


def resource_quality_correlation(df):
    """§5.6.2: association between resource expenditure and generation quality,
    reporting Pearson alongside Spearman.

    Spearman serves as a robustness check: RAGAS scores cluster heavily at the
    0 and 1 boundaries and so violate the normality assumption behind Pearson
    (Schober et al., 2018; Mukaka, 2012).
    """
    rows = []
    groups = [(label, df[df.architecture == label])
              for label in ARCHITECTURES.values()]
    groups.append(("Pooled", df))

    for label, subset in groups:
        for hardware in HARDWARE_METRICS:
            for quality in QUALITY_METRICS:
                r, p_r = pearsonr(subset[hardware], subset[quality])
                rho, p_rho = spearmanr(subset[hardware], subset[quality])
                rows.append({
                    "level": label,
                    "hardware_metric": hardware,
                    "quality_metric": quality,
                    "pearson_r": round(r, 4),
                    "pearson_p": round(p_r, 4),
                    "spearman_rho": round(rho, 4),
                    "spearman_p": round(p_rho, 4),
                    "n": len(subset),
                })
    return pd.DataFrame(rows)


def print_summary(decomposition, correlation):
    print("\n§5.6.1  Token-level latency decomposition (Pearson r)")
    print(f'{"Architecture":<14}{"Prefill":>22}{"Decode":>22}')
    for label in ARCHITECTURES.values():
        cells = []
        for phase in TOKEN_PHASES.values():
            row = decomposition[
                (decomposition.architecture == label) & (
                    decomposition.phase == phase)
            ].iloc[0]
            cells.append(f'{row.pearson_r:>13.3f} (p={row.p_value:.3f})')
        print(f'{label:<14}' + "".join(cells))

    print("\n§5.6.2  Latency x generation quality (Pearson r / Spearman rho)")
    print(f'{"Level":<14}' + "".join(f"{q[:15]:>17}" for q in QUALITY_METRICS))
    for level in list(ARCHITECTURES.values()) + ["Pooled"]:
        cells = []
        for quality in QUALITY_METRICS:
            row = correlation[
                (correlation.level == level)
                & (correlation.hardware_metric == "wall_clock_seconds")
                & (correlation.quality_metric == quality)
            ].iloc[0]
            cells.append(f'{row.pearson_r:>8.2f}/{row.spearman_rho:>8.2f}')
        print(f'{level:<14}' + "".join(cells))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_paired_data()
    df.to_csv(OUTPUT_DIR / "paired_per_question.csv", index=False)

    decomposition = latency_decomposition(df)
    decomposition.to_csv(OUTPUT_DIR / "latency_decomposition.csv", index=False)

    correlation = resource_quality_correlation(df)
    correlation.to_csv(
        OUTPUT_DIR / "resource_quality_correlation.csv", index=False)

    print(
        f"Paired {len(df)} rows ({len(ARCHITECTURES)} architectures x {len(df) // len(ARCHITECTURES)} items)")
    print_summary(decomposition, correlation)
    print(f"\nOutput written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
