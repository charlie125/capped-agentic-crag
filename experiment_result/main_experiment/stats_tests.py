"""Re-runs every significance test reported in Chapter 6 and checks it against the
value printed in the manuscript.

The correlation coefficients had a script already (`correlation_analysis.py`); the
paired Wilcoxon tests of §6.2 and §6.3, the Mann-Whitney comparison of §6.7 and the
Bonferroni thresholds of §6.1 did not, so the reported p-values could not be
reproduced from the repository. This script closes that gap: EXPECTED holds what the
manuscript claims, the run recomputes it from the raw records, and any drift is
reported as a mismatch rather than silently overwritten.

Two conventions matter for reproducing the published numbers:

  CPU time is avg_cpu_pct x wall_clock computed per question and only then averaged.
  avg_cpu_pct is a rate diluted by the sampling window, so the product of the two
  means is a different -- and wrong -- quantity.

  Bonferroni is applied within each family separately: seven metrics on the resource
  axis (.05/7) and eight testable comparisons on the quality axis (.05/8), the eight
  being the twelve cells of Table 6.3 less the four that carry no variance. A cell
  with no variance yields no test statistic, so it is not a test to correct for.

Usage:  python stats_tests.py        (runs from any working directory)
Output: stats_tests_results.json beside this script, plus a console report.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
ARCHITECTURES = ["naive", "uncapped", "capped"]

# Table 6.1's seven metrics. Context is the resource axis, so total_tokens -- which
# §6.2 also tests -- is derived from two of these rather than counted as an eighth.
RESOURCE_METRICS = [
    ("wall_clock_seconds", "End-to-end latency"),
    ("cpu_seconds", "CPU time"),
    ("avg_cpu_pct", "Mean CPU utilisation"),
    ("peak_ram_mb", "Peak RSS"),
    ("prompt_tokens", "Prefill tokens"),
    ("completion_tokens", "Decode tokens"),
    ("llm_calls", "LLM calls"),
]
QUALITY_METRICS = ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]

ALPHA = 0.05
RESOURCE_FAMILY = 7      # the seven metrics of Table 6.1
QUALITY_FAMILY = 8       # 12 cells of Table 6.3 less 4 with no variance

# What the manuscript prints, and where. A run that disagrees with any of these has
# either found an error in the text or changed the analysis.
EXPECTED = {
    "§6.2 其一 latency U-C answerable":        ("p", 0.0056),
    "§6.2 其二 total_tokens U-C answerable":   ("p", 0.0003),
    "§6.2 其五 total_tokens U-C all 25":       ("p", 0.798),
    "§6.2 其五 total_tokens U-C unanswerable": ("p", 0.0078),
    "§6.2 其六 peak_ram U-C unanswerable":     ("p", 0.0547),
    "§6.3 faithfulness N-U":                   ("p", 0.2583),
    "§6.3 faithfulness N-C":                   ("p", 0.7885),
    "§6.3 faithfulness U-C":                   ("p", 0.1814),
    "§6.3 answer_relevancy N-U":               ("p", 0.0046),
    "§6.3 answer_relevancy N-C":               ("p", 0.0395),
    "§6.3 answer_relevancy U-C":               ("p", 0.1556),
    "§6.3 context_recall N-U":                 ("p", 0.3173),
    "§6.3 context_recall N-C":                 ("p", 0.3173),
    "§6.7 一 wall_clock_seconds U":            ("U", 781.0),
    "§6.7 一 wall_clock_seconds rb":           ("rb", 0.306),
    "§6.7 一 cpu_seconds U":                   ("U", 772.0),
    "§6.7 一 cpu_seconds rb":                  ("rb", 0.291),
    "§6.7 一 peak_ram_mb U":                   ("U", 829.0),
    "§6.7 一 peak_ram_mb rb":                  ("rb", 0.386),
    "§6.7 一 total_tokens U":                  ("U", 829.0),
    "§6.7 一 total_tokens rb":                 ("rb", 0.386),
    "§6.7 三 pooled latency-AnsRel pearson":   ("r", -0.414),
    "§6.7 三 pooled latency-AnsRel spearman":  ("r", -0.097),
    "§6.7 三 capped total_tokens-CtxPrec":     ("r", -0.929),
    "§6.7 三 naive latency-AnsRel":            ("r", 0.440),
    "§6.7 三 uncapped latency-AnsRel":         ("r", -0.803),
    "§6.7 三 capped latency-AnsRel":           ("r", -0.744),
    "§5.6.1 naive latency-prefill":            ("r", 0.165),
    "§5.6.1 naive latency-decode":             ("r", 0.938),
    "§5.6.1 uncapped latency-prefill":         ("r", 0.990),
    "§5.6.1 uncapped latency-decode":          ("r", 0.875),
    "§5.6.1 capped latency-prefill":           ("r", 0.911),
    "§5.6.1 capped latency-decode":            ("r", 0.415),
    "collinearity naive prompt-completion":    ("r", 0.111),
    "collinearity uncapped prompt-completion": ("r", 0.888),
    "collinearity capped prompt-completion":   ("r", 0.305),
}


def load():
    """One row per question per architecture, hardware joined to RAGAS scores by id.

    The RAGAS CSVs carry no id column; their rows follow the order of the resource
    report, which the assertion below checks question by question rather than
    trusting.
    """
    rows = []
    for arch in ARCHITECTURES:
        res = json.load(open(SCRIPT_DIR / f"{arch}_resource_report.json"))
        rag = list(csv.DictReader(
            open(SCRIPT_DIR / f"ragas_results_{arch}.csv")))
        assert len(res) == len(
            rag), f"{arch}: {len(res)} records against {len(rag)} scores"
        for r, q in zip(res, rag):
            assert r["user_input"].strip() == q["user_input"].strip(), \
                f"{arch} id {r['id']}: resource record and RAGAS row are not the same question"
            row = {
                "arch": arch,
                "id": r["id"],
                # §5.2's three answerable categories against the unanswerable one
                "answerable": r["category"] != "unanswerable",
                # per question, then averaged -- never the product of two means
                "cpu_seconds": r["avg_cpu_pct"] / 100 * r["wall_clock_seconds"],
            }
            for k in ("wall_clock_seconds", "avg_cpu_pct", "peak_ram_mb",
                      "prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"):
                row[k] = float(r[k])
            for m in QUALITY_METRICS:
                # RAGAS returns 0.9999999999 where the computation lands on 1: a
                # floating-point artefact, not a score difference. Left raw it makes
                # the zero-variance columns of Table 6.3 look testable, on differences
                # of 1e-10. Six places is well beyond anything an 8B judge resolves.
                row[m] = round(float(q[m]), 6)
            rows.append(row)
    return rows


def paired(rows, arch_a, arch_b, metric, subset=None):
    """Wilcoxon signed-rank over the questions the two architectures share."""
    def series(arch):
        sel = [r for r in rows if r["arch"] == arch
               and (subset is None or r["answerable"] == subset)]
        return [r[metric] for r in sorted(sel, key=lambda r: r["id"])]

    a, b = series(arch_a), series(arch_b)
    diff = np.array(a) - np.array(b)
    if not np.any(diff):
        return {"n": len(a), "p": None, "note": "no variance -- not testable"}
    stat, p = wilcoxon(a, b)
    return {"n": len(a), "statistic": float(stat), "p": float(p),
            "n_a_greater": int(np.sum(diff > 0)), "n_b_greater": int(np.sum(diff < 0))}


def check(results):
    """Compare every recomputed value against what the manuscript prints."""
    report = []
    for label, (kind, claimed) in EXPECTED.items():
        got = results.get(label)
        if got is None:
            report.append((label, kind, claimed, None, "NOT COMPUTED"))
            continue
        # p-values and coefficients are printed rounded, so compare at the printed precision
        places = len(str(claimed).split(".")[-1])
        ok = round(got, places) == round(claimed, places)
        report.append((label, kind, claimed, got,
                      "match" if ok else "MISMATCH"))
    return report


def main():
    rows = load()
    R = {}

    # --- §6.2 resource axis: Uncapped against Capped -------------------------
    resource = {}
    for metric, label in RESOURCE_METRICS + [("total_tokens", "Total tokens")]:
        for subset, name in ((True, "answerable"), (False, "unanswerable"), (None, "all 25")):
            resource[f"{metric} [{name}]"] = paired(
                rows, "uncapped", "capped", metric, subset)
    R["§6.2 其一 latency U-C answerable"] = resource["wall_clock_seconds [answerable]"]["p"]
    R["§6.2 其二 total_tokens U-C answerable"] = resource["total_tokens [answerable]"]["p"]
    R["§6.2 其五 total_tokens U-C all 25"] = resource["total_tokens [all 25]"]["p"]
    R["§6.2 其五 total_tokens U-C unanswerable"] = resource["total_tokens [unanswerable]"]["p"]
    R["§6.2 其六 peak_ram U-C unanswerable"] = resource["peak_ram_mb [unanswerable]"]["p"]

    # --- §6.3 quality axis: three pairs on the answerable subset --------------
    quality = {}
    for metric in QUALITY_METRICS:
        for a, b, tag in (("naive", "uncapped", "N-U"), ("naive", "capped", "N-C"),
                          ("uncapped", "capped", "U-C")):
            out = paired(rows, a, b, metric, subset=True)
            quality[f"{metric} {tag}"] = out
            R[f"§6.3 {metric} {tag}"] = out["p"]
    testable = sum(1 for v in quality.values() if v["p"] is not None)

    # --- §6.7 一 Context Precision is binary: Mann-Whitney plus rank-biserial --
    mw = {}
    zero = [r for r in rows if r["context_precision"] == 0]
    one = [r for r in rows if r["context_precision"] == 1]
    for metric in ("wall_clock_seconds", "cpu_seconds", "peak_ram_mb", "total_tokens"):
        a = [r[metric] for r in zero]
        b = [r[metric] for r in one]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        # +1 means the 0 group ranks higher throughout
        rb = 2 * u / (len(a) * len(b)) - 1
        mw[metric] = {"n_zero": len(a), "n_one": len(b),
                      "median_zero": float(np.median(a)), "median_one": float(np.median(b)),
                      "U": float(u), "p": float(p), "rank_biserial": float(rb)}
        R[f"§6.7 一 {metric} U"] = float(u)
        R[f"§6.7 一 {metric} rb"] = float(rb)

    # --- §6.7 二 stratified Spearman, and 三 the pooled coefficients ----------
    strat = {}
    for stratum, sel in (("answerable", True), ("unanswerable", False)):
        for q in ("faithfulness", "answer_relevancy", "context_recall"):
            for h in ("wall_clock_seconds", "cpu_seconds", "peak_ram_mb", "total_tokens"):
                sub = [r for r in rows if r["answerable"] == sel]
                rho, p = spearmanr([r[h] for r in sub], [r[q] for r in sub])
                strat[f"{stratum} {h} x {q}"] = {
                    "rho": float(rho), "p": float(p), "n": len(sub)}

    pooled = {}
    lat = [r["wall_clock_seconds"] for r in rows]
    ans = [r["answer_relevancy"] for r in rows]
    r_p, p_p = pearsonr(lat, ans)
    rho_p, p_s = spearmanr(lat, ans)
    pooled["latency x AnsRel (n=75) pearson"] = {
        "r": float(r_p), "p": float(p_p)}
    pooled["latency x AnsRel (n=75) spearman"] = {
        "rho": float(rho_p), "p": float(p_s)}
    R["§6.7 三 pooled latency-AnsRel pearson"] = float(r_p)
    R["§6.7 三 pooled latency-AnsRel spearman"] = float(rho_p)

    cap = [r for r in rows if r["arch"] == "capped"]
    r_c, _ = pearsonr([r["total_tokens"] for r in cap], [
                      r["context_precision"] for r in cap])
    pooled["capped total_tokens x CtxPrec (n=25) pearson"] = {"r": float(r_c)}
    R["§6.7 三 capped total_tokens-CtxPrec"] = float(r_c)

    # --- §5.6.1 latency decomposition, and the collinearity check ------------
    decomp = {}
    for arch in ARCHITECTURES:
        sub = [r for r in rows if r["arch"] == arch]
        for tok, tag in (("prompt_tokens", "prefill"), ("completion_tokens", "decode")):
            r_v, p_v = pearsonr([r[tok] for r in sub], [
                                r["wall_clock_seconds"] for r in sub])
            decomp[f"{arch} latency x {tag}"] = {
                "r": float(r_v), "p": float(p_v)}
            R[f"§5.6.1 {arch} latency-{tag}"] = float(r_v)
        r_v, p_v = pearsonr([r["prompt_tokens"] for r in sub], [
                            r["completion_tokens"] for r in sub])
        decomp[f"{arch} prompt x completion"] = {
            "r": float(r_v), "p": float(p_v)}
        R[f"collinearity {arch} prompt-completion"] = float(r_v)

        rho, _ = spearmanr([r["wall_clock_seconds"]
                           for r in sub], [r["answer_relevancy"] for r in sub])
        r_v, _ = pearsonr([r["wall_clock_seconds"]
                          for r in sub], [r["answer_relevancy"] for r in sub])
        pooled[f"{arch} latency x AnsRel (n=25)"] = {
            "r": float(r_v), "spearman": float(rho)}
        R[f"§6.7 三 {arch} latency-AnsRel"] = float(r_v)

    # --- report --------------------------------------------------------------
    W = 46
    print("=" * 78)
    print("BONFERRONI THRESHOLDS (§6.1 其三) -- applied within each family separately")
    print(
        f"  resource axis: .05 / {RESOURCE_FAMILY} metrics            = {ALPHA/RESOURCE_FAMILY:.5f}")
    print(
        f"  quality axis : .05 / {QUALITY_FAMILY} testable comparisons  = {ALPHA/QUALITY_FAMILY:.5f}")
    print(f"  testable comparisons found on the quality axis: {testable}"
          f"  {'(matches the family size used)' if testable == QUALITY_FAMILY else '<-- DISAGREES'}")
    print(f"  exact-test floor at n=8: 2/2**8            = {2/2**8:.5f}"
          f"  ({'above' if 2/2**8 > ALPHA/RESOURCE_FAMILY else 'below'} the resource threshold)")

    print("\n" + "=" * 78)
    print("§6.2 RESOURCE AXIS -- paired Wilcoxon, Uncapped against Capped")
    thr = ALPHA / RESOURCE_FAMILY
    print(f"  {'metric [subset]':<{W}} {'n':>3} {'p':>9}  verdict")
    for k, v in resource.items():
        p = v["p"]
        mark = "-" if p is None else ("SIGNIFICANT" if p < thr else "n.s.")
        print(
            f"  {k:<{W}} {v['n']:>3} {'--' if p is None else f'{p:>9.4f}'}  {mark}")

    print("\n" + "=" * 78)
    print("§6.3 QUALITY AXIS -- paired Wilcoxon, answerable subset (n=17)")
    thr = ALPHA / QUALITY_FAMILY
    print(f"  {'comparison':<{W}} {'n':>3} {'p':>9}  verdict")
    for k, v in quality.items():
        p = v["p"]
        mark = "not testable" if p is None else (
            "SIGNIFICANT" if p < thr else "n.s.")
        print(
            f"  {k:<{W}} {v['n']:>3} {'--' if p is None else f'{p:>9.4f}'}  {mark}")

    print("\n" + "=" * 78)
    print("§6.7 一 CONTEXT PRECISION IS BINARY -- Mann-Whitney U with rank-biserial")
    print(
        f"  {'resource metric':<{W}} {'med=0':>9} {'med=1':>9} {'U':>7} {'p':>7} {'rb':>7}")
    for k, v in mw.items():
        print(f"  {k:<{W}} {v['median_zero']:>9.1f} {v['median_one']:>9.1f} "
              f"{v['U']:>7.0f} {v['p']:>7.3f} {v['rank_biserial']:>+7.3f}")

    print("\n" + "=" * 78)
    print("§6.7 二 STRATIFIED SPEARMAN -- 24 coefficients, trend description only")
    for stratum in ("answerable", "unanswerable"):
        vals = [v["rho"] for k, v in strat.items() if k.startswith(stratum)]
        sig = [k for k, v in strat.items() if k.startswith(stratum)
               and v["p"] < ALPHA]
        print(f"  {stratum:<14} n={strat[[k for k in strat if k.startswith(stratum)][0]]['n']:>3}"
              f"  range {min(vals):+.3f} to {max(vals):+.3f}"
              f"  |  reaching p<.05: {len(sig)}")

    print("\n" + "=" * 78)
    print("VERIFICATION AGAINST THE MANUSCRIPT")
    report = check(R)
    bad = [r for r in report if r[4] != "match"]
    for label, kind, claimed, got, verdict in report:
        if verdict != "match":
            print(f"  {verdict:<12} {label:<{W}} paper {claimed}  recomputed "
                  f"{'--' if got is None else round(got, 4)}")
    print(f"\n  {len(report) - len(bad)} of {len(report)} reported values reproduce exactly."
          + ("" if bad else "  No discrepancies."))

    out = SCRIPT_DIR / "stats_tests_results.json"
    json.dump({"thresholds": {"resource": ALPHA / RESOURCE_FAMILY,
                              "quality": ALPHA / QUALITY_FAMILY,
                              "exact_floor_n8": 2 / 2 ** 8,
                              "testable_quality_comparisons": testable},
               "resource_axis": resource, "quality_axis": quality,
               "context_precision_mannwhitney": mw, "stratified_spearman": strat,
               "pooled_and_per_architecture": pooled, "latency_decomposition": decomp,
               "verification": [{"claim": a, "kind": b, "paper": c,
                                 "recomputed": d, "verdict": e} for a, b, c, d, e in report]},
              open(out, "w"), indent=2)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
