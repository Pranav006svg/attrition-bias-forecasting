"""Descriptive 'hidden differential' analysis (primary set, all splits).

Per arm, reason-group rates (counts / started) over the primary-attrition
categories: AE, EFFICACY (lack of efficacy + disease progression), SUBJECT
(withdrawal by subject + lost to follow-up), OTHER_GROUP (rest). Groups sum to
the primary attrition rate.

Pairs: each experimental arm vs the trial's pooled control (same control rule as
the benchmark: highest-priority control type, pooled; experimental = EXPERIMENTAL/
OTHER arms, else non-control arms). Differential = experimental - control.

Trial level: total differential = the pair with max |total diff|; a trial has a
"hidden differential" when every pair has |total diff| < 0.05 but some pair has
|AE diff| >= 0.05 or |EFFICACY diff| >= 0.05. Subsets: all, placebo-controlled,
reliable pairs (experimental arm and pooled control both started >= 100; at trial
level only reliable pairs are used). Correlations of AE vs EFFICACY diff are
computed over pairs.

Outputs: results/tables/hidden_differential.csv, results/hidden_differential_summary.md

Usage:
    python -m src.eval.hidden_differential
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.data.build_benchmark import REASON_GROUPS, RELIABLE_MIN_STARTED, cat_col

SEED = 42
N_BOOT = 1000
THR = 0.05
GROUPS = list(REASON_GROUPS)
PROCESSED = Path("data/processed")


def pair_table(arms: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    a = arms.copy()
    for g, cats in REASON_GROUPS.items():
        a[f"n_{g}"] = a[[cat_col(c) for c in cats]].sum(axis=1)
    ncols = [f"n_{g}" for g in GROUPS]
    ctrl = a[a.meta_role == "control"].groupby("nct_id")[ncols + ["y_started"]].sum()
    exp = a[a.meta_role == "experimental"].join(ctrl, on="nct_id", rsuffix="_c")
    for g in GROUPS:
        exp[f"diff_{g}"] = exp[f"n_{g}"] / exp.y_started - exp[f"n_{g}_c"] / exp.y_started_c
    exp["diff_total"] = exp[[f"diff_{g}" for g in GROUPS]].sum(axis=1)
    exp["reliable"] = (exp.y_started >= RELIABLE_MIN_STARTED) & (exp.y_started_c >= RELIABLE_MIN_STARTED)
    exp = exp.merge(trials[["nct_id", "meta_control_type"]], on="nct_id")
    return exp[["nct_id", "arm_idx", "split", "meta_control_type", "reliable", "diff_total"]
               + [f"diff_{g}" for g in GROUPS]]


def trial_table(pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.assign(abs_total=pairs.diff_total.abs(), abs_ae=pairs.diff_ae.abs(),
                     abs_eff=pairs.diff_efficacy.abs())
    g = p.groupby("nct_id")
    t = pd.DataFrame({
        "split": g.split.first(), "control_type": g.meta_control_type.first(),
        "max_abs_total": g.abs_total.max(), "max_abs_ae": g.abs_ae.max(), "max_abs_efficacy": g.abs_eff.max()})
    t["total_small"] = t.max_abs_total < THR
    t["ae_large"] = t.max_abs_ae >= THR
    t["efficacy_large"] = t.max_abs_efficacy >= THR
    t["hidden"] = t.total_small & (t.ae_large | t.efficacy_large)
    return t


def boot_share(x: np.ndarray, rng) -> tuple:
    x = np.asarray(x, dtype=float)
    stats = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def main():
    arms = pd.read_parquet(PROCESSED / "arms.parquet")
    trials = pd.read_parquet(PROCESSED / "trials.parquet")
    trials = trials[trials.y_has_differential]
    pairs = pair_table(arms[arms.nct_id.isin(trials.nct_id)], trials)

    rng = np.random.default_rng(SEED)
    rows = []

    def add(subset, split, metric, value, lo=np.nan, hi=np.nan, n=None):
        rows.append({"subset": subset, "split": split, "metric": metric, "value": value,
                     "ci_low": lo, "ci_high": hi, "n": n})

    subsets = {
        "all": pairs,
        "placebo_controlled": pairs[pairs.meta_control_type == "PLACEBO_COMPARATOR"],
        "reliable_pairs": pairs[pairs.reliable],
        "placebo_controlled_reliable": pairs[(pairs.meta_control_type == "PLACEBO_COMPARATOR") & pairs.reliable],
    }
    for name, pp in subsets.items():
        for split in ["all", "train", "val", "test"]:
            ps = pp if split == "all" else pp[pp.split == split]
            if ps.empty:
                continue
            t = trial_table(ps)
            n = len(t)
            add(name, split, "n_trials", n, n=n)
            add(name, split, "n_pairs", len(ps), n=len(ps))
            for col, lab in [("hidden", "share_hidden_differential"), ("total_small", "share_total_diff_lt5"),
                             ("ae_large", "share_ae_diff_ge5"), ("efficacy_large", "share_efficacy_diff_ge5")]:
                add(name, split, lab, *boot_share(t[col].values, rng), n=n)
            small = t[t.total_small]
            if len(small):
                add(name, split, "share_hidden_among_total_diff_lt5", *boot_share(small.hidden.values, rng), n=len(small))
            if split == "all" and len(ps) > 2:
                r, pr = pearsonr(ps.diff_ae, ps.diff_efficacy)
                rho, prho = spearmanr(ps.diff_ae, ps.diff_efficacy)
                add(name, split, "pearson_ae_vs_efficacy_diff", float(r), n=len(ps))
                add(name, split, "pearson_p", float(pr), n=len(ps))
                add(name, split, "spearman_ae_vs_efficacy_diff", float(rho), n=len(ps))
                add(name, split, "spearman_p", float(prho), n=len(ps))
                # Pairs where both groups move >= 5 pp in opposite directions (masking each other).
                opp = (np.sign(ps.diff_ae) != np.sign(ps.diff_efficacy)) & (ps.diff_ae.abs() >= THR) & (ps.diff_efficacy.abs() >= THR)
                add(name, split, "share_pairs_ae_efficacy_opposite_ge5", *boot_share(opp.values, rng), n=len(ps))

    res = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    res.to_csv("results/tables/hidden_differential.csv", index=False)
    write_summary(res)


def write_summary(res):
    def v(subset, metric, split="all", ci=True, pct=True):
        r = res[(res.subset == subset) & (res.split == split) & (res.metric == metric)]
        if r.empty:
            return "n/a"
        r = r.iloc[0]
        if not pct:
            return f"{r.value:.1e}" if metric.endswith("_p") else f"{r.value:.3f}"
        s = f"{100 * r.value:.1f}%"
        return s + (f" [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}]" if ci and not np.isnan(r.ci_low) else "")

    subs = ["all", "placebo_controlled", "reliable_pairs", "placebo_controlled_reliable"]
    L = ["# Hidden differential attrition (descriptive, primary set m80, all splits)", "",
         "Hidden differential = every experimental-vs-control pair has |total primary-attrition diff| < 5 pp, "
         "but some pair has |AE diff| >= 5 pp or |EFFICACY diff| >= 5 pp. "
         "Reliable pairs: experimental arm and pooled control both started >= 100. "
         "95% CIs: 1,000 bootstrap resamples of trials (seed 42).", "",
         "| Subset | Trials | Pairs | Total diff < 5 pp | AE diff >= 5 pp | EFFICACY diff >= 5 pp | **Hidden differential** | Hidden, among total < 5 pp |",
         "|---|---:|---:|---|---|---|---|---|"]
    for s in subs:
        n = res[(res.subset == s) & (res.split == "all") & (res.metric == "n_trials")].value
        npairs = res[(res.subset == s) & (res.split == "all") & (res.metric == "n_pairs")].value
        L.append(f"| {s} | {int(n.iloc[0]):,} | {int(npairs.iloc[0]):,} | {v(s, 'share_total_diff_lt5')} | "
                 f"{v(s, 'share_ae_diff_ge5')} | {v(s, 'share_efficacy_diff_ge5')} | **{v(s, 'share_hidden_differential')}** | "
                 f"{v(s, 'share_hidden_among_total_diff_lt5')} |")
    L += ["", "Hidden-differential share by split:", "", "| Subset | train | val | test |", "|---|---|---|---|"]
    for s in subs:
        L.append(f"| {s} | " + " | ".join(v(s, "share_hidden_differential", sp, ci=False) for sp in ["train", "val", "test"]) + " |")
    L += ["", "AE vs EFFICACY differential, over experimental-control pairs:", "",
          "| Subset | Pearson r | Spearman rho | Pairs with AE and EFFICACY diffs >= 5 pp in opposite directions |",
          "|---|---|---|---|"]
    for s in subs:
        L.append(f"| {s} | {v(s, 'pearson_ae_vs_efficacy_diff', pct=False)} (p={v(s, 'pearson_p', pct=False)}) | "
                 f"{v(s, 'spearman_ae_vs_efficacy_diff', pct=False)} (p={v(s, 'spearman_p', pct=False)}) | "
                 f"{v(s, 'share_pairs_ae_efficacy_opposite_ge5')} |")
    Path("results/hidden_differential_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
