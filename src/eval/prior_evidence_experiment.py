"""Prior-evidence retrieval experiment (decisive check before Task 4).

C) Coverage of time-respecting drug evidence (experimental arms, by split/phase;
   trials with all experimental arms covered).
D) On the COVERED test subset (trials with a differential whose experimental arms
   all have >= 1 prior trial), and covered + reliable pairs (both arms started >= 100):
   flat LightGBM (base features) vs flat LightGBM + prior-evidence features, each
   with per-arm regressors for primary, AE and EFFICACY rates (fit on train, early
   stopping/grid on val). Metrics: total within-trial concordance; AE- and
   EFFICACY-direction concordance vs a trivial arm-role baseline (experimental arm
   has more AE withdrawals / fewer EFFICACY withdrawals than control); level-adjusted
   AE sign accuracy (trials with pooled AE rate >= 2%); flag_primary_10_sig AUROC/AUPRC.
   95% CIs from 1,000 trial-grouped bootstrap resamples, including paired
   differences (prior - base).
E) Random trial-grouped split (seed 42, temporal split sizes) vs temporal split for
   hidden-differential and flag prevalence, plus start-year trends in reporting
   granularity, to see whether the decline over time is reporting drift.

Usage:
    python -m src.eval.prior_evidence_experiment
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_benchmark import RELIABLE_MIN_STARTED, cat_col, CATEGORIES, PRIMARY_CATS
from src.eval import metrics as M
from src.eval.hidden_differential import pair_table, trial_table
from src.features.build_features import FeatureBuilder
from src.features.prior_evidence import PRIOR_COLS, compute_prior_features
from src.models.baselines import RATE_OBJECTIVES, concordance, fit_lgbm, predict, trial_differentials

SEED = 42
PROCESSED = Path("data/processed")
TABLES = Path("results/tables")
AE_LEVEL_MIN = 0.02
TARGETS = {"primary": "y_primary_attrition_rate", "ae": "y_ae_rate", "efficacy": "y_efficacy_rate"}
SPLITS = ["train", "val", "test"]


# ------------------------------------------------------------------ C) coverage
def coverage(arms, trials, P):
    rows = []
    exp = arms.meta_role == "experimental"
    cov = P.prior_n_trials > 0
    a = arms.assign(covered=cov, phase=arms.nct_id.map(trials.set_index("nct_id").phase))
    for s in ["all"] + SPLITS:
        m = exp & ((a.split == s) if s != "all" else True)
        rows.append({"section": "exp_arms_covered", "split": s, "group": "all",
                     "n": int(m.sum()), "value": float(a.covered[m].mean())})
        for ph, g in a[m].groupby("phase"):
            rows.append({"section": "exp_arms_covered", "split": s, "group": ph, "n": len(g),
                         "value": float(g.covered.mean())})
        t = trials[trials.y_has_differential & ((trials.split == s) if s != "all" else True)]
        allcov = a[exp].groupby("nct_id").covered.all().reindex(t.nct_id).fillna(False)
        rows.append({"section": "trials_all_exp_arms_covered", "split": s, "group": "all",
                     "n": len(t), "value": float(allcov.mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ D) models + evaluation
def fit_targets(Xtr, Xva, Xte, A):
    preds, info = {}, {}
    for name, col in TARGETS.items():
        model, inf = fit_lgbm(Xtr, A["train"][col].values, Xva, A["val"][col].values, RATE_OBJECTIVES, M.mae)
        preds[name], info[name] = predict(model, Xte), inf
    return preds, info


def pairs_of(arms):
    e = arms[arms.meta_role == "experimental"][["nct_id", "y_started"]].reset_index()
    c = arms[arms.meta_role == "control"][["nct_id", "y_started"]].reset_index()
    pr = e.merge(c, on="nct_id", suffixes=("_e", "_c"))
    pr["reliable"] = (pr.y_started_e >= RELIABLE_MIN_STARTED) & (pr.y_started_c >= RELIABLE_MIN_STARTED)
    return pr


def evaluate(arms, trials, covered_ids, preds):
    rows = []

    def add(subset, target, metric, model, fn, groups, *arrays):
        if len(arrays[0]) == 0:
            return
        v, lo, hi = M.bootstrap_ci(fn, groups, *arrays)
        rows.append({"subset": subset, "target": target, "metric": metric, "model": model, "value": v,
                     "ci_low": lo, "ci_high": hi, "n_rows": len(arrays[0]), "n_trials": len(np.unique(groups))})

    pr = pairs_of(arms)
    pr = pr[pr.nct_id.isin(covered_ids)]
    trivial_sign = {"primary": None, "ae": 1.0, "efficacy": -1.0}
    for subset, pm in [("covered", np.ones(len(pr), bool)), ("covered_reliable", pr.reliable.values)]:
        ie, ic, g = pr.index_e.values[pm], pr.index_c.values[pm], pr.nct_id.values[pm]
        for tname, col in TARGETS.items():
            y = arms[col].values
            td = y[ie] - y[ic]
            pb = preds["base"][tname][ie] - preds["base"][tname][ic]
            pp = preds["prior"][tname][ie] - preds["prior"][tname][ic]
            add(subset, f"pairwise_{tname}", "concordance", "flat_lgbm", concordance, g, td, pb)
            add(subset, f"pairwise_{tname}", "concordance", "flat_lgbm+prior", concordance, g, td, pp)
            add(subset, f"pairwise_{tname}", "concordance", "prior_minus_base",
                lambda t, a_, b_: concordance(t, b_) - concordance(t, a_), g, td, pb, pp)
            if trivial_sign[tname] is not None:
                add(subset, f"pairwise_{tname}", "concordance", "trivial_arm_role", concordance, g, td,
                    np.full(len(td), trivial_sign[tname]))

    # Level-adjusted AE sign: trials with pooled AE rate >= 2%.
    a = arms.assign(n_ae=arms[[cat_col("ADVERSE_EVENT")]].sum(axis=1))
    pooled = a.groupby("nct_id").n_ae.sum() / a.groupby("nct_id").y_started.sum()
    lvl_ids = set(pooled.index[pooled >= AE_LEVEL_MIN]) & set(covered_ids)
    sub = arms[arms.nct_id.isin(lvl_ids)]

    def gap(values):
        d = sub[["nct_id", "meta_role"]].assign(v=values[sub.index])
        e = d[d.meta_role == "experimental"].groupby("nct_id").v.mean()
        c = d[d.meta_role == "control"].groupby("nct_id").v.mean()
        return (e - c).dropna()

    true_gap = (sub.assign(n=a.n_ae[sub.index]).pipe(
        lambda x: x[x.meta_role == "experimental"].groupby("nct_id").y_ae_rate.mean()
        - x[x.meta_role == "control"].groupby("nct_id").n.sum() / x[x.meta_role == "control"].groupby("nct_id").y_started.sum()))
    true_gap = true_gap.dropna()
    gb, gp = gap(preds["base"]["ae"]).reindex(true_gap.index), gap(preds["prior"]["ae"]).reindex(true_gap.index)
    ids = true_gap.index.values
    for model, pg in [("flat_lgbm", gb.values), ("flat_lgbm+prior", gp.values),
                      ("trivial_arm_role", np.ones(len(ids)))]:
        add("covered_ae_level_ge2pct", "ae_gap_sign", "sign_accuracy", model, concordance, ids, true_gap.values, pg)
    add("covered_ae_level_ge2pct", "ae_gap_sign", "sign_accuracy", "prior_minus_base",
        lambda t, a_, b_: concordance(t, b_) - concordance(t, a_), ids, true_gap.values, gb.values, gp.values)

    # flag_primary_10_sig on covered trials (and covered reliable trials).
    tc = trials[trials.nct_id.isin(covered_ids)].set_index("nct_id")
    for subset, tt in [("covered", tc), ("covered_reliable", tc[tc.meta_reliable])]:
        yf = tt.y_flag_primary_10_sig.astype(int).values
        sb = trial_differentials(arms, preds["base"]["primary"]).reindex(tt.index).pred_absmax.values
        sp = trial_differentials(arms, preds["prior"]["primary"]).reindex(tt.index).pred_absmax.values
        for metric, fn in [("AUROC", M.auroc), ("AUPRC", M.auprc)]:
            add(subset, "flag_primary_10_sig", metric, "flat_lgbm", fn, tt.index.values, yf, sb)
            add(subset, "flag_primary_10_sig", metric, "flat_lgbm+prior", fn, tt.index.values, yf, sp)
            add(subset, "flag_primary_10_sig", metric, "prior_minus_base",
                lambda y_, a_, b_, f=fn: f(y_, b_) - f(y_, a_), tt.index.values, yf, sb, sp)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E) drift
def drift(arms, trials):
    rows = []
    t = trials[trials.y_has_differential].copy()
    pairs = pair_table(arms[arms.nct_id.isin(t.nct_id)], t)
    hid = trial_table(pairs).hidden
    t["hidden"] = t.nct_id.map(hid).fillna(False)
    flags = ["hidden", "y_flag_primary_5_sig", "y_flag_primary_10_sig", "y_flag_ae_5_sig", "y_flag_efficacy_5_sig"]
    rng = np.random.default_rng(SEED)

    # Random trial-grouped split with the temporal split sizes (over all benchmark trials).
    sizes = trials.split.value_counts()
    perm = rng.permutation(trials.nct_id.values)
    rand = {}
    start = 0
    for s in SPLITS:
        rand.update({n: s for n in perm[start:start + sizes[s]]})
        start += sizes[s]
    t["random_split"] = t.nct_id.map(rand)

    def share_ci(x):
        x = np.asarray(x, float)
        st = x[rng.integers(0, len(x), (1000, len(x)))].mean(1)
        return float(x.mean()), *map(float, np.percentile(st, [2.5, 97.5]))

    for scheme, col in [("temporal", "split"), ("random_grouped", "random_split")]:
        for s in SPLITS:
            g = t[t[col] == s]
            for f in flags:
                v, lo, hi = share_ci(g[f].astype(bool).values)
                rows.append({"section": f"prevalence_{scheme}", "group": s, "metric": f, "value": v,
                             "ci_low": lo, "ci_high": hi, "n": len(g)})

    # Start-year trends: prevalence and reporting-granularity indicators.
    bins = [0, 2007, 2010, 2014, 2016, 2019, 9999]
    labels = ["<=2007", "2008-10", "2011-14", "2015-16", "2017-19", ">=2020"]
    t["year_bin"] = pd.cut(t.start_year, bins=bins, labels=labels)
    a = arms.merge(t[["nct_id", "year_bin"]], on="nct_id")
    prim = [cat_col(c) for c in PRIMARY_CATS]
    for yb, g in t.groupby("year_bin", observed=True):
        ga = a[a.year_bin == yb]
        for f in flags:
            v, lo, hi = share_ci(g[f].astype(bool).values)
            rows.append({"section": "by_start_year", "group": yb, "metric": f, "value": v, "ci_low": lo,
                         "ci_high": hi, "n": len(g)})
        tot = ga[prim].values.sum()
        ind = {
            "mean_arm_started": ga.y_started.mean(),
            "median_arm_started": ga.y_started.median(),
            "mean_primary_rate": ga.y_primary_attrition_rate.mean(),
            "mean_ae_rate": ga.y_ae_rate.mean(),
            "mean_efficacy_rate": ga.y_efficacy_rate.mean(),
            "share_withdrawals_other_unspecified": ga[cat_col("OTHER_UNSPECIFIED")].sum() / tot,
            "share_withdrawals_ae": ga[cat_col("ADVERSE_EVENT")].sum() / tot,
            "mean_distinct_primary_categories_per_trial": (ga.groupby("nct_id")[prim].sum() > 0).sum(1).mean(),
            "share_placebo_controlled": (g.meta_control_type == "PLACEBO_COMPARATOR").mean(),
            "share_phase3": (g.phase == "PHASE3").mean(),
            "share_reliable": g.meta_reliable.mean(),
        }
        for k, v in ind.items():
            rows.append({"section": "reporting_by_start_year", "group": yb, "metric": k, "value": float(v),
                         "ci_low": np.nan, "ci_high": np.nan, "n": len(g)})
    return pd.DataFrame(rows)


def main():
    arms = pd.read_parquet(PROCESSED / "arms.parquet").sort_values(["nct_id", "arm_idx"]).reset_index(drop=True)
    trials = pd.read_parquet(PROCESSED / "trials.parquet")
    dates = pd.read_parquet("data/interim/trial_dates.parquet")
    P = compute_prior_features(arms, dates)
    print(f"unique normalized drugs: {P.attrs['unique_drugs']:,}; evidence links: {P.attrs['n_evidence_links']:,}")

    cov = coverage(arms, trials, P)
    cov = pd.concat([cov, pd.DataFrame([{"section": "drugs", "split": "all", "group": "unique_normalized_drugs",
                                         "n": np.nan, "value": P.attrs["unique_drugs"]}])], ignore_index=True)

    A = {s: arms[arms.split == s] for s in SPLITS}
    T = {s: trials[trials.split == s] for s in SPLITS}
    fb = FeatureBuilder().fit(T["train"], A["train"])
    X = {s: fb.arm_matrix(A[s], fb.trial_matrix(T[s])) for s in SPLITS}
    Xp = {s: X[s].join(P.loc[A[s].index, PRIOR_COLS]) for s in SPLITS}

    base, info_b = fit_targets(X["train"], X["val"], X["test"], A)
    prior, info_p = fit_targets(Xp["train"], Xp["val"], Xp["test"], A)

    te_arms = A["test"].reset_index(drop=True)
    te_trials = T["test"]
    exp_cov = (P.loc[A["test"].index, "prior_n_trials"].values > 0)
    te_arms = te_arms.assign(_cov=exp_cov)
    allcov = te_arms[te_arms.meta_role == "experimental"].groupby("nct_id")._cov.all()
    covered_ids = set(allcov.index[allcov]) & set(te_trials.nct_id[te_trials.y_has_differential])
    res = evaluate(te_arms, te_trials, covered_ids, {"base": base, "prior": prior})
    dr = drift(arms, trials)

    TABLES.mkdir(parents=True, exist_ok=True)
    cov.to_csv(TABLES / "prior_evidence_coverage.csv", index=False)
    res.to_csv(TABLES / "prior_evidence_results.csv", index=False)
    dr.to_csv(TABLES / "drift_analysis.csv", index=False)
    (TABLES / "prior_evidence_selected_params.json").write_text(
        json.dumps({"base": info_b, "prior": info_p}, indent=1, default=str), encoding="utf-8")
    write_summary(cov, res, dr, len(covered_ids))


def write_summary(cov, res, dr, n_cov):
    def ci(r):
        return f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"

    L = ["# Prior-evidence retrieval experiment", "",
         "Evidence = other benchmark trials containing the same normalized DRUG/BIOLOGICAL name (exact match) "
         "whose resultsFirstPostDate is strictly before the target trial's start date (asserted). "
         "resultsFirstPostDate is used only for this ordering, never as a feature.", "",
         f"Unique normalized drugs: {int(cov[cov.section == 'drugs'].value.iloc[0]):,}.", "",
         "## Coverage", "", "| Split | Experimental arms | Covered (>= 1 prior trial) | Trials (with differential) | All exp. arms covered |",
         "|---|---:|---:|---:|---:|"]
    for s in ["all"] + SPLITS:
        e = cov[(cov.section == "exp_arms_covered") & (cov.split == s) & (cov.group == "all")].iloc[0]
        t = cov[(cov.section == "trials_all_exp_arms_covered") & (cov.split == s)].iloc[0]
        L.append(f"| {s} | {int(e.n):,} | {100 * e.value:.1f}% | {int(t.n):,} | {100 * t.value:.1f}% |")
    L += ["", "Covered share of experimental arms by phase (all splits):", "", "| Phase | Exp. arms | Covered |", "|---|---:|---:|"]
    for _, r in cov[(cov.section == "exp_arms_covered") & (cov.split == "all") & (cov.group != "all")].sort_values("n", ascending=False).iterrows():
        L.append(f"| {r.group} | {int(r.n):,} | {100 * r.value:.1f}% |")

    L += ["", f"## Covered test subset ({n_cov:,} trials with a differential whose experimental arms are all covered)", "",
          "Models fit on train, selected on val. 95% CIs: 1,000 trial-grouped bootstrap resamples; "
          "prior_minus_base is the paired difference.", "",
          "| Subset | Target | Metric | trivial arm-role | flat_lgbm | flat_lgbm + prior | prior − base | n (trials / rows) |",
          "|---|---|---|---|---|---|---|---|"]
    for (subset, target, metric), g in res.groupby(["subset", "target", "metric"], sort=False):
        get = lambda m: ci(g[g.model == m].iloc[0]) if (g.model == m).any() else "–"
        n = g.iloc[0]
        L.append(f"| {subset} | {target} | {metric} | {get('trivial_arm_role')} | {get('flat_lgbm')} | "
                 f"{get('flat_lgbm+prior')} | {get('prior_minus_base')} | {int(n.n_trials):,} / {int(n.n_rows):,} |")

    L += ["", "## Drift check: temporal vs random trial-grouped split (prevalence, 95% CI)", "",
          "| Metric | Scheme | train | val | test |", "|---|---|---|---|---|"]
    for metric in dr[dr.section == "prevalence_temporal"].metric.unique():
        for scheme in ["temporal", "random_grouped"]:
            d = dr[(dr.section == f"prevalence_{scheme}") & (dr.metric == metric)].set_index("group")
            L.append(f"| {metric} | {scheme} | " + " | ".join(
                f"{100 * d.loc[s].value:.1f}% [{100 * d.loc[s].ci_low:.1f}, {100 * d.loc[s].ci_high:.1f}]" for s in SPLITS) + " |")
    yb = dr[dr.section.isin(["by_start_year", "reporting_by_start_year"])]
    groups = list(dict.fromkeys(yb.group))
    L += ["", "By start year:", "", "| Metric | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for metric in dict.fromkeys(yb.metric):
        d = yb[yb.metric == metric].set_index("group").value
        pct = metric in {"hidden"} or metric.startswith(("y_flag", "share_"))
        L.append(f"| {metric} | " + " | ".join((f"{100 * d[gr]:.1f}%" if pct else f"{d[gr]:.3f}") for gr in groups) + " |")
    n_row = yb[yb.metric == "hidden"].set_index("group").n
    L.append("| n trials | " + " | ".join(f"{int(n_row[gr]):,}" for gr in groups) + " |")
    Path("results/prior_evidence_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
