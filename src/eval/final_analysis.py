"""Final models, pre-specified analyses, conformal intervals, planning simulation, sensitivity.

A) Models (fit on train, early stopping / grid on val, reported on test):
   reference_class, trial_lgbm, flat_lgbm (base features), FINAL = flat_lgbm +
   time-respecting prior-evidence features (NaN when no evidence).
   Subsets: all, reliable, covered, covered_reliable.
B) Pre-specified primary endpoints (unadjusted, two-sided paired bootstrap):
   1. arm_rate_primary MAE, all test arms: FINAL vs reference_class
   2. composition mean KL, all test arms with >= 1 primary withdrawal: FINAL vs reference_class
   3. pairwise_primary concordance, covered_reliable pairs: FINAL vs flat_lgbm
   Every other FINAL-vs-comparator paired test is secondary; Holm-adjusted as one family.
C) Mondrian split-conformal 90% intervals for the primary arm rate (calibrated on val),
   bins by arm size (started 10-49, 50-99, 100-299, >=300), plus a normalized variant
   with score |y - p| / sqrt(p(1-p)/n). NOTE: arm size here is the observed number
   started, used as a proxy for the planned arm size available at design time.
D) Recruitment-planning simulation on test arms.
E) Sensitivity of the primary endpoints: m70, m90 matched sets and a random
   trial-grouped split of m80.

Usage:
    python -m src.eval.final_analysis [--summary-only]
    --summary-only  relabel tests and rewrite final_summary.md from results/tables/final_*.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_benchmark import FLAG_THRESHOLDS, RELIABLE_MIN_STARTED
from src.eval import metrics as M
from src.features.build_features import FeatureBuilder
from src.features.prior_evidence import PRIOR_COLS, compute_prior_features
from src.models.baselines import (KINDS, PROP_COLS, RATE_OBJECTIVES, concordance, concordance_pairs, fit_flat,
                                  fit_lgbm, predict, reference_class, renormalize, trial_differentials)

SEED = 42
ALPHA = 0.10
PROCESSED = Path("data/processed")
TABLES = Path("results/tables")
SPLITS = ["train", "val", "test"]
SUBSETS = ["all", "reliable", "covered", "covered_reliable"]
MODELS = ["reference_class", "trial_lgbm", "flat_lgbm", "FINAL"]
SIZE_BINS = [(10, 49), (50, 99), (100, 299), (300, None)]
MAX_ASSUMED_ATTRITION = 0.95
PRIMARY_ENDPOINTS = [
    ("all", "arm_rate_primary", "MAE", "reference_class"),
    ("all", "composition", "mean_KL", "reference_class"),
    ("covered_reliable", "pairwise_primary", "concordance", "flat_lgbm"),
]


# ------------------------------------------------------------------ data + models
def load_set(name="m80", split_scheme="temporal"):
    suffix = "" if name == "m80" else f"_{name}"
    arms = pd.read_parquet(PROCESSED / f"arms{suffix}.parquet")
    trials = pd.read_parquet(PROCESSED / f"trials{suffix}.parquet")
    if split_scheme == "random_grouped":
        rng = np.random.default_rng(SEED)
        sizes = trials.split.value_counts()
        perm = rng.permutation(trials.nct_id.values)
        new, start = {}, 0
        for s in SPLITS:
            new.update({n: s for n in perm[start:start + sizes[s]]})
            start += sizes[s]
        trials = trials.assign(split=trials.nct_id.map(new))
        arms = arms.assign(split=arms.nct_id.map(new))
    arms = arms.merge(trials[["nct_id", "phase", "mesh_branches"]], on="nct_id")
    arms = arms.sort_values(["nct_id", "arm_idx"]).reset_index(drop=True)
    return arms, trials


def fit_models(arms, trials, P, full=True):
    """Predictions for val/test: {split: {model: {"rate_primary", "rate_nonendpoint", "comp"}}}."""
    A = {s: arms[arms.split == s] for s in SPLITS}
    T = {s: trials[trials.split == s].reset_index(drop=True) for s in SPLITS}
    fb = FeatureBuilder().fit(T["train"], A["train"])
    TX = {s: fb.trial_matrix(T[s]) for s in SPLITS}
    AX = {s: fb.arm_matrix(A[s], TX[s]) for s in SPLITS}
    AXp = {s: AX[s].join(P.loc[A[s].index, PRIOR_COLS]) for s in SPLITS}
    Ar = {s: A[s].reset_index(drop=True) for s in SPLITS}
    AXr = {s: AX[s].reset_index(drop=True) for s in SPLITS}
    AXpr = {s: AXp[s].reset_index(drop=True) for s in SPLITS}
    preds, selected = {s: {} for s in ["val", "test"]}, {}

    for s in preds:  # reference class
        p = {}
        for k in KINDS:
            p[f"rate_{k}"] = reference_class(Ar["train"], Ar[s], [f"y_{k}_attrition_rate"])[0][:, 0]
        p["comp"] = renormalize(reference_class(Ar["train"], Ar[s], PROP_COLS, mask_col="y_has_composition")[0])
        preds[s]["reference_class"] = p

    fit_flat("FINAL", AXpr, Ar, preds, selected)
    if full:
        fit_flat("flat_lgbm", AXr, Ar, preds, selected)
        TXf = {s: TX[s].drop(columns="nct_id") for s in SPLITS}
        for s in preds:
            preds[s]["trial_lgbm"] = {"comp": None}
        for k in KINDS:
            ty = {s: Ar[s].groupby("nct_id")[f"y_{k}_attrition_rate"].mean().reindex(T[s].nct_id).values for s in SPLITS}
            model, info = fit_lgbm(TXf["train"], ty["train"], TXf["val"], ty["val"], RATE_OBJECTIVES, M.mae)
            selected[f"trial_lgbm_{k}"] = info
            for s in preds:
                tp = pd.Series(predict(model, TXf[s]), index=T[s].nct_id.values)
                preds[s]["trial_lgbm"][f"rate_{k}"] = tp.reindex(Ar[s].nct_id).values
    else:  # sensitivity: only what the primary endpoints need
        for s in preds:
            preds[s]["flat_lgbm"] = {}
        model, info = fit_lgbm(AXr["train"], Ar["train"].y_primary_attrition_rate.values, AXr["val"],
                               Ar["val"].y_primary_attrition_rate.values, RATE_OBJECTIVES, M.mae)
        selected["flat_lgbm_primary"] = info
        for s in preds:
            preds[s]["flat_lgbm"]["rate_primary"] = predict(model, AXr[s])
    covered = {}
    for s in ["val", "test"]:
        a = Ar[s].assign(_cov=P.loc[A[s].index, "prior_n_trials"].values > 0)
        allcov = a[a.meta_role == "experimental"].groupby("nct_id")._cov.all()
        covered[s] = set(allcov.index[allcov]) & set(T[s].nct_id[T[s].y_has_differential])
    return preds, Ar, T, covered, selected


# ------------------------------------------------------------------ evaluation specs
def specs(arms, trials, covered_ids, preds, subsets=SUBSETS, only=None):
    """Yield (subset, target, metric, fn, groups, y, {model: prediction array})."""
    td_all = trials[trials.y_has_differential].set_index("nct_id")
    ie, ic, pg, prel = concordance_pairs(arms)
    in_cov_arm = arms.nct_id.isin(covered_ids).values
    in_cov_pair = np.isin(pg, list(covered_ids))
    big = arms.y_started.values >= RELIABLE_MIN_STARTED
    g = arms.nct_id.values
    for subset in subsets:
        am = {"all": np.ones(len(arms), bool), "reliable": big, "covered": in_cov_arm,
              "covered_reliable": in_cov_arm & big}[subset]
        pm = {"all": np.ones(len(ie), bool), "reliable": prel, "covered": in_cov_pair,
              "covered_reliable": in_cov_pair & prel}[subset]
        td = {"all": td_all, "reliable": td_all[td_all.meta_reliable],
              "covered": td_all[td_all.index.isin(covered_ids)],
              "covered_reliable": td_all[td_all.index.isin(covered_ids) & td_all.meta_reliable]}[subset]
        tg = td.index.values
        for kind in KINDS:
            y = arms[f"y_{kind}_attrition_rate"].values
            pr = {m: p[f"rate_{kind}"] for m, p in preds.items() if f"rate_{kind}" in p}
            for metric, fn in [("MAE", M.mae), ("RMSE", M.rmse), ("Spearman", M.spearman)]:
                yield subset, f"arm_rate_{kind}", metric, fn, g[am], y[am], {m: v[am] for m, v in pr.items()}
            yield (subset, f"pairwise_{kind}", "concordance", concordance, pg[pm], (y[ie] - y[ic])[pm],
                   {m: (v[ie] - v[ic])[pm] for m, v in pr.items()})
        mc = arms.y_has_composition.values & am
        prc = {m: p["comp"][mc] for m, p in preds.items() if p.get("comp") is not None}
        if prc:
            T_ = arms.loc[mc, PROP_COLS].values
            yield subset, "composition", "mean_KL", lambda t, q: float(M.kl_rows(t, q).mean()), g[mc], T_, prc
            yield (subset, "composition", "mean_Aitchison", lambda t, q: float(M.aitchison_rows(t, q).mean()),
                   g[mc], T_, prc)
        for thr in FLAG_THRESHOLDS:
            yf = td[f"y_flag_primary_{thr}_sig"].astype(int).values
            sc = {m: trial_differentials(arms, p["rate_primary"]).reindex(td.index).pred_absmax.values
                  for m, p in preds.items() if "rate_primary" in p}
            for metric, fn in [("AUROC", M.auroc), ("AUPRC", M.auprc)]:
                yield subset, f"flag_primary_{thr}_sig", metric, fn, tg, yf, sc


def evaluate(arms, trials, covered_ids, preds, subsets=SUBSETS):
    est, tests = [], []
    for subset, target, metric, fn, groups, y, pr in specs(arms, trials, covered_ids, preds, subsets):
        if len(y) == 0:
            continue
        for m, p in pr.items():
            v, lo, hi = M.bootstrap_ci(fn, groups, y, p)
            est.append({"subset": subset, "target": target, "metric": metric, "model": m, "value": v,
                        "ci_low": lo, "ci_high": hi, "n_trials": len(np.unique(groups)), "n_rows": len(y)})
        if "FINAL" not in pr:
            continue
        for comp in [m for m in MODELS if m in pr and m != "FINAL"]:
            d, lo, hi, p, p_perc = M.paired_bootstrap_test(lambda yy, a, b, f=fn: f(yy, a) - f(yy, b), groups, y,
                                                           pr["FINAL"], pr[comp])
            primary = (subset, target, metric, comp) in PRIMARY_ENDPOINTS
            tests.append({"subset": subset, "target": target, "metric": metric, "comparison": f"FINAL - {comp}",
                          "family": "primary" if primary else "secondary", "diff": d, "ci_low": lo, "ci_high": hi,
                          "p_raw": p, "p_boot_percentile": p_perc, "n_trials": len(np.unique(groups)), "n_rows": len(y)})
    return pd.DataFrame(est), adjust_tests(pd.DataFrame(tests))


def differential_target(target):
    return target.startswith(("pairwise_", "flag_"))


def adjust_tests(tests):
    """Exclude non-meaningful comparisons and Holm-adjust the secondary family.

    trial_lgbm predicts one rate per trial, so its predicted within-trial differential is zero by
    construction; FINAL-vs-trial_lgbm tests on concordance and flags are labelled not_meaningful
    and excluded from the Holm family. Differential targets are compared only against
    reference_class and flat_lgbm.
    """
    if tests.empty:
        return tests
    tests = tests.copy()
    nm = (tests.comparison == "FINAL - trial_lgbm") & tests.target.map(differential_target)
    tests.loc[nm, "family"] = "not_meaningful"
    sec = tests.family == "secondary"
    tests["p_holm"] = np.nan
    tests.loc[sec, "p_holm"] = M.holm(tests.loc[sec, "p_raw"].values)
    return tests


def robustness_labels(sens):
    """Per primary endpoint: 'robust' if every sensitivity CI excludes 0 with the main-analysis sign."""
    labels = {}
    for i, (subset, target, metric, comp) in enumerate(PRIMARY_ENDPOINTS, 1):
        r = sens[(sens.target == target) & (sens.metric == metric) & (sens.comparison == f"FINAL - {comp}")]
        main = r[(r.matched_set == "m80") & (r.split_scheme == "temporal")].iloc[0]
        sign = np.sign(main["diff"])
        fails = [f"{x.matched_set}" + ("" if x.split_scheme == "temporal" else " random split")
                 for _, x in r.iterrows()
                 if not ((x.ci_low > 0 and sign > 0) or (x.ci_high < 0 and sign < 0))]
        labels[i] = ("robust across sensitivity analyses" if not fails else
                     f"exploratory, not robust (fails at {', '.join(fails)})")
    return labels


# ------------------------------------------------------------------ C) conformal
def size_bin(n):
    for lo, hi in SIZE_BINS:
        if n >= lo and (hi is None or n <= hi):
            return f"{lo}-{hi}" if hi else f">={lo}"
    return "<10"


def conformal(val_arms, val_pred, test_arms, test_pred):
    def q_hat(scores):
        n = len(scores)
        k = min(n, int(np.ceil((n + 1) * (1 - ALPHA))))
        return float(np.sort(scores)[k - 1])

    yv, yt = val_arms.y_primary_attrition_rate.values, test_arms.y_primary_attrition_rate.values
    bv, bt = val_arms.y_started.map(size_bin).values, test_arms.y_started.map(size_bin).values
    out = {}
    # Mondrian (per size bin) absolute-residual conformal.
    qs = {b: q_hat(np.abs(yv - val_pred)[bv == b]) for b in np.unique(bv)}
    q_t = np.array([qs[b] for b in bt])
    out["mondrian"] = (np.clip(test_pred - q_t, 0, 1), np.clip(test_pred + q_t, 0, 1))
    # Normalized: |y - p| / sqrt(p(1-p)/n), single calibration.
    def sd(p, n):
        pc = np.clip(p, 0.01, 0.99)
        return np.sqrt(pc * (1 - pc) / n)
    qn = q_hat(np.abs(yv - val_pred) / sd(val_pred, val_arms.y_started.values))
    w = qn * sd(test_pred, test_arms.y_started.values)
    out["normalized"] = (np.clip(test_pred - w, 0, 1), np.clip(test_pred + w, 0, 1))

    rows = []
    for name, (lo, hi) in out.items():
        cov = (yt >= lo) & (yt <= hi)
        for b in ["all"] + [size_bin(l) for l, _ in SIZE_BINS]:
            m = np.ones(len(yt), bool) if b == "all" else (bt == b)
            v, cl, ch = M.bootstrap_ci(lambda c: float(c.mean()), test_arms.nct_id.values[m], cov[m])
            rows.append({"method": name, "size_bin": b, "n_arms": int(m.sum()), "coverage": v, "cov_ci_low": cl,
                         "cov_ci_high": ch, "mean_width": float((hi - lo)[m].mean()),
                         "calib_quantile": qs.get(b, np.nan) if name == "mondrian" else qn})
    return pd.DataFrame(rows), out


# ------------------------------------------------------------------ D) planning simulation
def planning(test_arms, assumed: dict):
    started = test_arms.y_started.values.astype(float)
    completers = started - test_arms.y_withdrawals_primary.values
    g = test_arms.nct_id.values
    rows = []
    for name, a in assumed.items():
        a = np.minimum(np.broadcast_to(a, started.shape).astype(float), MAX_ASSUMED_ATTRITION)
        target = completers / (1 - a)
        rel = (target - started) / started
        for metric, fn in [("pct_under_recruited", lambda r: float((r < 0).mean() * 100)),
                           ("mean_over_recruitment_pct_among_not_under", lambda r: float(r[r >= 0].mean() * 100)
                            if (r >= 0).any() else np.nan),
                           ("mean_shortfall_pct_among_under", lambda r: float(-r[r < 0].mean() * 100)
                            if (r < 0).any() else np.nan),
                           ("mean_relative_error_pct", lambda r: float(r.mean() * 100))]:
            v, lo, hi = M.bootstrap_ci(fn, g, rel)
            rows.append({"strategy": name, "metric": metric, "value": v, "ci_low": lo, "ci_high": hi,
                         "n_arms": len(started)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def summary_only():
    """Rebuild final_summary.md (and the test-family labels) from the saved final_*.csv tables."""
    est = pd.read_csv(TABLES / "final_estimates.csv")
    tests = adjust_tests(pd.read_csv(TABLES / "final_paired_tests.csv"))
    tests.to_csv(TABLES / "final_paired_tests.csv", index=False)
    arms, trials = load_set("m80", "temporal")
    A = {s: arms[arms.split == s] for s in SPLITS}
    T = {s: trials[trials.split == s] for s in SPLITS}
    n_cov = int(est[(est.subset == "covered") & (est.target == "pairwise_primary") & (est.model == "FINAL")].n_trials.iloc[0])
    write_summary(est, tests, pd.read_csv(TABLES / "final_conformal.csv"),
                  pd.read_csv(TABLES / "final_planning_simulation.csv"), pd.read_csv(TABLES / "final_sensitivity.csv"),
                  A, T, n_cov)


def main():
    if "--summary-only" in sys.argv:
        return summary_only()
    dates = pd.read_parquet("data/interim/trial_dates.parquet")
    arms, trials = load_set("m80", "temporal")
    P = compute_prior_features(arms, dates)
    preds, A, T, covered, selected = fit_models(arms, trials, P, full=True)

    est, tests = evaluate(A["test"], T["test"], covered["test"], preds["test"])
    conf, intervals = conformal(A["val"], preds["val"]["FINAL"]["rate_primary"],
                                A["test"], preds["test"]["FINAL"]["rate_primary"])
    plan = planning(A["test"], {
        "fixed_20pct": 0.20,
        "reference_class_mean": preds["test"]["reference_class"]["rate_primary"],
        "FINAL_point": preds["test"]["FINAL"]["rate_primary"],
        "FINAL_conformal_upper_90": intervals["mondrian"][1],
    })

    sens_rows = []
    for set_name, scheme in [("m80", "temporal"), ("m70", "temporal"), ("m90", "temporal"), ("m80", "random_grouped")]:
        if (set_name, scheme) == ("m80", "temporal"):
            t = tests[tests.family == "primary"]
        else:
            a2, t2 = load_set(set_name, scheme)
            P2 = compute_prior_features(a2, dates)
            pr2, A2, T2, cov2, _ = fit_models(a2, t2, P2, full=False)
            needed = {s for s, *_ in PRIMARY_ENDPOINTS}
            _, t = evaluate(A2["test"], T2["test"], cov2["test"], pr2["test"], subsets=sorted(needed))
            t = t[t.apply(lambda r: (r.subset, r.target, r.metric, r.comparison.split(" - ")[1]) in PRIMARY_ENDPOINTS,
                          axis=1)]
        sens_rows.append(t.assign(matched_set=set_name, split_scheme=scheme))
        print(f"sensitivity {set_name}/{scheme} done")
    sens = pd.concat(sens_rows, ignore_index=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    est.to_csv(TABLES / "final_estimates.csv", index=False)
    tests.to_csv(TABLES / "final_paired_tests.csv", index=False)
    conf.to_csv(TABLES / "final_conformal.csv", index=False)
    plan.to_csv(TABLES / "final_planning_simulation.csv", index=False)
    sens.to_csv(TABLES / "final_sensitivity.csv", index=False)
    (TABLES / "final_selected_params.json").write_text(json.dumps(selected, indent=1, default=str), encoding="utf-8")
    write_summary(est, tests, conf, plan, sens, A, T, len(covered["test"]))


# ------------------------------------------------------------------ summary
def fmt(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def pfmt(p):
    if pd.isna(p):
        return "n/a"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def write_summary(est, tests, conf, plan, sens, A, T, n_covered):
    robust = robustness_labels(sens)
    L = ["# Forecasting arm-level attrition from pre-enrollment protocol data: final results", "",
         "## Setup", "",
         f"- Benchmark: ClinicalTrials.gov interventional randomized trials with posted results, >= 1 control arm, all "
         f"participant-flow groups matched to protocol arms (rapidfuzz score >= 80), every arm started >= 10.",
         f"- Temporal split by start year: train < 2015, validation 2015-2016, test >= 2017. Trials (arms): train "
         f"{len(T['train']):,} ({len(A['train']):,}), validation {len(T['val']):,} ({len(A['val']):,}), test "
         f"{len(T['test']):,} ({len(A['test']):,}).",
         "- Models were fit on train and hyperparameters were selected on validation; no model fitting or tuning used "
         "the test split.",
         "- Models: reference class (mean by phase x MeSH branch x arm type with back-off below 20 arms); trial-level "
         "LightGBM (one rate per trial); flat per-arm LightGBM on protocol features (flat_lgbm); FINAL = flat_lgbm plus "
         "time-respecting prior-evidence features (same normalized drug in earlier trials whose results were posted "
         "before the target trial started; missing when no evidence).",
         f"- Subsets: all; reliable (arms started >= 100; trials whose pooled control and every experimental arm started "
         f">= 100); covered (trials whose experimental arms all have prior evidence: {n_covered:,} test "
         "trials); covered_reliable.",
         "- 95% CIs: 1,000 bootstrap resamples of trials (seed 42), percentile intervals. Paired comparisons resample "
         "trials jointly for both models; p-values are two-sided Wald tests of the paired difference using its "
         "bootstrap standard error (percentile-bootstrap p-values with add-one correction are also given in "
         "results/tables/final_paired_tests.csv; with 1,000 resamples their resolution is ~0.002).",
         "- Transparency on test-set use: the test split was also used in earlier exploratory analyses of this "
         "project (baselines, arm-feature enrichment, prior-evidence experiment), and the choice of FINAL and of the "
         "primary endpoints below was made after seeing those results. In particular, endpoint 3 was first observed "
         "in the prior-evidence experiment. These endpoints are therefore pre-specified for this report but not "
         "confirmatory; a prospective evaluation on trials posting results after this analysis is needed.", "",
         "## Pre-specified primary endpoints (test set, unadjusted)", "",
         "| # | Endpoint | Subset | FINAL | Comparator | Difference (FINAL - comparator) | p | Status |",
         "|---|---|---|---|---|---|---|---|"]
    for i, (subset, target, metric, comp) in enumerate(PRIMARY_ENDPOINTS, 1):
        t = tests[(tests.subset == subset) & (tests.target == target) & (tests.metric == metric)
                  & (tests.comparison == f"FINAL - {comp}")].iloc[0]
        f = est[(est.subset == subset) & (est.target == target) & (est.metric == metric) & (est.model == "FINAL")].iloc[0]
        c = est[(est.subset == subset) & (est.target == target) & (est.metric == metric) & (est.model == comp)].iloc[0]
        L.append(f"| {i} | {target} {metric} | {subset} | {fmt(f.value, f.ci_low, f.ci_high)} | {comp}: "
                 f"{fmt(c.value, c.ci_low, c.ci_high)} | {fmt(t['diff'], t.ci_low, t.ci_high)} | {pfmt(t.p_raw)} | "
                 f"{robust[i]} |")
    L += ["", "Lower is better for MAE and KL; higher is better for concordance (0.5 = chance). Primary endpoints are "
          "reported without multiplicity adjustment. Status is derived from the sensitivity analyses below (a CI "
          "that includes 0 or flips sign in any matched set or split counts as a failure).", ""]

    L += ["## Model comparison (test set)", ""]
    targets = [("arm_rate_primary", "MAE"), ("arm_rate_primary", "Spearman"), ("arm_rate_nonendpoint", "MAE"),
               ("arm_rate_nonendpoint", "Spearman"), ("composition", "mean_KL"), ("composition", "mean_Aitchison"),
               ("pairwise_primary", "concordance"), ("pairwise_nonendpoint", "concordance"),
               ("flag_primary_10_sig", "AUROC"), ("flag_primary_10_sig", "AUPRC"),
               ("flag_primary_5_sig", "AUROC"), ("flag_primary_5_sig", "AUPRC")]
    for subset in SUBSETS:
        e = est[est.subset == subset]
        if e.empty:
            continue
        L += [f"### Subset: {subset}", "", "| Target | Metric | n (trials / rows) | " + " | ".join(MODELS) + " |",
              "|---|---|---|" + "---|" * len(MODELS)]
        for target, metric in targets:
            r = e[(e.target == target) & (e.metric == metric)]
            if r.empty:
                continue
            cells = []
            for m in MODELS:
                x = r[r.model == m]
                if m == "trial_lgbm" and differential_target(target):
                    cells.append("n/m†")
                    continue
                cells.append("–" if x.empty or pd.isna(x.value.iloc[0]) else
                             fmt(x.value.iloc[0], x.ci_low.iloc[0], x.ci_high.iloc[0]))
            L.append(f"| {target} | {metric} | {int(r.n_trials.iloc[0]):,} / {int(r.n_rows.iloc[0]):,} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["† Not meaningful: trial_lgbm assigns the same rate to every arm of a trial, so its within-trial "
          "differential is zero by construction (concordance 0.5, flag AUROC 0.5). Differential targets are compared "
          "only against reference_class and flat_lgbm.", ""]

    sec = tests[tests.family == "secondary"]
    n_nm = int((tests.family == "not_meaningful").sum())
    L += ["## Secondary paired comparisons (Holm-adjusted across the secondary family)", "",
          f"{len(sec)} secondary tests ({n_nm} FINAL-vs-trial_lgbm tests on concordance/flags excluded as not "
          "meaningful). Tests with Holm-adjusted p < 0.05:", "",
          "| Subset | Target | Metric | Comparison | Difference | p raw | p Holm |", "|---|---|---|---|---|---|---|"]
    for _, t in sec[sec.p_holm < 0.05].sort_values(["subset", "target", "metric"]).iterrows():
        L.append(f"| {t.subset} | {t.target} | {t.metric} | {t.comparison} | {fmt(t['diff'], t.ci_low, t.ci_high)} | "
                 f"{pfmt(t.p_raw)} | {pfmt(t.p_holm)} |")
    L += ["", "All secondary tests (raw and adjusted p) are in results/tables/final_paired_tests.csv.", ""]

    L += ["## Conformal prediction intervals for the primary arm attrition rate (FINAL, target 90%)", "",
          "Split-conformal, calibrated on validation arms. Mondrian = absolute-residual quantile per arm-size bin; "
          "normalized = single quantile of |y - p| / sqrt(p(1-p)/n). Arm size is the observed number started, used "
          "here as a proxy for the planned arm size that is known at design time.", "",
          "| Method | Arm size (started) | Test arms | Coverage | Mean width |", "|---|---|---:|---|---:|"]
    for _, r in conf.iterrows():
        L.append(f"| {r.method} | {r.size_bin} | {r.n_arms:,} | {fmt(r.coverage, r.cov_ci_low, r.cov_ci_high)} | "
                 f"{r.mean_width:.3f} |")
    L.append("")

    L += ["## Recruitment-planning simulation (test arms)", "",
          "Required completers = observed completers (started minus primary-attrition withdrawals). Recruitment target = "
          f"completers / (1 - assumed attrition), assumed attrition capped at {MAX_ASSUMED_ATTRITION:.2f}. Under-recruited = "
          "target below the number actually started.", "",
          "| Strategy | % arms under-recruited | Mean over-recruitment % (arms not under) | Mean shortfall % (arms under) |",
          "|---|---|---|---|"]
    for s in plan.strategy.unique():
        p = plan[plan.strategy == s].set_index("metric")
        cell = lambda m: fmt(p.loc[m].value, p.loc[m].ci_low, p.loc[m].ci_high, 1)
        L.append(f"| {s} | {cell('pct_under_recruited')} | {cell('mean_over_recruitment_pct_among_not_under')} | "
                 f"{cell('mean_shortfall_pct_among_under')} |")
    L.append("")

    fpath, mpath = TABLES / "final_recruitment_frontier.csv", TABLES / "final_recruitment_matched.csv"
    if fpath.exists() and mpath.exists():
        mt = pd.read_csv(mpath)
        L += ["### Recruitment trade-off frontier", "",
              "Each strategy is swept over its parameter (fixed attrition 0.05-0.50; reference class + margin 0-0.30; "
              "FINAL one-sided Mondrian conformal upper quantile at levels 0.50-0.95). Over-recruitment at matched "
              "under-recruitment rates (linear interpolation; paired trial-grouped bootstrap CIs):", "",
              "| Under-recruited | Metric | Fixed rule | Reference class | FINAL | FINAL - fixed | FINAL - reference |",
              "|---|---|---|---|---|---|---|"]
        for (u, metric), g in mt.groupby(["under_recruitment_pct", "metric"], sort=False):
            gi = g.set_index("comparison")
            cell = lambda k: fmt(gi.loc[k].value, gi.loc[k].ci_low, gi.loc[k].ci_high, 1)
            L.append(f"| {u:.0f}% | {metric} | {cell('fixed')} | {cell('reference_class')} | {cell('FINAL')} | "
                     f"{cell('FINAL - fixed')} | {cell('FINAL - reference_class')} |")
        L += ["", "Full frontier: results/tables/final_recruitment_frontier.csv.", ""]
    spath = TABLES / "final_shap_families.csv"
    if spath.exists():
        sf = pd.read_csv(spath)
        fams = ["design", "condition", "eligibility_text", "arm", "prior_evidence"]
        pv = sf.pivot(index="target", columns="family", values="share_of_total")[fams]
        nf = sf.drop_duplicates("family").set_index("family").n_features
        L += ["## Interpretability (TreeSHAP, FINAL, test arms)", "",
              "Share of summed mean |SHAP| by feature family (log-odds scale). Shares are sums over features and grow "
              "with family size; prior-evidence features are missing for most test arms.", "",
              "| Target | " + " | ".join(f"{f} ({int(nf[f])})" for f in fams) + " |", "|---|" + "---:|" * len(fams)]
        for t, r in pv.iterrows():
            L.append(f"| {t} | " + " | ".join(f"{100 * r[f]:.1f}%" for f in fams) + " |")
        L += ["", "Top-15 features per target: results/tables/final_shap_top15.csv.", ""]
    npath = TABLES / "final_shap_families_normalized.csv"
    if npath.exists():
        nf2 = pd.read_csv(npath)
        fams = ["design", "condition", "eligibility_text", "arm", "prior_evidence"]
        pv = nf2.pivot(index="target", columns="family", values="share_per_feature_normalized")[fams]
        L += ["### Robustness check: per-feature-normalized SHAP families", "",
              "Mean |SHAP| per feature within each family (family sum / number of features), as a share across "
              "families. This removes the advantage of large families (300 TF-IDF features).", "",
              "| Target | " + " | ".join(fams) + " |", "|---|" + "---:|" * len(fams)]
        for t, r in pv.iterrows():
            L.append(f"| {t} | " + " | ".join(f"{100 * r[f]:.1f}%" for f in fams) + " |")
        L.append("")
    apath = TABLES / "final_start_year_ablation.csv"
    if apath.exists():
        ab = pd.read_csv(apath)
        L += ["### Robustness check: start-year ablation", "",
              "flat_lgbm and FINAL refit without the start_year feature (same grid, selected on validation). "
              "Difference = without - with start_year (paired trial-grouped bootstrap, test split). For MAE and KL, "
              "lower is better; for concordance, higher is better.", "",
              "| Model | Target | Subset | With start_year | Without start_year | Difference | p | Without vs reference_class |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in ab.iterrows():
            vs = (f"{fmt(r.without_year_minus_reference, r.vs_ref_ci_low, r.vs_ref_ci_high)} (p={pfmt(r.vs_ref_p)})"
                  if pd.notna(r.get("without_year_minus_reference")) else "–")
            L.append(f"| {r.model} | {r.target} | {r.subset} | {fmt(r.with_year, r.with_year_ci_low, r.with_year_ci_high)} | "
                     f"{fmt(r.without_year, r.without_year_ci_low, r.without_year_ci_high)} | "
                     f"{fmt(r.diff_without_minus_with, r.diff_ci_low, r.diff_ci_high)} | {pfmt(r.p)} | {vs} |")
        L.append("")

    L += ["## Sensitivity of the primary endpoints", "",
          "| Matched set | Split | Endpoint | Subset | n trials | Difference (FINAL - comparator) | p |",
          "|---|---|---|---|---:|---|---|"]
    for _, t in sens.iterrows():
        L.append(f"| {t.matched_set} | {t.split_scheme} | {t.target} {t.metric} ({t.comparison}) | {t.subset} | "
                 f"{int(t.n_trials):,} | {fmt(t['diff'], t.ci_low, t.ci_high)} | {pfmt(t.p_raw)} |")
    L += ["", "The random trial-grouped split ignores calendar time for model fitting (prior evidence remains "
          "time-respecting) and is reported only as a sensitivity analysis.", ""]
    Path("results/final_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
