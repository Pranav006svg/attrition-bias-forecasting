"""Start-year ablation for FINAL and flat_lgbm (robustness check).

Refits flat_lgbm and FINAL without the start_year feature (same LightGBM grid and
early stopping on validation) and compares with the versions that include it, on the
test split: arm-rate MAE and composition KL (also versus reference_class, i.e. primary
endpoints 1 and 2) and within-trial pairwise concordance (all, reliable pairs).
Paired trial-grouped bootstrap (1,000 resamples): difference = without - with start_year.

Output: results/tables/final_start_year_ablation.csv

Usage:
    python -m src.eval.start_year_ablation
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import metrics as M
from src.eval.final_analysis import SPLITS, load_set
from src.eval.final_cache import load as load_cache
from src.features.build_features import FeatureBuilder
from src.features.prior_evidence import PRIOR_COLS, compute_prior_features
from src.models.baselines import PROP_COLS, concordance, concordance_pairs, fit_flat

TABLES = Path("results/tables")


def kl_mean(t, q):
    return float(M.kl_rows(t, q).mean())


def main():
    dates = pd.read_parquet("data/interim/trial_dates.parquet")
    arms, trials = load_set("m80", "temporal")
    P = compute_prior_features(arms, dates)
    A = {s: arms[arms.split == s] for s in SPLITS}
    T = {s: trials[trials.split == s].reset_index(drop=True) for s in SPLITS}
    fb = FeatureBuilder().fit(T["train"], A["train"])
    X = {s: fb.arm_matrix(A[s], fb.trial_matrix(T[s])).reset_index(drop=True) for s in SPLITS}
    Xp = {s: X[s].join(P.loc[A[s].index, PRIOR_COLS].reset_index(drop=True)) for s in SPLITS}
    Ar = {s: A[s].reset_index(drop=True) for s in SPLITS}

    preds, sel = {s: {} for s in ["val", "test"]}, {}
    fit_flat("flat_lgbm", X, Ar, preds, sel)
    fit_flat("flat_lgbm_no_year", {s: X[s].drop(columns="start_year") for s in SPLITS}, Ar, preds, sel)
    fit_flat("FINAL_no_year", {s: Xp[s].drop(columns="start_year") for s in SPLITS}, Ar, preds, sel)
    cache = load_cache()
    assert cache["A_test"].nct_id.equals(Ar["test"].nct_id), "cache/test row order mismatch"
    p = preds["test"]
    p["FINAL"] = cache["pred"]["test"]["FINAL"]
    p["reference_class"] = cache["pred"]["test"]["reference_class"]
    print("refits done")

    at = Ar["test"]
    g = at.nct_id.values
    y = at.y_primary_attrition_rate.values
    mc = at.y_has_composition.values
    Tc = at.loc[mc, PROP_COLS].values
    ie, ic, pg, prel = concordance_pairs(at)

    def specs():
        yield "arm_rate_primary MAE", "all", M.mae, g, y, lambda m: p[m]["rate_primary"]
        yield "composition mean_KL", "all", kl_mean, g[mc], Tc, lambda m: p[m]["comp"][mc]
        for sub, pm in [("all", np.ones(len(ie), bool)), ("reliable", prel)]:
            yield ("pairwise_primary concordance", sub, concordance, pg[pm], (y[ie] - y[ic])[pm],
                   lambda m, pm=pm: (p[m]["rate_primary"][ie] - p[m]["rate_primary"][ic])[pm])

    rows = []
    for base in ["flat_lgbm", "FINAL"]:
        ablated = f"{base}_no_year"
        for target, subset, fn, groups, truth, pred in specs():
            r = {"model": base, "target": target, "subset": subset, "n_trials": len(np.unique(groups)),
                 "n_rows": len(truth)}
            for tag, m in [("with_year", base), ("without_year", ablated)]:
                v, lo, hi = M.bootstrap_ci(fn, groups, truth, pred(m))
                r.update({f"{tag}": v, f"{tag}_ci_low": lo, f"{tag}_ci_high": hi})
            d, lo, hi, pw, _ = M.paired_bootstrap_test(lambda t, a, b: fn(t, a) - fn(t, b), groups, truth,
                                                        pred(ablated), pred(base))
            r.update({"diff_without_minus_with": d, "diff_ci_low": lo, "diff_ci_high": hi, "p": pw})
            if target != "pairwise_primary concordance":
                d, lo, hi, pw, _ = M.paired_bootstrap_test(lambda t, a, b: fn(t, a) - fn(t, b), groups, truth,
                                                            pred(ablated), pred("reference_class"))
                r.update({"without_year_minus_reference": d, "vs_ref_ci_low": lo, "vs_ref_ci_high": hi, "vs_ref_p": pw})
            rows.append(r)
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "final_start_year_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
