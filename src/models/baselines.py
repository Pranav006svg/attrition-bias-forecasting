"""Baselines for arm-level attrition forecasting.

Models (fit on train, hyperparameters selected on val, final numbers on test):
  global_mean      train mean rate / mean composition (sanity floor)
  reference_class  mean by (phase x MeSH branch x arm type), backing off to
                   (phase x arm type) -> (arm type) -> global when a cell has < 20 arms
  trial_lgbm       LightGBM on trial features predicting the trial's mean arm rate,
                   applied equally to every arm of the trial
  flat_lgbm        LightGBM per arm (trial + arm features + TF-IDF); one regressor for
                   the rate and one per reason proportion (renormalized to sum to 1)
  flat_lgbm_enriched  same, with enriched arm features (dose, dose rank, intervention
                   class, combination flag, active interventions, arm type x masking)

Evaluation subsets: all and reliable (arms / pairs / trials with arms started >= 100).
Extra targets: within-trial pairwise concordance; AE and EFFICACY reason-group
differentials and their flags (predicted group rate = rate x group proportion).

Targets: primary and nonendpoint arm rates; reason composition (primary categories,
arms with >= 1 withdrawal); trial differential derived from predicted arm rates
(control = unweighted mean over control arms); flags *_10_sig / *_5_sig scored by
predicted |absmax differential|.

Usage:
    python -m src.models.baselines [--summary-only]   # --summary-only: rewrite markdown from baselines.csv
"""
import itertools
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.build_benchmark import (FLAG_THRESHOLDS, GROUP_FLAG_TARGETS, GROUP_FLAG_THRESHOLD, PRIMARY_CATS,
                                      REASON_GROUPS, RELIABLE_MIN_STARTED, prop_col)
from src.eval import metrics as M
from src.features.build_features import FeatureBuilder

SEED = 42
PROCESSED = Path("data/processed")
TABLES = Path("results/tables")
KINDS = ["primary", "nonendpoint"]
PROP_COLS = [prop_col(c) for c in PRIMARY_CATS]
REF_MIN_ARMS = 20

GRID = {"num_leaves": [15, 63], "min_child_samples": [20, 100]}
RATE_OBJECTIVES = ["regression", "cross_entropy"]
BASE_PARAMS = {"learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
               "lambda_l2": 1.0, "seed": SEED, "deterministic": True, "force_row_wise": True,
               "num_threads": 4, "verbose": -1}
MAX_ROUNDS, EARLY_STOP = 3000, 100


# ---------------------------------------------------------------- LightGBM helpers
def fit_lgbm(Xtr, ytr, Xva, yva, objectives, select_metric):
    """Grid over objectives x GRID with early stopping on val; returns best model + info."""
    best = None
    for obj, (nl, mcs) in itertools.product(objectives, itertools.product(*GRID.values())):
        params = {**BASE_PARAMS, "objective": obj, "num_leaves": nl, "min_child_samples": mcs,
                  "metric": "l1" if obj == "regression" else "cross_entropy"}
        model = lgb.train(params, lgb.Dataset(Xtr, ytr), MAX_ROUNDS,
                          valid_sets=[lgb.Dataset(Xva, yva)],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        score = select_metric(yva, np.clip(model.predict(Xva, num_iteration=model.best_iteration), 0, 1))
        if best is None or score < best[0]:
            best = (score, model, {"objective": obj, "num_leaves": nl, "min_child_samples": mcs,
                                   "best_iteration": model.best_iteration, "val_score": score})
    return best[1], best[2]


def predict(model, X):
    return np.clip(model.predict(X, num_iteration=model.best_iteration), 0, 1)


# ---------------------------------------------------------------- reference class
def reference_class(train: pd.DataFrame, apply: pd.DataFrame, target_cols, mask_col=None):
    """Mean of target_cols by (phase, mesh, arm_type) with backoff; multi-branch arms averaged."""
    tr = train if mask_col is None else train[train[mask_col]]
    tr = tr.assign(mesh=tr.mesh_branches.str.split("|")).explode("mesh")
    levels = [["phase", "mesh", "arm_type"], ["phase", "arm_type"], ["arm_type"]]
    tables = []
    for keys in levels:
        g = tr.groupby(keys)
        means = g[target_cols].mean()
        n = g.size()  # arms per cell (an arm counts once in each of its MeSH branches)
        tables.append((keys, means[n.reindex(means.index) >= REF_MIN_ARMS]))
    glob = tr[target_cols].mean().values

    ap = apply.assign(mesh=apply.mesh_branches.str.split("|"), _row=np.arange(len(apply))).explode("mesh")
    out = np.full((len(ap), len(target_cols)), np.nan)
    level_used = np.full(len(ap), "global", dtype=object)
    for keys, tab in tables:
        todo = np.isnan(out[:, 0])
        if not todo.any():
            break
        idx = pd.MultiIndex.from_frame(ap.loc[todo, keys]) if len(keys) > 1 else pd.Index(ap.loc[todo, keys[0]])
        vals = tab.reindex(idx).values
        hit = ~np.isnan(vals[:, 0])
        rows = np.where(todo)[0][hit]
        out[rows] = vals[hit]
        level_used[rows] = "+".join(keys)
    miss = np.isnan(out[:, 0])
    out[miss] = glob
    res = pd.DataFrame(out, columns=target_cols).assign(_row=ap._row.values).groupby("_row").mean()
    return res.values, pd.Series(level_used).value_counts().to_dict()


# ---------------------------------------------------------------- derived trial quantities
def trial_differentials(arms: pd.DataFrame, pred_rate: np.ndarray) -> pd.DataFrame:
    """Predicted signed and absmax differential per trial from predicted arm rates."""
    d = arms[["nct_id", "meta_role"]].assign(p=pred_rate)
    ctrl = d[d.meta_role == "control"].groupby("nct_id").p.mean()
    exp = d[d.meta_role == "experimental"].join(ctrl.rename("c"), on="nct_id")
    exp["diff"] = exp.p - exp.c
    g = exp.groupby("nct_id")
    out = pd.DataFrame({"pred_signed": g.p.max() - g.c.first(), "pred_absmax": g["diff"].apply(lambda x: x.abs().max())})
    return out.round(12)  # drop floating-point residue (e.g. equal per-arm predictions)


# ---------------------------------------------------------------- evaluation
def concordance_pairs(arms: pd.DataFrame):
    """Within-trial (experimental, control-arm) row-index pairs; reliable = both arms started >= 100."""
    e = arms[arms.meta_role == "experimental"][["nct_id", "y_started"]].reset_index()
    c = arms[arms.meta_role == "control"][["nct_id", "y_started"]].reset_index()
    pr = e.merge(c, on="nct_id", suffixes=("_e", "_c"))
    rel = (pr.y_started_e >= RELIABLE_MIN_STARTED) & (pr.y_started_c >= RELIABLE_MIN_STARTED)
    return pr.index_e.values, pr.index_c.values, pr.nct_id.values, rel.values


def concordance(true_d, pred_d):
    """Share of pairs ordered correctly; true ties dropped, predicted ties count 0.5."""
    m = true_d != 0
    if not m.any():
        return np.nan
    t, p = np.sign(true_d[m]), np.sign(np.round(pred_d[m], 12))
    return float(np.mean(np.where(p == 0, 0.5, (p == t).astype(float))))


def evaluate(rows, split, model, arms, trials, preds):
    """Append metric rows for subsets all / reliable.

    preds = {"rate_<kind>": arr, "comp": (n_arms x 10) or None}; arms has a 0..n-1 index.
    reliable: arm rates on arms started >= 100; trial targets on trials whose pooled
    control and every experimental arm started >= 100; concordance on pairs with
    both arms started >= 100.
    """
    boot = split == "test"

    def add(subset, target, metric, fn, groups, *arrays):
        if len(arrays[0]) == 0:
            return
        if boot:
            v, lo, hi = M.bootstrap_ci(fn, groups, *arrays)
        else:
            v, lo, hi = fn(*arrays), np.nan, np.nan
        rows.append({"split": split, "subset": subset, "model": model, "target": target, "metric": metric,
                     "value": v, "ci_low": lo, "ci_high": hi, "n_rows": len(arrays[0]),
                     "n_trials": len(np.unique(groups))})

    g = arms.nct_id.values
    td_all = trials[trials.y_has_differential].set_index("nct_id")
    ie, ic, pg, prel = concordance_pairs(arms)
    group_rates = {}
    if preds.get("comp") is not None:
        for gname in GROUP_FLAG_TARGETS:
            idx = [PROP_COLS.index(prop_col(c)) for c in REASON_GROUPS[gname]]
            group_rates[gname] = preds["rate_primary"] * preds["comp"][:, idx].sum(axis=1)

    for subset in ["all", "reliable"]:
        am = np.ones(len(arms), bool) if subset == "all" else (arms.y_started.values >= RELIABLE_MIN_STARTED)
        td = td_all if subset == "all" else td_all[td_all.meta_reliable]
        tg = td.index.values
        pm = np.ones(len(ie), bool) if subset == "all" else prel
        for kind in KINDS:
            y = arms[f"y_{kind}_attrition_rate"].values
            p = preds[f"rate_{kind}"]
            add(subset, f"arm_rate_{kind}", "MAE", M.mae, g[am], y[am], p[am])
            add(subset, f"arm_rate_{kind}", "RMSE", M.rmse, g[am], y[am], p[am])
            add(subset, f"arm_rate_{kind}", "Spearman", M.spearman, g[am], y[am], p[am])
            add(subset, f"pairwise_{kind}", "concordance", concordance, pg[pm],
                (y[ie] - y[ic])[pm], (p[ie] - p[ic])[pm])

            pdiff = trial_differentials(arms, p).reindex(td.index)
            add(subset, f"differential_{kind}_absmax", "MAE", M.mae, tg,
                td[f"y_differential_{kind}_absmax"].values, pdiff.pred_absmax.values)
            add(subset, f"differential_{kind}_signed", "MAE", M.mae, tg,
                td[f"y_differential_{kind}"].values, pdiff.pred_signed.values)
            for thr in FLAG_THRESHOLDS:
                yf = td[f"y_flag_{kind}_{thr}_sig"].astype(int).values
                sc = pdiff.pred_absmax.values
                add(subset, f"flag_{kind}_{thr}_sig", "AUROC", M.auroc, tg, yf, sc)
                add(subset, f"flag_{kind}_{thr}_sig", "AUPRC", M.auprc, tg, yf, sc)

        for gname, gr in group_rates.items():
            pdiff = trial_differentials(arms, gr).reindex(td.index)
            add(subset, f"differential_{gname}_absmax", "MAE", M.mae, tg,
                td[f"y_differential_{gname}_absmax"].values, pdiff.pred_absmax.values)
            yf = td[f"y_flag_{gname}_{GROUP_FLAG_THRESHOLD}_sig"].astype(int).values
            sc = pdiff.pred_absmax.values
            add(subset, f"flag_{gname}_{GROUP_FLAG_THRESHOLD}_sig", "AUROC", M.auroc, tg, yf, sc)
            add(subset, f"flag_{gname}_{GROUP_FLAG_THRESHOLD}_sig", "AUPRC", M.auprc, tg, yf, sc)

        if preds.get("comp") is not None:
            m = arms.y_has_composition.values & am
            T = arms.loc[m, PROP_COLS].values
            P = preds["comp"][m]
            add(subset, "composition", "mean_KL", lambda t, q: float(M.kl_rows(t, q).mean()), g[m], T, P)
            add(subset, "composition", "mean_Aitchison", lambda t, q: float(M.aitchison_rows(t, q).mean()),
                g[m], T, P)


def renormalize(P):
    P = np.clip(P, 0, None)
    s = P.sum(axis=1, keepdims=True)
    return np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / P.shape[1])


def fit_flat(name, AX, A, preds, selected, models=None):
    """Per-arm LightGBM: rate regressors + one regressor per reason proportion.

    If a dict is passed as models, fitted boosters are stored under keys like "rate_primary" or a
    proportion column name (used for SHAP)."""
    for s in preds:
        preds[s][name] = {}
    for k in KINDS:
        col = f"y_{k}_attrition_rate"
        model, info = fit_lgbm(AX["train"], A["train"][col].values, AX["val"], A["val"][col].values,
                               RATE_OBJECTIVES, M.mae)
        selected[f"{name}_{k}"] = info
        if models is not None:
            models[f"rate_{k}"] = model
        for s in preds:
            preds[s][name][f"rate_{k}"] = predict(model, AX[s])
    mtr, mva = A["train"].y_has_composition.values, A["val"].y_has_composition.values
    comp = {s: np.zeros((len(A[s]), len(PROP_COLS))) for s in preds}
    for j, c in enumerate(PROP_COLS):
        model, info = fit_lgbm(AX["train"][mtr], A["train"].loc[mtr, c].values, AX["val"][mva],
                               A["val"].loc[mva, c].values, ["cross_entropy"],
                               lambda y, p: float(np.mean(-(y * np.log(np.clip(p, 1e-6, 1)) +
                                                            (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))))
        selected[f"{name}_{c}"] = info
        if models is not None:
            models[c] = model
        for s in preds:
            comp[s][:, j] = predict(model, AX[s])
    for s in preds:
        preds[s][name]["comp"] = renormalize(comp[s])


def main():
    if "--summary-only" in sys.argv:
        arms = pd.read_parquet(PROCESSED / "arms.parquet")
        trials = pd.read_parquet(PROCESSED / "trials.parquet")
        write_summary(pd.read_csv(TABLES / "baselines.csv"),
                      {s: arms[arms.split == s] for s in ["train", "val", "test"]},
                      {s: trials[trials.split == s] for s in ["train", "val", "test"]})
        return
    arms = pd.read_parquet(PROCESSED / "arms.parquet")
    trials = pd.read_parquet(PROCESSED / "trials.parquet")
    arms = arms.merge(trials[["nct_id", "phase", "mesh_branches"]], on="nct_id")
    arms = arms.sort_values(["nct_id", "arm_idx"]).reset_index(drop=True)
    A = {s: arms[arms.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    T = {s: trials[trials.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}

    fb = FeatureBuilder().fit(T["train"], A["train"])
    TX = {s: fb.trial_matrix(T[s]) for s in T}
    AX = {s: fb.arm_matrix(A[s], TX[s]) for s in A}
    TXf = {s: TX[s].drop(columns="nct_id") for s in TX}
    print(f"features: trial {TXf['train'].shape[1]}, arm {AX['train'].shape[1]}")

    preds = {s: {} for s in ["val", "test"]}
    selected = {}

    # global mean
    for s in preds:
        preds[s]["global_mean"] = {f"rate_{k}": np.full(len(A[s]), A["train"][f"y_{k}_attrition_rate"].mean())
                                   for k in KINDS}
        comp_mean = A["train"].loc[A["train"].y_has_composition, PROP_COLS].mean().values
        preds[s]["global_mean"]["comp"] = np.tile(comp_mean, (len(A[s]), 1))

    # reference class
    for s in preds:
        p = {}
        for k in KINDS:
            v, lv = reference_class(A["train"], A[s], [f"y_{k}_attrition_rate"])
            p[f"rate_{k}"] = v[:, 0]
            selected[f"reference_class_levels_{s}_{k}"] = lv
        v, _ = reference_class(A["train"], A[s], PROP_COLS, mask_col="y_has_composition")
        p["comp"] = renormalize(v)
        preds[s]["reference_class"] = p

    # trial-level LightGBM (target: mean arm rate of the trial)
    for s in preds:
        preds[s]["trial_lgbm"] = {"comp": None}
    for k in KINDS:
        ty = {s: A[s].groupby("nct_id")[f"y_{k}_attrition_rate"].mean().reindex(T[s].nct_id).values for s in A}
        model, info = fit_lgbm(TXf["train"], ty["train"], TXf["val"], ty["val"], RATE_OBJECTIVES, M.mae)
        selected[f"trial_lgbm_{k}"] = info
        for s in preds:
            tp = pd.Series(predict(model, TXf[s]), index=T[s].nct_id.values)
            preds[s]["trial_lgbm"][f"rate_{k}"] = tp.reindex(A[s].nct_id).values

    # flat per-arm LightGBM: base arm features, then enriched arm features
    fit_flat("flat_lgbm", AX, A, preds, selected)
    fbe = FeatureBuilder(enriched=True).fit(T["train"], A["train"])
    AXe = {s: fbe.arm_matrix(A[s], fbe.trial_matrix(T[s])) for s in A}
    print(f"enriched arm features: {AXe['train'].shape[1]}")
    fit_flat("flat_lgbm_enriched", AXe, A, preds, selected)

    rows = []
    for s in ["val", "test"]:
        for model_name, p in preds[s].items():
            print(f"evaluating {model_name} on {s}")
            evaluate(rows, s, model_name, A[s], T[s], p)
    res = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "baselines.csv", index=False)
    (TABLES / "baselines_selected_params.json").write_text(json.dumps(selected, indent=1, default=str),
                                                          encoding="utf-8")
    write_summary(res, A, T)


def level_auroc(arms, trials, gname):
    """AUROC of the true trial-mean reason-group rate for that group's flag (level-only reference)."""
    td = trials[trials.y_has_differential].set_index("nct_id")
    lvl = arms.groupby("nct_id")[f"y_{gname}_rate"].mean().reindex(td.index)
    return M.auroc(td[f"y_flag_{gname}_{GROUP_FLAG_THRESHOLD}_sig"].astype(int).values, lvl.values)


def write_summary(res, A, T):
    test = res[res.split == "test"]
    models = ["global_mean", "reference_class", "trial_lgbm", "flat_lgbm", "flat_lgbm_enriched"]

    def cell(target, metric, model, subset):
        r = test[(test.target == target) & (test.metric == metric) & (test.model == model) & (test.subset == subset)]
        if r.empty or pd.isna(r.value.iloc[0]):
            return "n/a"
        r = r.iloc[0]
        return f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"

    def n_of(target, subset):
        r = test[(test.target == target) & (test.subset == subset)]
        return f"{int(r.n_trials.iloc[0]):,} / {int(r.n_rows.iloc[0]):,}" if len(r) else "n/a"

    L = ["# Baselines (test split, primary benchmark m80)", "",
         f"Train/val/test trials: {len(T['train']):,} / {len(T['val']):,} / {len(T['test']):,}; "
         f"arms: {len(A['train']):,} / {len(A['val']):,} / {len(A['test']):,}. "
         "Values are point estimates with 95% bootstrap CIs (1,000 trial-grouped resamples, seed 42). "
         "Models were fit on train with hyperparameters selected on val; no tuning or selection used test.", "",
         "Subsets: **all**; **reliable** = arm rates on arms started >= 100, trial targets on trials whose pooled "
         "control and every experimental arm started >= 100, pairwise concordance on pairs with both arms >= 100. "
         "n = trials / rows (arms, pairs or trials).", ""]
    blocks = [("Arm attrition rate", [(f"arm_rate_{k}", m) for k in KINDS for m in ["MAE", "RMSE", "Spearman"]]),
              ("Within-trial pairwise concordance (experimental vs control arm ordering; 0.5 = chance)",
               [(f"pairwise_{k}", "concordance") for k in KINDS]),
              ("Reason composition (arms with >= 1 primary withdrawal; lower is better)",
               [("composition", "mean_KL"), ("composition", "mean_Aitchison")]),
              ("Trial differential MAE (trials with a computable differential)",
               [(f"differential_{k}_{v}", "MAE") for k in KINDS for v in ["absmax", "signed"]]
               + [(f"differential_{g}_absmax", "MAE") for g in GROUP_FLAG_TARGETS]),
              ("Differential-attrition flags (score = predicted |absmax differential|)",
               [(f"flag_{k}_{t}_sig", m) for k in KINDS for t in FLAG_THRESHOLDS for m in ["AUROC", "AUPRC"]]
               + [(f"flag_{g}_{GROUP_FLAG_THRESHOLD}_sig", m) for g in GROUP_FLAG_TARGETS for m in ["AUROC", "AUPRC"]])]
    for subset in ["all", "reliable"]:
        L += [f"# Subset: {subset}", ""]
        for title, items in blocks:
            L += [f"## {title}", "", "| Target | Metric | n | " + " | ".join(models) + " |",
                  "|---|---|---|" + "---|" * len(models)]
            for target, metric in items:
                L.append(f"| {target} | {metric} | {n_of(target, subset)} | "
                         + " | ".join(cell(target, metric, m, subset) for m in models) + " |")
            L.append("")
    L += ["## Effect of arm-feature enrichment (flat_lgbm_enriched minus flat_lgbm, test, point estimates)", "",
          "| Subset | Target | Metric | flat_lgbm | enriched | change |", "|---|---|---|---:|---:|---:|"]
    for subset in ["all", "reliable"]:
        b = test[(test.model == "flat_lgbm") & (test.subset == subset)].set_index(["target", "metric"]).value
        e = test[(test.model == "flat_lgbm_enriched") & (test.subset == subset)].set_index(["target", "metric"]).value
        for (target, metric), bv in b.items():
            ev = e.get((target, metric), np.nan)
            L.append(f"| {subset} | {target} | {metric} | {bv:.3f} | {ev:.3f} | {ev - bv:+.3f} |")
    L += ["", "Notes: trial_lgbm predicts one rate per trial, so its predicted differential is 0 for every trial "
          "(flag AUROC 0.5, concordance 0.5) and it has no composition model. AE/EFFICACY differentials use predicted "
          "group rates = predicted primary rate x predicted group proportion (models with a composition head only). "
          f"Composition KL uses prediction smoothing eps={M.KL_EPS}; Aitchison uses multiplicative zero replacement "
          f"delta={M.ZERO_DELTA}.", "",
          "Caveat on reason-group flags: these flags are largely determined by the trial's overall level of the "
          "reason group. On test, the TRUE trial-mean group rate alone ranks the flag with AUROC "
          + ", ".join(f"{g}: {level_auroc(A['test'], T['test'], g):.3f}" for g in GROUP_FLAG_TARGETS)
          + ". Their AUROCs therefore mainly reflect predicting reason-group levels, not which arm is higher; "
          "direction is measured by pairwise concordance.", ""]
    Path("results/baselines_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
