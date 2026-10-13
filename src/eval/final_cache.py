"""Refit FINAL (deterministic, same settings as final_analysis) and cache what Task 5 needs.

Writes data/interim/final_cache.pkl (gitignored) with:
  A_val, A_test            arm tables (reset index, aligned with predictions)
  pred                     {split: {"FINAL": {...}, "reference_class": {...}}}
  models                   FINAL boosters {"rate_primary", "rate_nonendpoint", <prop col>: ...}
  X_test, X_val            FINAL feature matrices (same column order the boosters were fit on)

Usage:
    python -m src.eval.final_cache
"""
import pickle
from pathlib import Path

import pandas as pd

from src.eval.final_analysis import SPLITS, load_set
from src.features.build_features import FeatureBuilder
from src.features.prior_evidence import PRIOR_COLS, compute_prior_features
from src.models.baselines import KINDS, PROP_COLS, fit_flat, reference_class, renormalize

CACHE = Path("data/interim/final_cache.pkl")


def build():
    dates = pd.read_parquet("data/interim/trial_dates.parquet")
    arms, trials = load_set("m80", "temporal")
    P = compute_prior_features(arms, dates)
    A = {s: arms[arms.split == s] for s in SPLITS}
    T = {s: trials[trials.split == s].reset_index(drop=True) for s in SPLITS}
    fb = FeatureBuilder().fit(T["train"], A["train"])
    X = {s: fb.arm_matrix(A[s], fb.trial_matrix(T[s])).join(P.loc[A[s].index, PRIOR_COLS]).reset_index(drop=True)
         for s in SPLITS}
    Ar = {s: A[s].reset_index(drop=True) for s in SPLITS}
    preds, selected, models = {s: {} for s in ["val", "test"]}, {}, {}
    fit_flat("FINAL", X, Ar, preds, selected, models=models)
    for s in preds:
        p = {f"rate_{k}": reference_class(Ar["train"], Ar[s], [f"y_{k}_attrition_rate"])[0][:, 0] for k in KINDS}
        p["comp"] = renormalize(reference_class(Ar["train"], Ar[s], PROP_COLS, mask_col="y_has_composition")[0])
        preds[s]["reference_class"] = p
    cache = {"A_val": Ar["val"], "A_test": Ar["test"], "pred": preds, "models": models,
             "X_val": X["val"], "X_test": X["test"]}
    CACHE.write_bytes(pickle.dumps(cache))
    return cache


def load():
    return pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build()


if __name__ == "__main__":
    c = build()
    print(f"cached FINAL: {len(c['models'])} boosters; test arms {len(c['A_test']):,}")
