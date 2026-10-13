"""Time-respecting prior-evidence features per arm.

For an arm of trial T (start date s), evidence = arms of OTHER benchmark trials
whose resultsFirstPostDate < s and that contain at least one of the arm's
normalized drugs (exact match). resultsFirstPostDate is used only for this
ordering and never becomes a feature. Evidence arms are pooled across the
target arm's drugs (each evidence arm counted once).

Features (NaN when no evidence):
  prior_n_trials        distinct evidence trials (0 when none)
  prior_n_arms          distinct evidence arms
  prior_rate_primary    mean primary attrition rate of evidence arms
  prior_rate_ae         mean AE-withdrawal rate
  prior_rate_efficacy   mean EFFICACY-withdrawal rate
  prior_diff_total      mean (evidence drug arm - its trial's pooled control) primary rate,
  prior_diff_ae         ... AE rate,
  prior_diff_efficacy   ... EFFICACY rate   (non-control evidence arms in trials with a control)
"""
import numpy as np
import pandas as pd

from src.data.build_benchmark import REASON_GROUPS, cat_col
from src.features.drugs import arm_drugs

PRIOR_COLS = ["prior_n_trials", "prior_n_arms", "prior_rate_primary", "prior_rate_ae", "prior_rate_efficacy",
              "prior_diff_total", "prior_diff_ae", "prior_diff_efficacy"]


def _arm_evidence_table(arms: pd.DataFrame) -> pd.DataFrame:
    """Per arm: rates and (arm - pooled control) differentials for evidence use."""
    a = arms[["nct_id", "arm_idx", "meta_role", "y_started", "y_primary_attrition_rate", "y_ae_rate",
              "y_efficacy_rate"]].copy()
    for g in ["ae", "efficacy"]:
        a[f"n_{g}"] = arms[[cat_col(c) for c in REASON_GROUPS[g]]].sum(axis=1)
    a["n_primary"] = arms.y_withdrawals_primary
    ctrl = a[a.meta_role == "control"].groupby("nct_id")[["n_primary", "n_ae", "n_efficacy", "y_started"]].sum()
    c = ctrl.reindex(a.nct_id)
    noncontrol = (a.meta_role != "control").values & c.y_started.notna().values
    for col, n in [("diff_total", "n_primary"), ("diff_ae", "n_ae"), ("diff_efficacy", "n_efficacy")]:
        own = a[n].values / a.y_started.values
        crate = c[n].values / c.y_started.values
        a[col] = np.where(noncontrol, own - crate, np.nan)
    return a


def compute_prior_features(arms: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """Prior-evidence features aligned to arms.index. arms: full benchmark (all splits)."""
    d = dates.set_index("nct_id")
    ev = _arm_evidence_table(arms)
    ev["post"] = ev.nct_id.map(d.results_first_post_date)
    ev["ev_key"] = ev.nct_id + "#" + ev.arm_idx.astype(str)

    drugs = arms.intervention_names.map(arm_drugs)
    long = pd.DataFrame({"row": np.repeat(arms.index.values, drugs.str.len()),
                         "drug": [x for l in drugs for x in l]})
    tgt = long.assign(t_nct=arms.loc[long.row, "nct_id"].values,
                      t_start=arms.loc[long.row, "nct_id"].map(d.start_date).values)
    src = long.assign(ev_row=long.row).drop(columns="row")
    ev_long = src.join(ev[["nct_id", "post", "ev_key", "y_primary_attrition_rate", "y_ae_rate", "y_efficacy_rate",
                           "diff_total", "diff_ae", "diff_efficacy"]], on="ev_row")

    m = tgt.merge(ev_long, on="drug")
    m = m[(m.post < m.t_start) & (m.nct_id != m.t_nct)]
    m = m.drop_duplicates(["row", "ev_key"])

    # Assertion against the source date table (not the merged columns).
    ev_post = m.nct_id.map(d.results_first_post_date).values
    tgt_start = m.t_nct.map(d.start_date).values
    assert (ev_post < tgt_start).all(), "evidence posted on/after target start"
    assert (m.nct_id.values != m.t_nct.values).all(), "trial used as its own evidence"

    g = m.groupby("row")
    out = pd.DataFrame(index=arms.index)
    out["prior_n_trials"] = g.nct_id.nunique().reindex(arms.index).fillna(0)
    out["prior_n_arms"] = g.ev_key.nunique().reindex(arms.index).fillna(0)
    for col, src_col in [("prior_rate_primary", "y_primary_attrition_rate"), ("prior_rate_ae", "y_ae_rate"),
                         ("prior_rate_efficacy", "y_efficacy_rate"), ("prior_diff_total", "diff_total"),
                         ("prior_diff_ae", "diff_ae"), ("prior_diff_efficacy", "diff_efficacy")]:
        out[col] = g[src_col].mean().reindex(arms.index)
    out.attrs["n_evidence_links"] = len(m)
    out.attrs["unique_drugs"] = len(set(long.drug))
    return out
