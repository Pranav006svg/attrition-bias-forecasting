"""Build the arm-level attrition benchmark.

Inclusion: interventional, randomized, >=2 arms, results posted, withdrawal
reasons present, >=1 control arm, every flow group matched to a protocol arm
(rapidfuzz score >= threshold; 80 primary, 70/90 sensitivity), >=1 matched
control arm, every matched arm STARTED >= 10, known start year.

Periods: core randomized periods only. Periods whose title matches
extension / open-label / OLE / follow-up / long-term are excluded unless the
trial has a single period. Denominator = STARTED of the first included period.

Column conventions in the outputs:
  nct_id, arm_idx, split   identifiers
  y_*                      labels (built from resultsSection)
  meta_*                   bookkeeping derived from resultsSection (never features)
  everything else          pre-enrollment features (checked by leakage_check.py)

Usage:
    python -m src.data.build_benchmark
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.feasibility import assign_groups
from src.data.reasons import CATEGORIES, eligible_pool_ids

SEED = 42
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
TABLES = Path("results/tables")

THRESHOLDS = {"m70": 70, "m80": 80, "m90": 90}
PRIMARY = "m80"
MIN_STARTED = 10
SPLIT_RULE = {"train": "start_year < 2015", "val": "2015 <= start_year <= 2016", "test": "start_year >= 2017"}

EXCLUDED_PERIOD = re.compile(r"extension|open[- ]?label|\bole\b|follow[- ]?up|long[- ]?term", re.IGNORECASE)
CONTROL_PRIORITY = ["PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION", "ACTIVE_COMPARATOR"]
EXPERIMENTAL_TYPES = {"EXPERIMENTAL", "OTHER"}
NOT_PRIMARY = {"SPONSOR_ADMINISTRATIVE", "NOT_ATTRITION"}
NOT_NONENDPOINT = NOT_PRIMARY | {"DEATH", "DISEASE_PROGRESSION"}

TRIAL_FEATURES = ["phase", "allocation", "masking", "intervention_model", "primary_purpose", "n_arms",
                  "sponsor_class", "mesh_branches", "healthy_volunteers", "sex", "min_age_years",
                  "max_age_years", "start_year", "eligibility_text", "brief_summary", "conditions"]
ARM_FEATURES = ["arm_type", "arm_label", "arm_description", "intervention_types", "intervention_names"]


def split_of(year: int) -> str:
    return "train" if year < 2015 else ("val" if year <= 2016 else "test")


def cat_col(c: str) -> str:
    return f"y_n_{c.lower()}"


def load():
    trials = pd.read_parquet(INTERIM / "protocol_trials.parquet")
    arms = pd.read_parquet(INTERIM / "protocol_arms.parquet")
    fg = pd.read_parquet(INTERIM / "flow_groups.parquet")
    periods = pd.read_parquet(INTERIM / "flow_periods.parquet")
    started = pd.read_parquet(INTERIM / "flow_started.parquet")
    dw = pd.read_parquet(INTERIM / "flow_dropwithdraws.parquet")
    mapping = pd.read_csv("annotation/reason_mapping.csv", keep_default_na=False)
    return trials, arms, fg, periods, started, dw, mapping


def build_trial(nct, t_arms, t_fg, t_periods, t_started, t_dw, reason_cat):
    """Arm rows + trial summary for one trial (before threshold filtering), or a drop reason."""
    flow_groups = t_fg.fillna("").to_dict("records")
    arm_recs = [{"label": a.arm_label, "description": a.arm_description,
                 "interventionNames": a.intervention_names.split("|") if a.intervention_names else []}
                for a in t_arms.fillna("").itertuples()]
    arm_idx, scores = assign_groups(flow_groups, arm_recs)
    if len(flow_groups) < 2:
        return None, "fewer_than_2_flow_groups"

    titles = t_periods.sort_values("period_idx")
    if len(titles) == 1:
        incl = list(titles.period_idx)
    else:
        incl = [p for p, ti in zip(titles.period_idx, titles.period_title) if not EXCLUDED_PERIOD.search(ti or "")]
    n_excl = len(titles) - len(incl)
    if not incl:
        return None, "all_periods_excluded"
    first = min(incl)
    st = t_started[t_started.period_idx == first].set_index("group_id")["started"]

    d = t_dw[t_dw.period_idx.isin(incl)].copy()
    d["category"] = d.raw_reason.map(reason_cat)
    if d.category.isna().any():
        raise ValueError(f"{nct}: unmapped reasons {d.raw_reason[d.category.isna()].unique()[:3]}")
    counts = d.groupby(["group_id", "category"])["n"].sum()

    rows = []
    for g, ai, sc in zip(flow_groups, arm_idx, scores):
        gid = g["group_id"]
        s = st.get(gid)
        row = {"nct_id": nct, "meta_flow_group_id": gid, "meta_match_score": float(sc),
               "arm_idx": int(ai), "y_started": None if s is None or pd.isna(s) else int(s)}
        for c in CATEGORIES:
            row[cat_col(c)] = int(counts.get((gid, c), 0))
        rows.append(row)
    a = pd.DataFrame(rows)
    if a.y_started.isna().any():
        return None, "missing_started"
    return {"arms": a, "n_periods": len(titles), "n_periods_excluded": n_excl}, None


def trial_labels(a: pd.DataFrame) -> dict:
    """Trial-level differential attrition from labeled arm rows."""
    types = a.arm_type.values
    ctrl_type = next((c for c in CONTROL_PRIORITY if c in set(types)), None)
    ctrl = a[a.arm_type == ctrl_type]
    exp = a[a.arm_type.isin(EXPERIMENTAL_TYPES)]
    fallback = exp.empty
    if fallback:
        exp = a[a.arm_type != ctrl_type]
    out = {"meta_control_type": ctrl_type, "meta_exp_fallback": bool(fallback),
           "meta_n_exp_arms": len(exp), "meta_n_control_arms": len(ctrl)}
    for kind in ["primary", "nonendpoint"]:
        cols = [cat_col(c) for c in CATEGORIES if c not in (NOT_PRIMARY if kind == "primary" else NOT_NONENDPOINT)]
        c_rate = ctrl[cols].values.sum() / ctrl.y_started.sum()
        e_rates = exp[cols].sum(axis=1) / exp.y_started
        diff = float(e_rates.max() - c_rate) if len(exp) else np.nan
        out[f"y_control_{kind}_rate"] = float(c_rate)
        out[f"y_differential_{kind}"] = diff
        out[f"y_differential_{kind}_absmax"] = float((e_rates - c_rate).abs().max()) if len(exp) else np.nan
        out[f"y_flag_{kind}_5pp"] = bool(abs(diff) >= 0.05) if not np.isnan(diff) else None
        out[f"y_flag_{kind}_10pp"] = bool(abs(diff) >= 0.10) if not np.isnan(diff) else None
    total = a[[cat_col(c) for c in CATEGORIES]].values.sum()
    out["y_sponsor_admin_share"] = float(a[cat_col("SPONSOR_ADMINISTRATIVE")].sum() / total) if total else np.nan
    return out


def main():
    trials, arms, fg, periods, started, dw, mapping = load()
    reason_cat = dict(zip(mapping.raw_reason, mapping.category))
    pool = eligible_pool_ids()
    funnel = {"eligible_pool": len(pool)}

    g_arms = {k: v for k, v in arms[arms.nct_id.isin(pool)].groupby("nct_id")}
    g_fg = {k: v for k, v in fg[fg.nct_id.isin(pool)].groupby("nct_id")}
    g_per = {k: v for k, v in periods[periods.nct_id.isin(pool)].groupby("nct_id")}
    g_st = {k: v for k, v in started[started.nct_id.isin(pool)].groupby("nct_id")}
    g_dw = {k: v for k, v in dw[dw.nct_id.isin(pool)].groupby("nct_id")}
    empty_st = started.iloc[:0]

    built, drops, period_info = [], {}, []
    for nct in tqdm(sorted(pool), desc="trials"):
        res, why = build_trial(nct, g_arms[nct].sort_values("arm_idx"), g_fg[nct], g_per[nct],
                               g_st.get(nct, empty_st), g_dw[nct], reason_cat)
        if why:
            drops[why] = drops.get(why, 0) + 1
            continue
        period_info.append({"nct_id": nct, "meta_n_periods": res["n_periods"],
                            "meta_n_periods_excluded": res["n_periods_excluded"]})
        built.append(res["arms"])
    funnel.update({f"dropped_{k}": v for k, v in drops.items()})
    all_arms = pd.concat(built, ignore_index=True)
    pinfo = pd.DataFrame(period_info)
    funnel["built"] = all_arms.nct_id.nunique()

    # Attach protocol arm features (unmatched flow groups have arm_idx -1 -> NaN features).
    all_arms = all_arms.merge(arms[["nct_id", "arm_idx"] + ARM_FEATURES], on=["nct_id", "arm_idx"], how="left")
    cat_cols = [cat_col(c) for c in CATEGORIES]
    prim = [cat_col(c) for c in CATEGORIES if c not in NOT_PRIMARY]
    nonend = [cat_col(c) for c in CATEGORIES if c not in NOT_NONENDPOINT]
    all_arms["y_withdrawals_total"] = all_arms[cat_cols].sum(axis=1)
    all_arms["y_primary_attrition_rate"] = all_arms[prim].sum(axis=1) / all_arms.y_started.replace(0, np.nan)
    all_arms["y_nonendpoint_attrition_rate"] = all_arms[nonend].sum(axis=1) / all_arms.y_started.replace(0, np.nan)

    tfeat = trials.set_index("nct_id")[TRIAL_FEATURES]
    per_trial = all_arms.groupby("nct_id").agg(
        min_score=("meta_match_score", "min"), min_started=("y_started", "min"),
        max_rate=("y_primary_attrition_rate", "max"),
        has_control=("arm_type", lambda s: s.isin(CONTROL_PRIORITY).any()))

    sets, stats_rows = {}, []
    for name, thr in THRESHOLDS.items():
        steps = [("matched_all_flow_groups", per_trial.min_score >= thr),
                 ("matched_control_arm", per_trial.has_control),
                 (f"all_arms_started_ge{MIN_STARTED}", per_trial.min_started >= MIN_STARTED),
                 ("rates_le_1", per_trial.max_rate <= 1),
                 ("known_start_year", tfeat.loc[per_trial.index, "start_year"].notna())]
        keep = pd.Series(True, index=per_trial.index)
        for step, mask in steps:
            keep &= mask.reindex(per_trial.index).fillna(False).astype(bool)
            funnel[f"{name}_after_{step}"] = int(keep.sum())
        ids = keep.index[keep]
        a = all_arms[all_arms.nct_id.isin(ids)].copy()
        t = pd.DataFrame([{"nct_id": k, **trial_labels(v)} for k, v in a.groupby("nct_id")])
        t = t.merge(tfeat, left_on="nct_id", right_index=True).merge(pinfo, on="nct_id")
        t["start_year"] = t.start_year.astype(int)
        t["split"] = t.start_year.map(split_of)
        a = a.merge(t[["nct_id", "split"]], on="nct_id")
        id_cols = ["nct_id", "arm_idx", "split"]
        a = a[id_cols + ARM_FEATURES + [c for c in a.columns if c.startswith(("y_", "meta_"))]]
        t = t[["nct_id", "split"] + TRIAL_FEATURES + [c for c in t.columns if c.startswith(("y_", "meta_"))]]
        a = a.sort_values(["nct_id", "arm_idx"]).reset_index(drop=True)
        t = t.sort_values("nct_id").reset_index(drop=True)
        sets[name] = (a, t)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, (a, t) in sets.items():
        suffix = "" if name == PRIMARY else f"_{name}"
        a.to_parquet(PROCESSED / f"arms{suffix}.parquet", index=False)
        t.to_parquet(PROCESSED / f"trials{suffix}.parquet", index=False)

    splits = {"seed": SEED, "rule": SPLIT_RULE, "primary_set": PRIMARY,
              "sets": {name: {s: sorted(t.nct_id[t.split == s]) for s in ["train", "val", "test"]}
                       for name, (a, t) in sets.items()}}
    (PROCESSED / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")

    write_stats(sets, funnel, pinfo)


def write_stats(sets, funnel, pinfo):
    rows = []

    def add(section, key, value):
        rows.append({"section": section, "key": str(key), "value": value})

    for k, v in funnel.items():
        add("funnel", k, v)
    a, t = sets[PRIMARY]
    add("periods", "trials_with_periods_excluded_primary_set", int((t.meta_n_periods_excluded > 0).sum()))
    add("periods", "trials_with_periods_excluded_all_built", int((pinfo.meta_n_periods_excluded > 0).sum()))
    add("periods", "trials_multi_period_primary_set", int((t.meta_n_periods > 1).sum()))

    for name, (sa, st) in sets.items():
        add("matched_sets", f"{name}_trials", len(st))
        add("matched_sets", f"{name}_arms", len(sa))

    add("overall", "trials", len(t))
    add("overall", "arms", len(a))
    for s in ["train", "val", "test"]:
        add("split_trials", s, int((t.split == s).sum()))
        add("split_arms", s, int((a.split == s).sum()))
    for ph, n in t.phase.value_counts().items():
        add("phase_trials", ph, int(n))
    add("design", "exp_arm_fallback_trials", int(t.meta_exp_fallback.sum()))
    add("design", "trials_without_differential_no_comparison_arm", int(t.y_differential_primary.isna().sum()))
    for ct, n in t.meta_control_type.value_counts().items():
        add("control_type", ct, int(n))

    for kind in ["primary", "nonendpoint"]:
        r = a[f"y_{kind}_attrition_rate"]
        for s in ["all", "train", "val", "test"]:
            rr = r if s == "all" else r[a.split == s]
            add(f"arm_{kind}_rate", f"{s}_mean", float(rr.mean()))
            add(f"arm_{kind}_rate", f"{s}_median", float(rr.median()))
        d = t[f"y_differential_{kind}"]
        add(f"trial_differential_{kind}", "mean", float(d.mean()))
        add(f"trial_differential_{kind}", "median", float(d.median()))
        add(f"trial_differential_{kind}", "abs_mean", float(d.abs().mean()))
        for pp in ["5pp", "10pp"]:
            f = t[f"y_flag_{kind}_{pp}"].astype(float)
            add(f"flag_prevalence_{kind}", f"{pp}_all", float(f.mean()))
            for s in ["train", "val", "test"]:
                add(f"flag_prevalence_{kind}", f"{pp}_{s}", float(f[t.split == s].mean()))
    add("trial_sponsor_admin_share", "mean", float(t.y_sponsor_admin_share.mean()))
    add("trial_sponsor_admin_share", "trials_gt0", int((t.y_sponsor_admin_share > 0).sum()))

    cat_cols = [cat_col(c) for c in CATEGORIES]
    tot = a[cat_cols].values.sum()
    for c in CATEGORIES:
        n = int(a[cat_col(c)].sum())
        add("category_subjects", c, n)
        add("category_share", c, n / tot)
        add("category_arms_with_any", c, int((a[cat_col(c)] > 0).sum()))

    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "dataset_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
