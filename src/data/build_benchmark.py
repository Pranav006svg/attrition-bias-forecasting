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
    python -m src.data.build_benchmark [--reuse]
    --reuse  skip the per-trial build and reuse data/interim/built_*.{parquet,json}
             (valid while extract outputs and reason_mapping.csv are unchanged)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
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
PRIMARY_CATS = [c for c in CATEGORIES if c not in NOT_PRIMARY]
FLAG_THRESHOLDS = [5, 10]  # percentage points; flag also requires Fisher p < 0.05
# Reason groups over the primary-attrition categories (they partition PRIMARY_CATS).
REASON_GROUPS = {
    "ae": ["ADVERSE_EVENT"],
    "efficacy": ["LACK_OF_EFFICACY", "DISEASE_PROGRESSION"],
    "subject": ["WITHDRAWAL_BY_SUBJECT", "LOST_TO_FOLLOW_UP"],
    "othergroup": ["DEATH", "PHYSICIAN_DECISION", "PROTOCOL_VIOLATION", "PRE_TREATMENT_EXCLUSION", "OTHER_UNSPECIFIED"],
}
GROUP_FLAG_TARGETS = ["ae", "efficacy"]
GROUP_FLAG_THRESHOLD = 5
RELIABLE_MIN_STARTED = 100

TRIAL_FEATURES = ["phase", "allocation", "masking", "intervention_model", "primary_purpose", "n_arms",
                  "sponsor_class", "mesh_branches", "healthy_volunteers", "sex", "min_age_years",
                  "max_age_years", "start_year", "eligibility_text", "brief_summary", "conditions"]
ARM_FEATURES = ["arm_type", "arm_label", "arm_description", "intervention_types", "intervention_names"]


def split_of(year: int) -> str:
    return "train" if year < 2015 else ("val" if year <= 2016 else "test")


def cat_col(c: str) -> str:
    return f"y_n_{c.lower()}"


def prop_col(c: str) -> str:
    return f"y_prop_{c.lower()}"


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


def arm_roles(arm_types: pd.Series):
    """(control_type, exp_fallback, role per arm) from the protocol arm types of a trial.

    Control = all arms of the highest-priority control type present (pooled).
    Experimental = EXPERIMENTAL/OTHER arms; if none, every non-control-type arm
    (fallback). Remaining arms (e.g. an active comparator next to a placebo) = other.
    """
    types = set(arm_types)
    ctrl_type = next((c for c in CONTROL_PRIORITY if c in types), None)
    is_exp = arm_types.isin(EXPERIMENTAL_TYPES)
    fallback = not is_exp.any()
    if fallback:
        is_exp = arm_types != ctrl_type
    role = np.where(arm_types == ctrl_type, "control", np.where(is_exp, "experimental", "other"))
    return ctrl_type, fallback, role


def trial_labels(a: pd.DataFrame) -> dict:
    """Trial-level differential attrition from labeled arm rows (a has meta_role)."""
    ctrl = a.meta_role.values == "control"
    exp = a.meta_role.values == "experimental"
    has_diff = bool(exp.any())
    out = {"meta_n_exp_arms": int(exp.sum()), "meta_n_control_arms": int(ctrl.sum()),
           "y_has_differential": has_diff}
    for kind in ["primary", "nonendpoint"]:
        cols = [cat_col(c) for c in CATEGORIES if c not in (NOT_PRIMARY if kind == "primary" else NOT_NONENDPOINT)]
        w = a[cols].sum(axis=1).values
        s = a.y_started.values
        c_w, c_s = w[ctrl].sum(), s[ctrl].sum()
        c_rate = c_w / c_s
        out[f"y_control_{kind}_rate"] = float(c_rate)
        if not has_diff:
            out[f"y_differential_{kind}"] = np.nan
            out[f"y_differential_{kind}_absmax"] = np.nan
            out[f"meta_fisher_p_{kind}"] = np.nan
            for thr in FLAG_THRESHOLDS:
                out[f"y_flag_{kind}_{thr}_sig"] = None
            continue
        e_rates = w[exp] / s[exp]
        diffs = e_rates - c_rate
        j = int(np.abs(diffs).argmax())
        absmax = float(abs(diffs[j]))
        ew, es = int(w[exp][j]), int(s[exp][j])
        p = fisher_exact([[ew, es - ew], [int(c_w), int(c_s - c_w)]])[1]
        out[f"y_differential_{kind}"] = float(e_rates.max() - c_rate)
        out[f"y_differential_{kind}_absmax"] = absmax
        out[f"meta_fisher_p_{kind}"] = float(p)
        for thr in FLAG_THRESHOLDS:
            out[f"y_flag_{kind}_{thr}_sig"] = bool(absmax >= thr / 100 and p < 0.05)
    # Reason-group differentials (experimental arm with max |group diff| vs pooled control).
    for gname in GROUP_FLAG_TARGETS:
        w = a[[cat_col(c) for c in REASON_GROUPS[gname]]].sum(axis=1).values
        s = a.y_started.values
        if not has_diff:
            out[f"y_differential_{gname}_absmax"] = np.nan
            out[f"meta_fisher_p_{gname}"] = np.nan
            out[f"y_flag_{gname}_{GROUP_FLAG_THRESHOLD}_sig"] = None
            continue
        c_w, c_s = w[ctrl].sum(), s[ctrl].sum()
        diffs = w[exp] / s[exp] - c_w / c_s
        j = int(np.abs(diffs).argmax())
        ew, es = int(w[exp][j]), int(s[exp][j])
        p = fisher_exact([[ew, es - ew], [int(c_w), int(c_s - c_w)]])[1]
        out[f"y_differential_{gname}_absmax"] = float(abs(diffs[j]))
        out[f"meta_fisher_p_{gname}"] = float(p)
        out[f"y_flag_{gname}_{GROUP_FLAG_THRESHOLD}_sig"] = bool(abs(diffs[j]) >= GROUP_FLAG_THRESHOLD / 100 and p < 0.05)
    s = a.y_started.values
    out["meta_reliable"] = bool(has_diff and s[ctrl].sum() >= RELIABLE_MIN_STARTED
                                and (s[exp] >= RELIABLE_MIN_STARTED).all())
    total = a[[cat_col(c) for c in CATEGORIES]].values.sum()
    out["y_sponsor_admin_share"] = float(a[cat_col("SPONSOR_ADMINISTRATIVE")].sum() / total) if total else np.nan
    return out


def build_all(trials, arms, fg, periods, started, dw, mapping):
    """Per-trial arm rows for the whole eligible pool (slow; cached by main)."""
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
    funnel["built"] = int(all_arms.nct_id.nunique())
    return all_arms, pd.DataFrame(period_info), funnel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    trials, arms, fg, periods, started, dw, mapping = load()
    cache = [INTERIM / "built_arms.parquet", INTERIM / "built_periods.parquet", INTERIM / "built_funnel.json"]
    if args.reuse and all(p.exists() for p in cache):
        all_arms, pinfo = pd.read_parquet(cache[0]), pd.read_parquet(cache[1])
        funnel = json.loads(cache[2].read_text(encoding="utf-8"))
        print("Reusing cached per-trial build.")
    else:
        all_arms, pinfo, funnel = build_all(trials, arms, fg, periods, started, dw, mapping)
        all_arms.to_parquet(cache[0], index=False)
        pinfo.to_parquet(cache[1], index=False)
        cache[2].write_text(json.dumps(funnel), encoding="utf-8")

    # Attach protocol arm features (unmatched flow groups have arm_idx -1 -> NaN features).
    all_arms = all_arms.merge(arms[["nct_id", "arm_idx"] + ARM_FEATURES], on=["nct_id", "arm_idx"], how="left")
    cat_cols = [cat_col(c) for c in CATEGORIES]
    prim = [cat_col(c) for c in PRIMARY_CATS]
    nonend = [cat_col(c) for c in CATEGORIES if c not in NOT_NONENDPOINT]
    all_arms["y_withdrawals_total"] = all_arms[cat_cols].sum(axis=1)
    all_arms["y_withdrawals_primary"] = all_arms[prim].sum(axis=1)
    all_arms["y_primary_attrition_rate"] = all_arms[prim].sum(axis=1) / all_arms.y_started.replace(0, np.nan)
    all_arms["y_nonendpoint_attrition_rate"] = all_arms[nonend].sum(axis=1) / all_arms.y_started.replace(0, np.nan)
    for gname, cats in REASON_GROUPS.items():
        all_arms[f"y_{gname}_rate"] = all_arms[[cat_col(c) for c in cats]].sum(axis=1) / all_arms.y_started
    # Reason composition over primary-attrition categories (arms with >= 1 such withdrawal).
    all_arms["y_has_composition"] = all_arms.y_withdrawals_primary >= 1
    denom = all_arms.y_withdrawals_primary.where(all_arms.y_has_composition)
    for c in PRIMARY_CATS:
        all_arms[prop_col(c)] = all_arms[cat_col(c)] / denom

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
        roles, tmeta = [], []
        for k, v in a.groupby("nct_id", sort=False):
            ctrl_type, fallback, role = arm_roles(v.arm_type)
            roles.append(pd.Series(role, index=v.index))
            tmeta.append({"nct_id": k, "meta_control_type": ctrl_type, "meta_exp_fallback": bool(fallback)})
        a["meta_role"] = pd.concat(roles)
        t = pd.DataFrame([{"nct_id": k, **trial_labels(v)} for k, v in a.groupby("nct_id")])
        t = t.merge(pd.DataFrame(tmeta), on="nct_id")
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
    splits = ["train", "val", "test"]

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
        add("matched_sets", f"{name}_differential_trials", int(st.y_has_differential.sum()))

    add("overall", "trials", len(t))
    add("overall", "arms", len(a))
    for s in splits:
        add("split_trials", s, int((t.split == s).sum()))
        add("split_arms", s, int((a.split == s).sum()))
    for ph, n in t.phase.value_counts().items():
        add("phase_trials", ph, int(n))
    add("features", "mesh_branch_unknown_trials", int((t.mesh_branches == "UNKNOWN").sum()))

    # Arm roles and differential-task eligibility.
    for ct, n in t.meta_control_type.value_counts().items():
        add("control_type", ct, int(n))
    for r, n in a.meta_role.value_counts().items():
        add("arm_roles", r, int(n))
    for s in ["all"] + splits:
        ts = t if s == "all" else t[t.split == s]
        add("differential_task", f"{s}_trials_with_differential", int(ts.y_has_differential.sum()))
        add("differential_task", f"{s}_trials_without_differential", int((~ts.y_has_differential).sum()))
        add("differential_task", f"{s}_exp_side_is_noncontrol_arms_fallback",
            int((ts.meta_exp_fallback & ts.y_has_differential).sum()))

    for kind in ["primary", "nonendpoint"]:
        r = a[f"y_{kind}_attrition_rate"]
        for s in ["all"] + splits:
            rr = r if s == "all" else r[a.split == s]
            add(f"arm_{kind}_rate", f"{s}_mean", float(rr.mean()))
            add(f"arm_{kind}_rate", f"{s}_median", float(rr.median()))
        td = t[t.y_has_differential]
        for col, lab in [(f"y_differential_{kind}", "signed"), (f"y_differential_{kind}_absmax", "absmax")]:
            add(f"trial_differential_{kind}", f"{lab}_mean", float(td[col].mean()))
            add(f"trial_differential_{kind}", f"{lab}_median", float(td[col].median()))
        for thr in FLAG_THRESHOLDS:
            f = td[f"y_flag_{kind}_{thr}_sig"].astype(float)
            add(f"flag_prevalence_{kind}", f"{thr}_sig_all", float(f.mean()))
            for s in splits:
                add(f"flag_prevalence_{kind}", f"{thr}_sig_{s}", float(f[td.split == s].mean()))
                add(f"flag_positives_{kind}", f"{thr}_sig_{s}", int(f[td.split == s].sum()))
    for s in ["all"] + splits:
        ts = t[t.y_has_differential] if s == "all" else t[t.y_has_differential & (t.split == s)]
        add("reliable_subset", f"{s}_trials", int(ts.meta_reliable.sum()))
        for gname in GROUP_FLAG_TARGETS:
            f = ts[f"y_flag_{gname}_{GROUP_FLAG_THRESHOLD}_sig"].astype(float)
            add(f"flag_prevalence_{gname}", f"{GROUP_FLAG_THRESHOLD}_sig_{s}", float(f.mean()))
            add(f"flag_positives_{gname}", f"{GROUP_FLAG_THRESHOLD}_sig_{s}", int(f.sum()))
    add("trial_sponsor_admin_share", "mean", float(t.y_sponsor_admin_share.mean()))
    add("trial_sponsor_admin_share", "trials_gt0", int((t.y_sponsor_admin_share > 0).sum()))

    # Reason composition (per-arm proportions over primary categories, equal arm weight).
    comp = a[a.y_has_composition]
    add("composition", "arms_with_composition_all", len(comp))
    for s in splits:
        add("composition", f"arms_with_composition_{s}", int((comp.split == s).sum()))
    for c in PRIMARY_CATS:
        add("composition_mean_prop", c, float(comp[prop_col(c)].mean()))

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
