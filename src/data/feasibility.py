"""Feasibility analysis of the raw ClinicalTrials.gov download.

Reads data/raw/page_*.json.gz and reports:
  - studies with results; randomized with >=2 arms
  - studies with participant-flow dropWithdraws reasons per flow group
  - studies with >=1 control arm
  - usable trials (all of the above) by phase and start year
  - top 100 raw withdrawal reason strings; number of unique reason strings
  - flow-group -> armGroup match rate (rapidfuzz + one-to-one assignment)

Outputs:
  results/tables/feasibility.csv        long-format table (section, key, value)
  results/feasibility_summary.md        human-readable summary
  data/interim/feasibility_studies.parquet  per-study flags (local only)

Usage:
    python -m src.data.feasibility [--raw data/raw]
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

CONTROL_TYPES = {"PLACEBO_COMPARATOR", "ACTIVE_COMPARATOR", "NO_INTERVENTION", "SHAM_COMPARATOR"}
MATCH_THRESHOLDS = [60, 70, 80, 90]
TOP_N_REASONS = 100


def norm_text(s):
    return " ".join((s or "").lower().split())


def match_score(fg: dict, arm: dict) -> float:
    """Best rapidfuzz similarity between a flow group and an arm group (0-100)."""
    ft, fd = norm_text(fg.get("title")), norm_text(fg.get("description"))
    al, ad = norm_text(arm.get("label")), norm_text(arm.get("description"))
    ai = norm_text(" ".join(arm.get("interventionNames", [])))
    scores = [fuzz.token_set_ratio(ft, al)]
    if fd and ad:
        scores.append(fuzz.token_set_ratio(fd, ad))
    scores.append(fuzz.token_set_ratio(f"{ft} {fd}", f"{al} {ad} {ai}"))
    return max(scores)


def match_groups(flow_groups: list, arm_groups: list):
    """One-to-one assignment maximizing similarity; returns assigned scores per flow group."""
    if not flow_groups or not arm_groups:
        return []
    S = np.array([[match_score(f, a) for a in arm_groups] for f in flow_groups])
    rows, cols = linear_sum_assignment(-S)
    assigned = np.zeros(len(flow_groups))  # unassigned flow groups (more FGs than arms) score 0
    assigned[rows] = S[rows, cols]
    return assigned.tolist()


def start_year(status_module: dict):
    d = (status_module.get("startDateStruct") or {}).get("date")
    try:
        return int(d[:4]) if d else None
    except ValueError:
        return None


def analyse_study(study: dict, reason_counter: Counter, reason_subjects: Counter) -> dict:
    ps = study.get("protocolSection", {})
    rs = study.get("resultsSection", {})
    nct = ps.get("identificationModule", {}).get("nctId")
    design = ps.get("designModule", {})
    arms = ps.get("armsInterventionsModule", {}).get("armGroups", []) or []
    allocation = (design.get("designInfo") or {}).get("allocation")
    phases = design.get("phases") or []

    pf = rs.get("participantFlowModule") or {}
    flow_groups = pf.get("groups", []) or []
    groups_with_reasons = set()
    n_dw_entries = 0
    for period in pf.get("periods", []) or []:
        for dw in period.get("dropWithdraws", []) or []:
            n_dw_entries += 1
            rtype = (dw.get("type") or "").strip()
            for r in dw.get("reasons", []) or []:
                gid = r.get("groupId")
                if gid:
                    groups_with_reasons.add(gid)
                try:
                    n = int(r.get("numSubjects") or 0)
                except ValueError:
                    n = 0
                reason_subjects[rtype] += n
            reason_counter[rtype] += 1

    arm_types = [a.get("type") for a in arms]
    rec = {
        "nct_id": nct,
        "has_results": bool(study.get("hasResults")),
        "allocation": allocation,
        "n_arms": len(arms),
        "randomized_2plus_arms": allocation == "RANDOMIZED" and len(arms) >= 2,
        "has_flow_module": bool(pf),
        "n_flow_groups": len(flow_groups),
        "n_dropwithdraw_entries": n_dw_entries,
        "n_flow_groups_with_reasons": len(groups_with_reasons),
        "has_dropwithdraw_reasons": len(groups_with_reasons) > 0,
        "has_control_arm": any(t in CONTROL_TYPES for t in arm_types),
        "phase": "|".join(phases) if phases else "NA",
        "start_year": start_year(ps.get("statusModule", {})),
    }
    rec["usable"] = rec["has_results"] and rec["randomized_2plus_arms"] and rec[
        "has_dropwithdraw_reasons"] and rec["has_control_arm"]
    rec["_flow_groups"] = flow_groups if rec["usable"] else None
    rec["_arms"] = arms if rec["usable"] else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--summary", default="results/feasibility_summary.md")
    ap.add_argument("--interim", default="data/interim")
    args = ap.parse_args()

    pages = sorted(Path(args.raw).glob("page_*.json.gz"))
    if not pages:
        raise SystemExit(f"No pages found in {args.raw}")

    reason_counter, reason_subjects = Counter(), Counter()
    recs = []
    for p in tqdm(pages, desc="pages"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            page = json.load(f)
        for s in page.get("studies", []):
            recs.append(analyse_study(s, reason_counter, reason_subjects))

    df = pd.DataFrame(recs)
    dup = int(df["nct_id"].duplicated().sum())
    df = df.drop_duplicates("nct_id").reset_index(drop=True)

    # Flow-group -> armGroup matching on usable trials.
    group_scores, trial_rows = [], []
    usable = df[df["usable"]]
    for _, r in tqdm(usable.iterrows(), total=len(usable), desc="matching"):
        sc = match_groups(r["_flow_groups"], r["_arms"])
        group_scores.extend(sc)
        trial_rows.append({
            "nct_id": r["nct_id"],
            "same_count": len(r["_flow_groups"]) == len(r["_arms"]),
            "min_score": min(sc) if sc else 0.0,
        })
    tdf = pd.DataFrame(trial_rows)
    gs = np.array(group_scores)

    rows = []

    def add(section, key, value):
        rows.append({"section": section, "key": str(key), "value": value})

    n_total = len(df)
    add("counts", "raw_pages", len(pages))
    add("counts", "duplicate_nct_ids_dropped", dup)
    add("counts", "studies_with_results", int(df["has_results"].sum()))
    add("counts", "randomized_2plus_arms", int(df["randomized_2plus_arms"].sum()))
    add("counts", "has_participant_flow_module", int(df["has_flow_module"].sum()))
    add("counts", "has_dropwithdraw_reasons_per_group", int(df["has_dropwithdraw_reasons"].sum()))
    add("counts", "has_dropwithdraw_reasons_2plus_groups", int((df["n_flow_groups_with_reasons"] >= 2).sum()))
    add("counts", "has_control_arm", int(df["has_control_arm"].sum()))
    add("counts", "usable_trials", int(df["usable"].sum()))

    for ph, n in usable["phase"].value_counts().sort_index().items():
        add("usable_by_phase", ph, int(n))
    yrs = usable["start_year"].fillna(-1).astype(int).value_counts().sort_index()
    for y, n in yrs.items():
        add("usable_by_start_year", "NA" if y == -1 else y, int(n))

    add("reasons", "n_unique_raw_strings", len(reason_counter))
    add("reasons", "n_unique_lowercased_stripped",
        len({norm_text(k) for k in reason_counter}))
    add("reasons", "n_dropwithdraw_entries_total", int(sum(reason_counter.values())))
    for i, (reason, n) in enumerate(reason_counter.most_common(TOP_N_REASONS), 1):
        rows.append({"section": "top_reasons", "key": reason, "value": n,
                     "rank": i, "subjects": reason_subjects[reason]})

    add("matching", "usable_trials_evaluated", len(tdf))
    add("matching", "flow_groups_evaluated", len(gs))
    add("matching", "trials_same_n_flowgroups_and_arms", int(tdf["same_count"].sum()) if len(tdf) else 0)
    add("matching", "median_group_score", float(np.median(gs)) if len(gs) else float("nan"))
    for t in MATCH_THRESHOLDS:
        add("matching", f"group_match_rate_ge{t}", float((gs >= t).mean()) if len(gs) else float("nan"))
        add("matching", f"trial_all_groups_matched_ge{t}",
            float((tdf["min_score"] >= t).mean()) if len(tdf) else float("nan"))

    out = pd.DataFrame(rows)
    Path(args.tables).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.tables) / "feasibility.csv", index=False)

    Path(args.interim).mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_flow_groups", "_arms"]).merge(tdf, on="nct_id", how="left").to_parquet(
        Path(args.interim) / "feasibility_studies.parquet", index=False)

    write_summary(out, Path(args.summary))
    print(Path(args.summary).read_text(encoding="utf-8"))


def write_summary(out: pd.DataFrame, path: Path):
    def sect(name):
        return out[out["section"] == name]

    def val(section, key):
        return sect(section).set_index("key").loc[key, "value"]

    c = sect("counts").set_index("key")["value"]
    total = c["studies_with_results"]
    L = ["# Feasibility summary", "",
         "Source: ClinicalTrials.gov API v2, INTERVENTIONAL studies with results.", "",
         "## Funnel", "", "| Criterion | Studies | % of with-results |", "|---|---:|---:|"]
    labels = {
        "studies_with_results": "Studies with results",
        "randomized_2plus_arms": "RANDOMIZED with >=2 arms",
        "has_participant_flow_module": "Has participantFlowModule",
        "has_dropwithdraw_reasons_per_group": "Has dropWithdraws reasons per flow group (>=1 group)",
        "has_dropwithdraw_reasons_2plus_groups": "Has dropWithdraws reasons for >=2 flow groups",
        "has_control_arm": "Has >=1 control arm (placebo/active/no-intervention/sham)",
        "usable_trials": "**Usable (randomized >=2 arms AND reasons AND control arm)**",
    }
    for k, lab in labels.items():
        L.append(f"| {lab} | {int(c[k]):,} | {100 * c[k] / total:.1f}% |")
    L += ["", f"Raw pages: {int(c['raw_pages'])}; duplicate NCT IDs dropped: {int(c['duplicate_nct_ids_dropped'])}.", ""]

    L += ["## Usable trials by phase", "", "| Phase | Trials |", "|---|---:|"]
    for _, r in sect("usable_by_phase").sort_values("value", ascending=False).iterrows():
        L.append(f"| {r['key']} | {int(r['value']):,} |")

    L += ["", "## Usable trials by start year", "", "| Start year | Trials |", "|---|---:|"]
    for _, r in sect("usable_by_start_year").iterrows():
        L.append(f"| {r['key']} | {int(r['value']):,} |")

    L += ["", "## Withdrawal reasons", "",
          f"- Unique raw reason strings: {int(val('reasons', 'n_unique_raw_strings')):,}",
          f"- Unique after lowercasing/whitespace normalization: {int(val('reasons', 'n_unique_lowercased_stripped')):,}",
          f"- Total dropWithdraws entries: {int(val('reasons', 'n_dropwithdraw_entries_total')):,}", "",
          "Top 100 raw reason strings (frequency = number of dropWithdraws entries; subjects = summed numSubjects):", "",
          "| Rank | Reason | Frequency | Subjects |", "|---:|---|---:|---:|"]
    for _, r in sect("top_reasons").iterrows():
        reason = str(r["key"]).replace("|", "\\|")
        L.append(f"| {int(r['rank'])} | {reason} | {int(r['value']):,} | {int(r['subjects']):,} |")

    m = sect("matching").set_index("key")["value"]
    L += ["", "## Flow-group to armGroup matching (usable trials)", "",
          "Score = max rapidfuzz token_set_ratio over (flow title vs arm label), (flow description vs arm "
          "description), and (title+description vs label+description+interventions); one-to-one assignment "
          "via Hungarian algorithm. Flow groups left unassigned (more flow groups than arms) score 0.", "",
          f"- Trials evaluated: {int(m['usable_trials_evaluated']):,}; flow groups evaluated: {int(m['flow_groups_evaluated']):,}",
          f"- Trials with same number of flow groups and arm groups: {int(m['trials_same_n_flowgroups_and_arms']):,} "
          f"({100 * m['trials_same_n_flowgroups_and_arms'] / max(m['usable_trials_evaluated'], 1):.1f}%)",
          f"- Median assigned group score: {m['median_group_score']:.1f}", "",
          "| Threshold | Flow-group match rate | Trials with all groups matched |", "|---:|---:|---:|"]
    for t in MATCH_THRESHOLDS:
        L.append(f"| >= {t} | {100 * m[f'group_match_rate_ge{t}']:.1f}% | {100 * m[f'trial_all_groups_matched_ge{t}']:.1f}% |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
