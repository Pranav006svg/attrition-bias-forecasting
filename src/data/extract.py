"""One pass over the raw API pages -> flat interim tables.

Protocol (pre-enrollment) fields and results (label-only) fields are written to
separate tables so feature tables can never pick up results fields by accident.

Outputs (data/interim/):
  protocol_trials.parquet   one row per study, pre-enrollment fields + eligibility-pool flags
  protocol_arms.parquet     one row per protocol arm group
  flow_groups.parquet       participant-flow groups (results; used for labels only)
  flow_periods.parquet      participant-flow periods (title, index)
  flow_started.parquet      STARTED milestone per period x group
  flow_dropwithdraws.parquet  withdrawal reason counts per period x group

Usage:
    python -m src.data.extract [--raw data/raw] [--out data/interim]
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CONTROL_TYPES = {"PLACEBO_COMPARATOR", "ACTIVE_COMPARATOR", "NO_INTERVENTION", "SHAM_COMPARATOR"}

# MeSH disease-category headings (C01-C26), F01 and F03, matched by descriptor name
# against each study's condition meshes + ancestors.
MESH_BRANCHES = {
    "Infections": "C01_INFECTIONS",
    "Neoplasms": "C04_NEOPLASMS",
    "Musculoskeletal Diseases": "C05_MUSCULOSKELETAL",
    "Digestive System Diseases": "C06_DIGESTIVE",
    "Stomatognathic Diseases": "C07_STOMATOGNATHIC",
    "Respiratory Tract Diseases": "C08_RESPIRATORY",
    "Otorhinolaryngologic Diseases": "C09_OTORHINOLARYNGOLOGIC",
    "Nervous System Diseases": "C10_NERVOUS_SYSTEM",
    "Eye Diseases": "C11_EYE",
    "Urogenital Diseases": "C12_UROGENITAL",
    "Male Urogenital Diseases": "C12_UROGENITAL",
    "Female Urogenital Diseases and Pregnancy Complications": "C13_FEMALE_UROGENITAL_PREGNANCY",
    "Female Urogenital Diseases": "C13_FEMALE_UROGENITAL_PREGNANCY",
    "Pregnancy Complications": "C13_FEMALE_UROGENITAL_PREGNANCY",
    "Cardiovascular Diseases": "C14_CARDIOVASCULAR",
    "Hemic and Lymphatic Diseases": "C15_HEMIC_LYMPHATIC",
    "Congenital, Hereditary, and Neonatal Diseases and Abnormalities": "C16_CONGENITAL_NEONATAL",
    "Skin and Connective Tissue Diseases": "C17_SKIN_CONNECTIVE",
    "Nutritional and Metabolic Diseases": "C18_NUTRITIONAL_METABOLIC",
    "Endocrine System Diseases": "C19_ENDOCRINE",
    "Immune System Diseases": "C20_IMMUNE",
    "Disorders of Environmental Origin": "C21_ENVIRONMENTAL",
    "Animal Diseases": "C22_ANIMAL",
    "Pathological Conditions, Signs and Symptoms": "C23_PATHOLOGICAL_SIGNS_SYMPTOMS",
    "Occupational Diseases": "C24_OCCUPATIONAL",
    "Chemically-Induced Disorders": "C25_CHEMICALLY_INDUCED",
    "Wounds and Injuries": "C26_WOUNDS_INJURIES",
    "Mental Disorders": "F03_MENTAL_DISORDERS",
    "Behavior and Behavior Mechanisms": "F01_BEHAVIOR",
    "Behavior": "F01_BEHAVIOR",
}

_AGE_UNITS = {"year": 1.0, "month": 1 / 12, "week": 7 / 365.25, "day": 1 / 365.25,
              "hour": 1 / (24 * 365.25), "minute": 1 / (60 * 24 * 365.25)}


def parse_age_years(s):
    """'18 Years' -> 18.0, '6 Months' -> 0.5, None/'N/A' -> None."""
    if not s:
        return None
    m = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", s)
    if not m:
        return None
    unit = m.group(2).lower().rstrip("s")
    if unit not in _AGE_UNITS:
        return None
    return float(m.group(1)) * _AGE_UNITS[unit]


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def start_year(status_module):
    d = (status_module.get("startDateStruct") or {}).get("date")
    return to_int(d[:4]) if d else None


def intervention_types(names):
    """'Drug: Aspirin' -> 'DRUG'; 'Dietary Supplement: X' -> 'DIETARY_SUPPLEMENT'."""
    out = []
    for n in names:
        if ":" in n:
            t = n.split(":", 1)[0].strip().upper().replace(" ", "_")
            if t not in out:
                out.append(t)
    return out


def extract_study(s, tables):
    ps = s.get("protocolSection", {})
    nct = ps.get("identificationModule", {}).get("nctId")
    design = ps.get("designModule", {})
    dinfo = design.get("designInfo") or {}
    elig = ps.get("eligibilityModule", {})
    arms = ps.get("armsInterventionsModule", {}).get("armGroups", []) or []
    cbm = s.get("derivedSection", {}).get("conditionBrowseModule", {}) or {}
    mesh_terms = {m.get("term") for m in (cbm.get("meshes", []) or []) + (cbm.get("ancestors", []) or [])}
    branches = sorted({MESH_BRANCHES[t] for t in mesh_terms if t in MESH_BRANCHES})

    pf = (s.get("resultsSection") or {}).get("participantFlowModule") or {}
    has_reasons = any(r.get("groupId")
                      for p in pf.get("periods", []) or []
                      for dw in p.get("dropWithdraws", []) or []
                      for r in dw.get("reasons", []) or [])

    tables["trials"].append({
        "nct_id": nct,
        # pre-enrollment protocol fields
        "study_type": design.get("studyType"),
        "phase": "|".join(design.get("phases") or []) or "NA",
        "allocation": dinfo.get("allocation"),
        "masking": (dinfo.get("maskingInfo") or {}).get("masking"),
        "intervention_model": dinfo.get("interventionModel"),
        "primary_purpose": dinfo.get("primaryPurpose"),
        "n_arms": len(arms),
        "sponsor_class": (ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor") or {}).get("class"),
        "mesh_branches": "|".join(branches) if branches else "NONE",
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "sex": elig.get("sex"),
        "min_age_years": parse_age_years(elig.get("minimumAge")),
        "max_age_years": parse_age_years(elig.get("maximumAge")),
        "start_year": start_year(ps.get("statusModule", {})),
        "eligibility_text": elig.get("eligibilityCriteria"),
        "brief_summary": ps.get("descriptionModule", {}).get("briefSummary"),
        "conditions": "|".join(ps.get("conditionsModule", {}).get("conditions", []) or []),
        # eligibility-pool flags (results presence used only for inclusion)
        "pool_has_results": bool(s.get("hasResults")),
        "pool_has_reasons": has_reasons,
        "pool_has_control_arm": any(a.get("type") in CONTROL_TYPES for a in arms),
    })

    for i, a in enumerate(arms):
        names = a.get("interventionNames", []) or []
        tables["arms"].append({
            "nct_id": nct,
            "arm_idx": i,
            "arm_type": a.get("type"),
            "arm_label": a.get("label"),
            "arm_description": a.get("description"),
            "intervention_types": "|".join(intervention_types(names)),
            "intervention_names": "|".join(names),
        })

    for g in pf.get("groups", []) or []:
        tables["flow_groups"].append({"nct_id": nct, "group_id": g.get("id"),
                                      "title": g.get("title"), "description": g.get("description")})
    for pi, p in enumerate(pf.get("periods", []) or []):
        tables["periods"].append({"nct_id": nct, "period_idx": pi, "period_title": p.get("title")})
        for m in p.get("milestones", []) or []:
            if (m.get("type") or "").strip().upper() == "STARTED":
                for a in m.get("achievements", []) or []:
                    tables["started"].append({"nct_id": nct, "period_idx": pi, "group_id": a.get("groupId"),
                                              "started": to_int(a.get("numSubjects"))})
        for dw in p.get("dropWithdraws", []) or []:
            for r in dw.get("reasons", []) or []:
                tables["dw"].append({"nct_id": nct, "period_idx": pi, "group_id": r.get("groupId"),
                                     "raw_reason": (dw.get("type") or "").strip(),
                                     "n": to_int(r.get("numSubjects")) or 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()

    tables = {k: [] for k in ["trials", "arms", "flow_groups", "periods", "started", "dw"]}
    for p in tqdm(sorted(Path(args.raw).glob("page_*.json.gz")), desc="pages"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for s in json.load(f).get("studies", []):
                extract_study(s, tables)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = {"trials": "protocol_trials", "arms": "protocol_arms", "flow_groups": "flow_groups",
             "periods": "flow_periods", "started": "flow_started", "dw": "flow_dropwithdraws"}
    for k, name in names.items():
        df = pd.DataFrame(tables[k])
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
