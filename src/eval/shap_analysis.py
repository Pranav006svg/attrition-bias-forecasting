"""SHAP (TreeExplainer) for FINAL on the test split.

Targets: primary arm attrition rate (all test arms) and the reason proportions for
ADVERSE_EVENT, WITHDRAWAL_BY_SUBJECT, LOST_TO_FOLLOW_UP, LACK_OF_EFFICACY,
PRE_TREATMENT_EXCLUSION (test arms with >= 1 primary withdrawal). Importance = mean
|SHAP| on the model's raw (log-odds, cross-entropy objective) scale.

Feature families:
  design            phase, allocation, masking, intervention model, purpose, n arms, sponsor
                    class, sex, healthy volunteers, age limits, start year
  condition         MeSH branch indicators (mesh_*)
  eligibility_text  TF-IDF of eligibility criteria + brief summary (tfidf_*)
  arm               arm type, intervention types (itype_*), number of interventions
  prior_evidence    prior_* features

Outputs: results/tables/final_shap_top15.csv, results/tables/final_shap_families.csv,
results/tables/final_shap_families_normalized.csv (mean |SHAP| per feature within each
family, i.e. family sum / number of features, and its share across families; this removes
the advantage of large families such as the 300 TF-IDF features).

Usage:
    python -m src.eval.shap_analysis [--normalized-only]   # --normalized-only: derive from saved family table
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.data.build_benchmark import prop_col
from src.eval.final_cache import load
from src.features.build_features import TRIAL_CAT, TRIAL_NUM

TARGETS = {"arm_rate_primary": "rate_primary"}
TARGETS.update({c.lower(): prop_col(c) for c in ["ADVERSE_EVENT", "WITHDRAWAL_BY_SUBJECT", "LOST_TO_FOLLOW_UP",
                                                  "LACK_OF_EFFICACY", "PRE_TREATMENT_EXCLUSION"]})
FAMILIES = ["design", "condition", "eligibility_text", "arm", "prior_evidence"]
TABLES = Path("results/tables")


def family(col):
    if col.startswith("prior_"):
        return "prior_evidence"
    if col.startswith("tfidf_"):
        return "eligibility_text"
    if col.startswith("mesh_"):
        return "condition"
    if col.startswith("itype_") or col in ("arm_type", "n_interventions"):
        return "arm"
    if col in TRIAL_CAT or col in TRIAL_NUM:
        return "design"
    raise ValueError(f"unassigned feature {col}")


def normalized_families():
    f = pd.read_csv(TABLES / "final_shap_families.csv")
    f["mean_abs_shap_per_feature"] = f.mean_abs_shap_sum / f.n_features
    f["share_per_feature_normalized"] = f.mean_abs_shap_per_feature / f.groupby("target").mean_abs_shap_per_feature.transform("sum")
    out = f[["target", "family", "n_features", "mean_abs_shap_per_feature", "share_per_feature_normalized",
             "share_of_total"]].rename(columns={"share_of_total": "share_summed"})
    out.to_csv(TABLES / "final_shap_families_normalized.csv", index=False)
    print(out.pivot(index="target", columns="family", values="share_per_feature_normalized")[FAMILIES].round(3).to_string())
    return out


def main():
    if "--normalized-only" in sys.argv:
        return normalized_families()
    c = load()
    X, A = c["X_test"], c["A_test"]
    fam = pd.Series({col: family(col) for col in X.columns})
    top_rows, fam_rows = [], []
    for name, key in TARGETS.items():
        rows = np.ones(len(X), bool) if key == "rate_primary" else A.y_has_composition.values
        Xs = X[rows]
        sv = shap.TreeExplainer(c["models"][key]).shap_values(Xs)
        imp = pd.Series(np.abs(sv).mean(0), index=X.columns)
        total = imp.sum()
        for rank, (feat, v) in enumerate(imp.sort_values(ascending=False).head(15).items(), 1):
            top_rows.append({"target": name, "rank": rank, "feature": feat, "family": fam[feat],
                             "mean_abs_shap": v, "share_of_total": v / total})
        fi = imp.groupby(fam).sum().reindex(FAMILIES, fill_value=0)
        for f, v in fi.items():
            fam_rows.append({"target": name, "family": f, "mean_abs_shap_sum": v, "share_of_total": v / total,
                             "n_features": int((fam == f).sum()), "n_rows": int(rows.sum())})
        print(f"{name}: done ({rows.sum():,} arms)")
    top, fams = pd.DataFrame(top_rows), pd.DataFrame(fam_rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    top.to_csv(TABLES / "final_shap_top15.csv", index=False)
    fams.to_csv(TABLES / "final_shap_families.csv", index=False)
    print(fams.pivot(index="target", columns="family", values="share_of_total")[FAMILIES].round(3).to_string())
    normalized_families()


if __name__ == "__main__":
    main()
