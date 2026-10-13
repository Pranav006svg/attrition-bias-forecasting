"""Human vs pipeline agreement on the blind reason-validation sample.

Joins annotation/validation_sample.csv (human labels, blind v2) with the local key
data/interim/.blind/validation_key.csv (tier + pipeline category) and reports,
per tier and overall: accuracy, Cohen's kappa, macro-F1 (95% bootstrap CIs,
1,000 resamples of strings, seed 42); binary agreement on whether a reason
counts toward primary / nonendpoint attrition; and dataset-level accuracy,
i.e. tier accuracies weighted by each tier's share of entries and subjects
(results/tables/reason_coverage.csv), since the sample is stratified by tier.

Outputs (v2; with --v1 the same names get a _v1_unblinded suffix):
  results/tables/reason_agreement.csv
  results/tables/reason_agreement_confusion.csv   (rows = human, cols = pipeline)
  results/tables/reason_agreement_disagreements.csv
  results/reason_agreement_summary.md

The v1 sample (annotation/validation_sample_v1_unblinded.csv) was labelled after
the annotator had seen its key; it is reported only as an unblinded pilot.

Usage:
    python -m src.eval.agreement          # blind v2
    python -m src.eval.agreement --v1     # reproduce the unblinded v1 pilot
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.data.build_benchmark import NOT_NONENDPOINT, NOT_PRIMARY
from src.data.reasons import CATEGORIES

SEED = 42
N_BOOT = 1000
TIERS = ["rule", "llm", "knn"]
TABLES = Path("results/tables")
SAMPLES = {
    "v2": {"sample": "annotation/validation_sample.csv", "key": "data/interim/.blind/validation_key.csv",
           "suffix": "", "title": "blind v2"},
    "v1": {"sample": "annotation/validation_sample_v1_unblinded.csv",
           "key": "data/interim/validation_sample_key_v1.csv", "suffix": "_v1_unblinded",
           "title": "UNBLINDED PILOT v1 (annotator had seen the key)",
           "note": "**UNBLINDED PILOT (v1).** The annotator had seen the pipeline key before labelling, so these "
                   "numbers are not a blind validation and must be reported only as a pilot. The blind validation "
                   "is v2 (annotation/validation_sample.csv)."},
}


def macro_f1(h, p):
    return f1_score(h, p, labels=sorted(set(h) | set(p)), average="macro", zero_division=0)


METRICS = {"accuracy": accuracy_score, "cohen_kappa": cohen_kappa_score, "macro_f1": macro_f1}


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI; unlike the percentile bootstrap it is not degenerate at 0% or 100%."""
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def boot(fn, h, p, rng):
    point = fn(h, p)
    stats = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(h), len(h))
        with np.errstate(all="ignore"):
            stats.append(fn(h[i], p[i]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    cfg = SAMPLES["v1" if "--v1" in sys.argv else "v2"]
    v = pd.read_csv(cfg["sample"])
    k = pd.read_csv(cfg["key"])
    if list(v.id) != list(k.id) or list(v.raw_reason) != list(k.raw_reason):
        raise SystemExit("validation_sample.csv and its key are not aligned")
    if v.category.isna().any():
        raise SystemExit(f"{v.category.isna().sum()} rows are not labelled yet")
    bad = set(v.category) - set(CATEGORIES)
    if bad:
        raise SystemExit(f"Unknown human categories: {bad}")
    d = k.rename(columns={"category": "pipeline"}).assign(human=v.category.values)

    rng = np.random.default_rng(SEED)
    rows = []
    groups = [("all", d)] + [(t, d[d.tier == t]) for t in TIERS]
    for name, g in groups:
        h, p = g.human.values, g.pipeline.values
        k = int((h == p).sum())
        lo, hi = clopper_pearson(k, len(g))
        rows.append({"group": name, "metric": "accuracy_exact_ci", "value": k / len(g), "ci_low": lo, "ci_high": hi,
                     "n": len(g)})
        for m, fn in METRICS.items():
            val, lo, hi = boot(fn, h, p, rng)
            rows.append({"group": name, "metric": m, "value": val, "ci_low": lo, "ci_high": hi, "n": len(g)})
        for lab, excl in [("primary_inclusion_agreement", NOT_PRIMARY), ("nonendpoint_inclusion_agreement", NOT_NONENDPOINT)]:
            hb, pb = ~np.isin(h, list(excl)), ~np.isin(p, list(excl))
            val, lo, hi = boot(accuracy_score, hb, pb, rng)
            rows.append({"group": name, "metric": lab, "value": val, "ci_low": lo, "ci_high": hi, "n": len(g)})

    # Dataset-level accuracy: tier accuracies weighted by tier coverage shares.
    cov = pd.read_csv(TABLES / "reason_coverage.csv").set_index("tier")
    acc = {t: accuracy_score(d[d.tier == t].human, d[d.tier == t].pipeline) for t in TIERS}
    for basis in ["entries", "subjects"]:
        w = cov.loc[TIERS, f"{basis}_share"]
        est = float(sum(w[t] * acc[t] for t in TIERS) / w.sum())
        # Stratified bootstrap: resample within each tier.
        stats = []
        for _ in range(N_BOOT):
            a = {}
            for t in TIERS:
                g = d[d.tier == t]
                i = rng.integers(0, len(g), len(g))
                a[t] = accuracy_score(g.human.values[i], g.pipeline.values[i])
            stats.append(sum(w[t] * a[t] for t in TIERS) / w.sum())
        lo, hi = np.percentile(stats, [2.5, 97.5])
        rows.append({"group": f"dataset_weighted_by_{basis}", "metric": "accuracy", "value": est,
                     "ci_low": float(lo), "ci_high": float(hi), "n": len(d)})

    res = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    sfx = cfg["suffix"]
    res.to_csv(TABLES / f"reason_agreement{sfx}.csv", index=False)
    conf = pd.crosstab(pd.Categorical(d.human, CATEGORIES), pd.Categorical(d.pipeline, CATEGORIES),
                       rownames=["human"], colnames=["pipeline"], dropna=False)
    conf.to_csv(TABLES / f"reason_agreement_confusion{sfx}.csv")
    dis = d[d.human != d.pipeline][["id", "tier", "raw_reason", "human", "pipeline", "frequency", "subjects"]]
    dis.sort_values(["tier", "subjects"], ascending=[True, False]).to_csv(
        TABLES / f"reason_agreement_disagreements{sfx}.csv", index=False)
    write_summary(res, conf, dis, cfg)


def write_summary(res, conf, dis, cfg):
    def fmt(group, metric):
        r = res[(res.group == group) & (res.metric == metric)].iloc[0]
        return f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"

    L = [f"# Reason-normalization agreement (human vs pipeline) - {cfg['title']}", ""]
    if cfg.get("note"):
        L += [f"> {cfg['note']}", ""]
    L += [
         "Sample of 100 raw reason strings (34 rule / 33 llm / 33 knn, frequency-stratified). "
         "95% CIs from 1,000 bootstrap resamples (seed 42); accuracy also with exact Clopper-Pearson intervals, "
         "which remain informative when a group scores 100%.", "",
         "| Group | n | Accuracy (exact Clopper-Pearson CI) | Accuracy (bootstrap CI) | Cohen's kappa | Macro-F1 | Primary-inclusion agreement | Nonendpoint-inclusion agreement |",
         "|---|---:|---|---|---|---|---|---|"]
    for g in ["all"] + TIERS:
        n = int(res[res.group == g].n.iloc[0])
        L.append(f"| {g} | {n} | {fmt(g, 'accuracy_exact_ci')} | {fmt(g, 'accuracy')} | {fmt(g, 'cohen_kappa')} | {fmt(g, 'macro_f1')} | "
                 f"{fmt(g, 'primary_inclusion_agreement')} | {fmt(g, 'nonendpoint_inclusion_agreement')} |")
    L += ["", "Dataset-level accuracy (tier accuracies weighted by tier coverage; stratified bootstrap):", "",
          f"- weighted by entries: {fmt('dataset_weighted_by_entries', 'accuracy')}",
          f"- weighted by subjects: {fmt('dataset_weighted_by_subjects', 'accuracy')}", "",
          f"Disagreements: {len(dis)} of {int(res[res.group == 'all'].n.iloc[0])} "
          f"(see results/tables/reason_agreement_disagreements{cfg['suffix']}.csv).", "",
          "Most frequent (human -> pipeline) confusions:", ""]
    pairs = dis.groupby(["human", "pipeline"]).size().sort_values(ascending=False).head(10)
    for (h, p), n in pairs.items():
        L.append(f"- {h} -> {p}: {n}")
    Path(f"results/reason_agreement_summary{cfg['suffix']}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
