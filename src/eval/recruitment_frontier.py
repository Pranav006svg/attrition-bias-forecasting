"""Recruitment trade-off frontier on test arms.

Required completers = observed completers (started - primary-attrition withdrawals).
Recruitment target = completers / (1 - assumed attrition), assumed attrition clipped to
[0, 0.95]. Under-recruited = target < number actually started.

Strategies (one frontier each):
  fixed            assumed attrition 0.05, 0.10, ..., 0.50
  reference_class  per-arm reference-class mean + uniform margin m in {0, 0.05, ..., 0.30}
  FINAL            Mondrian one-sided conformal upper quantile at levels
                   {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95}: per arm-size bin, the
                   ceil((n+1) level)/n quantile of signed validation residuals (y - p)
                   is added to the FINAL point prediction.
Metrics per point: % arms under-recruited, mean over-recruitment % among arms not under,
excess patients per 1,000 required completers (sum of max(target - started, 0) /
sum of completers x 1000). 95% CIs: 1,000 trial-grouped bootstrap resamples.
Matched comparison: over-recruitment interpolated (linearly in % under-recruited) at 10% and
20% under-recruitment; paired differences FINAL - fixed and FINAL - reference_class use the
same resamples for all strategies.

Usage:
    python -m src.eval.recruitment_frontier
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.final_analysis import size_bin
from src.eval.final_cache import load

SEED = 42
N_BOOT = 1000
MAX_A = 0.95
FIXED = np.round(np.arange(0.05, 0.501, 0.05), 2)
MARGINS = np.round(np.arange(0.0, 0.301, 0.05), 2)
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
MATCH_UNDER = [10.0, 20.0]
TABLES = Path("results/tables")


def conformal_upper(val_arms, val_pred, test_arms, test_pred, level):
    r = val_arms.y_primary_attrition_rate.values - val_pred
    bv, bt = val_arms.y_started.map(size_bin).values, test_arms.y_started.map(size_bin).values
    q = {}
    for b in np.unique(bv):
        rb = np.sort(r[bv == b])
        k = min(len(rb), int(np.ceil((len(rb) + 1) * level)))
        q[b] = rb[k - 1]
    return test_pred + np.array([q[b] for b in bt])


def point_metrics(rel, excess, completers):
    """rel, excess: (n_points, n_arms); completers: (n_arms,). Returns 3 arrays (n_points,)."""
    under = (rel < 0).mean(axis=1) * 100
    notu = rel >= 0
    over = np.where(notu.any(1), (rel * notu).sum(1) / np.maximum(notu.sum(1), 1) * 100, np.nan)
    exc = excess.sum(1) / completers.sum() * 1000
    return under, over, exc


def interp_at(under, values, target):
    """Linear interpolation of values at a given % under-recruited (NaN if outside the range)."""
    o = np.argsort(under)
    u, v = under[o], values[o]
    if target < u[0] or target > u[-1]:
        return np.nan
    return float(np.interp(target, u, v))


def main():
    c = load()
    A, At = c["A_val"], c["A_test"]
    started = At.y_started.values.astype(float)
    completers = started - At.y_withdrawals_primary.values
    fin_val, fin_test = c["pred"]["val"]["FINAL"]["rate_primary"], c["pred"]["test"]["FINAL"]["rate_primary"]
    ref_test = c["pred"]["test"]["reference_class"]["rate_primary"]

    points = []  # (strategy, parameter, assumed attrition per arm)
    for a in FIXED:
        points.append(("fixed", a, np.full(len(At), a)))
    for m in MARGINS:
        points.append(("reference_class", m, ref_test + m))
    for lv in LEVELS:
        points.append(("FINAL", lv, conformal_upper(A, fin_val, At, fin_test, lv)))
    assumed = np.clip(np.vstack([p[2] for p in points]), 0, MAX_A)
    target = completers / (1 - assumed)
    rel = (target - started) / started
    excess = np.maximum(target - started, 0)

    # Trial-grouped resampling indices (shared by all strategies).
    g = At.nct_id.values
    uniq, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(uniq), len(uniq))
        lens = counts[k]
        ends = np.cumsum(lens)
        boots.append(order[np.arange(ends[-1]) + np.repeat(starts[k] - (ends - lens), lens)])

    base = point_metrics(rel, excess, completers)
    bstats = [point_metrics(rel[:, i], excess[:, i], completers[i]) for i in boots]

    rows = []
    for j, (strat, param, _) in enumerate(points):
        row = {"strategy": strat, "parameter": param}
        for m_i, name in enumerate(["pct_under_recruited", "mean_over_recruitment_pct_not_under",
                                    "excess_patients_per_1000_completers"]):
            bs = np.array([b[m_i][j] for b in bstats])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            row.update({name: base[m_i][j], f"{name}_ci_low": lo, f"{name}_ci_high": hi})
        rows.append(row)
    front = pd.DataFrame(rows)
    front["n_arms"] = len(At)

    # Matched under-recruitment comparison.
    strat_idx = {s: [j for j, p in enumerate(points) if p[0] == s] for s in ["fixed", "reference_class", "FINAL"]}

    def matched(stats, target_under, metric_i):
        u = stats[0]
        return {s: interp_at(u[idx], stats[metric_i][idx], target_under) for s, idx in strat_idx.items()}

    comp_rows = []
    for tu in MATCH_UNDER:
        for metric_i, mname in [(1, "mean_over_recruitment_pct_not_under"), (2, "excess_patients_per_1000_completers")]:
            pt = matched(base, tu, metric_i)
            bs = [matched(b, tu, metric_i) for b in bstats]
            for s in ["fixed", "reference_class", "FINAL"]:
                v = np.array([b[s] for b in bs])
                lo, hi = np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)
                comp_rows.append({"under_recruitment_pct": tu, "metric": mname, "comparison": s, "value": pt[s],
                                  "ci_low": lo, "ci_high": hi})
            for other in ["fixed", "reference_class"]:
                d = np.array([b["FINAL"] - b[other] for b in bs])
                ok = np.isfinite(d)
                lo, hi = np.percentile(d[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
                comp_rows.append({"under_recruitment_pct": tu, "metric": mname, "comparison": f"FINAL - {other}",
                                  "value": pt["FINAL"] - pt[other], "ci_low": lo, "ci_high": hi,
                                  "boot_valid_share": ok.mean()})
    comp = pd.DataFrame(comp_rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    front.to_csv(TABLES / "final_recruitment_frontier.csv", index=False)
    comp.to_csv(TABLES / "final_recruitment_matched.csv", index=False)
    pd.set_option("display.width", 220)
    print(front.round(2).to_string(index=False))
    print(comp.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
