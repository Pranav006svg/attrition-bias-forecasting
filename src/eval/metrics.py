"""Evaluation metrics and trial-grouped bootstrap confidence intervals."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
N_BOOT = 1000
KL_EPS = 1e-6
ZERO_DELTA = 1e-3  # multiplicative zero replacement for Aitchison distance


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def spearman(y, p):
    if np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(y, p).statistic)


def _normalize(P):
    P = np.clip(P, 0, None)
    s = P.sum(axis=1, keepdims=True)
    return np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / P.shape[1])


def kl_rows(T, P):
    """KL(true || pred) per row; pred smoothed with KL_EPS; zero true terms contribute 0."""
    P = _normalize(np.clip(P, KL_EPS, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(T > 0, T * np.log(T / P), 0.0)
    return terms.sum(axis=1)


def _replace_zeros(X):
    """Multiplicative zero replacement (Martin-Fernandez et al.) on closed compositions."""
    X = _normalize(X)
    z = X <= 0
    nz = z.sum(axis=1, keepdims=True)
    return np.where(z, ZERO_DELTA, X * (1 - ZERO_DELTA * nz))


def aitchison_rows(T, P):
    def clr(X):
        L = np.log(_replace_zeros(X))
        return L - L.mean(axis=1, keepdims=True)
    return np.linalg.norm(clr(T) - clr(P), axis=1)


def auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def auprc(y, s):
    return float(average_precision_score(y, s)) if y.sum() > 0 else np.nan


def paired_bootstrap_test(fn, groups, *arrays, n_boot=N_BOOT, seed=SEED):
    """Paired trial-grouped bootstrap test of a difference statistic.

    fn(*arrays) returns a difference (e.g. metric(model A) - metric(model B)) computed on the
    same resample. Returns (point, ci_low, ci_high, p_wald, p_percentile):
      p_wald       two-sided normal test of point / bootstrap SE (continuous; used for Holm)
      p_percentile two-sided percentile-bootstrap p with add-one correction,
                   2 * (min(#diff <= 0, #diff >= 0) + 1) / (B + 1); its resolution is limited by B.
    """
    from scipy.stats import norm
    point, lo, hi, stats = bootstrap_ci(fn, groups, *arrays, n_boot=n_boot, seed=seed, return_stats=True)
    s = stats[~np.isnan(stats)]
    if len(s) < 2:
        return point, lo, hi, np.nan, np.nan
    se = float(np.std(s, ddof=1))
    p_wald = float(2 * norm.sf(abs(point) / se)) if se > 0 else (0.0 if point != 0 else 1.0)
    p_perc = min(1.0, 2 * (min((s <= 0).sum(), (s >= 0).sum()) + 1) / (len(s) + 1))
    return point, lo, hi, p_wald, float(p_perc)


def holm(pvalues):
    """Holm step-down adjusted p-values (NaN entries are ignored and returned as NaN)."""
    p = np.asarray(pvalues, float)
    out = np.full(len(p), np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def bootstrap_ci(fn, groups, *arrays, n_boot=N_BOOT, seed=SEED, return_stats=False):
    """Point estimate + 95% percentile CI, resampling whole groups (trials) with replacement.

    fn(*arrays_subset) -> float; groups: group id per row of every array.
    """
    groups = np.asarray(groups)
    point = fn(*arrays)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        g = rng.integers(0, len(uniq), len(uniq))
        lens = counts[g]
        ends = np.cumsum(lens)
        pos = np.arange(ends[-1]) + np.repeat(starts[g] - (ends - lens), lens)
        idx = order[pos]
        stats[b] = fn(*(a[idx] for a in arrays))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    if return_stats:
        return point, float(lo), float(hi), stats
    return point, float(lo), float(hi)
