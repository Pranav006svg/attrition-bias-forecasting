"""Paper figures (300 dpi PDF + PNG) and figure index with draft captions.

All numbers are read from results/tables/*.csv, data/processed/*.parquet and the FINAL
cache; nothing is typed in by hand. Style: white background, black text, DejaVu Sans,
categorical slots in fixed order from a validated colorblind-safe palette (dataviz
reference palette; adjacent CVD dE >= 9.1), one-hue blue ramp for sequential data.

Usage:
    python -m src.eval.figures
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.data.build_benchmark import cat_col, prop_col  # noqa: E402
from src.eval.final_cache import load  # noqa: E402

FIG = Path("results/figures")
TAB = Path("results/tables")
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
GRID = "#d9d9d9"
MUTED = "#555555"
ONE_COL, TWO_COL = 3.5, 7.16
MODEL_LABEL = {"reference_class": "Reference class", "trial_lgbm": "Trial-level LGBM", "flat_lgbm": "Per-arm LGBM",
               "FINAL": "FINAL (+ prior evidence)"}
MODEL_COLOR = {"reference_class": SLOTS[0], "flat_lgbm": SLOTS[1], "FINAL": SLOTS[2], "trial_lgbm": SLOTS[3]}
SUBSET_LABEL = {"all": "All", "reliable": "Reliable\n(arms ≥100)", "covered": "Covered\n(prior evidence)",
                "covered_reliable": "Covered +\nreliable"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "text.color": "black",
    "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "#333333",
    "axes.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
})

CAPTIONS = {}


def save(fig, name, caption):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    CAPTIONS[name] = caption


def stat(df, section, key):
    return float(df[(df.section == section) & (df.key == key)].value.iloc[0])


# ------------------------------------------------------------------ F1
def f1_flow():
    feas, ds = pd.read_csv(TAB / "feasibility.csv"), pd.read_csv(TAB / "dataset_stats.csv")
    steps = [
        ("Interventional studies with posted results", stat(feas, "counts", "studies_with_results")),
        ("Randomized, ≥2 arms, withdrawal reasons,\n≥1 control arm (eligible pool)", stat(ds, "funnel", "eligible_pool")),
        ("Participant flow parsed (≥2 flow groups,\ncore randomized periods)", stat(ds, "funnel", "built")),
        ("All flow groups matched to protocol arms\n(rapidfuzz score ≥80)", stat(ds, "funnel", "m80_after_matched_all_flow_groups")),
        ("≥1 matched control arm", stat(ds, "funnel", "m80_after_matched_control_arm")),
        ("Every arm started ≥10", stat(ds, "funnel", "m80_after_all_arms_started_ge10")),
        ("Arm rates ≤1 and known start year\n(primary benchmark)", stat(ds, "funnel", "m80_after_known_start_year")),
    ]
    fig, ax = plt.subplots(figsize=(TWO_COL, 6.4))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    step, h = 1.25, 0.95
    top = len(steps) * step
    ax.set_ylim(-1.9, top + 0.6)
    for i, (label, n) in enumerate(steps):
        y = top - i * step
        ax.add_patch(plt.Rectangle((1.2, y - h / 2), 5.0, h, fc="white", ec="#333333", lw=0.8))
        ax.text(3.7, y + 0.14, label, ha="center", va="center", fontsize=7.5, linespacing=1.15)
        ax.text(3.7, y - 0.3, f"n = {int(n):,} trials", ha="center", va="center", fontsize=7.5, weight="bold")
        if i:
            prev = steps[i - 1][1]
            ax.annotate("", xy=(3.7, y + h / 2), xytext=(3.7, y + step - h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.8))
            ax.text(6.45, y + step / 2, f"excluded {int(prev - n):,}", ha="left", va="center", fontsize=7, color=MUTED)
    splits = [(s, stat(ds, "split_trials", s), stat(ds, "split_arms", s)) for s in ["train", "val", "test"]]
    names = {"train": "Train (start <2015)", "val": "Validation (2015–2016)", "test": "Test (start ≥2017)"}
    for j, (s, nt, na) in enumerate(splits):
        x = 0.2 + j * 3.3
        ax.add_patch(plt.Rectangle((x, -1.75), 3.0, 0.8, fc="white", ec="#333333", lw=0.8))
        ax.text(x + 1.5, -1.2, names[s], ha="center", va="center", fontsize=7.5)
        ax.text(x + 1.5, -1.5, f"{int(nt):,} trials / {int(na):,} arms", ha="center", va="center", fontsize=7.5,
                weight="bold")
        ax.annotate("", xy=(x + 1.5, -0.95), xytext=(3.7, top - (len(steps) - 1) * step - h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.6))
    save(fig, "F1_benchmark_flow",
         "Construction of the arm-level attrition benchmark from ClinicalTrials.gov. Each box gives the number of "
         "trials remaining after the criterion; grey text gives the number excluded at that step. The primary "
         "benchmark is split by trial start year into train, validation and test sets.")


# ------------------------------------------------------------------ F2
GROUPS8 = [("Adverse event", ["ADVERSE_EVENT"]), ("Lack of efficacy / progression", ["LACK_OF_EFFICACY", "DISEASE_PROGRESSION"]),
           ("Death", ["DEATH"]), ("Withdrawal by subject", ["WITHDRAWAL_BY_SUBJECT"]),
           ("Lost to follow-up", ["LOST_TO_FOLLOW_UP"]), ("Protocol violation / physician", ["PROTOCOL_VIOLATION", "PHYSICIAN_DECISION"]),
           ("Pre-treatment exclusion", ["PRE_TREATMENT_EXCLUSION"]), ("Other / unspecified", ["OTHER_UNSPECIFIED"])]
PHASE_ORDER = ["EARLY_PHASE1", "PHASE1", "PHASE1|PHASE2", "PHASE2", "PHASE2|PHASE3", "PHASE3", "PHASE4", "NA"]
PHASE_LABEL = {"EARLY_PHASE1": "Early phase 1", "PHASE1": "Phase 1", "PHASE1|PHASE2": "Phase 1/2", "PHASE2": "Phase 2",
               "PHASE2|PHASE3": "Phase 2/3", "PHASE3": "Phase 3", "PHASE4": "Phase 4", "NA": "Not applicable"}


def f2_reasons_by_phase():
    arms = pd.read_parquet("data/processed/arms.parquet")
    trials = pd.read_parquet("data/processed/trials.parquet")
    a = arms[arms.y_has_composition].merge(trials[["nct_id", "phase"]], on="nct_id")
    for g, cats in GROUPS8:
        a[g] = a[[prop_col(c) for c in cats]].sum(axis=1)
    m = a.groupby("phase")[[g for g, _ in GROUPS8]].mean().reindex(PHASE_ORDER)
    n = a.groupby("phase").size().reindex(PHASE_ORDER)
    fig, ax = plt.subplots(figsize=(TWO_COL, 3.4))
    y = np.arange(len(m))[::-1]
    left = np.zeros(len(m))
    for k, (g, _) in enumerate(GROUPS8):
        ax.barh(y, m[g].values, left=left, height=0.62, color=SLOTS[k], edgecolor="white", linewidth=1.0, label=g)
        left += m[g].values
    ax.set_yticks(y, [f"{PHASE_LABEL[p]} (n={int(n[p]):,})" for p in m.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean per-arm share of primary-attrition withdrawals")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.2), handlelength=1.2, columnspacing=1.0)
    save(fig, "F2_reasons_by_phase",
         "Withdrawal-reason composition by trial phase. Bars show the mean per-arm proportion of primary-attrition "
         "withdrawals in each reason group, over arms with at least one such withdrawal (all splits; every arm "
         "weighted equally); n = number of arms.")


# ------------------------------------------------------------------ F3
def f3_drift():
    d = pd.read_csv(TAB / "drift_analysis.csv")
    r = d[d.section == "reporting_by_start_year"]
    bins = list(dict.fromkeys(r.group))
    tick = [b.replace("<=", "≤").replace(">=", "≥").replace("-", "–") for b in bins]
    get = lambda m: r[r.metric == m].set_index("group").value.reindex(bins).values
    x = np.arange(len(bins))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TWO_COL, 2.6))
    for k, (metric, label) in enumerate([("mean_ae_rate", "Adverse event"), ("mean_efficacy_rate", "Lack of efficacy / progression")]):
        v = get(metric) * 100
        a1.plot(x, v, color=SLOTS[k], marker="o", markersize=4, label=label)
        a1.text(x[-1] + 0.12, v[-1], label, color="black", fontsize=7, va="center")
    a1.set_xticks(x, tick, rotation=30)
    a1.set_ylabel("Mean per-arm withdrawal rate (%)")
    a1.set_ylim(bottom=0)
    a1.set_xlim(-0.3, len(bins) + 1.6)
    a1.set_title("a  Reason-specific withdrawal rates", loc="left")
    v = get("mean_distinct_primary_categories_per_trial")
    a2.plot(x, v, color=SLOTS[0], marker="o", markersize=4)
    a2.set_xticks(x, tick, rotation=30)
    a2.set_ylabel("Distinct reason categories per trial")
    a2.set_ylim(0, max(v) * 1.15)
    a2.set_title("b  Reporting granularity", loc="left")
    for ax in (a1, a2):
        ax.set_xlabel("Trial start year")
    fig.tight_layout()
    save(fig, "F3_temporal_drift",
         "Temporal drift in withdrawal reporting by trial start year (primary benchmark, trials with a computable "
         "differential). (a) Mean per-arm withdrawal rate attributed to adverse events and to lack of efficacy or "
         "disease progression. (b) Mean number of distinct primary-attrition reason categories reported per trial.")


# ------------------------------------------------------------------ F4 / F5
def ci_points(ax, est, target, metric, models, subsets, xlabel):
    offs = np.linspace(-0.25, 0.25, len(models))
    for k, mdl in enumerate(models):
        for i, s in enumerate(subsets):
            r = est[(est.subset == s) & (est.target == target) & (est.metric == metric) & (est.model == mdl)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = len(subsets) - 1 - i + offs[k]
            ax.errorbar(r.value, yy, xerr=[[r.value - r.ci_low], [r.ci_high - r.value]], fmt="o", ms=4.5,
                        color=MODEL_COLOR[mdl], ecolor=MODEL_COLOR[mdl], elinewidth=1.4, capsize=0,
                        label=MODEL_LABEL[mdl] if i == 0 else None)
    ax.set_yticks(range(len(subsets))[::-1], [SUBSET_LABEL[s] for s in subsets])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def f4_model_comparison():
    est = pd.read_csv(TAB / "final_estimates.csv")
    subsets = ["all", "reliable", "covered", "covered_reliable"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TWO_COL, 3.0), sharey=True)
    ci_points(a1, est, "arm_rate_primary", "MAE", ["reference_class", "trial_lgbm", "flat_lgbm", "FINAL"], subsets,
              "Mean absolute error (lower is better)")
    a1.set_title("a  Primary arm attrition rate", loc="left")
    ci_points(a2, est, "composition", "mean_KL", ["reference_class", "flat_lgbm", "FINAL"], subsets,
              "Mean KL divergence (lower is better)")
    a2.set_title("b  Reason composition", loc="left")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "F4_model_comparison",
         "Test-set accuracy of arm-level forecasts. (a) Mean absolute error of the primary arm attrition rate. "
         "(b) Mean Kullback-Leibler divergence between observed and predicted reason composition (arms with at "
         "least one primary-attrition withdrawal; the trial-level model has no composition output). Points are "
         "estimates with 95% trial-grouped bootstrap intervals.")


def f5_concordance():
    est = pd.read_csv(TAB / "final_estimates.csv")
    subsets = ["all", "reliable", "covered", "covered_reliable"]
    fig, ax = plt.subplots(figsize=(ONE_COL + 0.6, 3.0))
    ax.axvline(0.5, color="#333333", lw=1, ls=(0, (4, 3)))
    ci_points(ax, est, "pairwise_primary", "concordance", ["reference_class", "flat_lgbm", "FINAL"], subsets,
              "Within-trial concordance (experimental vs control)")
    ax.text(0.5, len(subsets) - 0.45, "chance", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xlim(0.40, 0.65)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2)
    save(fig, "F5_concordance",
         "Ordering of attrition between experimental and control arms within the same trial. Concordance is the "
         "share of experimental-control arm pairs whose attrition order is predicted correctly (0.5 = chance; dashed "
         "line). Subsets: all test trials; reliable pairs (both arms started ≥100); covered trials (all experimental "
         "arms have prior evidence for the same drug); both. The trial-level model is omitted because it predicts no "
         "within-trial difference by construction. Points: estimates with 95% trial-grouped bootstrap intervals.")


# ------------------------------------------------------------------ F6
def f6_conformal():
    c = pd.read_csv(TAB / "final_conformal.csv")
    bins = ["10-49", "50-99", "100-299", ">=300", "all"]
    labels = ["10–49", "50–99", "100–299", "≥300", "All arms"]
    methods = [("mondrian", "Mondrian (per size bin)"), ("normalized", "Normalized (binomial scale)")]
    x = np.arange(len(bins))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TWO_COL, 2.6))
    for k, (m, lab) in enumerate(methods):
        d = c[c.method == m].set_index("size_bin").reindex(bins)
        xx = x + (k - 0.5) * 0.25
        a1.errorbar(xx, d.coverage, yerr=[d.coverage - d.cov_ci_low, d.cov_ci_high - d.coverage], fmt="o", ms=4.5,
                    color=SLOTS[k], elinewidth=1.4, label=lab)
        a2.bar(xx, d.mean_width, width=0.25, color=SLOTS[k], edgecolor="white", linewidth=1.0, label=lab)
    a1.axhline(0.9, color="#333333", lw=1, ls=(0, (4, 3)))
    a1.text(-0.45, 0.905, "target 0.90", ha="left", va="bottom", fontsize=7, color=MUTED)
    a1.set_ylabel("Empirical coverage (test)")
    a1.set_ylim(0.55, 1.0)
    a1.set_title("a  Coverage", loc="left")
    a2.set_ylabel("Mean interval width (rate)")
    a2.set_title("b  Width", loc="left")
    a2.grid(axis="x", visible=False)
    for ax in (a1, a2):
        ax.set_xticks(x, labels)
        ax.set_xlabel("Arm size (participants started)")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "F6_conformal",
         "Split-conformal 90% prediction intervals for the primary arm attrition rate (FINAL model, calibrated on "
         "validation arms). (a) Empirical test coverage with 95% bootstrap intervals; dashed line = target. "
         "(b) Mean interval width. Mondrian intervals use a separate residual quantile per arm-size bin; normalized "
         "intervals scale residuals by the binomial standard error. Arm size is the observed number started, used "
         "as a proxy for the planned arm size.")


# ------------------------------------------------------------------ F7
def f7_frontier():
    f = pd.read_csv(TAB / "final_recruitment_frontier.csv")
    lab = {"fixed": "Fixed assumed attrition", "reference_class": "Reference class + margin",
           "FINAL": "FINAL conformal upper quantile"}
    fig, ax = plt.subplots(figsize=(ONE_COL + 0.6, 3.0))
    for k, s in enumerate(["fixed", "reference_class", "FINAL"]):
        d = f[f.strategy == s].sort_values("pct_under_recruited")
        x, y = d.pct_under_recruited, d.excess_patients_per_1000_completers
        ax.fill_between(x, d.excess_patients_per_1000_completers_ci_low, d.excess_patients_per_1000_completers_ci_high,
                        color=SLOTS[k], alpha=0.15, lw=0)
        ax.plot(x, y, color=SLOTS[k], marker="o", ms=3.5, label=lab[s])
    for u in [10, 20]:
        ax.axvline(u, color="#333333", lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel("Arms under-recruited (%)")
    ax.set_ylabel("Excess patients per 1,000 required completers")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    save(fig, "F7_recruitment_frontier",
         "Recruitment trade-off on test arms. Each point is a planning rule: recruitment target = required completers "
         "/ (1 − assumed attrition). Fixed rule: assumed attrition 5–50%; reference class: per-arm reference-class "
         "mean plus a margin of 0–30 percentage points; FINAL: one-sided Mondrian conformal upper quantile at levels "
         "0.50–0.95. Bands: 95% trial-grouped bootstrap intervals. Lower-left is better; dashed lines mark the 10% and "
         "20% under-recruitment rates used for matched comparison.")


# ------------------------------------------------------------------ F8
def f8_shap():
    fam = pd.read_csv(TAB / "final_shap_families.csv")
    order = ["arm_rate_primary", "adverse_event", "lack_of_efficacy", "withdrawal_by_subject", "lost_to_follow_up",
             "pre_treatment_exclusion"]
    tlabel = {"arm_rate_primary": "Arm attrition rate", "adverse_event": "Adverse event", "lack_of_efficacy":
              "Lack of efficacy", "withdrawal_by_subject": "Withdrawal by subject", "lost_to_follow_up":
              "Lost to follow-up", "pre_treatment_exclusion": "Pre-treatment exclusion"}
    families = ["design", "condition", "eligibility_text", "arm", "prior_evidence"]
    flabel = {"design": "Trial design", "condition": "Condition (MeSH)", "eligibility_text": "Eligibility / summary text",
              "arm": "Arm", "prior_evidence": "Prior evidence"}
    p = fam.pivot(index="target", columns="family", values="share_of_total").reindex(order)[families]
    nfeat = fam.drop_duplicates("family").set_index("family").n_features
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.8))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for k, f in enumerate(families):
        ax.barh(y, p[f].values, left=left, height=0.62, color=SLOTS[k], edgecolor="white", linewidth=1.0,
                label=f"{flabel[f]} ({int(nfeat[f])} features)")
        left += p[f].values
    ax.set_yticks(y, [tlabel[t] for t in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of total mean |SHAP| (test arms)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.42, -0.22))
    save(fig, "F8_shap_families",
         "Feature-family attribution for the FINAL model (TreeSHAP, test arms, log-odds scale). Bars show each "
         "family's share of the summed mean absolute SHAP value for the arm attrition rate and for five reason "
         "proportions. Family shares are sums over features and therefore grow with family size (feature counts in "
         "the legend); prior-evidence features are missing for about three quarters of test arms.")


# ------------------------------------------------------------------ F9
def f9_pred_vs_obs():
    c = load()
    y = c["A_test"].y_primary_attrition_rate.values
    p = c["pred"]["test"]["FINAL"]["rate_primary"]
    fig, ax = plt.subplots(figsize=(ONE_COL, 3.0))
    hb = ax.hexbin(p, y, gridsize=40, extent=(0, 1, 0, 1), bins="log", cmap=SEQ, mincnt=1, linewidths=0)
    ax.plot([0, 1], [0, 1], color="#333333", lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Predicted primary attrition rate (FINAL)")
    ax.set_ylabel("Observed primary attrition rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(False)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Arms per cell (log scale)")
    save(fig, "F9_predicted_vs_observed",
         f"Predicted versus observed primary attrition rate for {len(y):,} test arms (FINAL model). Hexagon shading "
         "gives the number of arms (log scale); dashed line = perfect agreement.")


# ------------------------------------------------------------------ S1
def s1_confusion():
    m = pd.read_csv(TAB / "reason_agreement_confusion.csv", index_col=0)
    short = {c: c.replace("_", " ").capitalize().replace("Pre treatment", "Pre-treatment")
             .replace("follow up", "follow-up") for c in m.index}
    fig, ax = plt.subplots(figsize=(TWO_COL * 0.8, 4.6))
    cmap = SEQ.copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_equal(m.values, 0), cmap=cmap, vmin=1, vmax=m.values.max())
    ax.set_xticks(np.arange(-0.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0]), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = int(m.values[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=7,
                        color="white" if v > m.values.max() * 0.5 else "black")
    ax.set_xticks(range(m.shape[1]), [short[c] for c in m.columns], rotation=45, ha="right")
    ax.set_yticks(range(m.shape[0]), [short[c] for c in m.index])
    ax.set_xlabel("Pipeline category")
    ax.set_ylabel("Human category (blind)")
    ax.grid(which="major", visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Strings")
    v = pd.read_csv(TAB / "reason_agreement.csv")
    acc = v[(v.group == "all") & (v.metric == "accuracy_exact_ci")].iloc[0]
    save(fig, "S1_reason_agreement_confusion",
         f"Blind validation of reason normalization: confusion matrix for 100 raw withdrawal-reason strings (34 rule, "
         f"33 hand-labelled, 33 nearest-neighbour tier) labelled by a blinded annotator. Accuracy {acc.value:.2f} "
         f"(exact 95% CI {acc.ci_low:.2f}–{acc.ci_high:.2f}).")


def write_index():
    L = ["# Figure index", "", "All figures: 300 dpi PNG and vector PDF, generated by `python -m src.eval.figures` from "
         "results/tables and data/processed. Captions are drafts.", ""]
    for name, cap in CAPTIONS.items():
        L += [f"## {name}", "", f"Files: `{name}.pdf`, `{name}.png`", "", f"Draft caption: {cap}", ""]
    (FIG / "index.md").write_text("\n".join(L), encoding="utf-8")


def main():
    for f in [f1_flow, f2_reasons_by_phase, f3_drift, f4_model_comparison, f5_concordance, f6_conformal,
              f7_frontier, f8_shap, f9_pred_vs_obs, s1_confusion]:
        f()
        print(f"{f.__name__} done")
    write_index()


if __name__ == "__main__":
    main()
