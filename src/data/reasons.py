"""Three-tier normalization of raw participant-flow withdrawal reasons.

Tier 1 (rule): case-insensitive normalization + anchored regex rules.
Tier 2 (llm):  unmatched keys sorted by subjects (desc) were labeled manually in
               batches; labels live in annotation/llm_labels.csv. The queue stops
               at 98% cumulative subject coverage or 3,000 keys.
Tier 3 (knn):  remaining keys take the category of the nearest Tier 1/2 key
               (all-MiniLM-L6-v2 embeddings, cosine); similarity < 0.6 ->
               OTHER_UNSPECIFIED.

Scope: reason strings of the eligible pool (interventional, randomized, >=2
arms, results posted, reasons present, >=1 control arm).

Usage:
    python -m src.data.reasons queue     # Tier 1 + write Tier 2 queue
    python -m src.data.reasons finalize  # Tier 2 labels + Tier 3 + outputs
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CATEGORIES = [
    "ADVERSE_EVENT", "LACK_OF_EFFICACY", "DISEASE_PROGRESSION", "DEATH",
    "WITHDRAWAL_BY_SUBJECT", "LOST_TO_FOLLOW_UP", "PHYSICIAN_DECISION",
    "PROTOCOL_VIOLATION", "PRE_TREATMENT_EXCLUSION", "SPONSOR_ADMINISTRATIVE",
    "NOT_ATTRITION", "OTHER_UNSPECIFIED",
]
TIER2_COVERAGE = 0.98
TIER2_MAX_KEYS = 3000
KNN_MODEL = "all-MiniLM-L6-v2"
KNN_MIN_SIM = 0.6

INTERIM = Path("data/interim")
ANNOT = Path("annotation")
LLM_LABELS = ANNOT / "llm_labels.csv"
QUEUE = INTERIM / "tier2_queue.csv"

_DEATH_WORDS = r"(death|deaths|died|dead|deceased|fatal|mortality)"

# Ordered (category, pattern); first full match wins. Patterns are anchored and
# deliberately conservative: compound or ambiguous strings fall through to Tier 2.
RULES = [
    ("NOT_ATTRITION", r"(on[- ]?going|still on[- ]?going|on[- ]?going (in|on|at) .*|on[- ]?going .*(study|treatment|therapy|follow[- ]?up|data cut.*)"
                      r"|still (in|on|receiving|participating) .*|continu(ing|es|ed) (in|on) (study|treatment|the study|therapy).*"
                      r"|remain(s|ed|ing)? (in|on) (study|treatment|the study|therapy).*|participant(s)? (is|are) still .*)"),
    ("DEATH", rf"({_DEATH_WORDS}|subject {_DEATH_WORDS}|patient {_DEATH_WORDS}|participant {_DEATH_WORDS}|{_DEATH_WORDS} \(.*\)"
              rf"|(subject|patient|participant) (has )?died|death due to .*|died due to .*|died of .*|death from .*|death - .*|death, .*)"),
    ("ADVERSE_EVENT", rf"(?!.*{_DEATH_WORDS})((serious |treatment[- ]emergent |non[- ]?serious )?adverse (event|events|reaction|reactions|experience|experiences|effect|effects)(?!.*progress).*"
                      r"|(sae|saes|ae|aes|teae|teaes)|(study )?(drug|treatment|medication)?[ -]?(related )?toxicity|toxicities|side[- ]effects?"
                      r"|abnormal (laboratory|lab|test|ecg|liver).*|intolerance( to .*)?|(drug|treatment) intolerance)"),
    ("DISEASE_PROGRESSION", rf"(?!.*{_DEATH_WORDS})(.*\bprogress(ion|ive|ed)\b.*|pd|symptomatic deterioration|clinical deterioration|deterioration of (disease|condition))"),
    ("LACK_OF_EFFICACY", r"(lack of (efficacy|effect|response|therapeutic (effect|response|efficacy)|improvement|benefit|clinical (benefit|response))"
                         r"|(insufficient|unsatisfactory|inadequate|poor) (therapeutic |clinical |treatment )?(effect|response|efficacy|benefit)"
                         r"|(treatment|therapeutic|therapy) failure|no (therapeutic |clinical )?(response|effect|improvement|benefit)|lack of efficacy.*|inefficacy)"),
    ("PRE_TREATMENT_EXCLUSION", r"(randomi[sz]ed (but|and) (not|never) (treated|dosed).*|not treated|never treated|not dosed|never dosed"
                                r"|did not receive (study |any )?(drug|treatment|intervention|study treatment|medication|dose).*"
                                r"|screen(ing)? fail(ure|ures|ed)?.*|ineligib(le|ility).*|not eligible.*|entry criteria not met"
                                r"|did not meet (the )?(entry|inclusion|eligibility|enrollment|study) criteria.*"
                                r"|(inclusion|exclusion|inclusion/exclusion|eligibility|entry) criteria not (met|fulfilled).*"
                                r"|randomi[sz]ed (in error|by mistake|erroneously).*|randomi[sz]ation error.*|mis[- ]?randomi[sz]ed.*)"),
    ("SPONSOR_ADMINISTRATIVE", r"((study|site|trial|center|centre|program|programme) (was )?(terminated|closed|stopped|discontinued|termination|closure).*"
                               r"|sponsor('s)? (decision|request|terminated).*|study termination.*|termination of (the )?(study|trial).*"
                               r"|administrative( problems?| reasons?| issues?| decision| withdrawal| discontinuation)?( by (the )?sponsor)?|sponsor decision.*|site closure.*|trial termination.*|study closed.*|study stopped.*)"),
    ("PROTOCOL_VIOLATION", r"((major |minor |significant )?protocol (violation|violations|deviation|deviations|non[- ]?compliance).*"
                           r"|(non|poor|lack of)[- ]?compliance.*|noncompliance.*|non[- ]?compliant.*|non[- ]?adheren(ce|t).*|poor adherence.*"
                           r"|did not comply.*|failure to comply.*)"),
    ("PHYSICIAN_DECISION", r"((physician|investigator|doctor|clinician|pi|investigators)('s|s')? (decision|discretion|judgement|judgment|request|choice)"
                           r"|decision (of|by) (the )?(physician|investigator|doctor)|(physician|investigator) decision.*)"),
    ("LOST_TO_FOLLOW_UP", r"(lost to (follow[- ]?up|followup|f/u).*|lost|ltfu|moved.*|relocat(ed|ion).*|migrat(ed|ion).*"
                          r"|missed (visit|visits|appointment|appointments|study visit).*|lost contact.*|unable to (contact|locate|reach).*)"),
    ("WITHDRAWAL_BY_SUBJECT", r"(withdrawal by (subject|participant|patient|parent|guardian|parent/guardian|legal guardian|subject/guardian).*"
                              r"|withdr[ae]w (his |her |their )?(informed )?consent.*|withdrawal of (informed )?consent.*|(informed )?consent withdrawn.*"
                              r"|(subject|participant|patient|parent|guardian)s?(/guardian|/parent)?('s|s')? (withdrew|withdrawal|decision|request|choice|withdrawn consent).*"
                              r"|personal reasons?|refused (further )?(treatment|participation|study|to continue).*|voluntary withdrawal.*|withdrew"
                              r"|(subject|participant|patient) (chose|decided|elected|wished) to (withdraw|discontinue|stop).*)"),
    ("OTHER_UNSPECIFIED", r"(others?|other reasons?|other,? not specified|other than specified( above)?|other miscellaneous reasons|miscellaneous( reasons?)?"
                          r"|missing|unclassified|unknown( reasons?)?|reason (not|un)(specified|known|given|reported|provided|recorded)|reasons? not (specified|given|reported|known|provided|recorded)"
                          r"|not specified|unspecified|pregnancy|pregnant|technical (problems?|issues?|difficulties)|alternative (therapy|treatment)s?"
                          r"|no reason( given| provided| specified)?|n/?a|other reason not specified)"),
]
_COMPILED = [(c, re.compile(p)) for c, p in RULES]


def normalize_key(s: str) -> str:
    s = (s or "").lower().replace("’", "'").replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s.strip(" .,:;*-_\"'")


_SPLIT = re.compile(r"\s*(?:,|/|;|\+|&|or|and)\s*")


def _match(key: str):
    for cat, rx in _COMPILED:
        if rx.fullmatch(key):
            return cat
    return None


def rule_category(key: str):
    """Rule category, or None if unmatched or if clauses map to conflicting categories."""
    cat = _match(key)
    if cat is None:
        return None
    parts = {_match(p) for p in _SPLIT.split(key) if len(p) > 2}
    parts.discard(None)
    if len(parts | {cat}) > 1:
        return None
    return cat


def eligible_pool_ids() -> set:
    t = pd.read_parquet(INTERIM / "protocol_trials.parquet")
    pool = t[(t.study_type == "INTERVENTIONAL") & (t.allocation == "RANDOMIZED") & (t.n_arms >= 2)
             & t.pool_has_results & t.pool_has_reasons & t.pool_has_control_arm]
    return set(pool.nct_id)


def reason_table() -> pd.DataFrame:
    """Raw reason strings in the pool with entry frequency and summed subjects."""
    dw = pd.read_parquet(INTERIM / "flow_dropwithdraws.parquet")
    dw = dw[dw.nct_id.isin(eligible_pool_ids())]
    freq = dw.drop_duplicates(["nct_id", "period_idx", "raw_reason"]).groupby("raw_reason").size()
    subj = dw.groupby("raw_reason")["n"].sum()
    r = pd.DataFrame({"frequency": freq, "subjects": subj}).reset_index()
    r["normalized_key"] = r["raw_reason"].map(normalize_key)
    return r


def key_table(r: pd.DataFrame) -> pd.DataFrame:
    k = r.groupby("normalized_key", as_index=False)[["frequency", "subjects"]].sum()
    k["rule"] = k["normalized_key"].map(rule_category)
    return k


def cmd_queue():
    r = reason_table()
    k = key_table(r)
    total = k.subjects.sum()
    covered = k.loc[k.rule.notna(), "subjects"].sum()
    print(f"raw strings: {len(r):,}; keys: {len(k):,}; rule keys: {k.rule.notna().sum():,}; "
          f"rule subject coverage: {covered / total:.4f}")
    un = k[k.rule.isna()].sort_values(["subjects", "frequency", "normalized_key"], ascending=[False, False, True])
    rows, cum = [], covered
    for _, x in un.iterrows():
        if cum / total >= TIER2_COVERAGE or len(rows) >= TIER2_MAX_KEYS:
            break
        cum += x.subjects
        rows.append({"rank": len(rows) + 1, "normalized_key": x.normalized_key,
                     "subjects": int(x.subjects), "frequency": int(x.frequency), "cum_coverage": cum / total})
    q = pd.DataFrame(rows)
    q.to_csv(QUEUE, index=False)
    print(f"Tier 2 queue: {len(q):,} keys -> cumulative coverage {q.cum_coverage.iloc[-1]:.4f}" if len(q) else "empty queue")


def embed(texts):
    from sentence_transformers import SentenceTransformer
    import torch
    torch.manual_seed(SEED)
    model = SentenceTransformer(KNN_MODEL, device="cpu")
    return model.encode(list(texts), batch_size=256, normalize_embeddings=True, show_progress_bar=True)


def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows, equal share from each of 4 frequency quartile strata."""
    if len(df) <= n:
        return df
    strata = pd.qcut(df["frequency"].rank(method="first"), 4, labels=False)
    per = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
    parts = []
    for s, m in zip(range(4), per):
        g = df[strata == s]
        parts.append(g.iloc[rng.choice(len(g), size=min(m, len(g)), replace=False)])
    return pd.concat(parts)


def cmd_finalize():
    r = reason_table()
    k = key_table(r)
    q = pd.read_csv(QUEUE)
    llm = pd.read_csv(LLM_LABELS)
    bad = set(llm.category) - set(CATEGORIES)
    if bad:
        sys.exit(f"Unknown categories in {LLM_LABELS}: {bad}")
    missing = set(q.normalized_key) - set(llm.normalized_key)
    if missing:
        sys.exit(f"{len(missing)} queued keys lack labels, e.g. {list(missing)[:5]}")
    llm = llm[llm.normalized_key.isin(set(q.normalized_key))].drop_duplicates("normalized_key")
    llm_map = dict(zip(llm.normalized_key, llm.category))

    k["category"] = k["rule"]
    k["method"] = np.where(k.rule.notna(), "rule", None)
    m2 = k.category.isna() & k.normalized_key.isin(llm_map)
    k.loc[m2, "category"] = k.loc[m2, "normalized_key"].map(llm_map)
    k.loc[m2, "method"] = "llm"

    labeled = k[k.category.notna()].reset_index(drop=True)
    rest = k[k.category.isna()].reset_index(drop=True)
    k["knn_similarity"] = np.nan
    if len(rest):
        E_lab = embed(labeled.normalized_key)
        E_rest = embed(rest.normalized_key)
        best_sim = np.empty(len(rest))
        best_idx = np.empty(len(rest), dtype=int)
        for i in range(0, len(rest), 2048):
            S = E_rest[i:i + 2048] @ E_lab.T
            best_idx[i:i + 2048] = S.argmax(1)
            best_sim[i:i + 2048] = S.max(1)
        cats = np.where(best_sim >= KNN_MIN_SIM, labeled.category.values[best_idx], "OTHER_UNSPECIFIED")
        idx = k.index[k.category.isna()]
        k.loc[idx, "category"] = cats
        k.loc[idx, "method"] = "knn"
        k.loc[idx, "knn_similarity"] = best_sim

    # Mapping at raw-string level.
    mapping = r.merge(k[["normalized_key", "category", "method"]], on="normalized_key", how="left")
    mapping = mapping[["raw_reason", "normalized_key", "category", "method", "frequency", "subjects"]]
    mapping = mapping.sort_values(["subjects", "frequency"], ascending=False)
    mapping.to_csv(ANNOT / "reason_mapping.csv", index=False)

    # Coverage per tier (by entries and by subjects).
    tot_f, tot_s = mapping.frequency.sum(), mapping.subjects.sum()
    rows = []
    for meth in ["rule", "llm", "knn"]:
        m = mapping[mapping.method == meth]
        rows.append({"tier": meth, "n_raw_strings": len(m), "n_keys": m.normalized_key.nunique(),
                     "entries": int(m.frequency.sum()), "entries_share": m.frequency.sum() / tot_f,
                     "subjects": int(m.subjects.sum()), "subjects_share": m.subjects.sum() / tot_s})
    cov = pd.DataFrame(rows)
    cov["entries_cum_share"] = cov.entries_share.cumsum()
    cov["subjects_cum_share"] = cov.subjects_share.cumsum()
    kn = k[k.method == "knn"]
    extra = [{"tier": "knn_below_threshold_to_OTHER", "n_raw_strings": int(mapping[(mapping.method == "knn")].normalized_key.isin(
                 kn.normalized_key[kn.knn_similarity < KNN_MIN_SIM]).sum()),
              "n_keys": int((kn.knn_similarity < KNN_MIN_SIM).sum()),
              "entries": int(kn.frequency[kn.knn_similarity < KNN_MIN_SIM].sum()),
              "entries_share": kn.frequency[kn.knn_similarity < KNN_MIN_SIM].sum() / tot_f,
              "subjects": int(kn.subjects[kn.knn_similarity < KNN_MIN_SIM].sum()),
              "subjects_share": kn.subjects[kn.knn_similarity < KNN_MIN_SIM].sum() / tot_s},
             {"tier": "total", "n_raw_strings": len(mapping), "n_keys": mapping.normalized_key.nunique(),
              "entries": int(tot_f), "entries_share": 1.0, "subjects": int(tot_s), "subjects_share": 1.0}]
    cov = pd.concat([cov, pd.DataFrame(extra)], ignore_index=True)
    cat_rows = mapping.groupby("category").agg(entries=("frequency", "sum"), subjects=("subjects", "sum")).reset_index()
    cat_rows["tier"] = "category:" + cat_rows.pop("category")
    cat_rows["entries_share"] = cat_rows.entries / tot_f
    cat_rows["subjects_share"] = cat_rows.subjects / tot_s
    cov = pd.concat([cov, cat_rows], ignore_index=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    cov.to_csv("results/tables/reason_coverage.csv", index=False)
    print(cov.to_string(index=False))

    # Validation sample: 100 raw strings per tier, frequency-stratified, no labels.
    vs_path = ANNOT / "validation_sample.csv"
    if vs_path.exists() and pd.read_csv(vs_path)["category"].notna().any():
        print(f"{vs_path} already contains manual labels; not regenerating it.")
        return
    rng = np.random.default_rng(SEED)
    parts = []
    for meth in ["rule", "llm", "knn"]:
        s = stratified_sample(mapping[mapping.method == meth].reset_index(drop=True), 100, rng)
        parts.append(s.assign(tier=meth))
    vs = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    vs.insert(0, "id", [f"V{i + 1:03d}" for i in range(len(vs))])
    vs[["id", "raw_reason"]].assign(category="").to_csv(ANNOT / "validation_sample.csv", index=False)
    # Key with tiers and predictions stays local so annotators stay blind.
    vs[["id", "raw_reason", "tier", "category", "frequency", "subjects"]].to_csv(
        INTERIM / "validation_sample_key.csv", index=False)
    print(f"validation sample: {len(vs)} strings ({vs.tier.value_counts().to_dict()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["queue", "finalize"])
    a = ap.parse_args()
    cmd_queue() if a.stage == "queue" else cmd_finalize()


if __name__ == "__main__":
    main()
