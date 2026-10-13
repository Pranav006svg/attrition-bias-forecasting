"""Feature matrices for the baselines (pre-enrollment fields only).

All vocabularies, category levels and the TF-IDF model are fit on the train
split only and then applied to val/test. Inputs are restricted to the
allowlisted protocol columns (TRIAL_FEATURES / ARM_FEATURES) on entry.

Trial features: categorical design fields, numeric fields, multi-hot MeSH
branches, TF-IDF (300) over eligibility criteria + brief summary.
Arm features (base): arm type, multi-hot intervention types, number of interventions.
Arm features (enriched=True) add:
  dose parsed from arm label -> intervention names -> description, normalized to mg
  (absolute mass), mg/kg, mg/m2 or IU; dose unit class; dose rank within trial
  (among arms sharing the trial's most common unit class); dose / trial max dose;
  intervention class (drug, biological, device, behavioral, procedure, placebo, other);
  combination-therapy flag; number of active (non-placebo) interventions;
  arm type x masking interaction.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.build_benchmark import ARM_FEATURES, TRIAL_FEATURES

TRIAL_CAT = ["phase", "allocation", "masking", "intervention_model", "primary_purpose",
             "sponsor_class", "sex", "healthy_volunteers"]
TRIAL_NUM = ["n_arms", "min_age_years", "max_age_years", "start_year"]
ARM_CAT = ["arm_type"]
TFIDF_MAX_FEATURES = 300

assert set(TRIAL_CAT + TRIAL_NUM) <= set(TRIAL_FEATURES)
assert set(ARM_CAT) <= set(ARM_FEATURES)

_NUM = r"(\d+(?:[.,]\d+)?)"
DOSE_RE = re.compile(_NUM + r"\s*(mg|mcg|µg|μg|ug|ng|g|iu|units?)\b(?:\s*/\s*(kg|m2|m²|m\^2|ml|l)\b)?", re.IGNORECASE)
MASS_TO_MG = {"mg": 1.0, "g": 1000.0, "mcg": 1e-3, "µg": 1e-3, "μg": 1e-3, "ug": 1e-3, "ng": 1e-6}
PLACEBO_RE = re.compile(r"\b(placebo|sham|vehicle|dummy|saline|matching tablet|matched tablet)\b", re.IGNORECASE)
COMBO_RE = re.compile(r"\s\+\s|\+|\bplus\b|\bcombination\b|\bcombined\b", re.IGNORECASE)
CLASS_PRECEDENCE = [("BIOLOGICAL", "BIOLOGICAL"), ("DRUG", "DRUG"), ("COMBINATION_PRODUCT", "DRUG"),
                    ("DEVICE", "DEVICE"), ("PROCEDURE", "PROCEDURE"), ("RADIATION", "PROCEDURE"),
                    ("BEHAVIORAL", "BEHAVIORAL")]
INTERVENTION_CLASSES = ["DRUG", "BIOLOGICAL", "DEVICE", "BEHAVIORAL", "PROCEDURE", "PLACEBO", "OTHER"]
DOSE_CLASSES = ["mass", "per_kg", "per_m2", "iu", "concentration", "none"]


def _as_cat_str(s: pd.Series) -> pd.Series:
    """Categorical values as plain strings; missing -> 'NA' level."""
    return s.astype(object).where(s.notna(), "NA").astype(str)


def _multi_hot(series: pd.Series, vocab: list, prefix: str) -> pd.DataFrame:
    sets = series.fillna("").map(lambda s: set(s.split("|")) if s else set())
    return pd.DataFrame({f"{prefix}{v}": sets.map(lambda x, v=v: int(v in x)) for v in vocab}, index=series.index)


def parse_dose(*texts):
    """First dose mention across texts (in priority order) -> (unit_class, value)."""
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        m = DOSE_RE.search(t)
        if not m:
            continue
        val = float(m.group(1).replace(",", "."))
        unit, per = m.group(2).lower(), (m.group(3) or "").lower()
        if unit in ("iu", "unit", "units"):
            return ("iu", val) if per in ("", "kg") else ("concentration", val)
        mg = val * MASS_TO_MG[unit]
        if per == "":
            return "mass", mg
        if per == "kg":
            return "per_kg", mg
        if per in ("m2", "m²", "m^2"):
            return "per_m2", mg
        return "concentration", mg
    return "none", np.nan


def _names(s):
    return [n for n in s.split("|") if n] if isinstance(s, str) else []


def intervention_class(arm_type, names, label):
    active = [n for n in names if not PLACEBO_RE.search(n)]
    if (names and not active) or (not names and arm_type in ("PLACEBO_COMPARATOR", "SHAM_COMPARATOR")) \
            or (not active and isinstance(label, str) and PLACEBO_RE.search(label)):
        return "PLACEBO"
    types = {n.split(":", 1)[0].strip().upper().replace(" ", "_") for n in active if ":" in n}
    for t, cls in CLASS_PRECEDENCE:
        if t in types:
            return cls
    return "OTHER"


def enriched_arm_features(a: pd.DataFrame) -> pd.DataFrame:
    """Per-arm enriched features from protocol arm text (a: nct_id + ARM_FEATURES)."""
    out = pd.DataFrame(index=a.index)
    doses = [parse_dose(r.arm_label, " ; ".join(_names(r.intervention_names)), r.arm_description)
             for r in a.itertuples()]
    out["dose_class"] = [d[0] for d in doses]
    val = np.array([d[1] for d in doses], dtype=float)
    for cls, col in [("mass", "dose_mg"), ("per_kg", "dose_mg_per_kg"), ("per_m2", "dose_mg_per_m2"), ("iu", "dose_iu")]:
        out[col] = np.where(out.dose_class == cls, val, np.nan)

    # Dose rank within trial among arms sharing the trial's most common dose class.
    tmp = pd.DataFrame({"nct_id": a.nct_id.values, "cls": out.dose_class.values, "val": val}, index=a.index)
    dosed = tmp[tmp.cls != "none"]
    modal = dosed.groupby("nct_id").cls.agg(lambda s: s.value_counts().index[0])
    same = dosed[dosed.cls.values == modal.reindex(dosed.nct_id).values]
    g = same.groupby("nct_id").val
    n_levels = g.transform("nunique")
    rank = g.rank(method="dense")
    out["dose_rank_pct"] = np.nan
    out.loc[same.index, "dose_rank_pct"] = np.where(n_levels > 1, (rank - 1) / (n_levels - 1).clip(lower=1), np.nan)
    out["dose_rel_trial_max"] = np.nan
    out.loc[same.index, "dose_rel_trial_max"] = same.val / g.transform("max").replace(0, np.nan)
    out["dose_is_trial_max"] = np.nan
    out.loc[same.index, "dose_is_trial_max"] = np.where(n_levels > 1, (same.val == g.transform("max")).astype(float), np.nan)
    out["n_dose_levels_trial"] = tmp.nct_id.map(same.groupby("nct_id").val.nunique()).fillna(0).values

    names = a.intervention_names.map(_names)
    out["intervention_class"] = [intervention_class(t, n, l) for t, n, l in zip(a.arm_type, names, a.arm_label)]
    out["n_active_interventions"] = names.map(lambda ns: sum(1 for n in ns if not PLACEBO_RE.search(n)))
    out["combination_therapy"] = ((out.n_active_interventions >= 2)
                                  | a.arm_label.fillna("").str.contains(COMBO_RE)).astype(int)
    return out


class FeatureBuilder:
    def __init__(self, enriched: bool = False):
        self.enriched = enriched

    @staticmethod
    def _restrict(df: pd.DataFrame, allowed) -> pd.DataFrame:
        """Keep only identifiers and allowlisted protocol columns."""
        keep = ["nct_id"] + [c for c in df.columns if c in allowed]
        return df[keep]

    def fit(self, trials_train: pd.DataFrame, arms_train: pd.DataFrame):
        t = self._restrict(trials_train, TRIAL_FEATURES)
        a = self._restrict(arms_train, ARM_FEATURES)
        self.cat_levels = {c: sorted(_as_cat_str(t[c]).unique()) for c in TRIAL_CAT}
        self.arm_levels = {c: sorted(_as_cat_str(a[c]).unique()) for c in ARM_CAT}
        self.mesh_vocab = sorted({b for s in t.mesh_branches for b in s.split("|")})
        self.itype_vocab = sorted({x for s in a.intervention_types.fillna("") for x in s.split("|") if x})
        self.tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True, min_df=5,
                                     stop_words="english", dtype=np.float32)
        self.tfidf.fit(self._text(t))
        if self.enriched:
            masking = a.nct_id.map(t.set_index("nct_id").masking)
            self.inter_levels = sorted((_as_cat_str(a.arm_type) + "|" + _as_cat_str(masking)).unique())
        return self

    @staticmethod
    def _text(t: pd.DataFrame) -> pd.Series:
        return t.eligibility_text.fillna("") + " " + t.brief_summary.fillna("")

    def trial_matrix(self, t: pd.DataFrame) -> pd.DataFrame:
        t = self._restrict(t, TRIAL_FEATURES)
        X = pd.DataFrame(index=t.index)
        for c in TRIAL_CAT:
            X[c] = pd.Categorical(_as_cat_str(t[c]), categories=self.cat_levels[c])
        for c in TRIAL_NUM:
            X[c] = t[c].astype(float)
        X = X.join(_multi_hot(t.mesh_branches, self.mesh_vocab, "mesh_"))
        tf = self.tfidf.transform(self._text(t)).toarray()
        X = X.join(pd.DataFrame(tf, index=t.index, columns=[f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]))
        X.insert(0, "nct_id", t.nct_id.values)
        return X

    def arm_matrix(self, a: pd.DataFrame, trial_X: pd.DataFrame) -> pd.DataFrame:
        a = self._restrict(a, ARM_FEATURES)
        X = pd.DataFrame(index=a.index)
        for c in ARM_CAT:
            X[c] = pd.Categorical(_as_cat_str(a[c]), categories=self.arm_levels[c])
        X = X.join(_multi_hot(a.intervention_types, self.itype_vocab, "itype_"))
        X["n_interventions"] = a.intervention_names.map(lambda s: len(_names(s)))
        if self.enriched:
            E = enriched_arm_features(a)
            E["dose_class"] = pd.Categorical(E.dose_class, categories=DOSE_CLASSES)
            E["intervention_class"] = pd.Categorical(E.intervention_class, categories=INTERVENTION_CLASSES)
            X = X.join(E)
        X["nct_id"] = a.nct_id.values
        X = X.merge(trial_X, on="nct_id", how="left").set_index(a.index)
        if self.enriched:
            inter = _as_cat_str(a.arm_type) + "|" + X.masking.astype(str)
            X["arm_type_x_masking"] = pd.Categorical(inter, categories=self.inter_levels)
        return X.drop(columns="nct_id")
